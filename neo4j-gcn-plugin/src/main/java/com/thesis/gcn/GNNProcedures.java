package com.thesis.gcn;

import org.neo4j.graphdb.*;
import org.neo4j.procedure.*;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Arrays;
import java.util.ArrayDeque;
import java.util.Queue;
import java.util.stream.Stream;

/**
 * Neo4j User Defined Procedure that performs server-side 1-hop mean aggregation
 * of neighbour feature vectors for each given seed node.
 *
 * <p>This is the "Option D hybrid" for GCN training: the expensive traversal and
 * running-sum step run inside the JVM against the native graph store.  Python then
 * receives one pre-aggregated vector per seed and applies the weight matrices with
 * full autograd intact.
 *
 * <h3>Feature formats supported</h3>
 * <ul>
 *   <li>{@code "f64[]"} — Neo4j {@code double[]} property (default ingestion format).</li>
 *   <li>{@code "byte[]"} — little-endian IEEE-754 float32 packed into a byte array
 *       (the compact ingestion format used in some experiments).</li>
 * </ul>
 *
 * <h3>Edge direction</h3>
 * The procedure follows <em>incoming</em> edges, i.e. it aggregates features of
 * nodes that point <em>to</em> the seed.  This matches the convention used by the
 * existing Python {@code Neo4jNeighborSampler}.
 *
 * <h3>Usage (Cypher)</h3>
 * <pre>{@code
 * CALL gnnProcedures.aggregation.neighbor.mean(
 *     $seed_ids,                -- List<Long>  application node IDs
 *     "id",                     -- property key that holds the node ID
 *     "embedding_bytes",        -- feature property key
 *     "byte[]",                 -- feature type: "byte[]" or "f64[]"
 *     "Paper",                  -- node label for the index lookup
 *     "CITES",                  -- relationship type; "" = any type
 *     10                        -- max neighbours (-1 = no limit)
 * ) YIELD nodeId, aggregatedFeatures
 * RETURN nodeId, aggregatedFeatures
 * }</pre>
 *
 * <p><b>Important:</b> For the index-based {@code findNode} lookup to work in
 * O(1), a uniqueness/index constraint on {@code (Label {nodeIdKey})} must exist.
 * Create it once per database:
 * <pre>{@code CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE }</pre>
 */
public class GNNProcedures {

    private static final String IN_DEGREE_PROPERTY = "in_degree";

    // -------------------------------------------------------------------------
    // Generic GNN inference infrastructure
    // -------------------------------------------------------------------------

    /**
     * A single aggregation primitive: graph-dependent, executed in Java.
     *
     * <p>Implementations receive the target node's ID, the IDs of its sampled
     * incoming neighbours, the current per-node feature map, and the stored
     * in-degree map.  They return the aggregated feature vector for the target
     * node (including any self-loop weighting), or {@code null} if no features
     * are available.
     */
    @FunctionalInterface
    private interface AggregationFn {
        double[] aggregate(Long nodeId,
                           List<Long> neighborIds,
                           Map<Long, double[]> features,
                           Map<Long, Long> inDegrees);
    }

    /**
     * Registry of named aggregation functions.
     *
     * <p>To support a new GNN type (e.g. GraphSAGE):
     * <ol>
     *   <li>Add an entry here mapping a string key to an {@link AggregationFn}.</li>
     *   <li>Add the matching entry to {@code CONV_AGGREGATION_MAP} in
     *       {@code create_inference_spec.py}.</li>
     *   <li>Export a new spec — no new procedure required.</li>
     * </ol>
     */
    private static final Map<String, AggregationFn> AGGREGATION_REGISTRY;
    static {
        Map<String, AggregationFn> r = new HashMap<>();

        // GCN-normalised aggregation with self-loop:
        //   self-weight  = 1 / (d̂_v)
        //   neighbour u  = 1 / sqrt(d̂_v · d̂_u)   where d̂ = in_degree + 1
        r.put("gcn_norm", (nodeId, neighborIds, features, inDegrees) -> {
            double[] selfFeat = features.get(nodeId);
            double   selfDeg  = inDegrees.getOrDefault(nodeId, 0L) + 1.0;
            double[] agg      = null;

            if (selfFeat != null) {
                agg = new double[selfFeat.length];
                accumulateScaledStatic(agg, selfFeat, 1.0 / selfDeg);
            }
            for (Long nbrId : neighborIds) {
                double[] nbrFeat = features.get(nbrId);
                if (nbrFeat == null) continue;
                if (agg == null) agg = new double[nbrFeat.length];
                double nbrDeg = inDegrees.getOrDefault(nbrId, 0L) + 1.0;
                accumulateScaledStatic(agg, nbrFeat, 1.0 / Math.sqrt(selfDeg * nbrDeg));
            }
            return agg;
        });

        // Mean aggregation (no self-loop, no normalisation):
        r.put("mean", (nodeId, neighborIds, features, inDegrees) -> {
            double[] sum = null;
            int count = 0;
            for (Long nbrId : neighborIds) {
                double[] nbrFeat = features.get(nbrId);
                if (nbrFeat == null) continue;
                if (sum == null) sum = new double[nbrFeat.length];
                accumulateScaledStatic(sum, nbrFeat, 1.0);
                count++;
            }
            if (sum == null || count == 0) return features.get(nodeId); // fall back to self
            for (int i = 0; i < sum.length; i++) sum[i] /= count;
            return sum;
        });

        AGGREGATION_REGISTRY = Collections.unmodifiableMap(r);
    }

    /** Static accumulate used inside lambda closures (lambdas cannot call instance methods). */
    private static void accumulateScaledStatic(double[] acc, double[] feat, double scale) {
        for (int i = 0; i < acc.length; i++) acc[i] += scale * feat[i];
    }

    // ─── Spec / weights caches (keyed by model-directory path) ───────────────

    /** Descriptor for one op in the execution plan. */
    private static class LayerSpec {
        final String op;         // "aggregate" | "linear" | "relu" | "tanh"
        final String method;     // aggregation method key  (aggregate ops only)
        final String weightKey;  // state-dict key for W    (linear ops only)
        final String biasKey;    // state-dict key for b    (linear ops only)

        LayerSpec(String op, String method, String weightKey, String biasKey) {
            this.op = op; this.method = method;
            this.weightKey = weightKey; this.biasKey = biasKey;
        }
    }

    /** Full model spec loaded from {@code spec.json}. */
    private static class GNNSpec {
        final int              numHops;
        final List<LayerSpec>  layers;

        GNNSpec(int numHops, List<LayerSpec> layers) {
            this.numHops = numHops;
            this.layers  = layers;
        }
    }

    /**
     * Flat tensor storage: shape + row-major float data.
     * Vectors (rank-1) and matrices (rank-2) are both stored here.
     */
    private static class Tensor {
        final int[]    shape;
        final double[] data;

        Tensor(int[] shape, double[] data) { this.shape = shape; this.data = data; }

        double[] asVector() { return data; }

        double[][] asMatrix() {
            int rows = shape[0], cols = shape[1];
            double[][] M = new double[rows][cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    M[i][j] = data[i * cols + j];
            return M;
        }
    }

    private static final ConcurrentHashMap<String, GNNSpec>              SPEC_CACHE    = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, Map<String, Tensor>>  WEIGHTS_CACHE = new ConcurrentHashMap<>();

    @Context
    public Transaction tx;

    // -------------------------------------------------------------------------
    // Result type
    // -------------------------------------------------------------------------

    public static class AggResult {
        /** Application-level node ID (same value that was passed in {@code seedIds}). */
        public final Long nodeId;
        /** Mean-aggregated feature vector across all sampled neighbours. */
        public final List<Double> aggregatedFeatures;
        /** Optional raw node label property value. */
        public final Object label;
        /** Optional raw feature vector from the seed node itself. */
        public final List<Double> nodeFeatures;

        public AggResult(Long nodeId, List<Double> aggregatedFeatures, Object label, List<Double> nodeFeatures) {
            this.nodeId = nodeId;
            this.aggregatedFeatures = aggregatedFeatures;
            this.label = label;
            this.nodeFeatures = nodeFeatures;
        }
    }

    /**
     * Result row for {@link #neighborSample}: one record containing full sampled
     * topology for a mini-batch.
     *
     * <p>Field names intentionally match the existing Python ETL contract:
     * {@code ordered_nodes} and {@code edge_pairs}.
     */
    public static class NeighborSampleResult {
        /** Global node IDs in encounter order: seeds first, then hop-1 new nodes, ... */
        public final List<Long> ordered_nodes;
        /** Sampled edges as [src_global_id, dst_global_id] pairs. */
        public final List<List<Long>> edge_pairs;
        /** Optional deepest-hop frontier node IDs in encounter order. */
        public final List<Long> frontier_nodes;
        /** Node IDs grouped by hop: index 0 = seeds, index 1 = hop-1 nodes, index 2 = hop-2 nodes, ... */
        public final List<List<Long>> nodes_by_hop;

        public NeighborSampleResult(List<Long> ordered_nodes, List<List<Long>> edge_pairs, List<Long> frontier_nodes, List<List<Long>> nodes_by_hop) {
            this.ordered_nodes = ordered_nodes;
            this.edge_pairs = edge_pairs;
            this.frontier_nodes = frontier_nodes;
            this.nodes_by_hop = nodes_by_hop;
        }
    }

    /**
     * Single-row batch result for sampled hybrid feature fetch.
     *
     * <p>Groups raw feature rows, label rows, and aggregated target rows so the
     * Python client receives one procedure row per mini-batch instead of one
     * row per node.
     */
    public static class SampledFetchBatchResult {
        public final List<Long> rawNodeIds;
        public final List<byte[]> rawNodeFeatures;
        public final List<Long> labelNodeIds;
        public final List<Object> labels;
        public final List<Long> targetNodeIds;
        public final List<byte[]> aggregatedFeatures;

        public SampledFetchBatchResult(
                List<Long> rawNodeIds,
            List<byte[]> rawNodeFeatures,
                List<Long> labelNodeIds,
                List<Object> labels,
                List<Long> targetNodeIds,
            List<byte[]> aggregatedFeatures
        ) {
            this.rawNodeIds = rawNodeIds;
            this.rawNodeFeatures = rawNodeFeatures;
            this.labelNodeIds = labelNodeIds;
            this.labels = labels;
            this.targetNodeIds = targetNodeIds;
            this.aggregatedFeatures = aggregatedFeatures;
        }
    }

    /**
     * Result row for {@link #gnnInfer}: one record per seed node.
     */
    public static class GNNInferenceResult {
        /** Application-level node ID. */
        public final Long nodeId;
        /** Argmax over the logit vector — the predicted class index. */
        public final Long predictedClass;
        /** Raw logit scores, one per class. */
        public final List<Double> logits;

        public GNNInferenceResult(Long nodeId, Long predictedClass, List<Double> logits) {
            this.nodeId = nodeId;
            this.predictedClass = predictedClass;
            this.logits = logits;
        }
    }

    /**
     * Result row for {@link #uploadModel}: one record confirming the upload.
     */
    public static class UploadModelResult {
        /** The resolved model directory path on the Neo4j server. */
        public final String modelDir;
        /** Number of weight tensors loaded. */
        public final Long numTensors;

        public UploadModelResult(String modelDir, Long numTensors) {
            this.modelDir = modelDir;
            this.numTensors = numTensors;
        }
    }

    // -------------------------------------------------------------------------
    // Procedure
    // -------------------------------------------------------------------------

    @Procedure(name = "gnnProcedures.sampling.neighbor.sample", mode = Mode.READ)
    @Description(
        "PyG-like homogeneous multi-hop neighbor sampling over incoming edges. "
        + "Returns one row with ordered_nodes and edge_pairs."
    )
    public Stream<NeighborSampleResult> neighborSample(
            @Name("seedIds") List<Long> seedIds,
            @Name("nodeIdKey") String nodeIdKey,
            @Name("nodeLabel") String nodeLabel,
            @Name("numNeighbors") List<Long> numNeighbors,
            @Name(value = "edgeType", defaultValue = "") String edgeType,
            @Name(value = "randomSeed", defaultValue = "42") Long randomSeed,
            @Name(value = "returnFrontier", defaultValue = "false") Boolean returnFrontier
    ) {
        if (seedIds == null || seedIds.isEmpty()) {
            return Stream.of(new NeighborSampleResult(new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>()));
        }
        if (nodeLabel == null || nodeLabel.isEmpty()) {
            throw new IllegalArgumentException("nodeLabel must be non-empty");
        }

        Label label = Label.label(nodeLabel);
        boolean hasEdgeType = edgeType != null && !edgeType.isEmpty();
        RelationshipType relType = hasEdgeType ? RelationshipType.withName(edgeType) : null;
        Random rng = new Random(randomSeed == null ? 42L : randomSeed);

        // Encounter-order node set mirrors pyg-lib Mapper insertion order.
        LinkedHashSet<Long> visited = new LinkedHashSet<>();
        List<Node> frontier = new ArrayList<>();

        // Seed initialization keeps user-provided order.
        for (Long seedId : seedIds) {
            Node seedNode = tx.findNode(label, nodeIdKey, seedId);
            if (seedNode == null) {
                continue;
            }
            Long globalSeedId = getNodeIdValue(seedNode, nodeIdKey);
            if (globalSeedId == null) {
                continue;
            }
            frontier.add(seedNode);
            visited.add(globalSeedId);
        }

        List<List<Long>> edgePairs = new ArrayList<>();
        List<Long> frontierNodeIds = new ArrayList<>();
        List<List<Long>> nodesByHop = new ArrayList<>();
        nodesByHop.add(new ArrayList<>(visited)); // hop 0 = seeds

        // Hop-wise expansion with per-hop fanout list.
        List<Long> fanouts = (numNeighbors == null) ? Collections.singletonList(-1L) : numNeighbors;
        for (Long kObj : fanouts) {
            long k = (kObj == null) ? -1L : kObj;
            if (frontier.isEmpty()) {
                break;
            }

            List<Node> nextFrontier = new ArrayList<>();
            LinkedHashSet<Long> nextFrontierIds = new LinkedHashSet<>();

            for (Node src : frontier) {
                Iterable<Relationship> rels = hasEdgeType
                        ? src.getRelationships(Direction.INCOMING, relType)
                        : src.getRelationships(Direction.INCOMING);

                List<Relationship> picked = sampleRelationships(rels, k, rng, label);

                Long srcId = getNodeIdValue(src, nodeIdKey);
                if (srcId == null) {
                    continue;
                }

                for (Relationship rel : picked) {
                    Node nbr = rel.getStartNode();
                    Long nbrId = getNodeIdValue(nbr, nodeIdKey);
                    if (nbrId == null) {
                        continue;
                    }

                    // Same orientation used by current Python ETL: [neighbor, seed/src].
                    List<Long> pair = new ArrayList<>(2);
                    pair.add(nbrId);
                    pair.add(srcId);
                    edgePairs.add(pair);

                    // New-node frontier dedup in first-encounter order.
                    if (!visited.contains(nbrId) && nextFrontierIds.add(nbrId)) {
                        nextFrontier.add(nbr);
                    }
                }
            }

            visited.addAll(nextFrontierIds);
            frontier = nextFrontier;
            frontierNodeIds = new ArrayList<>(nextFrontierIds);
            nodesByHop.add(frontierNodeIds);
        }

        return Stream.of(
                new NeighborSampleResult(
                        new ArrayList<>(visited),
                        edgePairs,
                        Boolean.TRUE.equals(returnFrontier) ? frontierNodeIds : new ArrayList<>(),
                        nodesByHop
                )
        );
    }

    @Procedure(name = "gnnProcedures.aggregation.neighbor.mean", mode = Mode.READ)
    @Description(
        "Mean-aggregate incoming neighbour feature vectors for each seed node. "
        + "Returns one row per seed with aggregated features and optional label/node features."
    )
    public Stream<AggResult> aggregateNeighbors(
            @Name("seedIds")                          List<Long> seedIds,
            @Name("nodeIdKey")                        String nodeIdKey,
            @Name("featureKey")                       String featureKey,
            @Name("featureType")                      String featureType,
            @Name("nodeLabel")                        String nodeLabel,
            @Name(value = "edgeType",       defaultValue = "") String edgeType,
            @Name(value = "maxNeighbors",   defaultValue = "-1") Long maxNeighbors,
            @Name(value = "targetKey",      defaultValue = "") String targetKey,
            @Name(value = "returnNode",     defaultValue = "false") Boolean returnNode,
            @Name(value = "returnLabel",    defaultValue = "false") Boolean returnLabel
    ) {
        boolean includeNode = Boolean.TRUE.equals(returnNode);
        boolean includeLabel = Boolean.TRUE.equals(returnLabel);
        if (includeLabel && (targetKey == null || targetKey.isEmpty())) {
            throw new IllegalArgumentException("targetKey must be non-empty when returnLabel=true");
        }

        Label label = Label.label(nodeLabel);
        boolean hasEdgeType = edgeType != null && !edgeType.isEmpty();
        RelationshipType relType = hasEdgeType ? RelationshipType.withName(edgeType) : null;

        // Fixed RNG seed so sampling is deterministic across calls with the
        // same inputs (matches pyg-lib's default behaviour).
        Random rng = new Random(42L);

        List<AggResult> results = new ArrayList<>(seedIds.size());

        for (Long seedId : seedIds) {
            Node seedNode = tx.findNode(label, nodeIdKey, seedId);
            if (seedNode == null) {
                // Emit a zero-length vector so downstream code detects the miss.
                results.add(new AggResult(seedId, new ArrayList<>(), null, null));
                continue;
            }

            // Sample neighbours via reservoir sampling (O(degree) time, O(k) space).
            Iterable<Relationship> rels = hasEdgeType
                    ? seedNode.getRelationships(Direction.INCOMING, relType)
                    : seedNode.getRelationships(Direction.INCOMING);

            List<Node> neighbours = reservoirSample(rels, maxNeighbors, rng);

            double[] agg = aggregateMean(neighbours, featureKey, featureType);

            if (agg == null) {
                // No neighbours had features — fall back to the seed's own features.
                agg = extractFeatures(seedNode, featureKey, featureType);
            }

            Object outLabel = includeLabel ? seedNode.getProperty(targetKey, null) : null;
            List<Double> outNodeFeatures = includeNode
                    ? toDoubleList(extractFeatures(seedNode, featureKey, featureType))
                    : null;

            results.add(new AggResult(seedId, toDoubleList(agg), outLabel, outNodeFeatures));
        }

        return results.stream();
    }

    @Procedure(name = "gnnProcedures.aggregation.neighbor.gcnNorm", mode = Mode.READ)
    @Description(
        "GCN-normalized 1-hop incoming aggregation for each seed node. "
        + "Adds a self-loop and applies symmetric degree normalization before returning one row per seed."
    )
    public Stream<AggResult> aggregateNeighborsGCNNorm(
            @Name("seedIds")                          List<Long> seedIds,
            @Name("nodeIdKey")                        String nodeIdKey,
            @Name("featureKey")                       String featureKey,
            @Name("featureType")                      String featureType,
            @Name("nodeLabel")                        String nodeLabel,
            @Name(value = "edgeType",       defaultValue = "") String edgeType,
            @Name(value = "maxNeighbors",   defaultValue = "-1") Long maxNeighbors,
            @Name(value = "targetKey",      defaultValue = "") String targetKey,
            @Name(value = "returnNode",     defaultValue = "false") Boolean returnNode,
            @Name(value = "returnLabel",    defaultValue = "false") Boolean returnLabel,
            @Name(value = "improved",       defaultValue = "false") Boolean improved
    ) {
        boolean includeNode = Boolean.TRUE.equals(returnNode);
        boolean includeLabel = Boolean.TRUE.equals(returnLabel);
        boolean useImproved = Boolean.TRUE.equals(improved);
        if (includeLabel && (targetKey == null || targetKey.isEmpty())) {
            throw new IllegalArgumentException("targetKey must be non-empty when returnLabel=true");
        }

        Label label = Label.label(nodeLabel);
        boolean hasEdgeType = edgeType != null && !edgeType.isEmpty();
        RelationshipType relType = hasEdgeType ? RelationshipType.withName(edgeType) : null;
        Random rng = new Random(42L);
        double selfLoopWeight = useImproved ? 2.0 : 1.0;

        List<AggResult> results = new ArrayList<>(seedIds.size());

        for (Long seedId : seedIds) {
            Node seedNode = tx.findNode(label, nodeIdKey, seedId);
            if (seedNode == null) {
                results.add(new AggResult(seedId, new ArrayList<>(), null, null));
                continue;
            }

            Iterable<Relationship> rels = hasEdgeType
                    ? seedNode.getRelationships(Direction.INCOMING, relType)
                    : seedNode.getRelationships(Direction.INCOMING);

            List<Node> neighbours = reservoirSample(rels, maxNeighbors, rng);
            double[] seedFeatures = extractFeatures(seedNode, featureKey, featureType);
            long seedDegree = getStoredIncomingDegreeOrCount(seedNode, relType, hasEdgeType, label);
            double seedDegreeHat = seedDegree + selfLoopWeight;

            double[] agg = null;

            if (seedFeatures != null) {
                agg = new double[seedFeatures.length];
                double selfWeight = selfLoopWeight / seedDegreeHat;
                accumulateScaled(agg, seedFeatures, selfWeight);
            }

            for (Node neighbour : neighbours) {
                double[] feat = extractFeatures(neighbour, featureKey, featureType);
                if (feat == null) {
                    continue;
                }

                if (agg == null) {
                    agg = new double[feat.length];
                }

                long neighborDegree = getStoredIncomingDegreeOrCount(neighbour, relType, hasEdgeType, label);
                double neighborDegreeHat = neighborDegree + selfLoopWeight;
                double weight = 1.0 / Math.sqrt(seedDegreeHat * neighborDegreeHat);
                accumulateScaled(agg, feat, weight);
            }

            Object outLabel = includeLabel ? seedNode.getProperty(targetKey, null) : null;
            List<Double> outNodeFeatures = includeNode
                    ? toDoubleList(seedFeatures)
                    : null;

            results.add(new AggResult(seedId, toDoubleList(agg), outLabel, outNodeFeatures));
        }

        return results.stream();
    }

    @Procedure(name = "gnnProcedures.aggregation.neighbor.sampledGcnNormFetchBatch", mode = Mode.READ)
    @Description(
        "Fetch raw sampled-node features, labels, and sampled GCN-normalized frontier aggregation in one batch row. "
        + "Returns grouped id/value payloads instead of one row per node."
    )
    public Stream<SampledFetchBatchResult> aggregateSampledNeighborsGCNNormFetchBatch(
            @Name("nodeIds")                            List<Long> nodeIds,
            @Name("rawNodeIds")                         List<Long> rawNodeIds,
            @Name("targetIds")                          List<Long> targetIds,
            @Name("edgePairs")                          List<List<Long>> edgePairs,
            @Name("frontierIds")                        List<Long> frontierIds,
            @Name("nodeIdKey")                          String nodeIdKey,
            @Name("featureKey")                         String featureKey,
            @Name("featureType")                        String featureType,
            @Name("nodeLabel")                          String nodeLabel,
            @Name(value = "targetKey", defaultValue = "") String targetKey,
            @Name(value = "returnLabel", defaultValue = "false") Boolean returnLabel,
            @Name(value = "improved", defaultValue = "false") Boolean improved
    ) {
        Label label = Label.label(nodeLabel);
        boolean includeLabel = Boolean.TRUE.equals(returnLabel) && targetKey != null && !targetKey.isEmpty();
        double selfLoopWeight = Boolean.TRUE.equals(improved) ? 2.0 : 1.0;
        List<Long> requestedIds = nodeIds == null ? Collections.emptyList() : nodeIds;
        List<Long> rawIds = rawNodeIds == null ? Collections.emptyList() : rawNodeIds;
        List<Long> targetIdsOrdered = targetIds == null ? Collections.emptyList() : targetIds;
        Set<Long> rawNodeSet = new HashSet<>(rawIds);
        Set<Long> targetSet = new HashSet<>(targetIdsOrdered);
        Set<Long> frontierSet = frontierIds == null ? Collections.emptySet() : new HashSet<>(frontierIds);

        Map<Long, List<Long>> incoming = new HashMap<>();
        Map<Long, Long> sampledInDegree = new HashMap<>();
        Set<Long> requiredNodeIds = new HashSet<>(requestedIds);
        if (edgePairs != null) {
            for (List<Long> pair : edgePairs) {
                if (pair == null || pair.size() < 2) {
                    continue;
                }
                Long srcId = pair.get(0);
                Long dstId = pair.get(1);
                if (srcId == null || dstId == null) {
                    continue;
                }
                if (!frontierSet.isEmpty() && !frontierSet.contains(srcId)) {
                    continue;
                }
                incoming.computeIfAbsent(dstId, ignored -> new ArrayList<>()).add(srcId);
                sampledInDegree.put(dstId, sampledInDegree.getOrDefault(dstId, 0L) + 1L);
                sampledInDegree.putIfAbsent(srcId, sampledInDegree.getOrDefault(srcId, 0L));
                requiredNodeIds.add(srcId);
            }
        }

        Map<Long, Node> nodesById = preloadNodesById(label, nodeIdKey, requiredNodeIds);

        List<Long> outRawIds = new ArrayList<>();
    List<byte[]> outRawFeatures = new ArrayList<>();
        List<Long> outLabelIds = new ArrayList<>();
        List<Object> outLabels = new ArrayList<>();
        List<Long> outTargetIds = new ArrayList<>();
    List<byte[]> outAggregated = new ArrayList<>();

        for (Long nodeId : requestedIds) {
            Node node = nodesById.get(nodeId);
            if (node == null) {
                continue;
            }

            double[] nodeFeatures = extractFeatures(node, featureKey, featureType);
            if (rawNodeSet.contains(nodeId) && nodeFeatures != null) {
                outRawIds.add(nodeId);
                outRawFeatures.add(extractFeaturesAsFloat32Bytes(node, featureKey, featureType));
            }

            if (includeLabel) {
                outLabelIds.add(nodeId);
                outLabels.add(node.getProperty(targetKey, null));
            }

            if (!targetSet.contains(nodeId)) {
                continue;
            }

            double targetDegreeHat = sampledInDegree.getOrDefault(nodeId, 0L) + selfLoopWeight;
            double[] agg = null;

            if (nodeFeatures != null) {
                agg = new double[nodeFeatures.length];
                double selfWeight = selfLoopWeight / targetDegreeHat;
                accumulateScaled(agg, nodeFeatures, selfWeight);
            }

            for (Long srcId : incoming.getOrDefault(nodeId, Collections.emptyList())) {
                Node srcNode = nodesById.get(srcId);
                if (srcNode == null) {
                    continue;
                }

                double[] srcFeatures = extractFeatures(srcNode, featureKey, featureType);
                if (srcFeatures == null) {
                    continue;
                }

                if (agg == null) {
                    agg = new double[srcFeatures.length];
                }

                double srcDegreeHat = sampledInDegree.getOrDefault(srcId, 0L) + selfLoopWeight;
                double weight = 1.0 / Math.sqrt(targetDegreeHat * srcDegreeHat);
                accumulateScaled(agg, srcFeatures, weight);
            }

            outTargetIds.add(nodeId);
            outAggregated.add(packFloat32Bytes(agg));
        }

        return Stream.of(new SampledFetchBatchResult(
                outRawIds,
                outRawFeatures,
                outLabelIds,
                outLabels,
                outTargetIds,
                outAggregated
        ));
    }

    // -------------------------------------------------------------------------
    // Generic GNN inference procedure
    // -------------------------------------------------------------------------

    /**
     * Run full GNN inference for any architecture whose message-passing layers
     * are registered in {@link #AGGREGATION_REGISTRY}.
     *
     * <h3>Model directory</h3>
     * Produce the two files with {@code src/create_inference_spec.py}:
     * <ul>
     *   <li>{@code spec.json}    — execution plan (architecture + layer sequence)</li>
     *   <li>{@code weights.bin}  — keyed binary tensor file</li>
     * </ul>
     *
     * <h3>Execution</h3>
     * <ol>
     *   <li>Count {@code aggregate} ops in the spec → {@code numHops}.</li>
     *   <li>Pre-fetch a {@code numHops}-hop subgraph from the seed nodes.</li>
     *   <li>Execute the layer sequence.  Each {@code aggregate} op folds one
     *       hop ring inward; subsequent {@code linear}/{@code relu} ops are
     *       applied only to the nodes that were just aggregated.</li>
     *   <li>Return the argmax prediction + logits for each seed node.</li>
     * </ol>
     *
     * <h3>Usage (Cypher)</h3>
     * <pre>{@code
     * CALL gnnProcedures.inference.run(
     *     $seed_ids,
     *     "id",                    -- nodeIdKey
     *     "embedding_bytes",       -- featureKey
     *     "byte[]",                -- featureType ("byte[]" or "f64[]")
     *     "Paper",                 -- nodeLabel
     *     "CITES",                 -- edgeType ("" = any type)
     *     "/path/to/model_dir/",   -- folder containing spec.json + weights.bin
     *     10                       -- maxNeighbors per hop (-1 = no limit)
     * ) YIELD nodeId, predictedClass, logits
     * }</pre>
     */
    @Procedure(name = "gnnProcedures.model.upload", mode = Mode.READ)
    @Description(
        "Upload a GNN model spec and weights from a remote client over the Bolt connection. "
        + "Parses both payloads, writes them to NEO4J_GNN_MODEL_DIR/<modelName>/ on the server, "
        + "and populates the in-memory cache so inference.run can use the model immediately."
    )
    public Stream<UploadModelResult> uploadModel(
            @Name("modelName")   String modelName,
            @Name("specJson")    String specJson,
            @Name("weightsBytes") byte[] weightsBytes
    ) {
        try {
            String modelDir = resolveModelDir(modelName);
            new java.io.File(modelDir).mkdirs();

            // Parse from in-memory bytes
            GNNSpec spec = parseSpecFromNode(new ObjectMapper().readTree(specJson));
            Map<String, Tensor> weights;
            try (java.io.ByteArrayInputStream bais = new java.io.ByteArrayInputStream(weightsBytes);
                 java.nio.channels.ReadableByteChannel ch = java.nio.channels.Channels.newChannel(bais)) {
                weights = parseWeightsFromChannel(ch);
            }

            // Write to disk for persistence across Neo4j restarts
            try (java.io.FileWriter fw = new java.io.FileWriter(new java.io.File(modelDir, "spec.json"))) {
                fw.write(specJson);
            }
            try (java.io.FileOutputStream fos = new java.io.FileOutputStream(new java.io.File(modelDir, "weights.bin"))) {
                fos.write(weightsBytes);
            }

            // Populate caches so inference.run works immediately without re-reading disk
            SPEC_CACHE.put(modelDir, spec);
            WEIGHTS_CACHE.put(modelDir, weights);

            return Stream.of(new UploadModelResult(modelDir, (long) weights.size()));
        } catch (IOException e) {
            throw new RuntimeException("Failed to upload model '" + modelName + "'.", e);
        }
    }

    @Procedure(name = "gnnProcedures.inference.run", mode = Mode.READ)
    @Description(
        "GNN inference for any architecture whose aggregation types are in "
        + "AGGREGATION_REGISTRY. Resolves modelName to $NEO4J_GNN_MODEL_DIR/<modelName>/."
    )
    public Stream<GNNInferenceResult> gnnInfer(
            @Name("seedIds")                                     List<Long> seedIds,
            @Name("nodeIdKey")                                   String nodeIdKey,
            @Name("featureKey")                                  String featureKey,
            @Name("featureType")                                 String featureType,
            @Name("nodeLabel")                                   String nodeLabel,
            @Name("modelName")                                   String modelName,
            @Name(value = "edgeType",        defaultValue = "")  String edgeType,
            @Name(value = "maxNeighbors",    defaultValue = "10") Long maxNeighbors
    ) {
        if (seedIds == null || seedIds.isEmpty()) return Stream.empty();
        try {
            String modelDir = resolveModelDir(modelName);
            return runEngine(
                    loadSpecCached(modelDir),
                    loadWeightsCached(modelDir),
                    seedIds, nodeIdKey, featureKey, featureType, nodeLabel, edgeType, maxNeighbors);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load model '" + modelName + "'.", e);
        }
    }

    /**
     * Resolve a model name to an absolute directory path using the
     * {@code NEO4J_GNN_MODEL_DIR} environment variable.
     *
     * <p>Set the variable once on the Neo4j server, e.g.:
     * <pre>  export NEO4J_GNN_MODEL_DIR=/var/lib/neo4j/gnn-models</pre>
     * Then {@code "my_gcn"} resolves to
     * {@code /var/lib/neo4j/gnn-models/my_gcn/}.
     */
    private static String resolveModelDir(String modelName) {
        // Check environment variable first, then JVM system property (-DNEO4J_GNN_MODEL_DIR=...)
        String base = System.getenv("NEO4J_GNN_MODEL_DIR");
        if (base == null || base.isBlank()) {
            base = System.getProperty("NEO4J_GNN_MODEL_DIR");
        }
        if (base == null || base.isBlank()) {
            throw new RuntimeException(
                    "NEO4J_GNN_MODEL_DIR is not set. Add to neo4j.conf:\n"
                    + "  server.jvm.additional=-DNEO4J_GNN_MODEL_DIR=/path/to/gnn_models");
        }
        return base + File.separator + modelName;
    }

    // ── Shared execution engine ───────────────────────────────────────────────

    /**
     * Core GNN inference engine. Runs the full layer plan against the live graph
     * using the current Neo4j transaction.
     */
    private Stream<GNNInferenceResult> runEngine(
            GNNSpec             spec,
            Map<String, Tensor> weights,
            List<Long>          seedIds,
            String              nodeIdKey,
            String              featureKey,
            String              featureType,
            String              nodeLabel,
            String              edgeType,
            long                maxNeighbors
    ) {
        int numHops = spec.numHops;

        Label            label       = Label.label(nodeLabel);
        boolean          hasEdgeType = edgeType != null && !edgeType.isEmpty();
        RelationshipType relType     = hasEdgeType ? RelationshipType.withName(edgeType) : null;
        Random           rng         = new Random(42L);

        // ── Step 1: Pre-fetch numHops-hop subgraph ────────────────────────────

        List<List<Long>>      hopNodes        = new ArrayList<>(numHops + 1);
        Map<Long, List<Long>> sampledIncoming = new HashMap<>();
        Map<Long, Node>       allNodes        = new LinkedHashMap<>();

        List<Long> seedLayer = new ArrayList<>();
        for (Long sid : seedIds) {
            Node n = tx.findNode(label, nodeIdKey, sid);
            if (n == null) continue;
            allNodes.put(sid, n);
            seedLayer.add(sid);
        }
        hopNodes.add(seedLayer);

        List<Node> frontier = new ArrayList<>(allNodes.values());
        for (int h = 0; h < numHops; h++) {
            List<Long> newLayer     = new ArrayList<>();
            List<Node> nextFrontier = new ArrayList<>();

            for (Node src : frontier) {
                Long srcId = getNodeIdValue(src, nodeIdKey);
                if (srcId == null) continue;

                Iterable<Relationship> rels = hasEdgeType
                        ? src.getRelationships(Direction.INCOMING, relType)
                        : src.getRelationships(Direction.INCOMING);

                List<Node> nbrs   = reservoirSample(rels, maxNeighbors, rng);
                List<Long> nbrIds = new ArrayList<>(nbrs.size());

                for (Node nbr : nbrs) {
                    Long nbrId = getNodeIdValue(nbr, nodeIdKey);
                    if (nbrId == null) continue;
                    nbrIds.add(nbrId);
                    if (!allNodes.containsKey(nbrId)) {
                        allNodes.put(nbrId, nbr);
                        newLayer.add(nbrId);
                        nextFrontier.add(nbr);
                    }
                }
                sampledIncoming.put(srcId, nbrIds);
            }

            hopNodes.add(newLayer);
            frontier = nextFrontier;
        }

        // ── Step 2: Extract raw features and in-degrees ───────────────────────

        Map<Long, double[]> currentH  = new HashMap<>(allNodes.size());
        Map<Long, Long>     inDegrees = new HashMap<>(allNodes.size());

        for (Map.Entry<Long, Node> e : allNodes.entrySet()) {
            Long nid  = e.getKey();
            Node node = e.getValue();
            double[] feat = extractFeatures(node, featureKey, featureType);
            if (feat != null) currentH.put(nid, feat);
            inDegrees.put(nid, getStoredIncomingDegreeOrCount(node, relType, hasEdgeType, label));
        }

        // ── Step 3: Execute the layer plan ────────────────────────────────────
        //
        // activeLevel  = deepest hop ring still being updated; decremented per aggregate.
        // lastAggLevel = ring depth used by linear/relu ops (floored at 0 so that
        //                pure-MLP specs with num_hops=0 still apply to seeds).

        int activeLevel  = numHops - 1;
        int lastAggLevel = Math.max(0, numHops - 1);

        for (LayerSpec layer : spec.layers) {

            if ("aggregate".equals(layer.op)) {
                AggregationFn fn = AGGREGATION_REGISTRY.get(layer.method);
                if (fn == null) throw new RuntimeException(
                        "Unknown aggregation '" + layer.method + "'. Add it to AGGREGATION_REGISTRY.");

                Map<Long, double[]> newH = new HashMap<>();
                for (int d = 0; d <= activeLevel; d++) {
                    for (Long nodeId : hopNodes.get(d)) {
                        List<Long> nbrs = sampledIncoming.getOrDefault(nodeId, Collections.emptyList());
                        double[] agg = fn.aggregate(nodeId, nbrs, currentH, inDegrees);
                        if (agg != null) newH.put(nodeId, agg);
                    }
                }
                currentH.putAll(newH);
                lastAggLevel = activeLevel;
                activeLevel--;

            } else if ("linear".equals(layer.op)) {
                double[][] W = weights.get(layer.weightKey).asMatrix();
                double[]   b = weights.get(layer.biasKey).asVector();
                for (int d = 0; d <= lastAggLevel; d++)
                    for (Long nid : hopNodes.get(d)) {
                        double[] h = currentH.get(nid);
                        if (h != null) currentH.put(nid, linearTransform(h, W, b));
                    }

            } else if ("relu".equals(layer.op)) {
                for (int d = 0; d <= lastAggLevel; d++)
                    for (Long nid : hopNodes.get(d)) {
                        double[] h = currentH.get(nid);
                        if (h != null) currentH.put(nid, relu(h));
                    }

            } else if ("tanh".equals(layer.op)) {
                for (int d = 0; d <= lastAggLevel; d++)
                    for (Long nid : hopNodes.get(d)) {
                        double[] h = currentH.get(nid);
                        if (h == null) continue;
                        double[] out = new double[h.length];
                        for (int i = 0; i < h.length; i++) out[i] = Math.tanh(h[i]);
                        currentH.put(nid, out);
                    }
            }
        }

        // ── Step 4: Emit predictions for seed nodes ───────────────────────────

        List<GNNInferenceResult> results = new ArrayList<>(seedIds.size());
        for (Long seedId : seedIds) {
            double[] logits = currentH.get(seedId);
            if (logits == null) continue;
            results.add(new GNNInferenceResult(seedId, argmax(logits), toDoubleList(logits)));
        }
        return results.stream();
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /** Read a numeric node ID property as Long. */
    private Long getNodeIdValue(Node node, String nodeIdKey) {
        Object id = node.getProperty(nodeIdKey, null);
        if (id instanceof Number) {
            return ((Number) id).longValue();
        }
        return null;
    }

    /**
     * Sample up to {@code k} relationships without replacement.
     *
     * <p>Applies optional neighbour label filtering before sampling so fanout is
     * computed over the same candidate set as the Cypher sampler.
     */
    private List<Relationship> sampleRelationships(
            Iterable<Relationship> rels,
            long k,
            Random rng,
            Label requiredNeighborLabel
    ) {
        List<Relationship> reservoir = new ArrayList<>();
        long seen = 0L;

        for (Relationship rel : rels) {
            Node nbr = rel.getStartNode();
            if (requiredNeighborLabel != null && !nbr.hasLabel(requiredNeighborLabel)) {
                continue;
            }

            seen++;

            if (k < 0) {
                reservoir.add(rel);
                continue;
            }

            int kk = (int) Math.min(k, Integer.MAX_VALUE);
            if (reservoir.size() < kk) {
                reservoir.add(rel);
            } else if (kk > 0) {
                long slot = (long) (rng.nextDouble() * seen);
                if (slot < kk) {
                    reservoir.set((int) slot, rel);
                }
            }
        }

        return reservoir;
    }

    /**
     * Reservoir-sample up to {@code k} neighbour nodes from a relationship iterable.
     * When {@code k < 0} all neighbours are collected.
     */
    private List<Node> reservoirSample(Iterable<Relationship> rels, long k, Random rng) {
        List<Node> reservoir = new ArrayList<>();
        long seen = 0L;

        for (Relationship rel : rels) {
            // INCOMING edge: the seed is endNode, the neighbour is startNode.
            Node neighbour = rel.getStartNode();
            seen++;

            if (k < 0 || reservoir.size() < (int) k) {
                reservoir.add(neighbour);
            } else {
                // Replace a random slot with probability k/seen.
                long slot = (long) (rng.nextDouble() * seen);
                if (slot < k) {
                    reservoir.set((int) slot, neighbour);
                }
            }
        }

        return reservoir;
    }

    /** Count incoming neighbours that match the same candidate set used by sampling. */
    private long countIncomingNeighbors(
            Node node,
            RelationshipType relType,
            boolean hasEdgeType,
            Label requiredNeighborLabel
    ) {
        long count = 0L;
        Iterable<Relationship> rels = hasEdgeType
                ? node.getRelationships(Direction.INCOMING, relType)
                : node.getRelationships(Direction.INCOMING);

        for (Relationship rel : rels) {
            Node nbr = rel.getStartNode();
            if (requiredNeighborLabel != null && !nbr.hasLabel(requiredNeighborLabel)) {
                continue;
            }
            count++;
        }

        return count;
    }

    /**
     * Use a persisted incoming degree when present, otherwise fall back to a live
     * traversal so the procedure remains backwards compatible.
     *
     * Assumes the stored property was computed over the same incoming edge set as
     * this procedure uses.
     */
    private long getStoredIncomingDegreeOrCount(
            Node node,
            RelationshipType relType,
            boolean hasEdgeType,
            Label requiredNeighborLabel
    ) {
        Object value = node.getProperty(IN_DEGREE_PROPERTY, null);
        if (value instanceof Number) {
            return ((Number) value).longValue();
        }
        return countIncomingNeighbors(node, relType, hasEdgeType, requiredNeighborLabel);
    }

    /** Preload all referenced nodes into a lookup map to avoid repeated indexed lookups in inner loops. */
    private Map<Long, Node> preloadNodesById(Label label, String nodeIdKey, Iterable<Long> nodeIds) {
        Map<Long, Node> nodesById = new HashMap<>();
        if (nodeIds == null) {
            return nodesById;
        }

        for (Long nodeId : nodeIds) {
            if (nodeId == null || nodesById.containsKey(nodeId)) {
                continue;
            }
            Node node = tx.findNode(label, nodeIdKey, nodeId);
            if (node != null) {
                nodesById.put(nodeId, node);
            }
        }
        return nodesById;
    }

    /** Add a scaled feature vector into an accumulator in place. */
    private void accumulateScaled(double[] acc, double[] feat, double scale) {
        for (int i = 0; i < acc.length; i++) {
            acc[i] += scale * feat[i];
        }
    }

    /**
     * Compute the element-wise mean of the feature vectors of all nodes in
     * {@code nodes}.  Returns {@code null} if the list is empty or no node
     * carries the requested property.
     */
    private double[] aggregateMean(List<Node> nodes, String featureKey, String featureType) {
        double[] sum = null;
        int count = 0;

        for (Node node : nodes) {
            double[] feat = extractFeatures(node, featureKey, featureType);
            if (feat == null) continue;

            if (sum == null) {
                sum = new double[feat.length];
            }
            for (int i = 0; i < feat.length; i++) {
                sum[i] += feat[i];
            }
            count++;
        }

        if (sum == null || count == 0) return null;

        for (int i = 0; i < sum.length; i++) {
            sum[i] /= count;
        }
        return sum;
    }

    /**
     * Extract the feature vector from a node property, handling both the
     * {@code "f64[]"} (double array) and {@code "byte[]"} (packed float32)
     * storage formats used in this codebase.
     *
     * @return a {@code double[]} of the feature values, or {@code null} if the
     *         property is absent.
     */
    private double[] extractFeatures(Node node, String featureKey, String featureType) {
        Object prop = node.getProperty(featureKey, null);
        if (prop == null) return null;

        if ("byte[]".equals(featureType)) {
            byte[] bytes = (byte[]) prop;
            int n = bytes.length / Float.BYTES;
            ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            double[] result = new double[n];
            for (int i = 0; i < n; i++) {
                result[i] = buf.getFloat();
            }
            return result;

        } else {
            // "f64[]" — Neo4j stores List<Double> properties as double[].
            double[] d = (double[]) prop;
            return d;
        }
    }

    /** Return node features as packed little-endian float32 bytes for transport efficiency. */
    private byte[] extractFeaturesAsFloat32Bytes(Node node, String featureKey, String featureType) {
        Object prop = node.getProperty(featureKey, null);
        if (prop == null) return null;

        if ("byte[]".equals(featureType)) {
            return (byte[]) prop;
        }
        return packFloat32Bytes((double[]) prop);
    }

    /** Pack a feature vector as little-endian float32 bytes for Bolt transport. */
    private byte[] packFloat32Bytes(double[] arr) {
        if (arr == null) return null;
        ByteBuffer buf = ByteBuffer.allocate(arr.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (double v : arr) {
            buf.putFloat((float) v);
        }
        return buf.array();
    }

    /** Box a primitive {@code double[]} to {@code List<Double>}. */
    private List<Double> toDoubleList(double[] arr) {
        if (arr == null) return new ArrayList<>(0);
        List<Double> list = new ArrayList<>(arr.length);
        for (double v : arr) list.add(v);
        return list;
    }

    // -------------------------------------------------------------------------
    // Inference math helpers
    // -------------------------------------------------------------------------

    /**
     * Affine transform: {@code out[i] = b[i] + Σ_j W[i][j] · in[j]}.
     *
     * <p>{@code W} has shape {@code [out_dim, in_dim]} — matching PyTorch's
     * convention ({@code weight.shape == [out_features, in_features]}).
     */
    private double[] linearTransform(double[] input, double[][] W, double[] b) {
        int outDim = W.length;
        double[] result = new double[outDim];
        for (int i = 0; i < outDim; i++) {
            result[i] = b[i];
            double[] row = W[i];
            for (int j = 0; j < input.length; j++) {
                result[i] += row[j] * input[j];
            }
        }
        return result;
    }

    /** Element-wise ReLU (returns a new array). */
    private double[] relu(double[] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) out[i] = x[i] > 0.0 ? x[i] : 0.0;
        return out;
    }

    /** Index of the largest value in {@code x}. */
    private long argmax(double[] x) {
        int best = 0;
        for (int i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
        return best;
    }

    // -------------------------------------------------------------------------
    // Spec + weights loading  (cached by model-directory path)
    // -------------------------------------------------------------------------

    private static GNNSpec loadSpecCached(String modelDir) throws IOException {
        GNNSpec cached = SPEC_CACHE.get(modelDir);
        if (cached != null) return cached;
        GNNSpec loaded = parseSpec(new File(modelDir, "spec.json"));
        GNNSpec prev   = SPEC_CACHE.putIfAbsent(modelDir, loaded);
        return prev != null ? prev : loaded;
    }

    private static Map<String, Tensor> loadWeightsCached(String modelDir) throws IOException {
        Map<String, Tensor> cached = WEIGHTS_CACHE.get(modelDir);
        if (cached != null) return cached;
        Map<String, Tensor> loaded = parseWeights(new File(modelDir, "weights.bin"));
        Map<String, Tensor> prev   = WEIGHTS_CACHE.putIfAbsent(modelDir, loaded);
        return prev != null ? prev : loaded;
    }

    // ── spec.json parsers ─────────────────────────────────────────────────────

    private static GNNSpec parseSpec(File specFile) throws IOException {
        return parseSpecFromNode(new ObjectMapper().readTree(specFile));
    }

    private static GNNSpec parseSpecFromNode(JsonNode root) {
        int numHops = root.path("num_hops").asInt(0);
        List<LayerSpec> layers = new ArrayList<>();
        for (JsonNode n : root.path("layers")) {
            layers.add(new LayerSpec(
                    n.path("op").asText(),
                    n.path("method").asText(null),
                    n.path("weight").asText(null),
                    n.path("bias").asText(null)));
        }
        return new GNNSpec(numHops, layers);
    }

    // ── weights.bin parsers ───────────────────────────────────────────────────
    //
    // Format (all little-endian):
    //   num_tensors : int32
    //   [repeated num_tensors times]
    //     key_length : int32
    //     key        : UTF-8 bytes[key_length]
    //     rank       : int32
    //     dims       : int32[rank]
    //     data       : float32[Π dims]   (row-major)

    private static Map<String, Tensor> parseWeights(File weightsFile) throws IOException {
        try (FileInputStream fis = new FileInputStream(weightsFile)) {
            return parseWeightsFromChannel(fis.getChannel());
        }
    }

    private static Map<String, Tensor> parseWeightsFromChannel(ReadableByteChannel ch)
            throws IOException {
        Map<String, Tensor> map = new HashMap<>();

        int numTensors = readInt(ch);
        for (int t = 0; t < numTensors; t++) {
            int    keyLen   = readInt(ch);
            byte[] keyBytes = readBytes(ch, keyLen);
            String key      = new String(keyBytes, java.nio.charset.StandardCharsets.UTF_8);

            int   rank  = readInt(ch);
            int[] dims  = new int[rank];
            int   total = 1;
            for (int d = 0; d < rank; d++) { dims[d] = readInt(ch); total *= dims[d]; }

            ByteBuffer buf = ByteBuffer.allocate(total * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            ch.read(buf);
            buf.flip();
            double[] data2 = new double[total];
            for (int i = 0; i < total; i++) data2[i] = buf.getFloat();

            map.put(key, new Tensor(dims, data2));
        }
        return map;
    }

    private static int readInt(ReadableByteChannel ch) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        ch.read(buf); buf.flip();
        return buf.getInt();
    }

    private static byte[] readBytes(ReadableByteChannel ch, int len) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(len);
        ch.read(buf);
        return buf.array();
    }

    // =========================================================================
    // METIS-style multilevel k-way graph partitioning
    // =========================================================================

    /**
     * Result row for {@link #metisPartition}.
     */
    public static class PartitionResult {
        /** Total number of nodes that received a partition assignment. */
        public final Long nodesPartitioned;
        /** Number of partitions (= numPartitions argument). */
        public final Long numPartitions;
        /** Edge cut of the final partition (each undirected edge counted once). */
        public final Long edgeCut;

        public PartitionResult(Long nodesPartitioned, Long numPartitions, Long edgeCut) {
            this.nodesPartitioned = nodesPartitioned;
            this.numPartitions    = numPartitions;
            this.edgeCut          = edgeCut;
        }
    }

    /**
     * Compressed-Sparse-Row graph used internally by the METIS algorithm.
     * Plain int[] arrays throughout to minimise GC pressure during coarsening.
     */
    private static final class MetisGraph {
        final int   n;       // number of vertices
        final int[] xadj;    // xadj[i]..xadj[i+1]-1 → neighbour range in adjncy/adjwgt
        final int[] adjncy;  // packed neighbour list
        final int[] adjwgt;  // edge weights (≥ 1)
        final int[] vwgt;    // vertex weights (≥ 1)

        MetisGraph(int n, int[] xadj, int[] adjncy, int[] adjwgt, int[] vwgt) {
            this.n = n; this.xadj = xadj; this.adjncy = adjncy;
            this.adjwgt = adjwgt; this.vwgt = vwgt;
        }

        int totalWeight() { int s = 0; for (int w : vwgt) s += w; return s; }
    }

    /**
     * Multilevel k-way graph partitioning (METIS-style).
     *
     * <p>Three phases:
     * <ol>
     *   <li><b>Coarsening</b> — Heavy-Edge Matching (HEM) repeatedly contracts the
     *       graph until it has ≤ max(40, 20k) nodes.</li>
     *   <li><b>Initial partition</b> — greedy BFS-based balanced k-way split on the
     *       coarsest graph.</li>
     *   <li><b>Uncoarsening + refinement</b> — partition is projected back to each
     *       finer level and improved with Fiduccia-Mattheyses (FM) boundary moves.</li>
     * </ol>
     *
     * <p>Writes an integer property ({@code rankIdProperty}, default {@code "rankId"})
     * in range {@code [0, numPartitions)} to every matched node.  Call this once before
     * training and use the property to shard seed nodes per worker.
     *
     * <h3>Example</h3>
     * <pre>{@code
     * CALL gnnProcedures.partition.metis(4, "Paper", "id", "CITES")
     * YIELD nodesPartitioned, numPartitions, edgeCut
     * }</pre>
     */
    @Procedure(name = "gnnProcedures.partition.metis", mode = Mode.WRITE)
    @Description(
        "Multilevel k-way graph partitioning (HEM coarsening + FM refinement). "
        + "Writes rankId ∈ [0, numPartitions) to every node."
    )
    public Stream<PartitionResult> metisPartition(
            @Name("numPartitions")                                     Long   numPartitions,
            @Name("nodeLabel")                                         String nodeLabel,
            @Name("nodeIdProperty")                                    String nodeIdProperty,
            @Name(value = "relType",        defaultValue = "")         String relType,
            @Name(value = "rankIdProperty", defaultValue = "rankId")   String rankIdProperty
    ) {
        int k = (int)(long) numPartitions;
        if (k < 2) throw new IllegalArgumentException("numPartitions must be >= 2");

        Label            lbl   = Label.label(nodeLabel);
        boolean          hasRt = relType != null && !relType.isEmpty();
        RelationshipType rt    = hasRt ? RelationshipType.withName(relType) : null;

        // ── 1. Assign contiguous indices to nodes ─────────────────────────────
        List<Long>        neoIds   = new ArrayList<>();
        Map<Long,Integer> idxOfNeo = new HashMap<>();

        try (ResourceIterator<Node> it = tx.findNodes(lbl)) {
            while (it.hasNext()) {
                Node   nd = it.next();
                Object r  = nd.getProperty(nodeIdProperty, null);
                if (r == null) continue;
                long nid = ((Number) r).longValue();
                if (idxOfNeo.putIfAbsent(nid, neoIds.size()) == null)
                    neoIds.add(nid);
            }
        }

        int n = neoIds.size();
        if (n == 0) return Stream.of(new PartitionResult(0L, (long) k, 0L));
        k = Math.min(k, n);   // can't have more partitions than nodes

        // ── 2. Build undirected adjacency ─────────────────────────────────────
        // Each undirected edge is processed once (u < v) and stored in both
        // directions in adjBuilder so the CSR is symmetric.
        @SuppressWarnings("unchecked")
        Map<Integer,Integer>[] adjBuilder = new HashMap[n];
        for (int i = 0; i < n; i++) adjBuilder[i] = new HashMap<>();

        try (ResourceIterator<Node> it = tx.findNodes(lbl)) {
            while (it.hasNext()) {
                Node   src    = it.next();
                Object rawSrc = src.getProperty(nodeIdProperty, null);
                if (rawSrc == null) continue;
                int u = idxOfNeo.get(((Number) rawSrc).longValue());

                Iterable<Relationship> rels = hasRt
                        ? src.getRelationships(rt)
                        : src.getRelationships();

                for (Relationship rel : rels) {
                    Node   dst    = rel.getOtherNode(src);
                    if (!dst.hasLabel(lbl)) continue;
                    Object rawDst = dst.getProperty(nodeIdProperty, null);
                    if (rawDst == null) continue;
                    Integer v = idxOfNeo.get(((Number) rawDst).longValue());
                    if (v == null || v == u) continue;
                    if (u >= v) continue;   // each undirected edge once
                    adjBuilder[u].merge(v, 1, Integer::sum);
                    adjBuilder[v].merge(u, 1, Integer::sum);
                }
            }
        }

        // ── 3. Convert to CSR ─────────────────────────────────────────────────
        MetisGraph graph = buildMetisCSR(n, adjBuilder);

        // ── 4. Multilevel k-way partition ─────────────────────────────────────
        int[] part = multilevelKwayPartition(graph, k, new Random(42L));

        // ── 5. Write rankId back to Neo4j ─────────────────────────────────────
        long edgeCut = computeMetisEdgeCut(graph, part);

        try (ResourceIterator<Node> it = tx.findNodes(lbl)) {
            while (it.hasNext()) {
                Node   nd  = it.next();
                Object raw = nd.getProperty(nodeIdProperty, null);
                if (raw == null) continue;
                Integer idx = idxOfNeo.get(((Number) raw).longValue());
                if (idx == null) continue;
                nd.setProperty(rankIdProperty, part[idx]);
            }
        }

        return Stream.of(new PartitionResult((long) n, (long) k, edgeCut));
    }

    // ── METIS: graph construction helpers ────────────────────────────────────

    private static MetisGraph buildMetisCSR(int n, Map<Integer,Integer>[] adj) {
        int[] xadj  = new int[n + 1];
        for (int i = 0; i < n; i++) xadj[i + 1] = xadj[i] + adj[i].size();
        int   m      = xadj[n];
        int[] adjncy = new int[m];
        int[] adjwgt = new int[m];
        int[] vwgt   = new int[n];
        Arrays.fill(vwgt, 1);
        for (int i = 0; i < n; i++) {
            int pos = xadj[i];
            for (Map.Entry<Integer,Integer> e : adj[i].entrySet()) {
                adjncy[pos] = e.getKey();
                adjwgt[pos] = e.getValue();
                pos++;
            }
        }
        return new MetisGraph(n, xadj, adjncy, adjwgt, vwgt);
    }

    private static long computeMetisEdgeCut(MetisGraph g, int[] part) {
        long cut = 0;
        for (int u = 0; u < g.n; u++)
            for (int j = g.xadj[u]; j < g.xadj[u + 1]; j++)
                if (part[u] != part[g.adjncy[j]]) cut += g.adjwgt[j];
        return cut / 2;   // each undirected edge counted from both sides
    }

    // ── METIS: multilevel orchestration ──────────────────────────────────────

    private static int[] multilevelKwayPartition(MetisGraph g, int k, Random rng) {
        // Coarsening phase: stack of (cmap, finer graph) pairs
        List<int[]>      cmapStack = new ArrayList<>();
        List<MetisGraph> fineStack = new ArrayList<>();
        MetisGraph cur = g;

        while (cur.n > Math.max(20 * k, 40)) {
            int prevN = cur.n;
            int[] cmap = hemCoarsen(cur, rng);
            int cn = 0;
            for (int c : cmap) if (c + 1 > cn) cn = c + 1;
            if (cn >= prevN) break;   // no further coarsening possible
            MetisGraph coarse = buildCoarseMetisGraph(cur, cmap, cn);
            cmapStack.add(cmap);
            fineStack.add(cur);
            cur = coarse;
        }

        // Initial k-way partition on the coarsest graph
        int[] part = bfsKwayPartition(cur, k);

        // Uncoarsening + refinement
        for (int lvl = cmapStack.size() - 1; lvl >= 0; lvl--) {
            int[]      cmap  = cmapStack.get(lvl);
            MetisGraph finer = fineStack.get(lvl);
            int[]      fp    = new int[finer.n];
            for (int u = 0; u < finer.n; u++) fp[u] = part[cmap[u]];
            fmRefine(finer, fp, k);
            part = fp;
        }
        return part;
    }

    // ── METIS: Heavy-Edge Matching coarsening ─────────────────────────────────

    /**
     * Returns {@code cmap[fine_node] = coarse_node} (0-based).
     *
     * <p>Processes nodes in random order.  For each unmatched node u, picks the
     * unmatched neighbour v with the highest edge weight (heavy-edge criterion).
     * Unmatched nodes are self-matched.
     */
    private static int[] hemCoarsen(MetisGraph g, Random rng) {
        int   n     = g.n;
        int[] match = new int[n];
        Arrays.fill(match, -1);

        // Fisher-Yates shuffle for random processing order
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1), tmp = order[i];
            order[i] = order[j]; order[j] = tmp;
        }

        int[] cmap = new int[n];
        int   cn   = 0;

        for (int u : order) {
            if (match[u] != -1) continue;

            // Heaviest unmatched neighbour
            int bestV = -1, bestW = -1;
            for (int j = g.xadj[u]; j < g.xadj[u + 1]; j++) {
                int v = g.adjncy[j], w = g.adjwgt[j];
                if (match[v] == -1 && w > bestW) { bestV = v; bestW = w; }
            }

            if (bestV >= 0) {
                match[u] = bestV; match[bestV] = u;
                cmap[u] = cmap[bestV] = cn++;
            } else {
                match[u] = u;
                cmap[u]  = cn++;
            }
        }
        return cmap;
    }

    /**
     * Build the coarsened graph from a fine graph and its HEM coarsening map.
     *
     * <p>Coarse vertex weight = sum of fine vertex weights in the matched pair.
     * Coarse edge weight = sum of fine edge weights between the two coarse vertices.
     * Internal edges (both endpoints map to the same coarse vertex) are dropped.
     */
    private static MetisGraph buildCoarseMetisGraph(MetisGraph g, int[] cmap, int cn) {
        int[] cvwgt = new int[cn];
        for (int u = 0; u < g.n; u++) cvwgt[cmap[u]] += g.vwgt[u];

        // Aggregate cross-partition edge weights; process each undirected edge once.
        // Key encodes the sorted coarse node pair: lo * cn + hi.
        Map<Long,Integer> edgeMap = new HashMap<>();
        for (int u = 0; u < g.n; u++) {
            int cu = cmap[u];
            for (int j = g.xadj[u]; j < g.xadj[u + 1]; j++) {
                int v = g.adjncy[j];
                if (u >= v) continue;          // each fine edge once
                int cv = cmap[v];
                if (cu == cv) continue;        // absorbed into coarse node
                int lo = Math.min(cu, cv), hi = Math.max(cu, cv);
                edgeMap.merge((long) lo * cn + hi, g.adjwgt[j], Integer::sum);
            }
        }

        // Build coarse CSR
        int[] deg = new int[cn];
        for (long key : edgeMap.keySet()) {
            int cu = (int)(key / cn), cv = (int)(key % cn);
            deg[cu]++; deg[cv]++;
        }
        int[] xadj = new int[cn + 1];
        for (int i = 0; i < cn; i++) xadj[i + 1] = xadj[i] + deg[i];

        int[] adjncy = new int[xadj[cn]];
        int[] adjwgt = new int[xadj[cn]];
        int[] pos    = Arrays.copyOf(xadj, cn);

        for (Map.Entry<Long,Integer> e : edgeMap.entrySet()) {
            long key = e.getKey(); int w = e.getValue();
            int  cu  = (int)(key / cn), cv = (int)(key % cn);
            adjncy[pos[cu]] = cv; adjwgt[pos[cu]] = w; pos[cu]++;
            adjncy[pos[cv]] = cu; adjwgt[pos[cv]] = w; pos[cv]++;
        }

        return new MetisGraph(cn, xadj, adjncy, adjwgt, cvwgt);
    }

    // ── METIS: initial BFS-based k-way partition ──────────────────────────────

    /**
     * Greedy balanced k-way partition via simultaneous BFS from k evenly-spaced seeds.
     *
     * <p>At each step, expands the lightest non-empty BFS queue, subject to a balance
     * guard that prevents any partition from exceeding 2× the target weight.
     * Disconnected components are handled by re-seeding into the lightest partition.
     */
    @SuppressWarnings("unchecked")
    private static int[] bfsKwayPartition(MetisGraph g, int k) {
        int   n      = g.n;
        int[] part   = new int[n];
        int[] partW  = new int[k];
        Arrays.fill(part, -1);

        int totalW  = g.totalWeight();
        int targetW = (totalW + k - 1) / k;

        Queue<Integer>[] queues = new ArrayDeque[k];
        int assigned = 0;
        for (int p = 0; p < k; p++) {
            queues[p] = new ArrayDeque<>();
            int seed = (int)((long) p * n / k);
            if (part[seed] == -1) {
                part[seed] = p; partW[p] += g.vwgt[seed];
                queues[p].add(seed); assigned++;
            }
        }

        while (assigned < n) {
            // Expand the lightest partition with a non-empty queue
            int chosen = -1;
            for (int p = 0; p < k; p++) {
                if (!queues[p].isEmpty() &&
                        (chosen == -1 || partW[p] < partW[chosen])) chosen = p;
            }

            if (chosen == -1) {
                // All queues empty (disconnected graph): restart from an unassigned node
                int minP = 0;
                for (int p = 1; p < k; p++) if (partW[p] < partW[minP]) minP = p;
                for (int u = 0; u < n; u++) {
                    if (part[u] == -1) {
                        part[u] = minP; partW[minP] += g.vwgt[u];
                        queues[minP].add(u); assigned++; break;
                    }
                }
                continue;
            }

            int u = queues[chosen].poll();
            for (int j = g.xadj[u]; j < g.xadj[u + 1]; j++) {
                int v = g.adjncy[j];
                if (part[v] != -1) continue;
                if (partW[chosen] + g.vwgt[v] > 2 * targetW) continue;
                part[v] = chosen; partW[chosen] += g.vwgt[v];
                queues[chosen].add(v); assigned++;
            }
        }
        return part;
    }

    // ── METIS: Fiduccia-Mattheyses boundary refinement ───────────────────────

    /**
     * FM-style refinement: sweep all nodes and greedily move each boundary node to
     * the adjacent partition that gives the largest positive gain, subject to a 5%
     * balance tolerance.  Repeats until no improvement or {@code MAX_PASSES} reached.
     *
     * <p>Gain of moving node {@code u} from partition {@code p} to {@code q}:
     * <pre>  gain = edgeWeightsTo(q) − edgeWeightsTo(p)</pre>
     */
    private static void fmRefine(MetisGraph g, int[] part, int k) {
        int   n        = g.n;
        int   totalW   = g.totalWeight();
        int   maxPartW = (int)(1.05 * totalW / k);   // 5% imbalance tolerance
        int[] partW    = new int[k];
        for (int u = 0; u < n; u++) partW[part[u]] += g.vwgt[u];

        final int MAX_PASSES = 10;
        for (int pass = 0; pass < MAX_PASSES; pass++) {
            boolean improved = false;

            for (int u = 0; u < n; u++) {
                int p = part[u];

                // Tally edge weight from u to each partition
                int[] edgesTo = new int[k];
                for (int j = g.xadj[u]; j < g.xadj[u + 1]; j++)
                    edgesTo[part[g.adjncy[j]]] += g.adjwgt[j];

                int selfEdges = edgesTo[p];

                // Best destination: highest gain within balance constraints
                int bestQ = -1, bestGain = 0;
                for (int q = 0; q < k; q++) {
                    if (q == p) continue;
                    if (partW[q] + g.vwgt[u] > maxPartW) continue;
                    if (partW[p] - g.vwgt[u] < 0)        continue;
                    int gain = edgesTo[q] - selfEdges;
                    if (gain > bestGain) { bestGain = gain; bestQ = q; }
                }

                if (bestQ >= 0) {
                    partW[p]     -= g.vwgt[u];
                    partW[bestQ] += g.vwgt[u];
                    part[u]       = bestQ;
                    improved      = true;
                }
            }
            if (!improved) break;
        }
    }
}
