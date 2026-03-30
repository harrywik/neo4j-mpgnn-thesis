package com.thesis.gcn;

import org.neo4j.graphdb.*;
import org.neo4j.procedure.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
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
public class NeighborAggregation {

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
     * Result row for {@link #signAggregate}: one row per (seed, hop) pair.
     *
     * <p>The Python sampler groups rows by {@code nodeId} and concatenates
     * {@code aggregatedFeatures} across hops to build the SIGN input tensor:
     * {@code [x_0 || x_1 || … || x_k]}.
     */
    public static class SIGNResult {
        /** Application-level node ID. */
        public final Long nodeId;
        /** Hop distance: 0 = seed's own features, 1 = 1-hop mean, 2 = 2-hop mean, … */
        public final Long hop;
        /** Mean-aggregated feature vector for nodes at this hop distance. */
        public final List<Double> aggregatedFeatures;

        public SIGNResult(Long nodeId, Long hop, List<Double> aggregatedFeatures) {
            this.nodeId = nodeId;
            this.hop = hop;
            this.aggregatedFeatures = aggregatedFeatures;
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

        public NeighborSampleResult(List<Long> ordered_nodes, List<List<Long>> edge_pairs) {
            this.ordered_nodes = ordered_nodes;
            this.edge_pairs = edge_pairs;
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
            @Name(value = "randomSeed", defaultValue = "42") Long randomSeed
    ) {
        if (seedIds == null || seedIds.isEmpty()) {
            return Stream.of(new NeighborSampleResult(new ArrayList<>(), new ArrayList<>()));
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
        }

        return Stream.of(new NeighborSampleResult(new ArrayList<>(visited), edgePairs));
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
            long seedDegree = countIncomingNeighbors(seedNode, relType, hasEdgeType, label);
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

                long neighborDegree = countIncomingNeighbors(neighbour, relType, hasEdgeType, label);
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

    @Procedure(name = "gnnProcedures.aggregation.neighbor.sampledGcnNorm", mode = Mode.READ)
    @Description(
        "GCN-normalized aggregation over an already-sampled subgraph. "
        + "Uses the provided sampled edge pairs instead of traversing the graph again, and returns one row per target node."
    )
    public Stream<AggResult> aggregateSampledNeighborsGCNNorm(
            @Name("targetIds")                          List<Long> targetIds,
            @Name("edgePairs")                          List<List<Long>> edgePairs,
            @Name("nodeIdKey")                          String nodeIdKey,
            @Name("featureKey")                         String featureKey,
            @Name("featureType")                        String featureType,
            @Name("nodeLabel")                          String nodeLabel,
            @Name(value = "improved", defaultValue = "false") Boolean improved
    ) {
        Label label = Label.label(nodeLabel);
        double selfLoopWeight = Boolean.TRUE.equals(improved) ? 2.0 : 1.0;

        Map<Long, List<Long>> incoming = new HashMap<>();
        Map<Long, Long> sampledInDegree = new HashMap<>();

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
                incoming.computeIfAbsent(dstId, ignored -> new ArrayList<>()).add(srcId);
                sampledInDegree.put(dstId, sampledInDegree.getOrDefault(dstId, 0L) + 1L);
                sampledInDegree.putIfAbsent(srcId, sampledInDegree.getOrDefault(srcId, 0L));
            }
        }

        List<AggResult> results = new ArrayList<>(targetIds.size());

        for (Long targetId : targetIds) {
            Node targetNode = tx.findNode(label, nodeIdKey, targetId);
            if (targetNode == null) {
                results.add(new AggResult(targetId, new ArrayList<>(), null, null));
                continue;
            }

            double[] targetFeatures = extractFeatures(targetNode, featureKey, featureType);
            double targetDegreeHat = sampledInDegree.getOrDefault(targetId, 0L) + selfLoopWeight;
            double[] agg = null;

            if (targetFeatures != null) {
                agg = new double[targetFeatures.length];
                double selfWeight = selfLoopWeight / targetDegreeHat;
                accumulateScaled(agg, targetFeatures, selfWeight);
            }

            for (Long srcId : incoming.getOrDefault(targetId, Collections.emptyList())) {
                Node srcNode = tx.findNode(label, nodeIdKey, srcId);
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

            results.add(new AggResult(targetId, toDoubleList(agg), null, null));
        }

        return results.stream();
    }

    // -------------------------------------------------------------------------
    // SIGN procedure
    // -------------------------------------------------------------------------

    /**
     * SIGN-style multi-hop aggregation.
     *
     * <p>Returns {@code hops + 1} rows per seed node:
     * <ul>
     *   <li>hop 0 — seed's own feature vector (no aggregation).</li>
     *   <li>hop 1 — mean of 1-hop incoming neighbours' features.</li>
     *   <li>hop k — mean of features at the k-hop shell (BFS frontier, new nodes only).</li>
     * </ul>
     *
     * <p>The Python sampler groups rows by {@code nodeId}, sorts by {@code hop}, then
     * concatenates to build the SIGN input {@code [x_0 || x_1 || … || x_k]}.
     *
     * <h3>Usage</h3>
     * <pre>{@code
    * CALL gnnProcedures.aggregation.sign.multiHop(
     *     $seed_ids, "id", "embedding_bytes", "byte[]", "Paper", "CITES", 2, 10
     * ) YIELD nodeId, hop, aggregatedFeatures
     * RETURN nodeId, hop, aggregatedFeatures ORDER BY nodeId, hop
     * }</pre>
     */
    @Procedure(name = "gnnProcedures.aggregation.sign.multiHop", mode = Mode.READ)
    @Description(
        "SIGN multi-hop aggregation. Returns one row per (seed, hop) with the mean-aggregated "
        + "feature vector for nodes at that hop distance. hop=0 is the seed's own features."
    )
    public Stream<SIGNResult> signAggregate(
            @Name("seedIds")                              List<Long> seedIds,
            @Name("nodeIdKey")                            String nodeIdKey,
            @Name("featureKey")                           String featureKey,
            @Name("featureType")                          String featureType,
            @Name("nodeLabel")                            String nodeLabel,
            @Name(value = "edgeType",            defaultValue = "") String edgeType,
            @Name(value = "hops",                defaultValue = "2") Long hops,
            @Name(value = "maxNeighborsPerHop",  defaultValue = "10") Long maxNeighborsPerHop
    ) {
        Label label = Label.label(nodeLabel);
        boolean hasEdgeType = edgeType != null && !edgeType.isEmpty();
        RelationshipType relType = hasEdgeType ? RelationshipType.withName(edgeType) : null;
        Random rng = new Random(42L);
        int k = (int) (long) hops;

        List<SIGNResult> results = new ArrayList<>(seedIds.size() * (k + 1));

        for (Long seedId : seedIds) {
            Node seedNode = tx.findNode(label, nodeIdKey, seedId);
            if (seedNode == null) continue;

            // hop 0: seed's own features (no aggregation).
            double[] ownFeat = extractFeatures(seedNode, featureKey, featureType);
            int featLen = ownFeat != null ? ownFeat.length : 0;
            results.add(new SIGNResult(seedId, 0L, toDoubleList(ownFeat)));

            // BFS: track which nodes have been seen (by element ID to avoid revisiting).
            Set<String> visited = new HashSet<>();
            visited.add(seedNode.getElementId());
            List<Node> frontier = Collections.singletonList(seedNode);

            for (int h = 1; h <= k; h++) {
                // Expand frontier: collect new nodes at this hop level.
                List<Node> shell = new ArrayList<>();
                Set<String> nextVisited = new HashSet<>(visited);

                for (Node node : frontier) {
                    Iterable<Relationship> rels = hasEdgeType
                            ? node.getRelationships(Direction.INCOMING, relType)
                            : node.getRelationships(Direction.INCOMING);
                    List<Node> sampled = reservoirSample(rels, maxNeighborsPerHop, rng);
                    for (Node nbr : sampled) {
                        String eid = nbr.getElementId();
                        if (!nextVisited.contains(eid)) {
                            nextVisited.add(eid);
                            shell.add(nbr);
                        }
                    }
                }

                // Aggregate mean over the new shell; emit zero vector if shell is empty.
                double[] agg = aggregateMean(shell, featureKey, featureType);
                if (agg == null) agg = new double[featLen];
                results.add(new SIGNResult(seedId, (long) h, toDoubleList(agg)));

                visited = nextVisited;
                frontier = shell;
                // If frontier is empty, pad remaining hops with zero vectors.
                if (frontier.isEmpty()) {
                    for (int rest = h + 1; rest <= k; rest++) {
                        results.add(new SIGNResult(seedId, (long) rest, toDoubleList(new double[featLen])));
                    }
                    break;
                }
            }
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

    /** Box a primitive {@code double[]} to {@code List<Double>}. */
    private List<Double> toDoubleList(double[] arr) {
        if (arr == null) return new ArrayList<>(0);
        List<Double> list = new ArrayList<>(arr.length);
        for (double v : arr) list.add(v);
        return list;
    }
}
