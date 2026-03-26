package com.thesis.gcn;

import org.neo4j.graphdb.*;
import org.neo4j.procedure.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
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
 * CALL custom.gcn.aggregateNeighbors(
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

        public AggResult(Long nodeId, List<Double> aggregatedFeatures) {
            this.nodeId = nodeId;
            this.aggregatedFeatures = aggregatedFeatures;
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

    // -------------------------------------------------------------------------
    // Procedure
    // -------------------------------------------------------------------------

    @Procedure(name = "custom.gcn.aggregateNeighbors", mode = Mode.READ)
    @Description(
        "Mean-aggregate incoming neighbour feature vectors for each seed node. "
        + "Returns one row per seed with the aggregated feature vector."
    )
    public Stream<AggResult> aggregateNeighbors(
            @Name("seedIds")                          List<Long> seedIds,
            @Name("nodeIdKey")                        String nodeIdKey,
            @Name("featureKey")                       String featureKey,
            @Name("featureType")                      String featureType,
            @Name("nodeLabel")                        String nodeLabel,
            @Name(value = "edgeType",       defaultValue = "") String edgeType,
            @Name(value = "maxNeighbors",   defaultValue = "-1") Long maxNeighbors
    ) {
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
                results.add(new AggResult(seedId, new ArrayList<>()));
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

            results.add(new AggResult(seedId, toDoubleList(agg)));
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
     * CALL custom.gcn.signAggregate(
     *     $seed_ids, "id", "embedding_bytes", "byte[]", "Paper", "CITES", 2, 10
     * ) YIELD nodeId, hop, aggregatedFeatures
     * RETURN nodeId, hop, aggregatedFeatures ORDER BY nodeId, hop
     * }</pre>
     */
    @Procedure(name = "custom.gcn.signAggregate", mode = Mode.READ)
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
