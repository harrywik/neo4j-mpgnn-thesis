

# Potential PyG / Neo4j contribution ideas

### Graphstore
implement `Neo4jGraphStore(driver)` that stores the graph. it has connection with the DB,
and therefore also has one method called `execute_cypher_query(query:str)`.

### Sampler
In order too optimize the sampling from the neo4j DB, the sampler can't be general. It has to specifically define a cypher query command. Instead of building the subgraph by iteratively sampling neighbors for the seed-nodes and sampled nodes by calling graphstore.getneighbor() for example that then would get called several times, and each time execute a cypher query.

`Neo4jBaseSampler` abstract class that contains a `attribute` unlike the current `BaseSampler`.
This abstract class can then be implemented in different ways to abstract away the problem of knowing cypher for the user. An implementation of Neo4jBaseSampler can for example be `Neo4jSimpleSampler` that has a `query:str` attribute that defines the sampling query, maybe also possible to insert parameters when creating `Neo4jSimpleSampler` to tweak the sapmling wihtout needing to know any cypher. One can also implement `Neo4jCustomSampler` that maybe takes an argument so that people that know cypher easily can optimize their sampling methods without generating new Sampling classes 