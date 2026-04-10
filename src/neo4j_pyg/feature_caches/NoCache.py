from neo4j_pyg.feature_caches.Neo4jCache import Neo4jCache


class NoCache(Neo4jCache):
    """Null-object cache — every lookup is a miss."""

    def get(self, key):
        return None

    def set(self, key, value):
        pass

    def delete(self, key):
        pass

    def clear(self):
        pass
