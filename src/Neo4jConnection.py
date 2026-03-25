from neo4j import GraphDatabase, NotificationMinimumSeverity

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(
            uri, 
            auth=(user, password),
            # encrypted=False,
            # fetch_size=-1,
            max_connection_lifetime=30 * 60, 
            max_connection_pool_size=50,
            connection_timeout=30.0,
            notifications_min_severity=NotificationMinimumSeverity.OFF,
        )

    def close(self):
        self.driver.close()

    def get_driver(self):
        return self.driver