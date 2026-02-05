import inspect
import pickle

# Import modules (for __file__)
import Neo4jGraphStore as Neo4jGraphStoreModule
import Neo4jSampler as Neo4jSamplerModule
import feature_stores.v002 as FeatureStoreModule

# Import classes (for instantiation/signatures)
from Neo4jGraphStore import Neo4jGraphStore
from Neo4jSampler import Neo4jSampler
from feature_stores.v002 import Neo4jFeatureStore as Neo4jFeatureStore002


def build_label_map(uri: str, user: str, pwd: str) -> dict[str, int]:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    q = "MATCH (n) RETURN DISTINCT n.subject AS s ORDER BY s ASC"
    with driver.session() as session:
        labels = [r["s"] for r in session.run(q)]
    driver.close()
    return {s: i for i, s in enumerate(labels)}


def print_module_and_sigs():
    print("=== Modules actually imported ===")
    print("Neo4jGraphStore module:", Neo4jGraphStoreModule.__file__)
    print("Neo4jSampler module   :", Neo4jSamplerModule.__file__)
    print("FeatureStore module   :", FeatureStoreModule.__file__)
    print()

    print("=== Class signatures ===")
    print("Neo4jGraphStore __init__:", inspect.signature(Neo4jGraphStore.__init__))
    print("Neo4jSampler __init__   :", inspect.signature(Neo4jSampler.__init__))

    # FeatureStore class name can differ; print what you imported:
    print("Neo4jFeatureStore002 __init__:", inspect.signature(Neo4jFeatureStore002.__init__))
    print()


def assert_no_driver_created(obj, name: str):
    """
    Sanity check: if your stores keep a cached driver (common attrs: driver/_driver),
    it should be None in the parent process before workers spawn.
    """
    suspicious_attrs = ["driver", "_driver", "session", "_session"]
    found = []
    for a in suspicious_attrs:
        if hasattr(obj, a):
            found.append((a, getattr(obj, a)))
    if found:
        print(f"--- {name} driver/session fields ---")
        for a, v in found:
            # We *want* None here.
            print(f"{name}.{a} =", type(v), "->", "None" if v is None else "NOT NONE (!!)")
        print()


def pickle_test(objs):
    print("=== Pickle test ===")
    for name, obj in objs:
        try:
            pickle.dumps(obj)
            print(name, "pickle OK")
        except Exception as e:
            print(name, "pickle FAIL:", repr(e))
            raise
    print()


def main():
    print_module_and_sigs()

    uri = "bolt://localhost:7687"
    user = "neo4j"
    pwd = "thesis-db-0-pw"

    label_map = build_label_map(uri, user, pwd)

    feature_store = Neo4jFeatureStore002(uri, user, pwd, label_map=label_map)
    graph_store = Neo4jGraphStore(uri, user, pwd)
    sampler = Neo4jSampler(graph_store, [10, 5])

    # Check whether anything eagerly created a driver/session:
    assert_no_driver_created(feature_store, "feature_store")
    assert_no_driver_created(graph_store, "graph_store")

    # Check picklability:
    pickle_test([
        ("graph_store", graph_store),
        ("feature_store", feature_store),
        ("sampler", sampler),
    ])

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
