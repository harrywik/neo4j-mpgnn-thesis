from Neo4jArrowClient import Neo4jArrowClient, to_pytorch_gpu
from dotenv import load_dotenv

def run_gpu_test():
    host = os.getenv("NEO4J_HOST", "localhost")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "password")
    db = os.getenv("NEO4J_DB", "neo4j")
    graph = os.getenv("GRAPH_NAME")

    try: 
        client = Neo4jArrowClient(host=host, database_name=db, username=user, password=pwd)
        node_props = ["features"] 
        table = client.fetch_graph_data(
            graph_name=graph, 
            node_properties=node_props, 
        )
        gpu_tensor = to_pytorch_gpu(table, col_name="features")

        print(f"Final Tensor Shape: {gpu_tensor.shape}")
        print(f"Device: {gpu_tensor.device}")
        print(f"Dtype: {gpu_tensor.dtype}")
        print("--- GPU Transfer Successful ---")

    except Exception as e:
        print(f"!!! Connection or Transfer Failed: {e} !!!")

if __name__ == "__main__":
    run_gpu_test()
