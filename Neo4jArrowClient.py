import pyarrow.flight as flight
import json
import torch
from torch.utils.dlpack import from_dlpack
from torch_geometric.data import Data

class Neo4jArrowClient:
    def __init__(self, host="localhost", port=8491, database_name="neo4j", username="neo4j", password="password"):
        self.location = f"grpc+tcp://{host}:{port}"
        self.client = flight.FlightClient(self.location)
        
        # Authentication (standard GDS Arrow uses Basic Auth)
        options = flight.FlightCallOptions(headers=[
            (b"authorization", f"Basic {self._encode_auth(username, password)}".encode())
        ])
        self.options = options
        self.database_name = database_name

    def fetch_graph_data(self, graph_name, node_properties):
        """
        Streams node features from GDS in-memory graph via Arrow Flight.
        """
        ticket_dict = {
            "name": "GET_COMMAND",
            "version": "v1",
            "body": {
                "database_name": self.database_name,
                "graph_name": graph_name,
                "procedure_name": "gds.graph.nodeProperties.stream",
                "configuration": {
                    "node_properties": node_properties,
                    "consecutive_ids": True # CRITICAL for PyG compatibility
                }
            }
        }
        ticket = flight.Ticket(json.dumps(ticket_dict))

        # Request the stream
        reader = self.client.do_get(ticket, self.options)
        
        # Read into Arrow Table
        table = reader.read_all()
        return table

    def _encode_auth(self, u, p):
        import base64
        return base64.b64encode(f"{u}:{p}".encode()).decode()

# --- Integration with GPU via DLPack ---

def to_pytorch_gpu(arrow_table, col_name: str = "features"):
    # Arrow -> DLPack -> PyTorch
    # Note: Arrow lives in CPU memory. DLPack avoids Py-copying, 
    # but .to('cuda') is still needed for the actual transfer.
    col = arrow_table.column(col_name)
    if len(col.chunks) > 1:
        col = col.combine_chunks()
    else:
        col = col.chunk(0)

    flat_values = col.values

    num_nodes = len(col)
    dim = len(flat_values) // num_nodes

    cpu_tensor_64 = cpu_tensor_flat.reshape(num_nodes, dim)
    cpu_tensor_32 = cpu_tensor_64.to(torch.float32)

    return cpu_tensor_32.to("cuda")
