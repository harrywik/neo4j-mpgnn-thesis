import pyarrow.flight as flight
import json
import torch
from torch_geometric.data import Data

class Neo4jArrowClient:
    def __init__(self, host="localhost", port=8491, username="neo4j", password="password"):
        self.location = f"grpc+tcp://{host}:{port}"
        self.client = flight.FlightClient(self.location)
        
        # Authentication (standard GDS Arrow uses Basic Auth)
        options = flight.FlightCallOptions(headers=[
            (b"authorization", f"Basic {self._encode_auth(username, password)}".encode())
        ])
        self.options = options

    def fetch_graph_data(self, graph_name, node_properties):
        """
        Streams node features from GDS in-memory graph via Arrow Flight.
        """
        ticket_dict = {
            "name": "GET_COMMAND",
            "version": "v1",
            "body": {
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

def to_pytorch_gpu(arrow_table):
    # Arrow -> DLPack -> PyTorch
    # Note: Arrow lives in CPU memory. DLPack avoids Py-copying, 
    # but .to('cuda') is still needed for the actual transfer.
    
    # Extract the feature columns
    feature_columns = [c for c in arrow_table.column_names if c != 'nodeId']
    
    # Convert Arrow to a PyTorch tensor via DLPack (Zero-copy on CPU)
    # Most production setups use .to_pandas() then torch.from_numpy() 
    # but for "ultra-scale," look at pyarrow.cuda for direct GPU buffers.
    features_np = arrow_table.select(feature_columns).to_pandas().values
    features_torch = torch.from_numpy(features_np).to('cuda')
    
    return features_torch
