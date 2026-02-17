import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch import Tensor

class GCN(nn.Module):
    def __init__(self, in_dim:int, hidden1_dim:int, hidden2_dim:int, out_dim:int, nbr_classes:int):
        super().__init__()
        self.GCN1 = GCNConv(in_dim, hidden1_dim, aggr="mean")
        self.GCN2 = GCNConv(hidden1_dim, hidden2_dim, aggr="mean")
        self.classifier = nn.Linear(hidden2_dim, nbr_classes)
    def forward(self, X:Tensor, edge_index:Tensor) -> Tensor:
        X = self.GCN1(X, edge_index).relu()
        output_GCN1 = self.GCN2(X, edge_index).relu()
        logits = self.classifier(output_GCN1)
        return logits