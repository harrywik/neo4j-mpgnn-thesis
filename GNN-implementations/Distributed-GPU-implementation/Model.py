import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch import Tensor

class GCN(nn.Module):
    def __init__(self, in_dim:int, hidden_dim:int, out_dim:int, nbr_classes:int):
        super().__init__()
        self.GCN1 = GCNConv(in_dim, hidden_dim)
        self.GCN2 = GCNConv(hidden_dim, out_dim)
        self.classifier = nn.Linear(out_dim, nbr_classes)
    def forward(self, X:Tensor, edge_index:Tensor) -> Tensor:
        X = self.GCN1(X, edge_index).relu()
        output_GCN1 = self.GCN2(X, edge_index).relu()
        logits = self.classifier(output_GCN1)
        return logits