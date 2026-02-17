import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch import Tensor

class GCN(nn.Module):
    def __init__(self, in_dim:int, hidden_dim1:int, hidden_dim2:int, nbr_classes:int):
        super().__init__()
        self.GCN1 = GCNConv(in_dim, hidden_dim1)
        self.GCN2 = GCNConv(hidden_dim1, hidden_dim2)
        self.classifier = nn.Linear(hidden_dim2, nbr_classes)
    def forward(self, X:Tensor, edge_index:Tensor) -> Tensor:
        X = self.GCN1(X, edge_index).relu_()
        X = self.GCN2(X, edge_index).relu_()
        logits = self.classifier(X)
        return logits