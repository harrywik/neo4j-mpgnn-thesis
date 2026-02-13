import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch import Tensor

class GCN(nn.Module):
    def __init__(self, in_dim:int, h1:int, h2:int, nbr_classes:int):
        super().__init__()
        self.GCN1 = GCNConv(in_dim, h1)
        self.GCN2 = GCNConv(h1, h2)
        self.classifier = nn.Linear(h2, nbr_classes)

    def forward(self, X:Tensor, edge_index:Tensor) -> Tensor:
        X = self.GCN1(X, edge_index).relu()
        output_GCN1 = self.GCN2(X, edge_index).relu()
        logits = self.classifier(output_GCN1)
        return logits