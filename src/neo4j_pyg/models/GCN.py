from random import seed

import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch import Tensor
import torch.nn.init as init
import torch

class GCN(nn.Module):
    def __init__(self, in_dim:int, hidden_dim1:int, hidden_dim2:int, nbr_classes:int, init_weights:bool=True):
        super().__init__()
        self.GCN1 = GCNConv(in_dim, hidden_dim1)
        self.GCN2 = GCNConv(hidden_dim1, hidden_dim2)
        self.classifier = nn.Linear(hidden_dim2, nbr_classes)
        if init_weights:
            self.reset_parameters()
        
    def forward(self, X:Tensor, edge_index:Tensor) -> Tensor:
        X = self.GCN1(X, edge_index).relu_()
        X = self.GCN2(X, edge_index).relu_()
        logits = self.classifier(X)
        return logits
    
    def reset_parameters(self):
        cpu_state = torch.get_rng_state()
        gpu_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        self.GCN1.reset_parameters()
        self.GCN2.reset_parameters()
        init.xavier_uniform_(self.classifier.weight)
        init.zeros_(self.classifier.bias)

        torch.set_rng_state(cpu_state)
        if gpu_state is not None:
            torch.cuda.set_rng_state_all(gpu_state)


class TinyGCN(nn.Module):
    def __init__(self, in_dim:int, hidden_dim:int, nbr_classes:int, init_weights:bool=True):
        super().__init__()
        self.GCN = GCNConv(in_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, nbr_classes)
        if init_weights:
            self.reset_parameters()
        
    def forward(self, X:Tensor, edge_index:Tensor) -> Tensor:
        X = self.GCN(X, edge_index).relu_()
        logits = self.classifier(X)
        return logits
    
    def reset_parameters(self):
        cpu_state = torch.get_rng_state()
        gpu_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        self.GCN.reset_parameters()
        init.xavier_uniform_(self.classifier.weight)
        init.zeros_(self.classifier.bias)

        torch.set_rng_state(cpu_state)
        if gpu_state is not None:
            torch.cuda.set_rng_state_all(gpu_state)