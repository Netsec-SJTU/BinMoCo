# @Time: 2022.12.28 15:31
# @Author: Bolun Wu (e-mail: bowenwu@sjtu.edu.cn)

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN


# GatedGCN
# ref: Benchmarking Graph Neural Networks
class GatedGCNKernel(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dims):
        super(GatedGCNKernel, self).__init__(aggr='add')
        self.A = nn.Linear(in_channels, out_channels)
        self.B = nn.Linear(in_channels, out_channels)
        self.C = nn.Linear(edge_dims, out_channels)
        self.D = nn.Linear(in_channels, out_channels)
        self.E = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        if isinstance(x, torch.Tensor):
            x = (x, x) # j -> i
        Ax = self.A(x[1])
        Dx = self.D(x[1])
        
        Bx = self.B(x[0])
        Ex = self.E(x[0])
        
        Ce = self.C(edge_attr)

        out = self.propagate(edge_index, B=Bx, D=Dx, E=Ex, edge_attr=Ce)
        out = out + Ax
        
        return out

    def message(self, B_j, D_i, E_j, edge_attr):
        gate = torch.sigmoid(D_i + E_j + edge_attr)
        return gate * B_j


class GatedGCN(BasicGNN):
    supports_edge_weight = False
    supports_edge_attr = True
    def init_conv(self, in_channels: int, out_channels: int, edge_dims: int, **kwargs):
        return GatedGCNKernel(in_channels, out_channels, edge_dims=edge_dims)

