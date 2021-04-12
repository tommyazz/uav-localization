import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNet(nn.Module):
    def __init__(self, layers_dim=None):
        super(DenseNet, self).__init__()
        fc_layers = []
        for in_dim, out_dim in zip(layers_dim[:-1], layers_dim[1:]):
            fc_layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(fc_layers)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
