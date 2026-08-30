import torch
import torch.nn as nn
import numpy as np


class PINN(nn.Module):
    def __init__(self, layers = [20, 20], const_alpha=-1):
        super(PINN, self).__init__()
        layer_sizes = [1] + layers + [2]
        modules = []
        for i in range(len(layer_sizes) - 1):
            modules.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            # Tanh activation layers
            if i < len(layer_sizes) - 2:
                modules.append(nn.Tanh())
                
        # Pack them into nn.Sequential
        self.net = nn.Sequential(*modules)
        
        self.log_a = nn.Parameter(torch.tensor(-1.0))
        self.log_b = nn.Parameter(torch.tensor(-1.0))
        self.log_c = nn.Parameter(torch.tensor(-1.0))
        self.log_d = nn.Parameter(torch.tensor(-1.0))

        if const_alpha <= 0:
            self.rho = nn.Parameter(torch.tensor(-1.0))
        else:
            assert const_alpha < 1, "0 < const_alpha < 1 to set alpha as a known constant, const_alpha <= 0 to train alpha" 
            self.rho = -torch.log((1-const_alpha) / const_alpha)
        
    def forward(self, x):
        return self.net(x)