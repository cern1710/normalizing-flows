import torch
from torch import nn
import torch.nn.functional as F

class Flow:
    def __init__(self):
        super().__init__()

class PlanarFlow:
    def __init__(self, dim: int):
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.scale = nn.Parameter(torch.Tensor(1, dim)) # equivalent to u
        self.bias = nn.Parameter(torch.Tensor(1))

class RadialFlow:
    def __init__(self, dim: int):
        self.z0 = nn.Parameter(torch.Tensor(1, dim))
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))

class NormalizingFlow:
    def __init__(self, K: int, latent_dim: int):
        super().__init__()

        self.transforms = nn.ModuleList(
            [RadialFlow(latent_dim) for _ in range(K)]
        )