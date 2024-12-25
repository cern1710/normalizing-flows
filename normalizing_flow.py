import torch
from torch import nn
import torch.nn.functional as F

class Flow(nn.Module):
    """
    Path traversed by random variables z_k with initial distribution q0(z_0).
    """
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply flow transform with change of variables."""
        raise NotImplementedError

    def logdet_jacobian(self, z: torch.Tensor) -> torch.Tensor:
        """logdet-Jacobian term for transformation of densities."""
        raise NotImplementedError

class PlanarFlow(Flow):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.scale = nn.Parameter(torch.Tensor(1, dim)) # equivalent to u
        self.bias = nn.Parameter(torch.Tensor(1))

        # some arbitrary initialisation
        nn.init.xavier_uniform_(self.weight)
        nn.init.uniform_(self.scale)
        nn.init.zeros_(self.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Planar transformation: f(z) = z + uh(wTz + b)."""
        linear = F.linear(z, self.weight, self.bias)
        return z + self.scale * torch.tanh(linear)  # h(z) = tanh(z)

    def logdet_jacobian(self, z: torch.Tensor) -> torch.Tensor:
        linear = F.linear(z, self.weight, self.bias)
        h_prime = 1 - torch.tanh(linear)**2 # derivative of tanh: 1 - tanh2
        psi = h_prime * self.weight
        return torch.log(torch.abs(1 + torch.matmul(psi, self.scale.t())))

class RadialFlow(Flow):
    def __init__(self, dim: int):
        super().__init__()
        self.z0 = nn.Parameter(torch.Tensor(1, dim))
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))

        # some arbitrary initialisation
        nn.init.normal_(self.z0)
        nn.init.constant_(self.alpha, 0.1)
        nn.init.zeros_(self.beta)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Radial transformation: f(z) = z + bh(a, r)(z-z0),
        where h(a, r) = 1/(a+r), r = |z-z0|.
        """
        radial = z - self.z0
        r = torch.norm(radial)
        h = 1 / (self.alpha + r)
        return z + self.beta * h * radial

    def logdet_jacobian(self, z: torch.Tensor) -> torch.Tensor:
        radial = z - self.z0
        r = torch.norm(radial)
        h = 1 / (self.alpha + r)
        h_prime = -1 / (self.alpha + r)**2
        return torch.log((1 + self.beta * h)**(radial.shape[1] - 1) *
                         (1 + self.beta * h + self.beta * h_prime * r))

class NormalizingFlow(nn.Module):
    def __init__(self, flow_len: int, latent_dim: int):
        super().__init__()
        self.transforms = nn.ModuleList(
            [PlanarFlow(latent_dim) for _ in range(flow_len)]
        )

    def forward(self, z: torch.Tensor):
        logdet = 0
        for transform in self.transforms:
            z = transform.forward(z)
            logdet += transform.logdet_jacobian(z)
        return z, logdet
