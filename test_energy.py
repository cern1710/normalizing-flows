import torch
import matplotlib.pyplot as plt

# Helper functions
def w1(z: torch.Tensor) -> torch.Tensor:
    """Sinusoidal oscillations"""
    return torch.sin(torch.pi * z[0] / 2)

def w2(z: torch.Tensor) -> torch.Tensor:
    """Generates a Gaussian peak at z1 = 1"""
    return 3 * torch.exp(-((z[0] - 1) / 0.6)**2 / 2)

def w3(z: torch.Tensor) -> torch.Tensor:
    """Sigmoidal transition at z1 > 0"""
    return 3 * torch.sigmoid((z[0] - 1) / 0.3)

# Test energy functions defined in table 1
def U1(z: torch.Tensor) -> torch.Tensor:
    def _exp(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((x) / 0.6)**2 / 2)

    ln_term = torch.log(_exp(z[0] - 2) + _exp(z[0] + 2))
    return ((z.norm() - 2) / 0.4)**2 / 2 - ln_term

def U2(z: torch.Tensor) -> torch.Tensor:
    return ((z[1] - w1(z)) / 0.4)**2 / 2

def U3(z: torch.Tensor) -> torch.Tensor:
    first_exp = torch.exp(-((z[1] - w1(z)) / 0.35)**2 / 2)
    second_exp = torch.exp(-((z[1] - w1(z) + w2(z)) / 0.35)**2 / 2)
    return -torch.log(first_exp + second_exp)

def U4(z: torch.Tensor) -> torch.Tensor:
    first_exp = torch.exp(-((z[1] - w1(z)) / 0.4)**2 / 2)
    second_exp = torch.exp(-((z[1] - w1(z) + w3(z)) / 0.35)**2 / 2)
    return -torch.log(first_exp + second_exp)

def plot_energy(values_list, titles, z1, z2):
    """Plot densities for each energy function in table 1"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        # Hacky way of reproducing the graphs: rotate 90 deg clockwise
        rot_values = torch.rot90(values_list[i])

        cf = ax.contourf(z1.numpy(), z2.numpy(),
                         rot_values.numpy(),
                         levels=100, cmap='jet')
        fig.colorbar(cf, ax=ax)
        ax.set_title(titles[i])
        ax.set_xlabel('z1')
        ax.set_ylabel('z2')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    z1 = z2 = torch.linspace(-4, 4, 100)
    z1_grid, z2_grid = torch.meshgrid(z1, z2, indexing='ij')
    z = torch.stack((z1_grid, z2_grid), dim=-1)
    U1_values = torch.zeros_like(z1_grid)
    U2_values = torch.zeros_like(z1_grid)
    U3_values = torch.zeros_like(z1_grid)
    U4_values = torch.zeros_like(z1_grid)

    for i in range(100):
        for j in range(100):
            z_point = z[i, j]
            U1_values[i, j] = U1(z_point)
            U2_values[i, j] = U2(z_point)
            U3_values[i, j] = U3(z_point)
            U4_values[i, j] = U4(z_point)

    potentials = [torch.exp(-U1_values), torch.exp(-U2_values),
                  torch.exp(-U3_values), torch.exp(-U4_values)]
    titles = ['U1 Potential', 'U2 Potential', 'U3 Potential', 'U4 Potential']

    plot_energy(potentials, titles, z1, z2)
