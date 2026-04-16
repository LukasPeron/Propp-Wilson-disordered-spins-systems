"""
This script investigates the magnetization of the Ising model on Erdős-Rényi random graphs using CFTP with Glauber heat bath dynamics. We vary the average degree of the graph and observe how the magnetization changes with inverse temperature beta, especially around the critical point.

Last edited: 2024-06-20
Author: L. Péron
"""

from cftp_bc import *

### CFTP Sampling for the Ising Model on Erdős-Rényi Graphs ###

N = 100
magnetizations_fixed_d = []
d_values = [2, 4, 6, 10]
lst_beta = []
lst_beta_critical = []
for d in d_values:
    G = nx.erdos_renyi_graph(N, d/N)
    lambda_bar = d/N * (N-2)
    beta_critical = np.arctanh(1/lambda_bar)
    lst_beta_critical.append(beta_critical)
    betas = np.linspace(0, beta_critical*1.1, 100)
    lst_beta.append(betas)
    couplings = np.ones((N, N)) - np.eye(N)
    magnetizations = []
    for beta in betas:
        temp = []
        for _ in range(50):
            config, time = CFTP_BC_disordered_optimized(beta=beta, G=G, coupling=couplings)
            magnetization = np.abs(np.mean(config))
            temp.append(magnetization)
        magnetizations.append(np.mean(temp))
    magnetizations_fixed_d.append(magnetizations)

### Plotting the results ###

for d, magnetizations, betas, beta_critical in zip(d_values, magnetizations_fixed_d, lst_beta, lst_beta_critical):
    color = plt.cm.Blues((d-1)/8)  # Normalize d to range [0, 1] for colormap
    plt.plot(betas, magnetizations, color=color)
    plt.vlines(beta_critical, -0.05, 1.05, color=color, linestyle='--', label=rf'$\beta_c(d={d})$')
plt.text(1.0, 0, "$N$=100\nErdős-Rényi Graph\nGlauber Heat Bath", fontsize=16)
plt.ylim(-0.05, 1.05)
plt.xlabel(r'Inverse Temperature $\beta$')
plt.ylabel('Magnetization')
plt.legend()
plt.savefig('../figures/ER_ferro_magnetization_comparison.png')