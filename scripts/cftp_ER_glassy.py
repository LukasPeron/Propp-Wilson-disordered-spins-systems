"""
This script implements the Coupling From The Past (CFTP) algorithm with Bounding Chains for sampling from the Gibbs distribution of a spin glass model defined on an Erdős-Rényi graph. The script generates a random Erdős-Rényi graph, assigns random couplings to the edges, and then uses CFTP to sample configurations at various inverse temperatures (betas). The energy per spin is computed for each sampled configuration and compared to the theoretical predictions obtained from the Theorem 17.3 of the book Information, Physics and Computation" from Mézard and Montanari. Finally, the results are plotted to visualize the behavior of the energy per spin as a function of the inverse temperature, highlighting the critical point where a phase transition occurs.

Author: L. Péron
Last edited: 2024-21-04
"""

from cftp_bc import *

### CFTP Sampling for a spin glass on an Erdős-Rényi graph ###

N = 100
d = 10
G = nx.erdos_renyi_graph(N, d/N)
couplings = np.random.choice([-1, 1], size=(G.number_of_nodes(), G.number_of_nodes())) * (np.ones((G.number_of_nodes(), G.number_of_nodes())) - np.eye(G.number_of_nodes()))
nb_edges = G.number_of_edges()
lambda_bar = d/N * (N-2)
beta_critical = np.arctanh(np.sqrt(1/lambda_bar))
betas = np.linspace(0, beta_critical*1.1, 100)
theoretical_energy_per_spin = np.array([-nb_edges/N * np.tanh(b) for b in betas])

mcmc_energy_per_spin = []

for b in betas:
    print(f"Processing beta={b:.3f}...")
    temp = []
    for _ in range(50):
        config, time = CFTP_BC_disordered_optimized(beta=b, G=G, coupling=couplings)
        energy = 0
        for i, j in G.edges():
            energy -= couplings[i, j] * config[i] * config[j]
        energy_per_spin = energy / N
        temp.append(energy_per_spin)
    mcmc_energy_per_spin.append(np.mean(temp))

### Plotting the results ###

plt.plot(betas[betas < beta_critical], theoretical_energy_per_spin[betas < beta_critical], '-r', label='Theoretical Energy per Spin')
plt.plot(betas[betas >= beta_critical], theoretical_energy_per_spin[betas >= beta_critical], '--r')
plt.plot(betas, mcmc_energy_per_spin, '+b', label='CFTP Energy per Spin')
plt.vlines(beta_critical, np.min(theoretical_energy_per_spin)*1.2, 0.1, color='k', linestyle='--', label=r'Critical Inverse Temperature $\beta_c$')
plt.ylim(np.min(theoretical_energy_per_spin)*1.1, 0.05)
plt.text(0, -1.5, '$N$=100\nErdős-Rényi Graph\nGlauber Heat Bath\n$J_{ij}\sim\\text{Unif}(\{\pm1\})$', fontsize=16)
plt.xlabel(r'Inverse Temperature $\beta$')
plt.ylabel('Energy per Spin')
plt.legend(loc="lower left")
plt.savefig('../figures/ER_glassy_energy_comparison.png')
plt.savefig('../figures/ER_glassy_energy_comparison.svg')