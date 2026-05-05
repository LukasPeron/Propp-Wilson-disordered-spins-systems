"""
This script implements the Coupling From The Past (CFTP) algorithm with Bounding Chains for sampling from the Gibbs distribution of a spin glass model defined on an Random Regular graph. The script generates a random Random Regular graph, assigns random couplings to the edges, and then uses CFTP to sample configurations at various inverse temperatures (betas). The energy per spin is computed for each sampled configuration and compared to the theoretical predictions obtained from the Theorem 17.3 of the book Information, Physics and Computation" from Mézard and Montanari. Finally, the results are plotted to visualize the behavior of the energy per spin as a function of the inverse temperature, highlighting the critical point where a phase transition occurs.

Author: L. Péron
"""

from cftp_bc import *

### CFTP Sampling for a spin glass on an Erdős-Rényi graph ###

N = 500
d = 4
G = nx.random_regular_graph(d, N)
couplings = np.triu(np.random.choice([-1, 1], size=(N, N)), k=1)
couplings += couplings.T  # account for the fact that the graph is not oriented
couplings = couplings * nx.to_numpy_array(G)  # set couplings to zero for non-edges

nb_edges = G.number_of_edges()
beta_BD = np.arctanh(1/d)
beta_SG = np.arctanh(np.sqrt(1/(d-1)))
print(f"beta_BD={beta_BD:.3f}, beta_SG={beta_SG:.3f}")
betas = np.linspace(0, beta_SG*1.1, 100)
beta_CFTP = np.linspace(0, beta_SG*1.2, 25)
theoretical_energy_per_spin = np.array([- d/2 * np.tanh(b) for b in betas])

mcmc_energy_per_spin = []

for b in beta_CFTP:
    print(f"Processing beta={b:.3f}...")
    temp = []
    for _ in range(1):
        config, time, __ = CFTP_BC_disordered_optimized(beta=b, G=G, coupling=couplings)
        energy = 0
        for i, j in G.edges():
            energy -= couplings[i, j] * config[i] * config[j] # account for the fact that the graph is not oriented
        energy_per_spin = energy / N
        temp.append(energy_per_spin)
    mcmc_energy_per_spin.append(np.mean(temp))

### Plotting the results ###

plt.plot(betas[betas < beta_SG], theoretical_energy_per_spin[betas < beta_SG], '-r', label='Theoretical Energy per Spin')
plt.plot(betas[betas >= beta_SG], theoretical_energy_per_spin[betas >= beta_SG], '--r')
plt.plot(beta_CFTP, mcmc_energy_per_spin, '+b', label='CFTP Energy per Spin')
plt.vlines(beta_BD, np.min(theoretical_energy_per_spin)*1.2, 0.1, color='k', linestyle='--', label=r'$\beta_{BD}$')
plt.vlines(beta_SG, np.min(theoretical_energy_per_spin)*1.2, 0.1, color='g', linestyle=':', label=r'$\beta_{SG}$')
plt.ylim(np.min(theoretical_energy_per_spin)*1.1, 0.05)
plt.text(0, -0.8, f'$N$={N}\nd={d}'+'\nRandom Regular Graph\nGlauber Heat Bath\n$J_{ij}\sim\\text{Unif}(\{\pm1\})$', fontsize=16)
plt.xlabel(r'Inverse Temperature $\beta$')
plt.ylabel('Energy per Spin')
plt.legend(loc="lower left")
plt.savefig(f'../figures/temp/RR_glassy_energy_comparison_d_{d}_N_{N}.png')
plt.savefig(f'../figures/temp/RR_glassy_energy_comparison_d_{d}_N_{N}.svg')
