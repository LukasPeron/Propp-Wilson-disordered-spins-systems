"""
This script implements the Coupling From The Past (CFTP) algorithm with Bounding Chains for sampling from the Gibbs distribution of a spin glass model defined on an Random Regular graph. The script generates a random Random Regular graph, assigns random couplings to the edges, and then uses CFTP to sample configurations at various inverse temperatures (betas). The energy per spin is computed for each sampled configuration and compared to the theoretical predictions obtained from the Theorem 17.3 of the book Information, Physics and Computation" from Mézard and Montanari. Finally, the results are plotted to visualize the behavior of the energy per spin as a function of the inverse temperature, highlighting the critical point where a phase transition occurs.

Author: L. Péron
"""

from cftp_bc import *

### CFTP Sampling for a spin glass on an Erdős-Rényi graph ###

d = 4
beta_BD = np.arctanh(1/d)
beta_SG = np.arctanh(np.sqrt(1/(d-1)))

print(f"beta_BD={beta_BD:.3f}, beta_SG={beta_SG:.3f}")
betas = np.linspace(0, beta_SG*1.1, 100)
beta_CFTP = np.linspace(0, beta_SG*1.1, 10)
theoretical_energy_per_spin = np.array([- d/2 * np.tanh(b) for b in betas])

plt.plot(betas[betas < beta_SG], theoretical_energy_per_spin[betas < beta_SG], '-r', label='RS Solution')
plt.plot(betas[betas >= beta_SG], theoretical_energy_per_spin[betas >= beta_SG], '--r')

marker_lst = ['o', 's', 'D', 'P']
color_lst = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

for k, N in enumerate([100, 200, 500, 750]):
    print(f"Processing N={N}...")
    G = nx.random_regular_graph(d, N)
    couplings = np.triu(np.random.choice([-1, 1], size=(N, N)), k=1)
    couplings += couplings.T  # account for the fact that the graph is not oriented
    couplings = couplings * nx.to_numpy_array(G)  # set couplings to zero for non-edges
    mcmc_energy_per_spin = []
    mcmc_energy_variances = []
    for b in beta_CFTP:
        print(f"Processing beta={b:.3f}...")
        temp = []
        for _ in range(25):
            config, time, __ = CFTP_BC_disordered_optimized(beta=b, G=G, coupling=couplings)
            energy = 0
            for i, j in G.edges():
                energy -= couplings[i, j] * config[i] * config[j] # account for the fact that the graph is not oriented
            energy_per_spin = energy / N
            temp.append(energy_per_spin)
        mcmc_energy_per_spin.append(np.mean(temp))
        mcmc_energy_variances.append(np.var(temp))

    ### Plotting the results ###
    plt.errorbar(beta_CFTP, mcmc_energy_per_spin, yerr=np.sqrt(mcmc_energy_variances), marker=marker_lst[k], color=color_lst[k], label=f'N={N}', linestyle='None')

plt.vlines(beta_BD, np.min(theoretical_energy_per_spin)*1.2, 0.1, color='k', linestyle='--', label=r'$\beta_{BD}$')
plt.vlines(beta_SG, np.min(theoretical_energy_per_spin)*1.2, 0.1, color='g', linestyle=':', label=r'$\beta_{SG}$')
plt.ylim(np.min(theoretical_energy_per_spin)*1.1, 0.05)
plt.xlabel(r'Inverse Temperature $\beta$')
plt.ylabel('Energy per Spin')
plt.legend(loc="lower left")
plt.savefig(save_path+'temp/RR_glassy_energy_comparison_d_{d}.png')
plt.savefig(save_path+'temp/RR_glassy_energy_comparison_d_{d}.svg')