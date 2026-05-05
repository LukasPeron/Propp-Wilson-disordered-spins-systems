"""
This script performs theoretical calculations for the Sherrington-Kirkpatrick (SK) model and compares them with results obtained from CFTP sampling.

Author: L. Péron
Last edited: 2024-21-04
"""

from scipy.special import gammaln
from cftp_bc import *

N = 100
G = nx.complete_graph(N)
couplings = np.random.normal(0, 1/np.sqrt(N), size=(N, N))
couplings = (couplings + couplings.T) / 2  # Ensure symmetry J_ij = J_ji
np.fill_diagonal(couplings, 0)  # No self-interaction

## theoretical part ##

"""
We want to compare our simulations to actual theoretical predictions for the SK model. We will compare :
- the energy per spin
- the Edwards-Anderson order parameter q

The Bubley-Dyer bound on temperature is given by the solution of the self-consistent equation :
$$ \beta * \sqrt{N}(1-\tanh^2(\beta)) = 1 - \frac{\tanh^2(\beta)\beta^2}{N} $$
"""

### CFTP sampling part ###

beta_SG = 1
beta = np.linspace(0, 1.1, 100)  # Range of inverse temperatures to test
theoretical_energy_per_spin = np.array([-b/2 for b in beta])  # Theoretical prediction for the energy per spin in the SK model within RS ansatz
theoretical_q = np.zeros_like(beta)  # Theoretical prediction for the Edwards-Anderson order parameter q in the SK model within RS ansatz
plt.figure(0)
plt.plot(beta[beta<=beta_SG], theoretical_energy_per_spin[beta<=beta_SG], '-r', label='RS Solution')
plt.plot(beta[beta>beta_SG], theoretical_energy_per_spin[beta>beta_SG], '--r')
plt.figure(1)
plt.plot(beta[beta<=beta_SG], theoretical_q[beta<=beta_SG], '-r', label='RS Solution')
plt.plot(beta[beta>beta_SG], theoretical_q[beta>beta_SG], '--r')
marker_lst = ['o', 's', 'D', 'P']
color_lst = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

for k, N in enumerate([100, 200]):
    print(f"Processing N={N}...")
    beta_BD = solve_self_consistent_beta(N, initial_guess=0)
    print(f"Bubley-Dyer bound for N={N}: beta_BD = {beta_BD:.4f}")
    beta_CFTP = np.linspace(0, beta_BD*1.05, 5)  # Inverse temperatures at which to perform CFTP sampling
    G = nx.complete_graph(N)
    couplings = np.random.normal(0, 1/np.sqrt(N), size=(N, N))
    couplings = (couplings + couplings.T) / 2  # Ensure symmetry J_ij = J_ji
    np.fill_diagonal(couplings, 0)  # No self-interaction
    mcmc_energy_per_spin = []
    mcmc_q = []
    mcmc_energy_variances = []
    mcmc_q_variances = []
    for b in beta_CFTP:
        print(f"Processing beta={b:.3f}...")
        temp_energy = []
        temp_q = []
        for _ in range(25):
            config, time, __ = CFTP_BC_disordered_optimized(beta=b, G=G, coupling=couplings)
            energy = 0
            for i, j in G.edges():
                energy -= couplings[i, j] * config[i] * config[j] # account for the fact that the graph is not oriented
            energy_per_spin = energy / N
            temp_energy.append(energy_per_spin)
            q = np.mean(config)**2  # Edwards-Anderson order parameter q is the square of the magnetization per spin
            temp_q.append(q)
        mcmc_energy_per_spin.append(np.mean(temp_energy))
        mcmc_q.append(np.mean(temp_q))
        mcmc_energy_variances.append(np.var(temp_energy))
        mcmc_q_variances.append(np.var(temp_q))
    plt.figure(0)
    plt.errorbar(beta_CFTP, mcmc_energy_per_spin, yerr=np.sqrt(mcmc_energy_variances), marker=marker_lst[k], color=color_lst[k], label=f'N={N}', linestyle='None')
    plt.figure(1)
    plt.errorbar(beta_CFTP, mcmc_q, yerr=np.sqrt(mcmc_q_variances), marker=marker_lst[k], color=color_lst[k], label=f'N={N}', linestyle='None')
plt.figure(0)
plt.vlines(beta_BD, np.min(theoretical_energy_per_spin)*1.2, 0.1, color='k', linestyle='--', label=r'$\beta_{BD}$')
plt.vlines(beta_SG, np.min(theoretical_energy_per_spin)*1.2, 0.1, color='g', linestyle=':', label=r'$\beta_{SG}$')
plt.ylim(np.min(theoretical_energy_per_spin)*1.1, 0.05)
plt.xlabel(r'Inverse Temperature $\beta$')
plt.ylabel('Energy per Spin')
plt.legend()
plt.savefig(f'../figures/temp/SK_energy_comparison.png')
plt.savefig(f'../figures/temp/SK_energy_comparison.svg')
plt.figure(1)
plt.vlines(beta_BD, np.min(theoretical_q)*1.2, 0.1, color='k', linestyle='--', label=r'$\beta_{BD}$')
plt.vlines(beta_SG, np.min(theoretical_q)*1.2, 0.1, color='g', linestyle=':', label=r'$\beta_{SG}$')
plt.xlabel(r'Inverse Temperature $\beta$')
plt.ylabel('$q_{EA}$')
plt.legend()
plt.savefig(save_path+'temp/SK_q_comparison.png')
plt.savefig(save_path+'temp/SK_q_comparison.svg')