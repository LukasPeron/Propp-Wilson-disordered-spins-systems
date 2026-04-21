"""
This script investigates the magnetization of the Ising model on Erdős-Rényi random graphs using CFTP with Glauber heat bath dynamics. We vary the average degree of the graph and observe how the magnetization changes with inverse temperature beta, especially around the critical point.

Last edited: 2026-21-04
Author: L. Péron
"""

from cftp_bc import *

### CFTP Sampling for the Ising Model on Erdős-Rényi Graphs ###

N = 64
magnetizations_fixed_d = []
d_values = [2,4,6,10]
lst_beta = []
lst_beta_critical = []
times_fixed_d = []
for d in d_values:
    G = nx.erdos_renyi_graph(N, d/N)
    lambda_bar = d/N * (N-2)
    beta_critical = np.arctanh(1/lambda_bar)
    lst_beta_critical.append(beta_critical)
    betas = np.linspace(0, beta_critical*1.05, 100)
    lst_beta.append(betas)
    couplings = np.ones((N, N)) - np.eye(N)
    magnetizations = []
    times = []
    for beta in betas:
        print(f"Processing d={d}, beta={beta:.3f}...")
        temp_mag = []
        times_temp = []
        for _ in range(1):
            # config, conv_time = CFTP_BC_disordered_optimized(beta=beta, G=G, coupling=couplings)
            config, conv_time = CFTP_BC_disordered_single_pass(beta=beta, G=G, coupling=couplings, t=-10**6)
            magnetization = np.abs(np.mean(config))
            temp_mag.append(magnetization)
            times_temp.append(-conv_time)
        magnetizations.append(np.mean(temp_mag))
        times.append(np.mean(times_temp))
    magnetizations_fixed_d.append(magnetizations)
    times_fixed_d.append(times)

### Plotting the results ###

for d, magnetizations, betas, beta_critical, times in zip(d_values, magnetizations_fixed_d, lst_beta, lst_beta_critical, times_fixed_d):
    color = plt.cm.Blues((d-1)/8)  # Normalize d to range [0, 1] for colormap
    plt.figure(0)
    plt.plot(betas, magnetizations, color=color)
    plt.vlines(beta_critical, -0.05, 1.05, color=color, linestyle='--', label=rf'$\beta_c(d={d})$')

    plt.figure(1)
    plt.plot(betas, times, color=color)
    plt.yscale('log')
    plt.vlines(beta_critical, 0, np.nanmax(times_fixed_d)*1.1, color=color, linestyle='--', label=rf'$\beta_c(d={d})$')

plt.figure(0)
plt.text(np.max(lst_beta_critical)*0.65, 0, f"$N$={N}\nErdős-Rényi Graph\nGlauber Heat Bath", fontsize=16)
plt.ylim(-0.05, 1.05)
plt.xlabel(r'Inverse Temperature $\beta$')
plt.ylabel('Magnetization')
plt.legend(loc="upper right")
plt.savefig('../figures/ER_ferro_magnetization_comparison_test.png')

plt.figure(1)
plt.text(np.nanmax(lst_beta_critical)*0.65, 1e3, f"$N$={N}\nErdős-Rényi Graph\nGlauber Heat Bath", fontsize=16)
plt.ylim(0, np.nanmax(times_fixed_d)*1.1)
plt.xlabel(r'Inverse Temperature $\beta$')
plt.ylabel('CFTP Convergence Time')
plt.legend(loc="upper right")
plt.savefig('../figures/ER_ferro_convergence_time_test.png')