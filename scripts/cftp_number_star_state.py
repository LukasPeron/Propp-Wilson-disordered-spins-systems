"""
This script runs the bounding chain algorithm forward in time to investigate the convergence properties of the bounding chain itself. We apply this to the Curie-Weiss model and Ising model on a ER graph and observe how the number of "star" states (undecided spins) evolves over time for different inverse temperatures (betas). This can provide insights into the mixing time and convergence behavior of the CFTP algorithm.

Author: L. Péron
"""

from cftp_bc import *

### Curie-Weiss model (i.e. complete graph) ###

# N = 100
# G = nx.complete_graph(N)
# couplings = (np.ones((N, N)) - np.eye(N)) / N
# beta_pm = 1
# betas_lower = np.linspace(0, beta_pm, 4)[:-1]
# betas_upper = np.linspace(beta_pm, beta_pm * 1.25, 4)
# betas = np.concatenate((betas_lower, betas_upper))

# for j, beta in enumerate(betas):
#     print(f"Processing beta={beta:.3f}...")
#     temp_state = []
#     temp_time = []
#     for _ in range(50):
#         config, _, nb_star_states = CFTP_BC_disordered_optimized(beta=beta, G=G, coupling=couplings)
#         temp_time.append(len(nb_star_states))
#         temp_state.append(nb_star_states)
#     max_time = max(temp_time)
#     for i in range(len(temp_state)):
#         if len(temp_state[i]) < max_time:
#             temp_state[i] += [temp_state[i][-1]] * (max_time - len(temp_state[i]))
#     time = np.arange(max_time)
#     color = cmc.berlin(j / (len(betas) - 1))
#     plt.figure(0)
#     plt.plot(time, np.nanmean(temp_state, axis=0), color=color, label=f"$\\beta$={beta:.2f}" if beta!=beta_pm else "$\\beta_c$",)
#     plt.fill_between(time, np.nanmean(temp_state, axis=0) - np.nanstd(temp_state, axis=0), np.nanmean(temp_state, axis=0) + np.nanstd(temp_state, axis=0), color=color, alpha=0.3, edgecolor=None)
# plt.figure(0)
# plt.text(1, 0, f"$N$={N}\nCurie-Weiss model\nGlauber Heat Bath", fontsize=16)
# plt.xscale('log')
# plt.xlabel('Time')
# plt.ylabel('Number of $\star$ States')
# plt.legend(loc="upper right")
# plt.savefig(f'../figures/CW_number_star_state.png')
# plt.savefig(f'../figures/CW_number_star_state.svg')

# ### Ising model on ER graph ###

N = 500
d = 4
G = nx.erdos_renyi_graph(N, d/N)
couplings = (np.ones((N, N)) - np.eye(N))
beta_pm = np.arctanh(1/np.mean([G.degree(v) for v in range(N)]))
print(f"beta_pm={beta_pm:.3f}")
betas_lower = np.linspace(0, beta_pm, 4)[:-1]
betas_upper = np.linspace(beta_pm, beta_pm * 1.1, 4)
betas = np.concatenate((betas_lower, betas_upper))
for j, beta in enumerate(betas):
    print(f"Processing beta={beta:.3f}...")
    temp_state = []
    temp_time = []
    for _ in range(50):
        config, _, nb_star_states = CFTP_BC_disordered_optimized(beta=beta, G=G, coupling=couplings)
        temp_time.append(len(nb_star_states))
        temp_state.append(nb_star_states)
    max_time = max(temp_time)
    for i in range(len(temp_state)):
        if len(temp_state[i]) < max_time:
            temp_state[i] += [temp_state[i][-1]] * (max_time - len(temp_state[i]))
    time = np.arange(max_time)
    color = cmc.berlin(j / (len(betas) - 1))
    plt.figure(1)
    plt.plot(time, np.nanmean(temp_state, axis=0), color=color, label=f"$\\beta$={beta:.2f}" if beta!=beta_pm else "$\\beta_c$",)
    plt.fill_between(time, np.nanmean(temp_state, axis=0) - np.nanstd(temp_state, axis=0), np.nanmean(temp_state, axis=0) + np.nanstd(temp_state, axis=0), color=color, alpha=0.3, edgecolor=None)
plt.figure(1)
plt.text(1, 0, f"$N$={N}\nd={d}\nErdős-Rényi Graph\nGlauber Heat Bath", fontsize=16)
plt.xscale('log')
plt.xlabel('Time')
plt.ylabel('Number of $\star$ States')
plt.legend(loc="upper right")
plt.savefig(f'../figures/ER_number_star_state_d_{d}_N_{N}.png')
plt.savefig(f'../figures/ER_number_star_state_d_{d}_N_{N}.svg')

### Random regular graph ###

# N = 100
# d = 3
# G = nx.random_regular_graph(d, N)
# couplings = (np.ones((N, N)) - np.eye(N))
# beta_pm = np.arctanh(1/(d-1))
# print(f"beta_pm={beta_pm:.3f}")
# betas_lower = np.linspace(0, beta_pm, 4)[:-1]
# betas_upper = np.linspace(beta_pm, beta_pm * 1.1, 4)
# betas = np.concatenate((betas_lower, betas_upper))
# for j, beta in enumerate(betas):
#     print(f"Processing beta={beta:.3f}...")
#     temp_state = []
#     temp_time = []
#     for _ in range(50):
#         config, _, nb_star_states = CFTP_BC_disordered_optimized(beta=beta, G=G, coupling=couplings)
#         temp_time.append(len(nb_star_states))
#         temp_state.append(nb_star_states)
#     max_time = max(temp_time)
#     for i in range(len(temp_state)):
#         if len(temp_state[i]) < max_time:
#             temp_state[i] += [temp_state[i][-1]] * (max_time - len(temp_state[i]))
#     time = np.arange(max_time)
#     color = cmc.berlin(j / (len(betas) - 1))
#     plt.figure(2)
#     plt.plot(time, np.nanmean(temp_state, axis=0), color=color, label=f"$\\beta$={beta:.2f}" if beta!=beta_pm else "$\\beta_c$",)
#     plt.fill_between(time, np.nanmean(temp_state, axis=0) - np.nanstd(temp_state, axis=0), np.nanmean(temp_state, axis=0) + np.nanstd(temp_state, axis=0), color=color, alpha=0.3, edgecolor=None)
# plt.figure(2)
# plt.text(1, 0, f"$N$={N}\nd={d}\nRandom Regular Graph\nGlauber Heat Bath", fontsize=16)
# plt.xscale('log')
# plt.xlabel('Time')
# plt.ylabel('Number of $\star$ States')
# plt.legend(loc="upper right")
# plt.savefig(f'../figures/RR_number_star_state_d_{d}_N_{N}.png')
# plt.savefig(f'../figures/RR_number_star_state_d_{d}_N_{N}.svg')