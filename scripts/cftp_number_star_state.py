"""
This script runs the bounding chain algorithm forward in time to investigate the convergence properties of the bounding chain itself. We apply this to the Curie-Weiss model and Ising model on a ER graph and observe how the number of "star" states (undecided spins) evolves over time for different inverse temperatures (betas). This can provide insights into the mixing time and convergence behavior of the CFTP algorithm.

Author: L. Péron
Last edited: 2024-21-04
"""

from cftp_bc import *

### Curie-Weiss model (i.e. complete graph) ###

N = 100
G = nx.complete_graph(N)
couplings = 1/2 * (np.ones((N, N)) - np.eye(N)) / N
betas = np.linspace(0, 1.25, 6)

for beta in betas:
    print(f"Processing beta={beta:.3f}...")
    temp_state = []
    temp_time = []
    for _ in range(10):
        config, _, nb_star_states = CFTP_BC_disordered_optimized(beta=beta, G=G, coupling=couplings)
        temp_time.append(len(nb_star_states))
        temp_state.append(nb_star_states)
    max_time = max(temp_time)
    for i in range(len(temp_state)):
        if len(temp_state[i]) < max_time:
            temp_state[i] += [temp_state[i][-1]] * (max_time - len(temp_state[i]))
    time = np.arange(max_time)
    color = plt.cm.Blues(beta / max(betas))
    plt.figure(0)
    plt.plot(time, np.nanmean(temp_state, axis=0), color=color, label=rf'$\beta={beta:.2f}$')
plt.figure(0)
plt.text(1, 0, f"$N$={N}\nCurie-Weiss model\nGlauber Heat Bath", fontsize=16)
plt.xscale('log')
plt.xlabel('Time')
plt.ylabel('Number of $\star$ States')
plt.legend(loc="upper right")
plt.savefig(f'../figures/CW_number_star_state.png')
plt.savefig(f'../figures/CW_number_star_state.svg')

### Ising model on ER graph ###
N = 100
d = 10
G = nx.erdos_renyi_graph(N, d/N)
couplings = (np.ones((N, N)) - np.eye(N))
beta_critical = np.arctanh(1/np.mean([G.degree(v) for v in range(N)]))
betas = np.linspace(0, beta_critical * (4/3), 5)
for beta in betas:
    print(f"Processing beta={beta:.3f}...")
    temp_state = []
    temp_time = []
    for _ in range(10):
        config, _, nb_star_states = CFTP_BC_disordered_optimized(beta=beta, G=G, coupling=couplings)
        temp_time.append(len(nb_star_states))
        temp_state.append(nb_star_states)
    max_time = max(temp_time)
    for i in range(len(temp_state)):
        if len(temp_state[i]) < max_time:
            temp_state[i] += [temp_state[i][-1]] * (max_time - len(temp_state[i]))
    time = np.arange(max_time)
    color = plt.cm.Blues(beta / max(betas))
    plt.figure(1)
    plt.plot(time, np.nanmean(temp_state, axis=0), color=color, label=rf'$\beta={beta:.2f}$')
plt.figure(1)
plt.text(1, 0, f"$N$={N}\nd={d}\nErdős-Rényi Graph\nGlauber Heat Bath", fontsize=16)
plt.xscale('log')
plt.xlabel('Time')
plt.ylabel('Number of $\star$ States')
plt.legend(loc="upper right")
plt.savefig(f'../figures/ER_number_star_state_d_{d}_N_{N}.png')
plt.savefig(f'../figures/ER_number_star_state_d_{d}_N_{N}.svg')