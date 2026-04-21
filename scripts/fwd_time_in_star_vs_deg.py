"""
This script runs the bounding chain algorithm forward in time to investigate the convergence properties of the bounding chain itself. We apply this on an Erdős-Rényi graph and observe what is the average time spent in a star state as a function of the node degree and the inverse temperature (beta). This can provide insights into the mixing time and convergence behavior of the CFTP algorithm.

Author: L. Péron
Last edited: 2024-21-04
"""

from cftp_bc import *

N = 100
d = 10
G = nx.erdos_renyi_graph(N, d/N)
deg_of_nodes = [G.degree(v) for v in range(N)]
couplings = (np.ones((N, N)) - np.eye(N))
beta_critical = np.arctanh(1/np.mean(deg_of_nodes))
betas = np.linspace(0, beta_critical * (4/3), 5)

for beta in betas:
    print(f"Processing beta={beta:.3f}...")
    temp_time = []
    for _ in range(10):
        config, _, time_in_star_state = CFTP_BC_disordered_time_in_star(beta=beta, G=G, coupling=couplings)
        temp_time.append(time_in_star_state)
    time_in_star_state_mean = np.nanmean(temp_time, axis=0)
    # for each degree, compute the average time in star state for nodes with that degree
    degree_time_in_star = {}
    for deg, time in zip(deg_of_nodes, time_in_star_state_mean):
        if deg not in degree_time_in_star:
            degree_time_in_star[deg] = []
        degree_time_in_star[deg].append(time)
    degree_time_in_star_mean = {deg: np.nanmean(times) for deg, times in degree_time_in_star.items()} 
    # Plot the average time in star state as a function of degree
    color = plt.cm.Blues(beta / max(betas))
    plt.plot(list(degree_time_in_star_mean.keys()), list(degree_time_in_star_mean.values()), 'o',color=color, label=rf'$\beta={beta:.2f}$')
plt.text(5, 3, f"$N$={N}\nd={d}\nErdős-Rényi Graph\nGlauber Heat Bath", fontsize=16)
plt.xlabel('Degree of Node')
plt.ylabel('Average Time in $\star$ State')
plt.yscale('log')
plt.legend()
plt.savefig(f'../figures/forward_time_in_star_state_degree_ER_d_{d}_N_{N}_log.png')
plt.savefig(f'../figures/forward_time_in_star_state_degree_ER_d_{d}_N_{N}_log.svg')