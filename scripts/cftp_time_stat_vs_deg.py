"""
This script runs the bounding chain algorithm forward in time to investigate the convergence properties of the bounding chain itself. We apply this on an Erdős-Rényi graph and observe what is the average time spent in a star state as a function of the node degree and the inverse temperature (beta). This can provide insights into the mixing time and convergence behavior of the CFTP algorithm.

Author: L. Péron
"""

from cftp_bc import *

N = 500
d = 4
G = nx.erdos_renyi_graph(N, d/N)
deg_of_nodes = [G.degree(v) for v in range(N)]
couplings = (np.ones((N, N)) - np.eye(N))
beta_pm = np.arctanh(1/np.mean(deg_of_nodes))
print(f"beta_pm={beta_pm:.3f}")
betas_lower = np.linspace(0, beta_pm, 4)[:-1]
betas_upper = np.linspace(beta_pm, beta_pm * 1.25, 4)
betas = np.concatenate((betas_lower, betas_upper))

for i, beta in enumerate(betas):
    print(f"Processing beta={beta:.3f}...")
    temp_time = []
    for _ in range(50):
        config, _, time_in_star_state = CFTP_BC_disordered_time_in_star(beta=beta, G=G, coupling=couplings)
        temp_time.append(time_in_star_state)
    time_in_star_state_mean = np.nanmean(temp_time, axis=0)
    time_in_star_state_std = np.nanstd(temp_time, axis=0)
    degree_time_in_star = {}
    for deg, time in zip(deg_of_nodes, time_in_star_state_mean):
        if deg not in degree_time_in_star:
            degree_time_in_star[deg] = []
        degree_time_in_star[deg].append(time)
    degree_time_in_star_mean = {deg: np.nanmean(times) for deg, times in degree_time_in_star.items()}
    degree_time_in_star_std = {deg: np.nanstd(times) for deg, times in degree_time_in_star.items()}
    # Plot the average time in star state as a function of degree
    color = cmc.berlin(i / (len(betas) - 1))
    # organize the data for plotting
    degrees = sorted(degree_time_in_star_mean.keys())
    times = [degree_time_in_star_mean[deg] for deg in degrees]
    times_std = [degree_time_in_star_std[deg] for deg in degrees]
    plt.plot(degrees, times, label=f"$\\beta$={beta:.2f}" if beta!=beta_pm else "$\\beta_c$", color=color)
    plt.fill_between(degrees, np.array(times) - np.array(times_std), np.array(times) + np.array(times_std), color=color, alpha=0.3, edgecolor=None)

plt.text(5, 3, f"$N$={N}\nd={d}\nErdős-Rényi Graph\nGlauber Heat Bath", fontsize=16)
plt.xlabel('Degree of Node')
plt.ylabel('Average Time in $\star$ State')
plt.yscale('log')
plt.legend()
plt.savefig(save_path+'algo/ER_time_stat_vs_deg_d_{d}_N_{N}.png')
plt.savefig(save_path+'algo/ER_time_stat_vs_deg_d_{d}_N_{N}.svg')