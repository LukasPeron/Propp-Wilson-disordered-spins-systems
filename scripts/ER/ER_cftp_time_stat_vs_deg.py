"""
Run the bounding-chain dynamics on an Erdős-Rényi graph and measure,
for each node, how long it stays in the star state (0).

The script then groups the measured star times by the degree of each
node and plots the average star time versus degree.

Author: L. Péron
"""

import os

from utils.cftp_func import *


N = 1000
average_degree = 4
beta_uni = np.arctanh(1/(average_degree-1))
beta = 0.3
n_runs = 1

G, couplings = create_ER_graph(N, average_degree)
degrees = np.array([G.degree(node) for node in G.nodes()], dtype=np.int32)


node_star_times, final_steps = run_star_time_experiment(
	beta=beta,
	G=G,
	couplings=couplings,
	n_runs=n_runs,
	max_iter=max_nb_iter_MCMC2_BC,
)

mean_star_time_per_node = node_star_times.mean(axis=0)

degree_values = np.unique(degrees)
mean_star_time_by_degree = np.array([
	mean_star_time_per_node[degrees == degree].mean()
	for degree in degree_values
])
std_star_time_by_degree = np.array([
	mean_star_time_per_node[degrees == degree].std()
	for degree in degree_values
])

fig, ax = plt.subplots()
ax.scatter(degrees, mean_star_time_per_node, alpha=0.4, s=35, color="tab:blue", label="Nodes")
ax.set_xlabel(rf"Node degree (ER graph $N={N}$, $d=4$), $\beta=0.3$")
ax.set_ylabel("Average time in $\star$ state")
# ax.set_title(f"ER graph: N={N}, <k>={average_degree}, $\\beta$={beta}")
plt.savefig(save_path+"figures/temp/test_cftp_time_in_star_state.png", dpi=300)

print(f"Mean coalescence steps over {n_runs} runs: {np.mean(final_steps):.1f}")