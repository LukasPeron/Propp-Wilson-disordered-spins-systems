"""
This script runs the bounding chain algorithm forward in time to investigate the convergence properties of the bounding chain itself. We apply this to the Ising model on a ER graph and observe how the number of "star" states (undecided spins) evolves over time for different inverse temperatures (betas). This can provide insights into the mixing time and convergence behavior of the CFTP algorithm.

Author: L. Péron
"""
from cftp_my_lib import *

N = 500
d = 4
G = nx.erdos_renyi_graph(N, d/N)
couplings = np.triu(np.random.choice([-1, 1], size=(N, N)), k=1)
couplings += couplings.T  # account for the fact that the graph is not oriented
couplings = couplings * nx.to_numpy_array(G)

# beta_BD = np.arctanh(1/np.mean([G.degree(v) for v in range(N)]))
beta_BD = np.arctanh(1/d)
beta_SG = np.arctanh(np.sqrt(1/d))
max_beta = beta_SG * 1.1
save_name = "ER"

Nb_star(N=N, G=G, couplings=couplings, beta_c=beta_SG, max_beta=max_beta, save_name=save_name, n_runs=25)

Plot_nb_star(save_name=save_name, beta_BD=beta_BD, beta_c=beta_SG, max_beta=max_beta)