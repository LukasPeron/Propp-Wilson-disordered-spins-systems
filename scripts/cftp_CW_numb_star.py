"""
This script runs the bounding chain algorithm forward in time to investigate the convergence properties of the bounding chain itself. We apply this to the Curie-Weiss model and observe how the number of "star" states (undecided spins) evolves over time for different inverse temperatures (betas). This can provide insights into the mixing time and convergence behavior of the CFTP algorithm.

Author: L. Péron
"""

from cftp_my_lib import *

N = 100
G = nx.complete_graph(N)
couplings = (np.ones((N, N)) - np.eye(N)) / N
beta_pm = 1
max_beta = beta_pm * 1.5
save_name = "CW"

Nb_star(N=N, G=G, couplings=couplings, beta_c=beta_pm, max_beta=max_beta, save_name=save_name, n_runs=25)

Plot_nb_star(save_name=save_name, beta_BD=np.nan, beta_c=beta_pm, max_beta=max_beta)