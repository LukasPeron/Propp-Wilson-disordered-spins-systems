"""
This script runs the bounding chain algorithm forward in time to investigate the convergence properties of the bounding chain itself. We apply this to the SK model and observe how the number of "star" states (undecided spins) evolves over time for different inverse temperatures (betas). This can provide insights into the mixing time and convergence behavior of the CFTP algorithm.

Author: L. Péron
"""

from cftp_my_lib import *

N = 100
G = nx.complete_graph(N)
couplings = np.random.normal(0, 1/np.sqrt(N), size=(N, N))
couplings = (couplings + couplings.T) / 2
np.fill_diagonal(couplings, 0)

beta_BD = solve_self_consistent_beta(N, initial_guess=0)
beta_SG = 1
print(f"Bubley-Dyer bound for N={N}: beta_BD = {beta_BD:.4f}")
max_beta = beta_SG * 1.1

Nb_star(N=N, G=G, couplings=couplings, beta_c=beta_BD, max_beta=max_beta, save_name="SK", n_runs=25)

Plot_nb_star(save_name="SK", beta_BD=beta_BD, beta_c=beta_SG, max_beta=max_beta)