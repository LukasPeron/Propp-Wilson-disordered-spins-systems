"""
This script is made to run the BC algorithm for the SK model at different values of beta and a fixed size, and to analyze the number of star states throughout the algorithm. 
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
max_beta = beta_BD * 3
betas = np.linspace(0, max_beta, 15, endpoint=True)
save_name = "SK"

for beta in betas:
    Nb_star(N=N, beta=beta, G=G, couplings=couplings, beta_c=beta_BD, max_beta=max_beta, save_name=save_name, n_runs=10)

Plot_nb_star(save_name=save_name, beta_BD=beta_BD, beta_c=beta_SG, max_beta=max_beta)