"""
This script is made to run the BC algorithm for the Curie-Weiss model at different values of beta and a fixed size, and to analyze the number of star states throughout the algorithm. 
"""

from cftp_my_lib import *

N = 100
G = nx.complete_graph(N)
couplings = (np.ones((N, N)) - np.eye(N)) / N
beta_pm = 1
max_beta = beta_pm * 1.5
save_name = "CW"
betas = np.linspace(0, max_beta, 15, endpoint=True)
for beta in betas:
    Nb_star(N=N, beta=beta, G=G, couplings=couplings, beta_c=beta_pm, max_beta=max_beta, save_name=save_name, n_runs=10)

Plot_nb_star(save_name=save_name, beta_BD=np.nan, beta_c=beta_pm, max_beta=max_beta)