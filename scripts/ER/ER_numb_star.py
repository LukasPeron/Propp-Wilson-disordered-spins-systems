"""
This script is made to run the BC algorithm for a disordered spin system on an ER graph at different values of beta and a fixed size, and to analyze the number of star states throughout the algorithm. 
"""

from utils.cftp_func import *

N = 500
d = 4

G, couplings = create_ER_graph(N=N, d=d)

beta_BD = np.arctanh(1/np.max([G.degree(v) for v in range(N)]))
beta_uni = np.arctanh(1/d)
beta_SG = np.arctanh(np.sqrt(1/d))
max_beta = beta_uni * 1.2
print(f"beta_BD={beta_BD:.3f}", f"beta_SG={beta_SG:.3f}, beta_uni={beta_uni:.3f}", f"max_beta={max_beta:.3f}")
save_name = "ER"
betas = np.linspace(0, max_beta, 15, endpoint=True)

# for beta in betas:
#     Nb_star(N=N, beta=beta, G=G, couplings=couplings, beta_c=beta_SG, save_name=save_name, n_runs=10)

Plot_nb_star(save_name=save_name, beta_BD=beta_BD, beta_c=beta_SG, max_beta=max_beta, sampler=F_beta_Glauber)