"""
This script is made to run the BC algorithm for a disordered spin system on an RR graph at different values of beta and a fixed size, and to analyze the number of star states throughout the algorithm. 
"""

from cftp_my_lib import *

N = 500
d = 4
G = nx.random_regular_graph(d, N)
couplings = np.triu(np.random.choice([-1, 1], size=(N, N)), k=1)
couplings += couplings.T  # account for the fact that the graph is not oriented
couplings = couplings * nx.to_numpy_array(G)

beta_BD = np.arctanh(1/d)
beta_SG = np.arctanh(np.sqrt(1/(d-1)))
beta_try = np.arctanh(1/(d-1))
print(f"beta_BD={beta_BD:.3f}", f"beta_SG={beta_SG:.3f}", f"beta_try={beta_try:.3f}")
max_beta = beta_try * 1.2
betas = np.linspace(0, max_beta, 15, endpoint=True)
save_name = "RR"

for beta in betas:
    Nb_star(N=N, beta=beta, G=G, couplings=couplings, beta_c=beta_SG, max_beta=max_beta, save_name=save_name, n_runs=10)

Plot_nb_star(save_name=save_name, beta_BD=beta_BD, beta_c=beta_SG, max_beta=max_beta)