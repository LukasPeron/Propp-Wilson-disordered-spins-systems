"""
This script is made to run the BC algorithm for the Ising model with or without disordred in d=2 and d=3 at different values of beta and a fixed size, and to analyze the number of star states throughout the algorithm. 
"""

from cftp_my_lib import *

d = 3
save_name = "RL_sg"
print(f"Running for d={d} and save_name={save_name}")
if d==2:
    N = 10**2
    beta_pm = 1/2 * (np.log(1 + np.sqrt(2)))
    L = int(round(N**(1/d)))
    dim = (L, L)
if d==3:
    N = 5**3
    beta_pm = 0.2216544
    L = int(round(N**(1/d)))
    dim = (L, L, L)
G = nx.grid_graph(dim)
N = G.number_of_nodes()
if save_name == "RL_ferr":
    couplings = np.ones((N, N)) - np.eye(N)
elif save_name == "RL_sg":
    couplings = np.random.choice([-1, 1], size=(N, N)) * nx.to_numpy_array(G)
    couplings = (couplings + couplings.T) / 2  # Ensure symmetry J_ij = J_ji
    np.fill_diagonal(couplings, 0)
max_beta = beta_pm*1.5
betas = np.linspace(0, max_beta, 15, endpoint=True)
for beta in betas:
    Nb_star(N=N, beta=beta, G=G, couplings=couplings, beta_c=beta_pm, max_beta=max_beta, save_name=save_name, n_runs=10, d=d)

Plot_nb_star(save_name=save_name, beta_BD=np.nan, beta_c=beta_pm, max_beta=max_beta, d=d)