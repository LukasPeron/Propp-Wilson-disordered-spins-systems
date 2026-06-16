"""
This script is made to run the BC algorithm for the Ising model with or without disordred in d=2 and d=3 at different values of beta and a fixed size, and to analyze the number of star states throughout the algorithm. 
"""

from utils.cftp_func import *

for save_name in ["RL_ferr"]:
    for d in [2, 3]:
        if d==2:
            L = 10
            beta_pm = 1/2 * (np.log(1 + np.sqrt(2)))
        elif d==3:
            L = 5
            beta_pm = 0.221654626
        G, couplings = create_lattice_graph(L=L, d=d, model=save_name)

        print(f"Running for d={d} and save_name={save_name}")

        max_beta = beta_pm*1.5
        betas = np.linspace(0, max_beta, 15, endpoint=True)
        # for beta in betas:
        #     Nb_star(N=N, beta=beta, G=G, couplings=couplings, beta_c=beta_pm, save_name=save_name, n_runs=10, d=d)

        print("Plotting...")

        Plot_nb_star(save_name=save_name, beta_BD=np.nan, beta_c=beta_pm, max_beta=max_beta, d=d, sampler=F_beta_Glauber)