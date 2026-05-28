"""
This script is made to run the BC algorithm for the Ising model with or without disordred in d=2 and d=3 at different values of beta and a fixed size, and to analyze the number of star states throughout the algorithm. 
"""

from utils.cftp_func import *

for d in [3]:
    for model in ['RL_sg']:
        if d==2:
            L = 32
            if model=="RL_ferr":
                beta_pm = 1/2 * (np.log(1 + np.sqrt(2)))
                beta_sg = None
                beta_uni = None
                beta_dm = None
            else: # no spin glass transition in 2D
                beta_uni = np.arctanh(1/(2*d-1))
                beta_sg = 1
                beta_dm = 1/1.69
        elif d==3:
            L = 16
            if model=="RL_ferr":
                beta_pm = 0.221654626
                beta_sg = None
                beta_uni = None
                beta_dm = None
            else:
                beta_sg = 1/1.3
                beta_uni = np.arctanh(1/(2*d-1))
                beta_dm = 1/3.89
        G, couplings = create_lattice_graph(L=L, d=d, model=model)
        if model=="RL_ferr":
            max_beta = beta_pm*1.2
            betas = np.linspace(0, beta_pm*3, 100, endpoint=True)
        else:
            max_beta = beta_sg
            betas = np.linspace(0, beta_sg, 100, endpoint=True)

        for beta in betas:
            Nb_star(N=L**d, beta=beta, G=G, couplings=couplings, beta_c=None, save_name=model, n_runs=25, init_mode="single_star", d=d)

        print("Plotting...")

        Plot_nb_star(save_name=model, beta_BD=None, beta_c=None, max_beta=None, init_mode="single_star", d=d, beta_sg=beta_sg, beta_uni=beta_uni, beta_dm=beta_dm)