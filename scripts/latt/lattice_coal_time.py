"""
This script is made to analyze the coalescence time of the BC algorithm for the Ising model with or without disordred in d=2 and d=3 at different sizes and different values of beta. 
"""

from utils.cftp_func import *

for save_name in ["RL_ferr", "RL_sg"]:
    print(f"process {save_name}")
    for d in [2, 3]:
        print(f"process d={d}")
        if d == 2:
            # classic is [100, 200, 500, 750, 1000, 1500, 2000, 2250, 2500, 2750, 3000]
            N_list = [10**2, 14**2, 23**2, 27**2, 32**2, 39**2, 45**2, 48**2, 50**2, 52**2, 55**2]
            if save_name == "RL_ferr":
                beta_c = 1/2 * (np.log(1 + np.sqrt(2)))
                beta_uni = beta_c # when there is a phase transition, the uniqueness threshold is the same as the critical one
            else:
                beta_c = np.nan # no phase transition for the 2D spin glass
                beta_uni = np.arctanh(1/(2*d-1)) # = arctanh(1/d-1)
            beta_BD = np.arctanh(1/(2*d)) # = arctanh(1/deg max)
            
        if d == 3:
            N_list = [5**3, 6**3, 8**3, 9**3, 10**3, 12**3, 13**3, 14**3, 15**3]
            if save_name == "RL_ferr":
                beta_c = 0.221654626
                beta_uni = beta_c # when there is a phase transition, the uniqueness threshold is the same as the critical one
            else:
                beta_c = 0.769230769 # = 1/1.3
                beta_uni = np.arctanh(1/(2*d-1)) # = arctanh(1/d-1)
            beta_BD = np.arctanh(1/(2*d)) # = arctanh(1/deg max)
        if save_name == "RL_ferr":
            max_beta = beta_c * 1.2
        else:
            max_beta = beta_BD * 1.5
        print(f"beta_c = {beta_c}", f"beta_BD = {beta_BD}", max_beta)
        print(save_name)
        betas = np.linspace(0, max_beta, 50, endpoint=True)

        # for beta in betas:
        #     Coal_time(N_list=N_list, beta=beta, d=d, max_beta=max_beta, save_name=save_name, n_runs=10, sampler=F_beta_Metropolis)

        Plot_coal_time(save_name=save_name, beta_BD=beta_BD, beta_c=beta_c, beta_uni=beta_uni, max_beta=max_beta, sampler=F_beta_Glauber, d=d)