"""
This script is made to analyze the coalescence time of the BC algorithm for a disordered spin system on a RR graph at different sizes and different values of beta. 
"""

from utils.cftp_func import *

N_list = [100, 200, 500, 750, 1000, 1500, 2000, 2250, 2500, 2750, 3000]
d = 4
beta_SG = np.arctanh(np.sqrt(1/(d-1)))
beta_BD = np.arctanh(1/d) # = arctanh(1/deg max)
beta_uni = np.arctanh(1/(d-1)) # = arctanh(1/d-1)
max_beta = beta_SG * 1.2
save_name = "RR"
print(f"beta_BD = {beta_BD}", max_beta)
betas = np.linspace(0, max_beta, 25, endpoint=True)

for beta in betas:
    Coal_time(N_list=N_list, beta=beta, d=d, max_beta=max_beta, save_name=save_name, n_runs=10, sampler=F_beta_Metropolis)

Plot_coal_time(save_name=save_name, beta_BD=beta_BD, beta_c=beta_SG, beta_uni=beta_uni, max_beta=max_beta, sampler=F_beta_Metropolis, d=d)