"""
This script is made to analyze the coalescence time of the BC algorithm for the Curie-Weiss model at different sizes and different values of beta. 
"""

from utils.cftp_func import *

N_list = [100, 200, 500, 750, 1000, 1500, 2000, 2250, 2500, 2750, 3000]
beta_pm = 1.0
max_beta = beta_pm * 1.3
save_name = "CW"
print(f"beta_BD = {beta_pm}", max_beta)
betas = np.linspace(0, max_beta, 25, endpoint=True)

# for beta in betas:
#     Coal_time(N_list=N_list, beta=beta, d=np.nan, max_beta=max_beta, save_name=save_name, n_runs=10, sampler=F_beta_Metropolis)

Plot_coal_time(save_name=save_name, beta_BD = np.nan, beta_c=beta_pm, beta_uni=beta_pm, max_beta=max_beta, sampler=F_beta_Glauber)