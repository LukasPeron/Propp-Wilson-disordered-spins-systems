"""
This script is made to analyze the coalescence time of the BC algorithm for the SK model at different sizes and different values of beta. 
"""

from utils.cftp_func import *

N_list = [100, 200, 500, 750, 1000, 1500, 2000, 2250, 2500, 2750, 3000]
beta_SG = 1
beta_BD = solve_self_consistent_beta(max(N_list), initial_guess=0)
beta_uni = beta_SG # for the SK model, the uniqueness threshold is the same as the critical one
max_beta = 0.3
save_name = "SK"
print(f"beta_BD = {beta_BD}", max_beta)
betas = np.linspace(0, max_beta, 25, endpoint=True)
# for beta in betas:
#     Coal_time(N_list=N_list, beta=beta, d=np.nan, max_beta=max_beta, save_name=save_name, n_runs=10, sampler=F_beta_Metropolis)

Plot_coal_time(save_name=save_name, beta_BD = beta_BD, beta_c=beta_SG, beta_uni=beta_uni, max_beta=max_beta, sampler=F_beta_Glauber)