"""
This script is made to analyze the coalescence time of the BC algorithm for the SK model at different sizes and different values of beta. 
"""

from cftp_my_lib import *

N_list = [100, 200, 500, 750, 1000]
beta_SG = 1
beta_BD = solve_self_consistent_beta(max(N_list), initial_guess=0)
max_beta = beta_BD * 5
save_name = "SK"
print(f"beta_BD = {beta_BD}", max_beta)
betas = np.linspace(0, max_beta, 51, endpoint=True)
for beta in betas:
    Coal_time(N_list=N_list, beta=beta, d=np.nan, beta_c=beta_SG, max_beta=max_beta, save_name=save_name, n_runs=10)

Plot_coal_time(save_name=save_name, beta_BD = beta_BD, beta_c=beta_SG, max_beta=max_beta)