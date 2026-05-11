"""
This script is made to analyze the coalescence time of the BC algorithm for a disordered spin system on a RR graph at different sizes and different values of beta. 
"""

from cftp_my_lib import *

N_list = [100, 200, 500, 750, 1000]
d = 4
beta_SG = np.arctanh(np.sqrt(1/(d-1)))
beta_BD = np.arctanh(1/d)
max_beta = beta_SG * 1.2
save_name = "RR"
print(f"beta_BD = {beta_BD}", max_beta)
betas = np.linspace(0, max_beta, 51, endpoint=True)
for beta in betas:
    Coal_time(N_list=N_list, beta=beta, d=d, beta_c=beta_SG, max_beta=max_beta, save_name=save_name, n_runs=10)

Plot_coal_time(save_name=save_name, beta_BD = beta_BD, beta_c=beta_SG, max_beta=max_beta)