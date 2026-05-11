"""
This script is made to analyze the coalescence time of the BC algorithm for the Ising model with or without disordred in d=2 and d=3 at different sizes and different values of beta. 
"""

from cftp_my_lib import *
d=3
if d == 2:
    N_list = [10**2, 14**2, 23**2, 27**2, 32**2]
    beta_pm = 1/2 * (np.log(1 + np.sqrt(2)))
if d == 3:
    N_list = [5**3, 6**3, 8**3, 9**3, 10**3]
    beta_pm = 0.2216544
max_beta = beta_pm * 1.5
save_name = "RL_ferr"
print(f"beta_BD = {beta_pm}", max_beta)
print(save_name)
betas = np.linspace(0, max_beta, 51, endpoint=True)

for beta in betas:
    Coal_time(N_list=N_list, beta=beta, d=d, beta_c=beta_pm, max_beta=max_beta, save_name=save_name, n_runs=10)

Plot_coal_time(save_name=save_name, beta_BD = np.nan, beta_c=beta_pm, max_beta=max_beta, d=d)