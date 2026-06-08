"""
This script is made to analyze the coalescence time of the BC algorithm for a disordered spin system on an RR graph at different sizes and different values of beta. 
"""

from utils.cftp_func import *

d=4
beta_SG = np.arctanh(np.sqrt(1/d))
beta_BD = np.arctanh(1/d)
betas = np.linspace(0, beta_SG*1.2, 15)
N_list = [100, 200, 500, 750, 1000]

# for beta in betas:
#     Physics(model="RR", N_list=N_list, beta=beta, save_name="RR", n_runs=10, d=d)

Plot_physics(model="RR", save_name="RR", d=4)
