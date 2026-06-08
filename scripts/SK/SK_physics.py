"""
This script is made to run physics simulation for the SK model with the CFTP BC algorithm at different values of beta and different sizes
"""

from utils.cftp_func import *

beta_SG = 1
betas = np.linspace(0, 0.3, 15)
N_list = [100, 200, 500, 750, 1000]

# for beta in betas:
#     Physics(model="SK", N_list=N_list, beta=beta, save_name="SK", n_runs=25)

Plot_physics(model="SK", save_name="SK")