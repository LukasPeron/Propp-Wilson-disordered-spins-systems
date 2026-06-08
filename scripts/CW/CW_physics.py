"""
This script is made to run physics simulation for the Curie-Weiss model with the CFTP BC algorithm at different values of beta and different sizes
"""

from utils.cftp_func import *

betas = np.linspace(0, 1.5, 15)
N_list = [100, 200, 500, 750, 1000]
save_name = "CW"
model = "CW"
# for beta in betas:
#     Physics(model=model, N_list=N_list, beta=beta, save_name=save_name, n_runs=10)

Plot_physics(model=model, save_name=save_name)