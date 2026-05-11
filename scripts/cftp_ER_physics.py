"""
This script is made to run physics simulation for a disordered spin system on an ER graph with the CFTP BC algorithm at different values of beta and different sizes
"""

from cftp_my_lib import *
d=4
beta_SG = np.arctanh(np.sqrt(1/d))
betas = np.linspace(0, beta_SG*1.2, 15)
N_list = [100, 200, 500, 750, 1000]
save_name = "ER"
model = "ER"
for beta in betas:
    Physics(model=model, N_list=N_list, beta=beta, beta_list=betas, save_name=save_name, n_runs=10, d=d)

Plot_physics(model=model, save_name=save_name, d=4)