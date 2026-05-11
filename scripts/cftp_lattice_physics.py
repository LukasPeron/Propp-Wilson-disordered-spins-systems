"""
This script is made to run physics simulation for the Ising model with or without disordred in d=2 and d=3 with the CFTP BC algorithm at different values of beta and different sizes
"""

from cftp_my_lib import *

d = 3
save_name = "RL_sg"
model = "RL_sg"
print(f"Running physics calculations for {model} in {d} dimensions...")
if d == 2:
    N_list = [10**2, 14**2, 23**2, 27**2, 32**2]
    beta_pm = 1/2 * np.log(1 + np.sqrt(2))
if d == 3:
    N_list = [5**3, 6**3, 8**3, 9**3, 10**3]
    beta_pm = 0.2216544
betas = np.linspace(0, beta_pm*1.5, 15)

for beta in betas:
    Physics(model=model, N_list=N_list, beta=beta, save_name=save_name, n_runs=10, d=d)

Plot_physics(model=model, save_name=save_name, d=d)