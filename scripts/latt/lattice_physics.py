"""
This script is made to run physics simulation for the Ising model with or without disordred in d=2 and d=3 with the CFTP BC algorithm at different values of beta and different sizes
"""

from utils.cftp_func import *

# model = 'RL_sg'  # 'RL_ferr', 'RL_sg'
# save_name = model
# d = 2
for d in [2, 3]:
    for model in ['RL_ferr', 'RL_sg']:
        save_name = model
        print(f"Running physics calculations for {model} in {d} dimensions...")
        if d == 2:
            N_list = [10**2, 14**2, 23**2, 27**2, 32**2]
            beta_pm = 1/2 * np.log(1 + np.sqrt(2))
        if d == 3:
            N_list = [5**3, 6**3, 8**3, 9**3, 10**3]
            if model == 'RL_ferr':
                beta_pm = 0.221654626
            else:
                beta_pm = 0.769230769 # = 1/1.3
        betas = np.linspace(0, beta_pm*1.2, 15)

        # for beta in betas:
        #     print(f"Running physics calculations for beta={beta:.3f}...")
        #     Physics(model=model, N_list=N_list, beta=beta, save_name=save_name, n_runs=10, d=d)

        Plot_physics(model=model, save_name=save_name, d=d)