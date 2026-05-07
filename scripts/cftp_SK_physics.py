"""
This script performs theoretical calculations for the Sherrington-Kirkpatrick (SK) model and compares them with results obtained from CFTP sampling.

Author: L. Péron
Last edited: 2024-21-04
"""

from cftp_my_lib import *

beta_SG = 1
beta_CFTP = np.linspace(0, beta_SG*1.1, 10)
N_list = [100, 200, 500, 750, 1000]

Physics(model="SK", N_list=N_list, beta_list=beta_CFTP, save_name="SK", n_runs=25)

Plot_physics(model="SK", save_name="SK")