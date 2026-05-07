"""
This script is made to analyze the coalescence time of the CFTP algorithm for the following spin systems:
- Erdős-Rényi graph with random couplings
Three plot are done for each model: a heatmap of the coalescence time as a function of beta and N, a plot of the coalescence time as a function of beta for different values of N, and a plot of the coalescence time as a function of N for different values of beta.
Author: L. Péron
"""

from cftp_my_lib import *

N_list = [100, 200, 500, 750, 1000]
d = 4
beta_SG = np.arctanh(np.sqrt(1/d))
beta_BD = np.arctanh(1/d)
max_beta = beta_SG * 1.1
save_name = "ER"

Coal_time(N_list=N_list, d=d, beta_c=beta_SG, max_beta=max_beta, save_name=save_name, n_runs=25)

Plot_coal_time(save_name=save_name, beta_BD = beta_BD, beta_c=beta_SG, max_beta=max_beta)