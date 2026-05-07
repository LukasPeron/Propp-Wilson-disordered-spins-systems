"""
This script performs theoretical calculations for the Curie-Weiss model and compares them with results obtained from CFTP sampling. It computes the partition function, magnetization, and magnetization variance for a complete graph of 100 nodes across a range of inverse temperatures (beta). The results are averaged over multiple runs to reduce noise and are intended to be plotted for comparison with theoretical predictions.

Author: L. Péron
"""

from cftp_my_lib import *

beta_CFTP = np.linspace(0, 1.5, 10)
N_list = [100, 200, 500, 750, 1000]
save_name = "CW"
model = "CW"

Physics(model=model, N_list=N_list, beta_list=beta_CFTP, save_name=save_name, n_runs=25)

Plot_physics(model=model, save_name=save_name)