"""
This script implements the Coupling From The Past (CFTP) algorithm with Bounding Chains for sampling from the Gibbs distribution of a spin glass model defined on an Erdős-Rényi graph. The script generates a random Erdős-Rényi graph, assigns random couplings to the edges, and then uses CFTP to sample configurations at various inverse temperatures (betas). The energy per spin is computed for each sampled configuration and compared to the theoretical predictions obtained from the Theorem 17.3 of the book Information, Physics and Computation" from Mézard and Montanari. Finally, the results are plotted to visualize the behavior of the energy per spin as a function of the inverse temperature, highlighting the critical point where a phase transition occurs.

Author: L. Péron
"""

from cftp_my_lib import *
d=4
beta_SG = np.arctanh(np.sqrt(1/d))
beta_CFTP = np.linspace(0, beta_SG*1.1, 10)
N_list = [100, 200, 500, 750, 1000]
save_name = "ER"
model = "ER"

Physics(model=model, N_list=N_list, beta_list=beta_CFTP, save_name=save_name, n_runs=25, d=d)

Plot_physics(model=model, save_name=save_name, d=4)
