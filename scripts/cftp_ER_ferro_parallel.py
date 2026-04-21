"""
This script investigates the magnetization of the Ising model on Erdős-Rényi random graphs using CFTP with Glauber heat bath dynamics. We vary the average degree of the graph and observe how the magnetization changes with inverse temperature beta, especially around the critical point.

Last edited: 2024-18-04
Author: L. Péron
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import multiprocessing as mp
from cftp_bc import *

def process_d(d):
    """
    Worker function to process a single value of d.
    """
    N = 100
    G = nx.erdos_renyi_graph(N, d/N)
    lambda_bar = d/N * (N-2)
    beta_critical = np.arctanh(1/lambda_bar)
    
    betas = np.linspace(0, beta_critical*1.05, 100)
    couplings = np.ones((N, N)) - np.eye(N)
    
    magnetizations = []
    times = []
    
    for beta in betas:
        print(f"Processing d={d}, beta={beta:.3f}...")
        temp_mag = []
        times_temp = []
        # Keeping your loop for structural consistency, 
        # though range(1) runs exactly once.
        for _ in range(50):
            config, conv_time = CFTP_BC_disordered_optimized(beta=beta, G=G, coupling=couplings)
            magnetization = np.abs(np.mean(config))
            temp_mag.append(magnetization)
            times_temp.append(-conv_time)
            
        magnetizations.append(np.mean(temp_mag))
        times.append(np.mean(times_temp))
        
    # Return all the data needed for plotting
    return d, magnetizations, betas, beta_critical, times


if __name__ == '__main__':
    d_values = [2, 4, 6, 10]
    
    # Initialize the multiprocessing pool to use 4 CPUs
    with mp.Pool(processes=4) as pool:
        # map() blocks until all processes are complete and returns results in order
        results = pool.map(process_d, d_values)
        
    # Unpack the results for plotting
    lst_beta_critical = [res[3] for res in results]
    times_fixed_d = [res[4] for res in results]
    
    ### Plotting the results ###
    for res in results:
        d, magnetizations, betas, beta_critical, times = res
        
        color = plt.cm.Blues((d-1)/8)  # Normalize d to range [0, 1] for colormap
        
        plt.figure(0)
        plt.plot(betas, magnetizations, color=color)
        plt.vlines(beta_critical, -0.05, 1.05, color=color, linestyle='--', label=rf'$\beta_c(d={d})$')

        plt.figure(1)
        plt.plot(betas, times, color=color)
        plt.vlines(beta_critical, 1, np.nanmax(times_fixed_d)*1.1, color=color, linestyle='--', label=rf'$\beta_c(d={d})$')

        plt.figure(2)
        plt.plot(betas, times, color=color)
        plt.vlines(beta_critical, 1, np.nanmax(times_fixed_d)*1.1, color=color, linestyle='--', label=rf'$\beta_c(d={d})$')

    # Formatting Figure 0 (Magnetization)
    plt.figure(0)
    plt.text(np.max(lst_beta_critical)*0.65, 0, "$N$=100\nErdős-Rényi Graph\nGlauber Heat Bath", fontsize=16)
    plt.ylim(-0.05, 1.05)
    plt.xlabel(r'Inverse Temperature $\beta$')
    plt.ylabel('Magnetization')
    plt.legend(loc="upper right")
    plt.savefig('../figures/ER_ferro_magnetization_comparison.png')
    plt.savefig('../figures/ER_ferro_magnetization_comparison.svg')

    # Formatting Figure 1 (Convergence Time)
    plt.figure(1)
    plt.text(np.nanmax(lst_beta_critical)*0.65, np.nanmax(times_fixed_d)*0.1, "$N$=100\nErdős-Rényi Graph\nGlauber Heat Bath", fontsize=16)
    plt.ylim(1, np.nanmax(times_fixed_d)*1.1)
    plt.xlabel(r'Inverse Temperature $\beta$')
    plt.ylabel('CFTP Convergence Time')
    plt.legend(loc="upper right")
    plt.savefig('../figures/ER_ferro_convergence_time.png')
    plt.savefig('../figures/ER_ferro_convergence_time.svg')

    plt.figure(2)
    plt.yscale('log')
    plt.text(np.nanmax(lst_beta_critical)*0.65, np.nanmax(times_fixed_d)*0.1, "$N$=100\nErdős-Rényi Graph\nGlauber Heat Bath", fontsize=16)
    plt.ylim(1, np.nanmax(times_fixed_d)*1.1)
    plt.xlabel(r'Inverse Temperature $\beta$')
    plt.ylabel('CFTP Convergence Time')
    plt.legend(loc="upper right")
    plt.savefig('../figures/ER_ferro_convergence_time_log.png')
    plt.savefig('../figures/ER_ferro_convergence_time_log.svg')