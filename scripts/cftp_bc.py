"""
This script implements the Coupling From The Past (CFTP) algorithm with bounding chains for sampling from the Gibbs distribution of disordered spin systems, such as the Ising model on a general graph with arbitrary couplings. The implementation is optimized for efficiency and can handle large systems.

Last edited: 2024-21-04
Author: L. Péron
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.figsize': (8, 6),
    'font.size': 22,
    'lines.linewidth': 4,
    'axes.grid': True,
    'legend.fontsize': 12,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.markersize': 10,
    'font.family': 'serif',
    'font.serif': ['Latin Modern Roman'],
    'mathtext.fontset': 'cm',
    'axes.spines.top': False,
    'axes.spines.right': False,
    # remove the box around the legend
    'legend.frameon': False
})

### Sampling functions ###

def F_beta_Glauber(beta, U):
    return 1 / (1 + np.exp(2 * beta * U))

def F_beta_Metropolis(beta, U):
    if U <= 0:
        return 1
    else:
        return np.exp(-2 * beta * U)

### CFTP with Bounding Chains for Disordered Spin Systems ###

def CFTP_BC_disordered_optimized(beta, G, coupling):
    t_max = 2**20  # A large negative number to prevent infinite loops in case of non-coalescence
    t = -1
    random_node = []
    random_spin_value = []
    random_real = []
    star = [-1, +1]
    N = G.number_of_nodes()
    while t > -t_max:  # A large negative number to prevent infinite loops in case of non-coalescence
        nb_star_state = []
        Y = [[-1, +1] for _ in range(N)]  # Initialize the bounding chains to cover all configurations
        for timestep in range(t, 1):
            nb_star_state.append(sum(1 for v in range(N) if Y[v] == star))
            while len(random_real) < -t:
                random_real.append(np.random.rand())  # Generate random numbers for updates
                random_node.append(np.random.randint(N))
                random_spin_value.append(np.random.choice([-1, 1]))
            actual_random_node = random_node[-timestep-1]
            actual_random_spin_value = random_spin_value[-timestep-1]
            actual_random_real = random_real[-timestep-1]

            if Y[actual_random_node] != [actual_random_spin_value]:
                h_bar = 0
                m = 0
                for neighbor in G.neighbors(actual_random_node):
                    if Y[neighbor] != star:
                        h_bar += coupling[actual_random_node, neighbor]*Y[neighbor][0]
                    else:
                        m += np.abs(coupling[actual_random_node, neighbor])
                h_minus = h_bar + actual_random_spin_value*m # The "minus" case corresponds to the +s_n value because the sampler are decreasing functions of the local field
                h_plus = h_bar - actual_random_spin_value*m
                if actual_random_real < F_beta_Glauber(beta, -2*actual_random_spin_value*h_plus):
                    Y[actual_random_node] = [actual_random_spin_value]
                elif actual_random_real > F_beta_Glauber(beta, -2*actual_random_spin_value*h_minus) and Y[actual_random_node] == [-actual_random_spin_value]:
                    continue
                else:
                    Y[actual_random_node] = star
                
        # Check for coalescence
        if all(len(Y[v])==1 for v in range(N)):
            # complete nb_star_state with zeros for the remaining time steps after coalescence
            return np.array([Y[v][0] for v in range(N)]), t, nb_star_state  # Return the coalesced configuration
        else:
            t *= 2  # Double the time window for the next iteration
    print("Warning: CFTP did not coalesce after a large number of iterations.")
    return np.array([np.nan for _ in range(N)]), np.nan, [np.nan for _ in range(-t)]

def CFTP_BC_disordered_time_in_star(beta, G, coupling):
    t_max = 2**20  # A large negative number to prevent infinite loops in case of non-coalescence
    t = -1
    random_node = []
    random_spin_value = []
    random_real = []
    star = [-1, +1]
    N = G.number_of_nodes()
    while t > -t_max:  # A large negative number to prevent infinite loops in case of non-coalescence
        time_in_star_state = [0 for _ in range(N)]
        Y = [[-1, +1] for _ in range(N)]  # Initialize the bounding chains to cover all configurations
        for timestep in range(t, 1):
            while len(random_real) < -t:
                random_real.append(np.random.rand())  # Generate random numbers for updates
                random_node.append(np.random.randint(N))
                random_spin_value.append(np.random.choice([-1, 1]))
            actual_random_node = random_node[-timestep-1]
            actual_random_spin_value = random_spin_value[-timestep-1]
            actual_random_real = random_real[-timestep-1]
            if Y[actual_random_node] == star:
                time_in_star_state[actual_random_node] += 1
            if Y[actual_random_node] != [actual_random_spin_value]:
                h_bar = 0
                m = 0
                for neighbor in G.neighbors(actual_random_node):
                    if Y[neighbor] != star:
                        h_bar += coupling[actual_random_node, neighbor]*Y[neighbor][0]
                    else:
                        m += np.abs(coupling[actual_random_node, neighbor])
                h_minus = h_bar + actual_random_spin_value*m # The "minus" case corresponds to the +s_n value because the sampler are decreasing functions of the local field
                h_plus = h_bar - actual_random_spin_value*m
                if actual_random_real < F_beta_Glauber(beta, -2*actual_random_spin_value*h_plus):
                    Y[actual_random_node] = [actual_random_spin_value]
                elif actual_random_real > F_beta_Glauber(beta, -2*actual_random_spin_value*h_minus) and Y[actual_random_node] == [-actual_random_spin_value]:
                    continue
                else:
                    Y[actual_random_node] = star
                
        # Check for coalescence
        if all(len(Y[v])==1 for v in range(N)):
            return np.array([Y[v][0] for v in range(N)]), t, time_in_star_state  # Return the coalesced configuration
        else:
            t *= 2  # Double the time window for the next iteration
    print("Warning: CFTP did not coalesce after a large number of iterations.")
    return np.array([np.nan for _ in range(N)]), np.nan, [np.nan for _ in range(N)]