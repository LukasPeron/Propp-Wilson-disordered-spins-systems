"""
This script implements the Coupling From The Past (CFTP) algorithm with bounding chains for sampling from the Gibbs distribution of disordered spin systems, such as the Ising model on a general graph with arbitrary couplings. The implementation is optimized for efficiency and can handle large systems.

Author: L. Péron
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

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
    return 1 / (1 + np.exp(beta * U))

def F_beta_Metropolis(beta, U):
    if U <= 0:
        return 1
    else:
        return np.exp(-beta * U)

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
        for timestep in range(t, 0):
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
            print(f"Coalescence achieved at time {-t}.")
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
        for timestep in range(t, 0):
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
            print(f"Coalescence achieved at time {-t}.")
            return np.array([Y[v][0] for v in range(N)]), t, time_in_star_state  # Return the coalesced configuration
        else:
            t *= 2  # Double the time window for the next iteration
    print("Warning: CFTP did not coalesce after a large number of iterations.")
    return np.array([np.nan for _ in range(N)]), np.nan, [np.nan for _ in range(N)]

def MCMC2_fwd(beta, G, couplings, n_iter=100000):
    N = G.number_of_nodes()
    config = np.random.choice([-1, 1], size=N)  # Random initial configuration
    for _ in range(n_iter):
        if _ % (n_iter // 10) == 0:
            print(f"Iteration {_}/{n_iter}")
        v = np.random.randint(N)  # Randomly select a node
        s = np.random.choice([-1, 1])  # Randomly select a spin value
        r = np.random.rand()  # Random number for acceptance
        if config[v] != s:
            h = sum(couplings[v, neighbor] * config[neighbor] for neighbor in G.neighbors(v))
            if r < F_beta_Glauber(beta, 2*config[v]*h):
                config[v] = -config[v]  # Flip the spin
    magnetization = np.mean(config)
    print(f"Final magnetization: {magnetization:.4f}")
    return config, magnetization

def BC_fwd(beta, G, couplings, n_iter=100000):
    N = G.number_of_nodes()
    star = [1, -1]
    options = [[1], [-1], star]
    random_indices = np.random.choice(len(options), size=N)
    config = [options[i] for i in random_indices]
    for _ in range(n_iter):
        if _ % (n_iter // 10) == 0:
            print(f"Iteration {_}/{n_iter}")
        v = np.random.randint(N)  # Randomly select a node
        s = np.random.choice([-1, 1])  # Randomly select a spin value
        r = np.random.rand()  # Random number for acceptance
        if config[v] != [s]:
            h_bar = 0
            m = 0
            for neighbor in G.neighbors(v):
                if config[neighbor] != star:
                    h_bar += couplings[v, neighbor]*config[neighbor][0]
                else:
                    m += np.abs(couplings[v, neighbor])
            h_minus = h_bar + s*m # The "minus" case corresponds to the +s_n value because the sampler are decreasing functions of the local field
            h_plus = h_bar - s*m
            if r < F_beta_Glauber(beta, -2*s*h_plus):
                config[v] = [s]
            elif r > F_beta_Glauber(beta, -2*s*h_minus) and config[v] == [-s]:
                continue
            else:
                config[v] = star
        if all(len(config[v])==1 for v in range(N)):
            print(f"Coalescence achieved at iteration {_}")
            magnetization = np.mean([config[v][0] for v in range(N)])
            print(f"Final magnetization: {magnetization:.4f}")
            return np.array([config[v][0] for v in range(N)]), magnetization
    print("Warning: Coalescence not achieved after a large number of iterations.")
    return np.array([np.nan for _ in range(N)]), np.nan