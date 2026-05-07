"""
This script implements the Coupling From The Past (CFTP) algorithm with bounding chains for sampling from the Gibbs distribution of disordered spin systems, such as the Ising model on a general graph with arbitrary couplings. The implementation is optimized for efficiency and can handle large systems.

Author: L. Péron
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import cmcrameri.cm as cmc
from scipy.optimize import fsolve
from scipy.special import gammaln

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

def solve_self_consistent_beta(N, initial_guess=1.0):
    """
    Finds the root beta for the self-consistent equation given a finite N.
    """
    def equation(beta):
        # Left-hand side: beta * sqrt(N) * (1 - tanh^2(beta))
        lhs = beta * np.sqrt(N) * (1 - np.tanh(beta)**2)
        
        # Right-hand side: 1 - (tanh^2(beta) * beta^2) / N
        rhs = 1 - (np.tanh(beta)**2 * beta**2) / N
        
        # We want to find where lhs == rhs, so we return lhs - rhs
        return lhs - rhs

    # fsolve returns an array, so we extract the first (and only) element
    beta_solution, info, ier, mesg = fsolve(equation, x0=initial_guess, full_output=True)
    
    if ier == 1:
        return beta_solution[0]
    else:
        raise ValueError(f"fsolve failed to converge for N={N}: {mesg}") 

def max_deg_ER_graph(N, d, initial_guess=5.0):
    def equation(k):
        lhs = k*(np.log(d)-np.log(k)+1)
        rhs = d - np.log(N)
        return lhs - rhs
    k_solution, info, ier, mesg = fsolve(equation, x0=initial_guess, full_output=True)
    if ier == 1:
        return k_solution[0]
    else:
        raise ValueError(f"fsolve failed to converge for max degree: {mesg}")

save_path = "/home/lperon/cftp_dis_spin/"
max_nb_iter_MCMC2_BC = 2**20

def F_beta_Glauber(beta, U):
    return 1 / (1 + np.exp(beta * U))

def F_beta_Metropolis(beta, U):
    if U <= 0:
        return 1
    else:
        return np.exp(-beta * U)

def theoretical_magnetization_CW(beta, N):
    val_m = np.arange(-N, N+1, 2) / N
    km = N / 2 * (1 + val_m)
    log_prefac = gammaln(N + 1) - gammaln(km + 1) - gammaln(N - km + 1)
    energy = -N * val_m**2 / 2 
    exponents = log_prefac - beta * energy
    max_exp = np.max(exponents)
    terms = np.exp(exponents - max_exp)
    mag_sum = np.sum(np.abs(val_m) * terms)
    Z_shifted = np.sum(terms) 
    return mag_sum / Z_shifted

def CFTP_MCMC2_BC(beta, G, coupling, max_nb_iter_MCMC2_BC=max_nb_iter_MCMC2_BC):
    t = -1
    random_node = []
    random_spin_value = []
    random_real = []
    star = [-1, +1]
    N = G.number_of_nodes()
    while t > -max_nb_iter_MCMC2_BC:  # A large negative number to prevent infinite loops in case of non-coalescence
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
    return np.array([np.nan for _ in range(N)]), np.nan, nb_star_state

def MCMC2_fwd(beta, G, couplings, max_nb_iter_MCMC2=max_nb_iter_MCMC2_BC):
    N = G.number_of_nodes()
    config = np.random.choice([-1, 1], size=N)  # Random initial configuration
    for _ in range(max_nb_iter_MCMC2):
        if _ % (max_nb_iter_MCMC2 // 10) == 0:
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

def MCMC2_BC_fwd(beta, G, couplings, max_nb_iter_MCMC2_BC=max_nb_iter_MCMC2_BC):
    N = G.number_of_nodes()
    star = [1, -1]
    options = [[1], [-1], star]
    random_indices = np.random.choice(len(options), size=N)
    config = [options[i] for i in random_indices]
    nb_star_state = []
    nb_star_state.append(sum(1 for v in range(N) if config[v] == star))
    for timestep in range(max_nb_iter_MCMC2_BC):
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
        nb_star_state.append(sum(1 for v in range(N) if config[v] == star))
        if all(len(config[v])==1 for v in range(N)):
            print(f"Coalescence achieved at iteration {timestep}")
            return nb_star_state, timestep
    print("Warning: Coalescence not achieved after a large number of iterations.")
    return nb_star_state, max_nb_iter_MCMC2_BC

def Nb_star(N, G, couplings, beta_c, max_beta, save_name, n_runs=25, save_path=save_path):
    """
    Runs the bounding chain algorithm forward in time and saves the data to a CSV.
    """
    # Create the beta array. We ensure we capture points around 0 to max_beta.
    # You can adjust the density of the linspace as needed.
    betas_lower = np.linspace(0, beta_c, 11)[:-1]
    betas_upper = np.linspace(beta_c, max_beta, 10, endpoint=True)
    betas = np.concatenate((betas_lower, betas_upper))
    
    all_data = []
    
    for beta in betas:
        print(f"Processing beta={beta:.3f}...")
        temp_state = []
        temp_time = []
        
        # Run multiple simulations for the current beta
        for _ in range(n_runs):
            nb_star_states, __ = MCMC2_BC_fwd(beta, G, couplings)
            temp_time.append(len(nb_star_states))
            temp_state.append(nb_star_states)
            
        max_time = max(temp_time)
        
        # Pad the states so they all match the max_time length
        for i in range(len(temp_state)):
            if len(temp_state[i]) < max_time:
                temp_state[i] += [temp_state[i][-1]] * (max_time - len(temp_state[i]))
                
        time = np.arange(max_time)
        mean_state = np.nanmean(temp_state, axis=0)
        std_state = np.nanstd(temp_state, axis=0)
        
        # Store data points
        for t, m, s in zip(time, mean_state, std_state):
            all_data.append({
                'beta': beta,
                'time': t,
                'mean_state': m,
                'std_state': s
            })
            
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(all_data)

    if save_name == "CW" or save_name == "SK":
        save_path+="data/algo/complet/"
    elif save_name == "ER" or save_name == "RR":
        save_path+="data/algo/random/"

    csv_path = save_path+f"{save_name}_nb_star.csv"
    df.to_csv(csv_path, index=False)
    print(f"Data successfully saved to {csv_path}")
    
    return df

def Plot_nb_star(save_name, beta_BD, beta_c, max_beta, df=None, save_path=save_path):
    """
    Reads the generated data points and creates the convergence plot.
    """
    # Load from CSV if the dataframe isn't directly passed
    if save_name == "CW" or save_name == "SK":
        access_path = save_path+"data/algo/complet/"+f"{save_name}_nb_star"
        save_path+="figures/algo/complet/"
    elif save_name == "ER" or save_name == "RR":
        access_path = save_path+"data/algo/random/"+f"{save_name}_nb_star"
        save_path+="figures/algo/random/"

    if df is None:
        df = pd.read_csv(f"{access_path}.csv")
        
    fig, ax = plt.subplots()
    norm = mpl.colors.Normalize(vmin=0, vmax=max_beta)
    
    betas = df['beta'].unique()
    beta_no_zero = None
    
    # Plot data for each beta
    for beta in betas:
        beta_data = df[df['beta'] == beta]
        time = beta_data['time'].values
        mean_state = beta_data['mean_state'].values
        std_state = beta_data['std_state'].values
        
        # Check if this is the lowest beta where the mean star states never reach 0
        if beta_no_zero is None and np.min(mean_state) > 0:
            beta_no_zero = beta
            
        color = cmc.berlin(norm(beta))
        
        ax.plot(time, mean_state, color=color)
        ax.fill_between(time, 
                        np.maximum(0, mean_state - std_state), 
                        mean_state + std_state, 
                        color=color, alpha=0.3, edgecolor=None)

    ax.set_xscale('log')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'Number of $\star$ States')

    # --- Colorbar implementation ---
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmc.berlin)
    sm.set_array([])

    # Define standard ticks including Bubley-Dyer and Phase Transition betas
    ticks = [0, beta_BD, beta_c, max_beta]
    tick_labels = [r'$\beta=0$', r'$\beta_{BD}$', r'$\beta_c$', f'{max_beta:.2f}']

    # Add the tracked lowest beta to the ticks if it was found
    if beta_no_zero is not None and beta_no_zero not in ticks:
        ticks.append(beta_no_zero)
        tick_labels.append(f'{beta_no_zero:.2f}*')

    # Sort ticks and labels so they appear in correct ascending order on the colorbar axis
    sorted_pairs = sorted(zip(ticks, tick_labels))
    sorted_ticks = [p[0] for p in sorted_pairs]
    sorted_labels = [p[1] for p in sorted_pairs]

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_ticks(sorted_ticks)
    cbar.set_ticklabels(sorted_labels)
    
    # Save outputs
    plt.savefig(save_path+f"{save_name}_nb_star.png", dpi=300)
    plt.savefig(save_path+f"{save_name}_nb_star.svg")

def Coal_time(N_list, d, beta_c, max_beta, save_name, n_runs=25, save_path=save_path):
    """
    Runs the coalescence time analysis for the Curie-Weiss model and saves data to a CSV.
    """
    betas_lower = np.linspace(0, beta_c, 11)[:-1]
    betas_upper = np.linspace(beta_c, max_beta, 10, endpoint=True)
    betas = np.concatenate((betas_lower, betas_upper))
    
    all_data = []

    for N in N_list:
        print(f"Processing N={N}...")
        if save_name == "CW" or save_name == "SK":
            G = nx.complete_graph(N)
            if save_name == "CW":
                couplings = (np.ones((N, N)) - np.eye(N)) / N
            else:  
                couplings = np.random.normal(0, 1/np.sqrt(N), size=(N, N))
                couplings = (couplings + couplings.T) / 2  # Ensure symmetry J_ij = J_ji
                np.fill_diagonal(couplings, 0)
        elif save_name == "ER" or save_name == "RR":
            if save_name == "ER":
                G = nx.erdos_renyi_graph(N, d/N)
            if save_name == "RR":
                G = nx.random_regular_graph(d, N)
            couplings = np.triu(np.random.choice([-1, 1], size=(N, N)), k=1)
            couplings += couplings.T  # account for the fact that the graph is not oriented
            couplings = couplings * nx.to_numpy_array(G)

        for beta in betas:
            print(f"  Processing beta={beta:.2f}...")
            coal_time_temp = []
            log_T_coal_over_N_temp = []
            
            for _ in range(n_runs):
                _, coal_time = MCMC2_BC_fwd(beta, G, couplings)
                coal_time_temp.append(coal_time)
                log_T_coal_over_N_temp.append(np.log(coal_time) / N)
                
            # Store all stats for this (N, beta) configuration
            all_data.append({
                'N': N,
                'beta': beta,
                'mean_coal_time': np.nanmean(coal_time_temp),
                'std_coal_time': np.sqrt(np.nanvar(coal_time_temp)),
                'mean_log_coal_N': np.nanmean(log_T_coal_over_N_temp),
                'std_log_coal_N': np.sqrt(np.nanvar(log_T_coal_over_N_temp))
            })
            
    df = pd.DataFrame(all_data)
    if save_name == "CW" or save_name == "SK":
        save_path+="data/algo/complet/"
    elif save_name == "ER" or save_name == "RR":
        save_path+="data/algo/random/"

    csv_path = save_path+f"{save_name}_coal_time.csv"
    df.to_csv(csv_path, index=False)
    print(f"Data successfully saved to {csv_path}")
    
    return df

def Plot_coal_time(save_name, beta_BD, beta_c, max_beta, df=None, save_path=save_path, max_nb_iter_MCMC2_BC=max_nb_iter_MCMC2_BC):
    """
    Reads the generated data points and creates the 4 coalescence time plots.
    """
    if save_name == "CW" or save_name == "SK":
        access_path = save_path+"data/algo/complet/"+f"{save_name}_coal_time"
        save_path+="figures/algo/complet/"
    elif save_name == "ER" or save_name == "RR":
        access_path = save_path+"data/algo/random/"+f"{save_name}_coal_time"
        save_path+="figures/algo/random/"

    if df is None:
        df = pd.read_csv(f"{access_path}.csv")
        
    N_list = sorted(df['N'].unique())
    betas = sorted(df['beta'].unique())
    
    # Recover beta_no_zero (lowest beta where mean coalescence time hits the 2**20 ceiling)
    beta_no_zero = None
    ceiling_mask = df['mean_coal_time'] == max_nb_iter_MCMC2_BC
    if ceiling_mask.any():
        beta_no_zero = df[ceiling_mask]['beta'].min()

    # --- Plot 0: Heatmap ---
    plt.figure(0)
    pivot_df = df.pivot(index='N', columns='beta', values='mean_coal_time')
    coalescence_times_matrix = pivot_df.values
    
    plt.imshow(coalescence_times_matrix, aspect='auto', origin='lower', 
               extent=[betas[0], betas[-1], min(N_list), max(N_list)], 
               cmap=plt.cm.viridis, norm=LogNorm())
    plt.vlines(beta_c, min(N_list), max(N_list), color='k', linestyle='--', label=r'$\beta_c$')
    plt.vlines(beta_BD, min(N_list), max(N_list), color='k', linestyle=':', label=r'$\beta_{BD}$')
    plt.colorbar(label='Coalescence Time')
    plt.xlabel(r'Inverse Temperature $\beta$')
    plt.ylabel('System Size $N$')
    plt.savefig(save_path+f'{save_name}_heatmap.png')
    plt.savefig(save_path+f'{save_name}_heatmap.svg')
    plt.close()

    # --- Plot 1: Coalescence time vs Beta ---
    plt.figure(1)
    for N in N_list:
        sub_df = df[df['N'] == N]
        color_val = (N - min(N_list)) / (max(N_list) - min(N_list)) if max(N_list) > min(N_list) else 0.5
        color = plt.cm.viridis(color_val)
        plt.plot(sub_df['beta'], sub_df['mean_coal_time'], label=f'$N$={N}', color=color)
        plt.fill_between(sub_df['beta'], 
                         sub_df['mean_coal_time'] - sub_df['std_coal_time'], 
                         sub_df['mean_coal_time'] + sub_df['std_coal_time'], 
                         color=color, alpha=0.3)
                         
    plt.vlines(beta_c, np.nanmin(coalescence_times_matrix), np.nanmax(coalescence_times_matrix), 
               color='k', linestyle='--', label=r'$\beta_c$')
    plt.vlines(beta_BD, np.nanmin(coalescence_times_matrix), np.nanmax(coalescence_times_matrix), 
               color='k', linestyle=':', label=r'$\beta_{BD}$')
    plt.xlabel(r'Inverse Temperature $\beta$')
    plt.yscale('log')
    plt.ylabel('Coalescence Time')
    plt.legend()
    plt.savefig(save_path+f'{save_name}_vs_beta.png')
    plt.savefig(save_path+f'{save_name}_vs_beta.svg')
    plt.close()

    # --- Plot 2: Coalescence time vs N ---
    fig, ax = plt.subplots()
    for i, beta in enumerate(betas):
        sub_df = df[df['beta'] == beta].sort_values('N')
        color = cmc.berlin(i / len(betas))
        label_str = fr'$\beta$={beta:.2f}' if abs(beta - beta_c) > 1e-5 else fr'$\beta=\beta_c$'
        
        ax.plot(sub_df['N'], sub_df['mean_coal_time'], label=label_str, color=color)
        ax.fill_between(sub_df['N'], 
                        sub_df['mean_coal_time'] - sub_df['std_coal_time'], 
                        sub_df['mean_coal_time'] + sub_df['std_coal_time'], 
                        color=color, alpha=0.3)

    norm = mpl.colors.Normalize(vmin=0, vmax=max_beta)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmc.berlin)
    sm.set_array([])
    ticks = [0, beta_BD, beta_c, max_beta]
    tick_labels = ['0', r'$\beta_{BD}$', r'$\beta_c$', f'{max_beta:.2f}']

    if beta_no_zero is not None and beta_no_zero not in ticks:
        ticks.append(beta_no_zero)
        tick_labels.append(f'{beta_no_zero:.2f}*')

    sorted_pairs = sorted(zip(ticks, tick_labels))
    sorted_ticks = [p[0] for p in sorted_pairs]
    sorted_labels = [p[1] for p in sorted_pairs]
    
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_ticks(sorted_ticks)
    cbar.set_ticklabels(sorted_labels)
    cbar.set_label(r'$\beta$')

    ax.set_xlabel('System Size $N$')
    ax.set_yscale('log')
    ax.set_ylabel('Coalescence Time')
    fig.savefig(save_path+f'{save_name}_vs_N.png')
    fig.savefig(save_path+f'{save_name}_vs_N.svg')
    plt.close(fig)

    # --- Plot 3: log(T_coal)/N vs Beta ---
    plt.figure(3)
    for N in N_list:
        sub_df = df[df['N'] == N]
        color_val = (N - min(N_list)) / (max(N_list) - min(N_list)) if max(N_list) > min(N_list) else 0.5
        color = plt.cm.viridis(color_val)
        plt.plot(sub_df['beta'], sub_df['mean_log_coal_N'], label=f'$N$={N}', color=color)
        plt.fill_between(sub_df['beta'], 
                         sub_df['mean_log_coal_N'] - sub_df['std_log_coal_N'], 
                         sub_df['mean_log_coal_N'] + sub_df['std_log_coal_N'], 
                         color=color, alpha=0.3)
                         
    max_log_val = df['mean_log_coal_N'].max()
    y_max = max_log_val * 10 if max_log_val > 0 else 10 # Guard against extreme tight limits
    plt.vlines(beta_c, 0, y_max, color='k', linestyle='--', label=r'$\beta_c$')
    plt.vlines(beta_BD, 0, y_max, color='k', linestyle=':', label=r'$\beta_{BD}$')
    plt.xlabel(r'Inverse Temperature $\beta$')
    plt.ylabel(r'$\log(T_{coal})/N$')
    plt.legend()
    plt.savefig(save_path+f'{save_name}_log_T_over_N.png')
    plt.savefig(save_path+f'{save_name}_log_T_over_N.svg')
    plt.close()

    # --- Plot 4: log(T_coal)/N vs N at fixed beta ---
    fig, ax = plt.subplots()
    for i, beta in enumerate(betas):
        sub_df = df[df['beta'] == beta].sort_values('N')
        color = cmc.berlin(i / len(betas))
        label_str = fr'$\beta$={beta:.2f}' if abs(beta - beta_c) > 1e-5 else fr'$\beta=\beta_c$'
        
        ax.plot(sub_df['N'], sub_df['mean_log_coal_N'], label=label_str, color=color)
        ax.fill_between(sub_df['N'], 
                        sub_df['mean_log_coal_N'] - sub_df['std_log_coal_N'], 
                        sub_df['mean_log_coal_N'] + sub_df['std_log_coal_N'], 
                        color=color, alpha=0.3)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_beta)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmc.berlin)
    sm.set_array([])
    ticks = [0, beta_BD, beta_c, max_beta]
    tick_labels = ['0', r'$\beta_{BD}$', r'$\beta_c$', f'{max_beta:.2f}']
    if beta_no_zero is not None and beta_no_zero not in ticks:
        ticks.append(beta_no_zero)
        tick_labels.append(f'{beta_no_zero:.2f}*')
    sorted_pairs = sorted(zip(ticks, tick_labels))
    sorted_ticks = [p[0] for p in sorted_pairs]
    sorted_labels = [p[1] for p in sorted_pairs]
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_ticks(sorted_ticks)
    cbar.set_ticklabels(sorted_labels)
    cbar.set_label(r'$\beta$')
    ax.set_xlabel('System Size $N$')
    ax.set_ylabel(r'$\log(T_{coal})/N$')
    fig.savefig(save_path+f'{save_name}_log_T_over_N_vs_N.png')
    fig.savefig(save_path+f'{save_name}_log_T_over_N_vs_N.svg')
    plt.close(fig)

def Physics(model, N_list, beta_list, save_name, n_runs=25, save_path=save_path, **kwargs):
    """
    Runs CFTP sampling and theoretical calculations for the specified model.
    model options: 'CW' (Curie-Weiss), 'ER' (Erdős-Rényi), 'SK' (Sherrington-Kirkpatrick)
    """
    all_data = []
    
    for N in N_list:
        print(f"Processing {model} model with N={N}...")
        
        # Setup Graph and Couplings based on model
        if model == 'CW':
            G = nx.complete_graph(N)
            couplings = (np.ones((N, N)) - np.eye(N)) / N
        elif model == 'ER' or model == 'RR':
            d = kwargs.get('d', 4)
            if model == 'ER':
                G = nx.erdos_renyi_graph(N, d/N)
            if model == 'RR':
                G = nx.random_regular_graph(d, N)
            couplings = np.triu(np.random.choice([-1, 1], size=(N, N)), k=1)
            couplings += couplings.T
            couplings = couplings * nx.to_numpy_array(G)
        elif model == 'SK':
            G = nx.complete_graph(N)
            couplings = np.random.normal(0, 1/np.sqrt(N), size=(N, N))
            couplings = (couplings + couplings.T) / 2
            np.fill_diagonal(couplings, 0)
        else:
            raise ValueError("Invalid model. Choose 'CW', 'ER', 'RR', or 'SK'.")

        for b in beta_list:
            print(f"  Processing beta={b:.3f}...")
            temp_metrics = {'mag': [], 'energy': [], 'q': []}
            
            for _ in range(n_runs):
                if model == 'CW':
                    config, time, _ = CFTP_MCMC2_BC(beta=b, G=G, coupling=couplings)
                    temp_metrics['mag'].append(np.abs(np.nanmean(config)))
                else:
                    config, time, _ = CFTP_MCMC2_BC(beta=b, G=G, coupling=couplings)
                    # Energy calculation
                    energy = 0
                    for i, j in G.edges():
                        energy -= couplings[i, j] * config[i] * config[j]
                    temp_metrics['energy'].append(energy / N)
                    temp_metrics['q'].append(np.mean(config)**2)
                    
            all_data.append({
                'model': model,
                'N': N,
                'beta': b,
                'mag_mean': np.nanmean(temp_metrics['mag']) if model == 'CW' else np.nan,
                'mag_var': np.nanvar(temp_metrics['mag']) if model == 'CW' else np.nan,
                'energy_mean': np.nanmean(temp_metrics['energy']) if model in ['ER', 'RR', 'SK'] else np.nan,
                'energy_var': np.nanvar(temp_metrics['energy']) if model in ['ER', 'RR', 'SK'] else np.nan,
                'q_mean': np.nanmean(temp_metrics['q']) if model in ['ER', 'RR', 'SK'] else np.nan,
                'q_var': np.nanvar(temp_metrics['q']) if model in ['ER', 'RR', 'SK'] else np.nan,
            })
            
    df = pd.DataFrame(all_data)
    if save_name == "CW" or save_name == "SK":
        save_path+="data/physics/complet/"
    elif save_name == "ER" or save_name == "RR":
        save_path+="data/physics/random/"

    csv_path = save_path+f"{save_name}_physics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    return df

def Plot_physics(model, save_name, df=None, save_path=save_path, **kwargs):
    """
    Generates plots comparing theoretical predictions with MCMC data.
    """
    if save_name == "CW" or save_name == "SK":
        access_path = save_path+"data/physics/complet/"+f"{save_name}_physics"
        save_path+="figures/physics/complet/"
    elif save_name == "ER" or save_name == "RR":
        access_path = save_path+"data/physics/random/"+f"{save_name}_physics"
        save_path+="figures/physics/random/"

    if df is None:
        df = pd.read_csv(f"{access_path}.csv")
        
    N_list = sorted(df['N'].unique())
    marker_lst = ['o', 's', 'D', 'P', 'X']
    color_lst = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']
    
    if model == 'CW':
        fig, axes = plt.subplots(nrows=len(N_list), ncols=1, sharex=True, figsize=(8, 8))
        beta_theo = np.linspace(0, 1.3, 100)
        
        for i, N in enumerate(N_list):
            ax = axes[i] if len(N_list) > 1 else axes
            sub_df = df[df['N'] == N]
            
            theo_mags = [theoretical_magnetization_CW(b, N) for b in beta_theo]
            ax.plot(beta_theo, theo_mags, color=color_lst[i % len(color_lst)], linestyle='-')
            ax.errorbar(sub_df['beta'], sub_df['mag_mean'], yerr=np.sqrt(sub_df['mag_var']), 
                        marker=marker_lst[i % len(marker_lst)], color=color_lst[i % len(color_lst)], linestyle='None')
            
            ax.vlines(1.0, -0.05, 1.05, color='k', linestyle='--', label=r'$\beta_c$' if i == 0 else None)
            if i < 3: ax.axhline(0, color='k', linestyle='-', linewidth=1)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel(r'$m(\beta)$'+f'\n$N$={N}', rotation=0, labelpad=20)
            if i == 0: ax.legend(loc='upper left')
            
        plt.xlabel(r'Inverse Temperature $\beta$')
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1, hspace=0)
        plt.savefig(save_path+f"{save_name}_magnetization.png")
        plt.savefig(save_path+f"{save_name}_magnetization.svg")
        
    elif model == 'ER' or model == 'RR':
        d = kwargs.get('d', 4)
        if model == 'ER':
            beta_SG = np.arctanh(np.sqrt(1/d))
            beta_obs = np.arctanh(1/d)
            deg_max = [np.floor(max_deg_ER_graph(N, d)) for N in N_list]
            beta_BD = [np.arctanh(1/deg) for deg in deg_max]
        else:
            beta_SG = np.arctanh(np.sqrt(1/(d-1)))
            beta_BD = np.arctanh(1/d)
        betas_theo = np.linspace(0, beta_SG*1.1, 100)
        theo_energy = np.array([-d/2 * np.tanh(b) for b in betas_theo])
        
        plt.figure()
        plt.plot(betas_theo[betas_theo < beta_SG], theo_energy[betas_theo < beta_SG], '-r', label='RS Solution')
        plt.plot(betas_theo[betas_theo >= beta_SG], theo_energy[betas_theo >= beta_SG], '--r')
        
        for i, N in enumerate(N_list):
            sub_df = df[df['N'] == N]
            plt.errorbar(sub_df['beta'], sub_df['energy_mean'], yerr=np.sqrt(sub_df['energy_var']), 
                         marker=marker_lst[i % len(marker_lst)], color=color_lst[i % len(color_lst)], 
                         label=f'N={N}', linestyle='None')  
            if model == 'ER':
                plt.axvspan(np.min(beta_BD), np.max(beta_BD), color='lightgray', alpha=0.5, label=r'$\beta_{BD}$ region' if i == 0 else None)
        plt.vlines(beta_obs if model=="ER" else beta_BD, np.min(theo_energy)*1.2, 0.1, color='k', linestyle=':', label=r'$\tanh^{-1}(1/d)$' if model=="ER" else r'$\beta_{BD}$')
        plt.vlines(beta_SG, np.min(theo_energy)*1.2, 0.1, color='k', linestyle='--', label=r'$\beta_{SG}$')
        plt.ylim(np.min(theo_energy)*1.1, 0.05)
        plt.xlabel(r'Inverse Temperature $\beta$')
        plt.ylabel('Energy per Spin')
        plt.legend(loc=(0.62, 0.46))
        plt.savefig(save_path+f"{save_name}_energy.png")
        plt.savefig(save_path+f"{save_name}_energy.svg")
        
    elif model == 'SK':
        beta_SG = 1
        betas_theo = np.linspace(0, 1.1, 100)
        theo_energy = np.array([-b/2 for b in betas_theo])
        theo_q = np.zeros_like(betas_theo)
        
        # Plot Energy
        fig1, ax1 = plt.subplots()
        ax1.plot(betas_theo[betas_theo <= beta_SG], theo_energy[betas_theo <= beta_SG], '-r', label='RS Solution')
        ax1.plot(betas_theo[betas_theo > beta_SG], theo_energy[betas_theo > beta_SG], '--r')
        
        # Plot Q
        fig2, ax2 = plt.subplots()
        ax2.plot(betas_theo[betas_theo <= beta_SG], theo_q[betas_theo <= beta_SG], '-r', label='RS Solution')
        ax2.plot(betas_theo[betas_theo > beta_SG], theo_q[betas_theo > beta_SG], '--r')
        
        for i, N in enumerate(N_list):
            sub_df = df[df['N'] == N]
            ax1.errorbar(sub_df['beta'], sub_df['energy_mean'], yerr=np.sqrt(sub_df['energy_var']), 
                         marker=marker_lst[i % len(marker_lst)], color=color_lst[i % len(color_lst)], label=f'N={N}', linestyle='None')
            ax2.errorbar(sub_df['beta'], sub_df['q_mean'], yerr=np.sqrt(sub_df['q_var']), 
                         marker=marker_lst[i % len(marker_lst)], color=color_lst[i % len(color_lst)], label=f'N={N}', linestyle='None')
                         
        ax1.vlines(beta_SG, np.min(theo_energy)*1.2, 0.1, color='g', linestyle=':', label=r'$\beta_{SG}$')
        ax1.set_ylim(np.min(theo_energy)*1.1, 0.05)
        ax1.set_xlabel(r'Inverse Temperature $\beta$')
        ax1.set_ylabel('Energy per Spin')
        ax1.legend()
        fig1.savefig(save_path+f"{save_name}_energy.png")
        fig1.savefig(save_path+f"{save_name}_energy.svg")
        
        ax2.vlines(beta_SG, -0.1, 0.1, color='g', linestyle=':', label=r'$\beta_{SG}$')
        ax2.set_xlabel(r'Inverse Temperature $\beta$')
        ax2.set_ylabel(r'$q_{EA}$')
        ax2.legend()
        fig2.savefig(save_path+f"{save_name}_q.png")
        fig2.savefig(save_path+f"{save_name}_q.svg")
