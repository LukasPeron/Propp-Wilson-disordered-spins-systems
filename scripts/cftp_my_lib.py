"""
This script implements all the needed function for the project on the perfect simulation for disordered spin systems. 
"""

import numpy as np
from numba import njit
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

@njit
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

def theoretical_mag_lattice(beta, d):
    # For the 2D lattice, we can use the Onsager solution for the infinite lattice as an approximation
    # The critical beta for the 2D Ising model is beta_c = log(1 + sqrt(2)) / 2
    if d==2:
        beta_c = np.log(1 + np.sqrt(2)) / 2
        if beta < beta_c:
            return 0.0
        else:
            return (1 - np.sinh(2 * beta)**(-4))**(1/8)
    
    # for the 3D lattice, there is no exact solution, but we can use numerical estimates for the critical beta
    # beta_c for 3D is approximately 0.2216544
    if d==3:
        beta_c_3D = 0.2216544
        if beta < beta_c_3D:
            return 0.0
        else:
            # We can use a simple power law fit for the magnetization near the critical point
            return (1 - (beta_c_3D / beta)**2)**0.326419

@njit
def _cftp_numba_core(beta, N, adj_indices, adj_indptr, adj_weights, adj_abs_weights, M_init, max_iter):
    # Pre-allocate random arrays to their maximum possible size to avoid resizing
    v_rand = np.empty(max_iter, dtype=np.int32)
    s_rand = np.empty(max_iter, dtype=np.int8)
    r_rand = np.empty(max_iter, dtype=np.float64)
    
    rand_generated_count = 0
    T_start = 1  # Represents time t = -1
    
    # We will declare this here so it survives the while loop
    final_nb_stars = np.zeros(0, dtype=np.int32) 
    
    while T_start <= max_iter:
        # 1. Generate NEW random variables ONLY for the newly expanded time window
        # Index i corresponds to time -(i+1). (e.g., index 0 is t = -1)
        while rand_generated_count < T_start:
            v_rand[rand_generated_count] = np.random.randint(0, N)
            s_rand[rand_generated_count] = 1 if np.random.rand() < 0.5 else -1
            r_rand[rand_generated_count] = np.random.rand()
            rand_generated_count += 1
            
        # 2. Re-initialize the state for this new CFTP attempt
        config = np.zeros(N, dtype=np.int8)  # 0 represents the 'star' state
        num_stars = N
        H_bar = np.zeros(N, dtype=np.float64)
        M = M_init.copy()
        
        nb_star_state = np.zeros(T_start, dtype=np.int32)
        
        # 3. Run the simulation FORWARD from -T_start up to time -1
        step_idx = 0
        # Iterate backwards through our random array (which moves forward in time)
        for i in range(T_start - 1, -1, -1):
            nb_star_state[step_idx] = num_stars
            step_idx += 1
            
            v = v_rand[i]
            s = s_rand[i]
            r = r_rand[i]
            
            current_state = config[v]
            
            if current_state != s:
                h_bar = H_bar[v]
                m = M[v]

                h_minus = h_bar + s * m
                h_plus = h_bar - s * m

                # Evaluate Transition
                if r < F_beta_Glauber(beta, -2 * s * h_plus):
                    new_state = s
                elif r > F_beta_Glauber(beta, -2 * s * h_minus) and current_state == -s:
                    new_state = -s
                else:
                    new_state = 0

                # Dynamic Updates (O(1) execution if state changes)
                if current_state != new_state:
                    config[v] = new_state

                    if current_state == 0:
                        num_stars -= 1
                    elif new_state == 0:
                        num_stars += 1

                    start_idx = adj_indptr[v]
                    end_idx = adj_indptr[v + 1]
                    for j in range(start_idx, end_idx):
                        u = adj_indices[j]
                        w = adj_weights[j]
                        aw = adj_abs_weights[j]

                        # Strip old state
                        if current_state != 0:
                            H_bar[u] -= w * current_state
                        else:
                            M[u] -= aw

                        # Inject new state
                        if new_state != 0:
                            H_bar[u] += w * new_state
                        else:
                            M[u] += aw

        # 4. Check for coalescence exactly at t = 0 (after the loop)
        if num_stars == 0:
            return config, -T_start, nb_star_state
            
        # 5. If not coalesced, double the time window and try again
        T_start *= 2
        final_nb_stars = nb_star_state # Keep the longest trajectory for debugging
        
    # Failure condition (t > 0 used as a flag for failure)
    return np.zeros(N, dtype=np.int8), 1, final_nb_stars

def CFTP_MCMC2_BC(beta, G, couplings, max_nb_iter_MCMC2_BC=max_nb_iter_MCMC2_BC):
    N = G.number_of_nodes()

    # NEW: Map tuple/arbitrary nodes to integer indices 0...N-1
    # This guarantees perfect alignment with the couplings matrix
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # 1. Flatten the NetworkX Graph (CSR Format)
    edges_count = sum(1 for v in node_list for _ in G.neighbors(v))
    adj_indices = np.zeros(edges_count, dtype=np.int32)
    adj_weights = np.zeros(edges_count, dtype=np.float64)
    adj_abs_weights = np.zeros(edges_count, dtype=np.float64)
    adj_indptr = np.zeros(N + 1, dtype=np.int32)
    M_init = np.zeros(N, dtype=np.float64)
    
    ptr = 0
    for i, v_node in enumerate(node_list):
        adj_indptr[i] = ptr
        for u_node in G.neighbors(v_node):
            j = node_to_idx[u_node] # Convert neighbor node name to integer index
            
            # Access couplings using the integer indices
            w = couplings[j, i] 
            aw = np.abs(w)
            
            adj_indices[ptr] = j
            adj_weights[ptr] = w
            adj_abs_weights[ptr] = aw
            ptr += 1
            
            M_init[i] += np.abs(couplings[i, j])
            
    adj_indptr[N] = ptr
    
    # 2. Run the Numba Core (Untouched)
    config, final_t, nb_star_state = _cftp_numba_core(
        beta, N, 
        adj_indices, adj_indptr, adj_weights, adj_abs_weights, 
        M_init, max_nb_iter_MCMC2_BC
    )
    
    # 3. Handle Output and Failures cleanly
    if final_t > 0: # Custom flag indicating max_iter was hit without coalescence
        print("Warning: CFTP did not coalesce after a large number of iterations.")
        return np.full(N, np.nan), np.nan, nb_star_state.tolist()
        
    print(f"Coalescence achieved at time {final_t}.")
    return config, final_t, nb_star_state.tolist()

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

@njit
def _mcmc_numba_core(beta, N, adj_indices, adj_indptr, adj_weights, adj_abs_weights, M_init, max_iter):
    config = np.zeros(N, dtype=np.int8)
    num_stars = N
    
    H_bar = np.zeros(N, dtype=np.float64)
    M = M_init.copy()
    
    nb_star_state = np.zeros(max_iter + 1, dtype=np.int32)
    nb_star_state[0] = num_stars
    
    for timestep in range(max_iter):
        # Numba is extremely fast at scalar random generation
        v = np.random.randint(0, N)
        s = 1 if np.random.rand() < 0.5 else -1
        r = np.random.rand()
        
        current_state = config[v]

        if current_state != s:
            h_bar = H_bar[v]
            m = M[v]

            h_minus = h_bar + s * m
            h_plus = h_bar - s * m

            # Evaluate Transition
            if r < F_beta_Glauber(beta, -2 * s * h_plus):
                new_state = s
            elif r > F_beta_Glauber(beta, -2 * s * h_minus) and current_state == -s:
                new_state = -s
            else:
                new_state = 0

            # Apply Updates
            if current_state != new_state:
                config[v] = new_state

                if current_state == 0:
                    num_stars -= 1
                elif new_state == 0:
                    num_stars += 1

                # Loop through neighbors using the CSR-style index pointers
                start_idx = adj_indptr[v]
                end_idx = adj_indptr[v + 1]
                
                for i in range(start_idx, end_idx):
                    u = adj_indices[i]
                    w = adj_weights[i]
                    aw = adj_abs_weights[i]

                    # Strip old state
                    if current_state != 0:
                        H_bar[u] -= w * current_state
                    else:
                        M[u] -= aw

                    # Inject new state
                    if new_state != 0:
                        H_bar[u] += w * new_state
                    else:
                        M[u] += aw

        nb_star_state[timestep + 1] = num_stars

        # Coalescence check
        if num_stars == 0:
            # Slicing creates a new array up to the current timestep
            return nb_star_state[:timestep + 2], timestep

    return nb_star_state, np.nan

def MCMC2_BC_fwd(beta, G, couplings, max_nb_iter_MCMC2_BC=max_nb_iter_MCMC2_BC):
    N = G.number_of_nodes()
    
    # NEW: Map tuple nodes from grid_graph to integer indices 0...N-1
    # nx.to_numpy_array(G) naturally uses list(G.nodes()) for its ordering, 
    # so we must match this exact sequence to keep couplings aligned.
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # 1. Count total edges to allocate flat arrays
    edges_count = sum(1 for v in node_list for _ in G.neighbors(v))
    
    # 2. Allocate CSR-style flat arrays
    adj_indices = np.zeros(edges_count, dtype=np.int32)
    adj_weights = np.zeros(edges_count, dtype=np.float64)
    adj_abs_weights = np.zeros(edges_count, dtype=np.float64)
    adj_indptr = np.zeros(N + 1, dtype=np.int32)
    M_init = np.zeros(N, dtype=np.float64)
    
    # 3. Populate arrays using the mapping
    ptr = 0
    for i, v_node in enumerate(node_list):
        adj_indptr[i] = ptr
        for u_node in G.neighbors(v_node):
            j = node_to_idx[u_node] # Get integer index of the neighbor
            
            # Access couplings using the integer indices
            w = couplings[j, i]
            aw = np.abs(w)
            
            adj_indices[ptr] = j
            adj_weights[ptr] = w
            adj_abs_weights[ptr] = aw
            ptr += 1
            
            M_init[i] += np.abs(couplings[i, j])
            
    adj_indptr[N] = ptr # Cap off the pointer array
    
    # 4. Pass compiled structures to the high-speed core
    result_states, final_step = _mcmc_numba_core(
        beta, N, 
        adj_indices, adj_indptr, adj_weights, adj_abs_weights, 
        M_init, max_nb_iter_MCMC2_BC
    )
    
    if final_step < max_nb_iter_MCMC2_BC:
        print(f"Coalescence achieved at iteration {final_step}")
    else:
        print("Warning: Coalescence not achieved after max iterations.")
        
    return list(result_states), final_step

def Nb_star(N, beta, G, couplings, beta_c, max_beta, save_name, n_runs=25, save_path=save_path, **kwargs):
    """
    Runs the bounding chain algorithm forward in time and saves the data to a CSV.
    """
    # Create the beta array. We ensure we capture points around 0 to max_beta.
    # You can adjust the density of the linspace as needed.
    d = kwargs.get('d', None)  # Extract dimension if provided, default to None
    all_data = []
    # for beta in betas:
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
    
    # open the CSV to write the data in it if it already exists, otherwise create it and write the data
    if save_name == "CW" or save_name == "SK":
        save_path+="data/algo/complet/"
    elif save_name == "ER" or save_name == "RR":
        save_path+="data/algo/random/"
    elif "RL" in save_name:
        save_path+=f"data/algo/latt/d{d}"
        if "ferr" in save_name:
            save_path+="/ferr/"
        elif "sg" in save_name:
            save_path+="/sg/"

    csv_path = save_path+f"{save_name}_nb_star.csv"
    try:
        df_existing = pd.read_csv(csv_path)
        df_new = pd.DataFrame(all_data)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
        print(f"Data successfully appended to {csv_path}")
    except FileNotFoundError:
        df = pd.DataFrame(all_data)
        df.to_csv(csv_path, index=False)
        print(f"Data successfully saved to {csv_path}")

def Plot_nb_star(save_name, beta_BD, beta_c, max_beta, df=None, save_path=save_path, **kwargs):
    """
    Reads the generated data points and creates the convergence plot.
    """
    # Load from CSV if the dataframe isn't directly passed
    d = kwargs.get('d', None)  # Extract dimension if provided, default to None
    if save_name == "CW" or save_name == "SK":
        access_path = save_path+"data/algo/complet/"+f"{save_name}_nb_star"
        save_path+="figures/algo/complet/"
    elif save_name == "ER" or save_name == "RR":
        access_path = save_path+"data/algo/random/"+f"{save_name}_nb_star"
        save_path+="figures/algo/random/"
    elif "RL" in save_name:
        access_path = save_path+f"data/algo/latt/d{d}/"
        if "ferr" in save_name:
            access_path+="ferr/"+f"{save_name}_nb_star"
            save_path+=f"figures/algo/latt/d{d}/ferr/"
        elif "sg" in save_name:
            access_path+="sg/"+f"{save_name}_nb_star"
            save_path+=f"figures/algo/latt/d{d}/sg/"

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
        cmap = cmc.berlin

        ax.plot(time, mean_state, color=color)
        ax.fill_between(time, 
                        np.maximum(0, mean_state - std_state), 
                        mean_state + std_state, 
                        color=color, alpha=0.3, edgecolor=None)

    ax.set_xscale('log')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'Number of $\star$ States')

    # --- Colorbar implementation ---
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Define standard ticks including Bubley-Dyer and Phase Transition betas
    ticks = [0, beta_BD, max_beta]
    tick_labels = [r'$\beta=0$', r'$\beta_{BD}$', f'{max_beta:.2f}']

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

def Coal_time(N_list, beta, d, beta_c, max_beta, save_name, n_runs=25, save_path=save_path):
    """
    Runs the coalescence time analysis for the Curie-Weiss model and saves data to a CSV.
    """
    
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
        elif "RL" in save_name:
            if d == 2:
                L = int(round(np.sqrt(N)))
                dim = (L, L)
            if d == 3:
                L = int(round(N**(1/3)))
                dim = (L, L, L)
            G = nx.grid_graph(dim)
            actual_N = G.number_of_nodes()
            if save_name == "RL_ferr":
                couplings = np.ones((actual_N, actual_N)) - np.eye(actual_N)
            elif save_name == "RL_sg":
                couplings = np.random.choice([-1, 1], size=(actual_N, actual_N)) * nx.to_numpy_array(G)
                couplings = (couplings + couplings.T) / 2  # Ensure symmetry J_ij = J_ji
                np.fill_diagonal(couplings, 0)

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
            
    # open the CSV to write the data in it if it already exists, otherwise create it and write the data
    if save_name == "CW" or save_name == "SK":
        save_path+="data/algo/complet/"
    elif save_name == "ER" or save_name == "RR":
        save_path+="data/algo/random/"
    elif "RL" in save_name:
        save_path+=f"data/algo/latt/d{d}"
        if "ferr" in save_name:
            save_path+="/ferr/"
        elif "sg" in save_name:
            save_path+="/sg/"

    csv_path = save_path+f"{save_name}_coal_time.csv"
    try:
        df_existing = pd.read_csv(csv_path)
        df_new = pd.DataFrame(all_data)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
        print(f"Data successfully appended to {csv_path}")
    except FileNotFoundError:
        df = pd.DataFrame(all_data)
        df.to_csv(csv_path, index=False)
        print(f"Data successfully saved to {csv_path}")

def Plot_coal_time(save_name, beta_BD, beta_c, max_beta, df=None, save_path=save_path, max_nb_iter_MCMC2_BC=max_nb_iter_MCMC2_BC, **kwargs):
    """
    Reads the generated data points and creates the 4 coalescence time plots.
    """
    d = kwargs.get('d', None)  # Extract dimension if provided, default to None
    if save_name == "CW" or save_name == "SK":
        access_path = save_path+"data/algo/complet/"+f"{save_name}_coal_time"
        save_path+="figures/algo/complet/"
    elif save_name == "ER" or save_name == "RR":
        access_path = save_path+"data/algo/random/"+f"{save_name}_coal_time"
        save_path+="figures/algo/random/"
    elif "RL" in save_name:
        access_path = save_path+f"data/algo/latt/d{d}/"
        if "ferr" in save_name:
            access_path+="ferr/"+f"{save_name}_coal_time"
            save_path+=f"figures/algo/latt/d{d}/ferr/"
        elif "sg" in save_name:
            access_path+="sg/"+f"{save_name}_coal_time"
            save_path+=f"figures/algo/latt/d{d}/sg/"

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

def Physics(model, N_list, beta, save_name, n_runs=25, save_path=save_path, **kwargs):
    """
    Runs CFTP sampling and theoretical calculations for the specified model.
    model options: 'CW' (Curie-Weiss), 'ER' (Erdős-Rényi), 'SK' (Sherrington-Kirkpatrick)
    """
    d = kwargs.get('d', None)  # Extract dimension if provided, default to None
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
        elif "RL" in model:
            if d == 2:
                L = int(round(np.sqrt(N)))
                dim = (L, L)
            if d == 3:
                L = int(round(N**(1/3)))
                dim = (L, L, L)
            G = nx.grid_graph(dim)
            actual_N = G.number_of_nodes()
            if "ferr" in model:
                couplings = np.ones((actual_N, actual_N)) - np.eye(actual_N)
            elif "sg" in model:
                couplings = np.random.choice([-1, 1], size=(actual_N, actual_N)) * nx.to_numpy_array(G)
                couplings = (couplings + couplings.T) / 2
                np.fill_diagonal(couplings, 0)
        else:
            raise ValueError("Invalid model. Choose 'CW', 'RL', 'ER', 'RR', or 'SK'.")

        temp_metrics = {'mag': [], 'energy': [], 'q': []}
        
        for _ in range(n_runs):
            if model == 'CW' or model == 'RL_ferr':
                config, time, _ = CFTP_MCMC2_BC(beta=beta, G=G, couplings=couplings)
                temp_metrics['mag'].append(np.abs(np.nanmean(config)))
            else:
                config, time, _ = CFTP_MCMC2_BC(beta=beta, G=G, couplings=couplings)
                node_list = list(G.nodes())
                node_to_idx = {node: idx for idx, node in enumerate(node_list)}
                energy = 0
                for u_node, v_node in G.edges():
                    i = node_to_idx[u_node]
                    j = node_to_idx[v_node]
                    energy -= couplings[i, j] * config[i] * config[j]
                temp_metrics['energy'].append(energy / N)
                temp_metrics['q'].append(np.mean(config)**2)
                
        all_data.append({
            'model': model,
            'N': N,
            'beta': beta,
            'mag_mean': np.nanmean(temp_metrics['mag']) if model == 'CW' or model=="RL_ferr" else np.nan,
            'mag_var': np.nanvar(temp_metrics['mag']) if model == 'CW' or model=="RL_ferr" else np.nan,
            'energy_mean': np.nanmean(temp_metrics['energy']) if model in ['ER', 'RR', 'SK', 'RL_sg'] else np.nan,
            'energy_var': np.nanvar(temp_metrics['energy']) if model in ['ER', 'RR', 'SK', 'RL_sg'] else np.nan,
            'q_mean': np.nanmean(temp_metrics['q']) if model in ['ER', 'RR', 'SK', 'RL_sg'] else np.nan,
            'q_var': np.nanvar(temp_metrics['q']) if model in ['ER', 'RR', 'SK', 'RL_sg'] else np.nan,
        })
            
    if save_name == "CW" or save_name == "SK":
        save_path+="data/physics/complet/"
    elif save_name == "ER" or save_name == "RR":
        save_path+="data/physics/random/"
    elif "RL" in save_name:
        save_path+=f"data/physics/latt/d{d}"
        if "ferr" in save_name:
            save_path+="/ferr/"
        elif "sg" in save_name:
            save_path+="/sg/"

    csv_path = save_path+f"{save_name}_physics.csv"
    try:
        df_existing = pd.read_csv(csv_path)
        df_new = pd.DataFrame(all_data)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
        print(f"Data successfully appended to {csv_path}")
    except FileNotFoundError:
        df = pd.DataFrame(all_data)
        df.to_csv(csv_path, index=False)
        print(f"Data successfully saved to {csv_path}")

def Plot_physics(model, save_name, df=None, save_path=save_path, **kwargs):
    """
    Generates plots comparing theoretical predictions with MCMC data.
    """
    d = kwargs.get('d', None)  # Extract dimension if provided, default to None
    if save_name == "CW" or save_name == "SK":
        access_path = save_path+"data/physics/complet/"+f"{save_name}_physics"
        save_path+="figures/physics/complet/"
    elif save_name == "ER" or save_name == "RR":
        access_path = save_path+"data/physics/random/"+f"{save_name}_physics"
        save_path+="figures/physics/random/"
    elif "RL" in save_name:
        access_path = save_path+f"data/physics/latt/d{d}/"
        if "ferr" in save_name:
            access_path+="ferr/"+f"{save_name}_physics"
            save_path+=f"figures/physics/latt/d{d}/ferr/"
        elif "sg" in save_name:
            access_path+="sg/"+f"{save_name}_physics"
            save_path+=f"figures/physics/latt/d{d}/sg/"

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
            if i < len(N_list)-1: 
                ax.axhline(0, color='k', linestyle='-', linewidth=1)
            ax.set_yticks([0])
            ax.set_ylim(-0.05, 1)
            ax.set_ylabel(r'$m(\beta)$'+f'\n$N$={N}' if i==0 else f'\n$N$={N}', rotation=0, labelpad=20)
            if i == 0: 
                ax.legend(loc='upper left')
            
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

    elif "RL" in model:
        if "ferr" in model:
            fig, axes = plt.subplots(nrows=len(N_list), ncols=1, sharex=True, figsize=(8, 8))
            beta_theo = np.linspace(0, 1.0, 100)
            
            for i, N in enumerate(N_list):
                ax = axes[i] if len(N_list) > 1 else axes
                sub_df = df[df['N'] == N]
                
                theo_mags = [theoretical_mag_lattice(beta, d) for beta in beta_theo]
                ax.plot(beta_theo, theo_mags, color=color_lst[i % len(color_lst)], linestyle='-')
                ax.errorbar(sub_df['beta'], sub_df['mag_mean'], yerr=np.sqrt(sub_df['mag_var']), 
                            marker=marker_lst[i % len(marker_lst)], color=color_lst[i % len(color_lst)], linestyle='None')
                
                if d == 2:
                    beta_c = 1/2 * np.log(1 + np.sqrt(2))
                elif d == 3:
                    beta_c = 0.2216544
                ax.vlines(beta_c, -0.05, 1.05, color='k', linestyle='--', label=r'$\beta_c$' if i == 0 else None)
                if i < len(N_list)-1: 
                    ax.axhline(0, color='k', linestyle='-', linewidth=1)
                ax.set_yticks([0])
                ax.set_ylim(-0.05, 1)
                ax.set_ylabel(r'$m(\beta)$'+f'\n$N$={N}' if i==0 else f'\n$N$={N}', rotation=0, labelpad=20)
                if i == 0: 
                    ax.legend(loc='upper left')
                
            plt.xlabel(r'Inverse Temperature $\beta$')
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1, hspace=0)
            plt.savefig(save_path+f"{save_name}_magnetization.png")
            plt.savefig(save_path+f"{save_name}_magnetization.svg")
        elif "sg" in model:
            d = kwargs.get('d', 4)
            if d == 2:
                beta_SG = 0
                beta_c = 1/2 * np.log(1 + np.sqrt(2))
            elif d == 3:
                beta_SG = 0.9
                beta_c = 0.2216544
            betas_theo = np.linspace(0, beta_SG*1.1, 100)
            theo_energy = np.array([-d/2 * np.tanh(beta) for beta in betas_theo])
            
            plt.figure()
            plt.plot(betas_theo[betas_theo < beta_SG], theo_energy[betas_theo < beta_SG], '-r', label='RS Solution')
            plt.plot(betas_theo[betas_theo >= beta_SG], theo_energy[betas_theo >= beta_SG], '--r')
            
            for i, N in enumerate(N_list):
                sub_df = df[df['N'] == N]
                plt.errorbar(sub_df['beta'], sub_df['energy_mean'], yerr=np.sqrt(sub_df['energy_var']), 
                             marker=marker_lst[i % len(marker_lst)], color=color_lst[i % len(color_lst)], 
                             label=f'N={N}', linestyle='None')  
                
            plt.vlines(beta_c, np.min(theo_energy)*1.2, 0.05, color='k', linestyle='--', label=r'$\beta_c$')
            if d == 3:
                plt.vlines(beta_SG, np.min(theo_energy)*1.2, 0.05, color='g', linestyle=':', label=r'$\beta_{SG}$')
            # plt.ylim(np.min(theo_energy)*1.1, 0.05)
            plt.xlabel(r'Inverse Temperature $\beta$')
            plt.ylabel('Energy per Spin')
            plt.legend(loc=(0.62, 0.46))
            plt.savefig(save_path+f"{save_name}_energy.png")
            plt.savefig(save_path+f"{save_name}_energy.svg")