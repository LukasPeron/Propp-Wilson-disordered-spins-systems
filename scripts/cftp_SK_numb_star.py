"""
This script runs the bounding chain algorithm forward in time to investigate the convergence properties of the bounding chain itself. We apply this to the SK model and observe how the number of "star" states (undecided spins) evolves over time for different inverse temperatures (betas). This can provide insights into the mixing time and convergence behavior of the CFTP algorithm.

Author: L. Péron
"""

from cftp_bc import *

N = 100
G = nx.complete_graph(N)
couplings = np.random.normal(0, 1/np.sqrt(N), size=(N, N))
couplings = (couplings + couplings.T) / 2  # Ensure symmetry J_ij = J_ji
np.fill_diagonal(couplings, 0)  # No self-interaction

beta_BD = solve_self_consistent_beta(N, initial_guess=0)
print(f"Bubley-Dyer bound for N={N}: beta_BD = {beta_BD:.4f}")
max_beta = beta_BD * 3
betas_lower = np.linspace(0, beta_BD, 4)[:-1]
betas_upper = np.linspace(beta_BD, max_beta, 10, endpoint=True)
betas = np.concatenate((betas_lower, betas_upper))

# Variable to store the lowest beta where the star states never reach 0
beta_no_zero = None
norm = mpl.colors.Normalize(vmin=0, vmax=max_beta)

fig, ax = plt.subplots()

for j, beta in enumerate(betas):
    print(f"Processing beta={beta:.3f}...")
    temp_state = []
    temp_time = []
    
    for _ in range(25):
        nb_star_states, __ = BC_fwd(beta, G, couplings)
        temp_time.append(len(nb_star_states))
        temp_state.append(nb_star_states)
        
    max_time = max(temp_time)
    
    for i in range(len(temp_state)):
        if len(temp_state[i]) < max_time:
            temp_state[i] += [temp_state[i][-1]] * (max_time - len(temp_state[i]))
            
    time = np.arange(max_time)
    mean_state = np.nanmean(temp_state, axis=0)
    std_state = np.nanstd(temp_state, axis=0)
    
    # Check if this is the lowest beta where the mean star states never reach 0
    if beta_no_zero is None and np.min(mean_state) > 0:
        beta_no_zero = beta
        
    color = cmc.berlin(norm(beta))
    
    # Plotting lines without labels since we are using a colorbar
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

# Define standard ticks based on the SK model parameters
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

plt.savefig(save_path+f'algo/SK_number_star_state_N_{N}.png')
plt.savefig(save_path+f'algo/SK_number_star_state_N_{N}.svg')