"""
This script is made to analyze the coalescence time of the CFTP algorithm for the following spin systems:
- Sherrington-Kirkpatrick (SK) model (complete graph with Gaussian couplings)
Three plots are done for each model: a heatmap of the coalescence time as a function of beta and N, a plot of the coalescence time as a function of beta for different values of N, and a plot of the coalescence time as a function of N for different values of beta.
Author: L. Péron
"""

from cftp_bc import *

### Sherrington-Kirkpatrick (SK) model ###

N_values = [100, 200, 500, 750]

# Calculate a reference beta_BD using the largest N for our beta grid
N_ref = max(N_values)
beta_BD = solve_self_consistent_beta(N_ref, initial_guess=0)
print(f"Reference Bubley-Dyer bound for N={N_ref}: beta_BD = {beta_BD:.4f}")

max_beta = beta_BD * 3

# Split betas to get points around beta_BD
betas_lower = np.linspace(0, beta_BD, 4)[:-1]
betas_upper = np.linspace(beta_BD, max_beta, 10)
betas = np.concatenate((betas_lower, betas_upper))

T_coal_vs_beta_N = {}
log_T_coal_over_N_vs_beta_N = {}
beta_no_zero = None

for N in N_values:
    print(f"Processing SK model with N={N}...")
    G = nx.complete_graph(N)
    
    # Gaussian couplings for SK model
    couplings = np.random.normal(0, 1/np.sqrt(N), size=(N, N))
    couplings = (couplings + couplings.T) / 2  # Ensure symmetry J_ij = J_ji
    np.fill_diagonal(couplings, 0)  # No self-interaction
    
    coalescence_times = []
    coalescence_times_std = []
    log_T_coal_over_N = []
    log_T_coal_over_N_std = []
    
    for beta in betas:
        print(f"  Processing beta={beta:.3f}...")
        coal_time_temp = []
        log_T_coal_over_N_temp = []
        
        for _ in range(25):  # Run multiple trials to get an average coalescence time
            _, coal_time = BC_fwd(beta, G, couplings)
            coal_time_temp.append(coal_time)
            log_T_coal_over_N_temp.append(np.log(coal_time) / N)
            
        coalescence_times.append(np.nanmean(coal_time_temp))
        log_T_coal_over_N.append(np.nanmean(log_T_coal_over_N_temp))
        coalescence_times_std.append(np.sqrt(np.nanvar(coal_time_temp)))
        log_T_coal_over_N_std.append(np.sqrt(np.nanvar(log_T_coal_over_N_temp)))
        
        # Track the lowest beta where the mean coalescence time hits the 2**20 ceiling
        if beta_no_zero is None and np.nanmean(coal_time_temp) == 2**20:
            beta_no_zero = beta

    T_coal_vs_beta_N[N] = (coalescence_times, coalescence_times_std)
    log_T_coal_over_N_vs_beta_N[N] = (log_T_coal_over_N, log_T_coal_over_N_std)

# Plotting the results for the SK model

# Heatmap of coalescence time as a function of beta and N
plt.figure(0)
coalescence_times_matrix = np.array([T_coal_vs_beta_N[N][0] for N in sorted(T_coal_vs_beta_N.keys())])
plt.imshow(coalescence_times_matrix, aspect='auto', origin='lower', extent=[betas[0], betas[-1], min(T_coal_vs_beta_N.keys()), max(T_coal_vs_beta_N.keys())], cmap=plt.cm.viridis, norm=LogNorm())
plt.vlines(beta_BD, min(N_values), max(N_values), color='k', linestyle='--', label=r'$\beta_{BD}$')
plt.colorbar(label='Coalescence Time')
plt.xlabel(r'Inverse Temperature $\beta$')
plt.ylabel('System Size $N$')
plt.savefig(save_path + 'temp/SK_coal_time_heatmap.png')
plt.savefig(save_path + 'temp/SK_coal_time_heatmap.svg')

# Plot of coalescence time as a function of beta for different values of N
plt.figure(1)
for N in sorted(T_coal_vs_beta_N.keys()):
    coalescence_times, coalescence_times_std = T_coal_vs_beta_N[N]
    color = plt.cm.viridis((N - min(T_coal_vs_beta_N.keys())) / (max(T_coal_vs_beta_N.keys()) - min(T_coal_vs_beta_N.keys())))
    plt.plot(betas, coalescence_times, label=f'$N$={N}', color=color)
    plt.fill_between(betas, np.array(coalescence_times) - np.array(coalescence_times_std), np.array(coalescence_times) + np.array(coalescence_times_std), color=color, alpha=0.3)
plt.vlines(beta_BD, min(coalescence_times_matrix.flatten()), max(coalescence_times_matrix.flatten())*10, color='k', linestyle='--', label=r'$\beta_{BD}$')
plt.ylim(min(coalescence_times_matrix.flatten())/2, max(coalescence_times_matrix.flatten())*2)
plt.xlabel(r'Inverse Temperature $\beta$')
plt.yscale('log')
plt.ylabel('Coalescence Time')
plt.legend()
plt.savefig(save_path + 'temp/SK_coal_time_vs_beta.png')
plt.savefig(save_path + 'temp/SK_coal_time_vs_beta.svg')

# Plot of coalescence time as a function of N for different values of beta
fig, ax = plt.subplots()
for i, beta in enumerate(betas):
    coalescence_times = [T_coal_vs_beta_N[N][0][i] for N in sorted(T_coal_vs_beta_N.keys())]
    color = cmc.berlin(i / len(betas))
    ax.plot(sorted(T_coal_vs_beta_N.keys()), coalescence_times, label=fr'$\beta$={beta:.2f}' if beta!=beta_BD else r'$\beta_{BD}$', color=color)
    ax.fill_between(sorted(T_coal_vs_beta_N.keys()), np.array(coalescence_times) - np.array([T_coal_vs_beta_N[N][1][i] for N in sorted(T_coal_vs_beta_N.keys())]), np.array(coalescence_times) + np.array([T_coal_vs_beta_N[N][1][i] for N in sorted(T_coal_vs_beta_N.keys())]), color=color, alpha=0.3)

# Add custom Colorbar implementation
norm = mpl.colors.Normalize(vmin=0, vmax=max_beta)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmc.berlin)
sm.set_array([])
ticks = [0, beta_BD, max_beta]
tick_labels = ['0', r'$\beta_{BD}$', f'{max_beta:.2f}']

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
# Legend removed to favor the new colorbar
fig.savefig(save_path + 'temp/SK_coal_time_vs_N.png')
fig.savefig(save_path + 'temp/SK_coal_time_vs_N.svg')

# Plot of log(T_coal/N) as a function of beta for different values of N
plt.figure(3)
for N in sorted(log_T_coal_over_N_vs_beta_N.keys()):
    log_T_coal_over_N, log_T_coal_over_N_std = log_T_coal_over_N_vs_beta_N[N]
    color = plt.cm.viridis((N - min(log_T_coal_over_N_vs_beta_N.keys())) / (max(log_T_coal_over_N_vs_beta_N.keys()) - min(log_T_coal_over_N_vs_beta_N.keys())))
    plt.plot(betas, log_T_coal_over_N, label=f'$N$={N}', color=color)
    plt.fill_between(betas, np.array(log_T_coal_over_N) - np.array(log_T_coal_over_N_std), np.array(log_T_coal_over_N) + np.array(log_T_coal_over_N_std), color=color, alpha=0.3)
plt.vlines(beta_BD, min(log_T_coal_over_N)-1, max(log_T_coal_over_N)+1, color='k', linestyle='--', label=r'$\beta_{BD}$')
plt.xlabel(r'Inverse Temperature $\beta$')
plt.ylabel(r'$\log(T_{coal})/N$')
plt.legend()
plt.savefig(save_path + 'temp/SK_log_T_coal_over_N_vs_beta.png')
plt.savefig(save_path + 'temp/SK_log_T_coal_over_N_vs_beta.svg')
