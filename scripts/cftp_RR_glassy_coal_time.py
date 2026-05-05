"""
This script is made to analyze the coalescence time of the CFTP algorithm for the following spin systems:
- Erdős-Rényi graph with random couplings
Three plot are done for each model: a heatmap of the coalescence time as a function of beta and N, a plot of the coalescence time as a function of beta for different values of N, and a plot of the coalescence time as a function of N for different values of beta.
Author: L. Péron
"""

from cftp_bc import *

### Curie-Weiss model (complete graph with uniform couplings) ###

d = 4
beta_BD = np.arctanh(1/d)
betas_lower = np.linspace(0, beta_BD, 4)[:-1]
max_beta = beta_BD * 2
betas_upper = np.linspace(beta_BD, max_beta, 10)
betas = np.concatenate((betas_lower, betas_upper))

T_coal_vs_beta_N = {}
log_T_coal_over_N_vs_beta_N = {}
beta_no_zero = None

for N in [100, 200, 500, 750]:
    print(f"Processing RR model with N={N}...")
    G = nx.random_regular_graph(d, N)
    couplings = np.triu(np.random.choice([-1, 1], size=(N, N)), k=1)
    couplings += couplings.T  # account for the fact that the graph is not oriented
    couplings = couplings * nx.to_numpy_array(G)  # set couplings to zero for non-edges
    coalescence_times = []
    coalescence_times_std = []
    log_T_coal_over_N = []
    log_T_coal_over_N_std = []
    for beta in betas:
        print(f"  Processing beta={beta:.2f}...")
        coal_time_temp = []
        log_T_coal_over_N_temp = []
        for _ in range(25):  # Run multiple trials to get an average coalescence time
            _, coal_time= BC_fwd(beta, G, couplings)
            coal_time_temp.append(coal_time)
            log_T_coal_over_N_temp.append(np.log(coal_time) / N)
        coalescence_times.append(np.nanmean(coal_time_temp))
        log_T_coal_over_N.append(np.nanmean(log_T_coal_over_N_temp))
        coalescence_times_std.append(np.sqrt(np.nanvar(coal_time_temp)))
        log_T_coal_over_N_std.append(np.sqrt(np.nanvar(log_T_coal_over_N_temp)))
        if beta_no_zero is None and np.nanmean(coal_time_temp) == 2**20:
            beta_no_zero = beta
    T_coal_vs_beta_N[N] = (coalescence_times, coalescence_times_std)
    log_T_coal_over_N_vs_beta_N[N] = (log_T_coal_over_N, log_T_coal_over_N_std)

# Plotting the results for the Curie-Weiss model

# Heatmap of coalescence time as a function of beta and N

plt.figure(0)
coalescence_times_matrix = np.array([T_coal_vs_beta_N[N][0] for N in sorted(T_coal_vs_beta_N.keys())])
plt.imshow(coalescence_times_matrix, aspect='auto', origin='lower', extent=[betas[0], betas[-1], min(T_coal_vs_beta_N.keys()), max(T_coal_vs_beta_N.keys())], cmap=plt.cm.viridis, norm=LogNorm())
plt.vlines(beta_BD, 100, 750, color='k', linestyle='--', label=r'$\beta=\tanh^{-1}(1/d)$')
plt.colorbar(label='Coalescence Time')
plt.xlabel(r'Inverse Temperature $\beta$')
plt.ylabel('System Size $N$')
plt.savefig(save_path+f'temp/RR_coal_time_heatmap_d_{d}.png')
plt.savefig(save_path+f'temp/RR_coal_time_heatmap_d_{d}.svg')

# Plot of coalescence time as a function of beta for different values of N

plt.figure(1)
for N in sorted(T_coal_vs_beta_N.keys()):
    coalescence_times, coalescence_times_std = T_coal_vs_beta_N[N]
    color = plt.cm.viridis((N - min(T_coal_vs_beta_N.keys())) / (max(T_coal_vs_beta_N.keys()) - min(T_coal_vs_beta_N.keys())))
    plt.plot(betas, coalescence_times, label=f'$N$={N}', color=color)
    plt.fill_between(betas, np.array(coalescence_times) - np.array(coalescence_times_std), np.array(coalescence_times) + np.array(coalescence_times_std), color=color, alpha=0.3)
plt.vlines(beta_BD, 0, max(coalescence_times_matrix.flatten())*10, color='k', linestyle='--', label=r'$\beta=\tanh^{-1}(1/d)$')
plt.ylim(min(coalescence_times_matrix.flatten())/2, max(coalescence_times_matrix.flatten())*2)
plt.xlabel(r'Inverse Temperature $\beta$')
plt.yscale('log')
plt.ylabel('Coalescence Time')
plt.legend()
plt.savefig(save_path+f'temp/RR_coal_time_vs_beta_d_{d}.png')
plt.savefig(save_path+f'temp/RR_coal_time_vs_beta_d_{d}.svg')

# Plot of coalescence time as a function of N for different values of beta
fig, ax = plt.subplots()
for i, beta in enumerate(betas):
    coalescence_times = [T_coal_vs_beta_N[N][0][i] for N in sorted(T_coal_vs_beta_N.keys())]
    color = cmc.berlin(i / len(betas))
    ax.plot(sorted(T_coal_vs_beta_N.keys()), coalescence_times, label=fr'$\beta$={beta:.2f}' if beta!=beta_BD else r'$\beta=\tanh^{-1}(1/d)$', color=color)
    ax.fill_between(sorted(T_coal_vs_beta_N.keys()), np.array(coalescence_times) - np.array([T_coal_vs_beta_N[N][1][i] for N in sorted(T_coal_vs_beta_N.keys())]), np.array(coalescence_times) + np.array([T_coal_vs_beta_N[N][1][i] for N in sorted(T_coal_vs_beta_N.keys())]), color=color, alpha=0.3)

norm = mpl.colors.Normalize(vmin=0, vmax=max_beta)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmc.berlin)
sm.set_array([])
ticks = [0, beta_BD, max_beta]
tick_labels = ['0', r'$\beta_c$', f'{max_beta:.2f}']
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
fig.savefig(save_path+f'temp/RR_coal_time_vs_N_d_{d}.png')
fig.savefig(save_path+f'temp/RR_coal_time_vs_N_d_{d}.svg')

# Plot of log(T_coal/N) as a function of beta for different values of N

plt.figure(3)
for N in sorted(log_T_coal_over_N_vs_beta_N.keys()):
    log_T_coal_over_N, log_T_coal_over_N_std = log_T_coal_over_N_vs_beta_N[N]
    color = plt.cm.viridis((N - min(log_T_coal_over_N_vs_beta_N.keys())) / (max(log_T_coal_over_N_vs_beta_N.keys()) - min(log_T_coal_over_N_vs_beta_N.keys())))
    plt.plot(betas, log_T_coal_over_N, label=f'$N$={N}', color=color)
    plt.fill_between(betas, np.array(log_T_coal_over_N) - np.array(log_T_coal_over_N_std), np.array(log_T_coal_over_N) + np.array(log_T_coal_over_N_std), color=color, alpha=0.3)
plt.vlines(beta_BD, 0, max(log_T_coal_over_N)*10, color='k', linestyle='--', label=r'$\beta=\tanh^{-1}(1/d)$')
plt.xlabel(r'Inverse Temperature $\beta$')
plt.ylabel(r'$\log(T_{coal})/N$')
plt.legend()
plt.savefig(save_path+f'temp/RR_log_T_coal_over_N_vs_beta_d_{d}.png')
plt.savefig(save_path+f'temp/RR_log_T_coal_over_N_vs_beta_d_{d}.svg')
