"""
This script performs theoretical calculations for the Curie-Weiss model and compares them with results obtained from CFTP sampling. It computes the partition function, magnetization, and magnetization variance for a complete graph of 100 nodes across a range of inverse temperatures (beta). The results are averaged over multiple runs to reduce noise and are intended to be plotted for comparison with theoretical predictions.

Author: L. Péron
"""

from scipy.special import gammaln
from cftp_bc import *

### Theoretical calculations for the Curie-Weiss model ###

def partition_function(beta, N):
    val_m = np.arange(-N, N+1, 2) / N  
    km = N / 2 * (1 + val_m)           
    
    log_prefac = gammaln(N + 1) - gammaln(km + 1) - gammaln(N - km + 1)
    energy = -N * val_m**2 / 2 
    exponents = log_prefac - beta * energy
    
    # We still use the shift to prevent intermediate overflow, 
    # though note that for very large N, Z itself may still exceed float64 limits (~10^308).
    max_exp = np.max(exponents)
    Z = np.exp(max_exp) * np.sum(np.exp(exponents - max_exp))
    return Z

def theoretical_magnetization(beta, N):
    val_m = np.arange(-N, N+1, 2) / N
    km = N / 2 * (1 + val_m)
    
    log_prefac = gammaln(N + 1) - gammaln(km + 1) - gammaln(N - km + 1)
    energy = -N * val_m**2 / 2 
    exponents = log_prefac - beta * energy
    
    # The Log-Sum-Exp trick: subtract max exponent to prevent overflow
    max_exp = np.max(exponents)
    terms = np.exp(exponents - max_exp)
    
    # The exp(max_exp) mathematically cancels out between the numerator and denominator
    mag_sum = np.sum(np.abs(val_m) * terms)
    Z_shifted = np.sum(terms) 
    
    return mag_sum / Z_shifted

### CFTP sampling for the Curie-Weiss model ###

beta = np.linspace(0, 1.25, 100)
beta_CFTP = np.linspace(0, 1.25, 15)
marker_lst = ['o', 's', 'D', 'P']
color_lst = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
linestyle_lst = ['-', '--', '-.', ':']
N_values = [100, 200, 500, 750]

# Initialize the figure with 4 stacked subplots sharing the x-axis
fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8,8))

for i, N in enumerate(N_values):
    print(f"Processing N={N}...")
    G = nx.complete_graph(N)
    couplings = (np.ones((N, N)) - np.eye(N)) / N
    magnetizations = []
    magnetization_variances = []
    for b in beta_CFTP:
        print(f"Processing beta={b:.2f}...")
        temp = []
        for _ in range(2):  # Average over multiple runs to reduce noise
            config, time, _ = CFTP_BC_disordered_optimized(beta=b, G=G, coupling=couplings)
            magnetization = np.abs(np.nanmean(config))
            temp.append(magnetization)
        magnetizations.append(np.nanmean(temp))
        magnetization_variances.append(np.nanvar(temp))

    ### Plotting the results ###
    ax = axes[i] # Target the specific subplot for this N
    marker = marker_lst[i]
    color = color_lst[i]
    linestyle = linestyle_lst[i]
    theoretical_magnetizations = [theoretical_magnetization(bb, N) for bb in beta]
    
    # Plot theory and CFTP on the current subplot
    ax.plot(beta, theoretical_magnetizations, linestyle=linestyle, color=color)
    ax.errorbar(beta_CFTP, magnetizations, yerr=np.sqrt(magnetization_variances), marker=marker, color=color, linestyle='None')

    # Add reference line, limits, labels, and legends per subplot
    ax.vlines(1.0, -0.05, 1.05, color='k', linestyle='--', label=r'$\beta_c$' if i == 0 else None)  # Only label the critical line in the first subplot
    if i<3:
        ax.axhline(0, color='k', linestyle='-', linewidth=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(r'$m(\beta)$'+f'\n$N$={N}' if i == 0 else f'$N$={N}', rotation=0, labelpad=20)
    if i == 0:
        ax.legend(loc='upper left')
    if i == 0:
        ax.set_yticks([0.0, 1.0])
    else:
        ax.set_yticks([0.0])

# Set the x-axis label only on the bottom-most subplot
axes[-1].set_xlabel(r'Inverse Temperature $\beta$')

# Adjust layout so subplots don't overlap
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1, hspace=0)
# plt.tight_layout()

plt.savefig(save_path+'temp/CW_magnetization_comparison.png')
plt.savefig(save_path+'temp/CW_magnetization_comparison.svg')