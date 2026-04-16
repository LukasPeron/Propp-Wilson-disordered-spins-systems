"""
This script performs theoretical calculations for the Curie-Weiss model and compares them with results obtained from CFTP sampling. It computes the partition function, magnetization, and magnetization variance for a complete graph of 100 nodes across a range of inverse temperatures (beta). The results are averaged over multiple runs to reduce noise and are intended to be plotted for comparison with theoretical predictions.

Last edited: 2024-06-20
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

def theoretical_magnetization_variance(beta, N):
    val_m = np.arange(-N, N+1, 2) / N
    km = N / 2 * (1 + val_m)
    
    log_prefac = gammaln(N + 1) - gammaln(km + 1) - gammaln(N - km + 1)
    energy = -N * val_m**2 / 2 
    exponents = log_prefac - beta * energy
    
    # The Log-Sum-Exp trick
    max_exp = np.max(exponents)
    terms = np.exp(exponents - max_exp)
    
    Z_shifted = np.sum(terms)
    mean_mag = np.sum(np.abs(val_m) * terms) / Z_shifted
    mean_mag_squared = np.sum((np.abs(val_m)**2) * terms) / Z_shifted
    
    return mean_mag_squared - mean_mag**2

### CFTP sampling for the Curie-Weiss model ###

N = 100
G = nx.complete_graph(N)
couplings = 1/2 * (np.ones((N, N)) - np.eye(N)) / N

betas = np.linspace(0, 1.2, 100)
magnetizations = []
theoretical_magnetizations = [theoretical_magnetization(b, N) for b in betas]
theoretical_magnetization_variances = [theoretical_magnetization_variance(b, N) for b in betas]
for beta in betas:
    print(f"Processing beta={beta:.2f}...")
    temp = []
    for _ in range(50):  # Average over multiple runs to reduce noise
        config, time = CFTP_BC_disordered_optimized(beta=beta, G=G, coupling=couplings)
        magnetization = np.abs(np.mean(config))
        temp.append(magnetization)
    magnetizations.append(np.mean(temp))

### Plotting the results ###

plt.plot(betas, theoretical_magnetizations, '-r', label='Theoretical Magnetization')
plt.fill_between(betas, np.array(theoretical_magnetizations) - np.sqrt(theoretical_magnetization_variances),
                 np.array(theoretical_magnetizations) + np.sqrt(theoretical_magnetization_variances), color='red', alpha=0.2, label='Theoretical Variance')
plt.plot(betas, magnetizations, '+b', label='CFTP Magnetization')
plt.vlines(1.0, -0.05, 1.05, color='k', linestyle='--', label=r'Critical Inverse Temperature')
plt.text(0, 0.82, f"$N$=100\nCurie-Weiss Model\nGlauber Heat Bath", fontsize=16)
plt.ylim(-0.05, 1.05)
plt.xlabel(r'Inverse Temperature $\beta$')
plt.ylabel('Magnetization')
plt.legend(loc="center left")
plt.savefig('../figures/CW_magnetization_comparison.png')
