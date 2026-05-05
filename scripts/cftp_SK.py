"""
This script performs theoretical calculations for the Sherrington-Kirkpatrick (SK) model and compares them with results obtained from CFTP sampling.

Author: L. Péron
Last edited: 2024-21-04
"""

from scipy.special import gammaln
from cftp_bc import *

N = 100
G = nx.complete_graph(N)
couplings = np.random.normal(0, 1/np.sqrt(N), size=(N, N))
couplings = (couplings + couplings.T) / 2  # Ensure symmetry J_ij = J_ji
np.fill_diagonal(couplings, 0)  # No self-interaction

## theoretical part ##

"""
We want to compare our simulations to actual theoretical predictions for the SK model. We will compare :
- the free energy per spin
- the Edwards-Anderson order parameter q
- the distribution of overlaps P(q)
"""

def free_energy_per_spin(beta):
    """
    Compute the free energy per spin for the SK model using the replica method.
    
    Parameters:
    beta (float): Inverse temperature
    
    Returns:
    float: Free energy per spin
    """
    # The free energy per spin in the SK model can be computed using the replica method.
    # For simplicity, we will use the known result for the free energy per spin in the SK model.
    
    # The free energy per spin is given by:
    # f = - (beta^2 / 4) * (1 - q^2) + (1/2) * log(1 - q)
    
    # We need to find q that minimizes the free energy. This can be done by solving the self-consistent equation for q.
    
    def self_consistent_q(q):
        return np.tanh(beta * np.sqrt(q))**2
    
    from scipy.optimize import fixed_point
    q = fixed_point(self_consistent_q, 0.5)  # Initial guess for q
    
    free_energy = - (beta**2 / 4) * (1 - q**2) + (1/2) * np.log(1 - q)
    
    return free_energy

def edwards_anderson_order_parameter(beta):
    """
    Compute the Edwards-Anderson order parameter q for the SK model.
    
    Parameters:
    beta (float): Inverse temperature
    
    Returns:
    float: Edwards-Anderson order parameter q
    """
    # The Edwards-Anderson order parameter q can be computed using the self-consistent equation:
    # q = <s_i s_j> = tanh^2(beta * sqrt(q))
    
    def self_consistent_q(q):
        return np.tanh(beta * np.sqrt(q))**2
    
    from scipy.optimize import fixed_point
    q = fixed_point(self_consistent_q, 0.5)  # Initial guess for q
    
    return q

def overlap_distribution(beta, num_samples=1000):
    """
    Compute the distribution of overlaps P(q) for the SK model using CFTP sampling.
    
    Parameters:
    beta (float): Inverse temperature
    num_samples (int): Number of samples to generate
    
    Returns:
    np.array: Distribution of overlaps P(q)
    """
    # We will use CFTP sampling to generate samples from the Gibbs distribution of the SK model.
    
    # Generate samples using CFTP
    samples = CFTP_BC_disordered_optimized(beta, G, couplings)
    
    # Compute overlaps between pairs of samples
    overlaps = []
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            overlap = np.mean(samples[i] * samples[j])
            overlaps.append(overlap)
    
    # Compute the distribution of overlaps P(q)
    hist, bin_edges = np.histogram(overlaps, bins=50, density=True)
    
    return hist, bin_edges

### CFTP sampling part ###

N = 100
G = nx.complete_graph(N)
couplings = np.random.normal(0, 1/np.sqrt(N), size=(N, N))
couplings = (couplings + couplings.T) / 2  # Ensure symmetry J_ij = J_ji
np.fill_diagonal(couplings, 0)  # No self-interaction

beta = np.linspace(0, 1.2, 13)  # Range of inverse temperatures to test
free_energies = []
order_parameters = []
for b in beta:
    free_energies.append(free_energy_per_spin(b))
    order_parameters.append(edwards_anderson_order_parameter(b))
    
# Plot the results
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(beta, free_energies, label='Free Energy per Spin')
plt.xlabel('Inverse Temperature (beta)')
plt.ylabel('Free Energy per Spin')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(beta, order_parameters, label='Edwards-Anderson Order Parameter q')
plt.xlabel('Inverse Temperature (beta)')
plt.ylabel('Edwards-Anderson Order Parameter q')
plt.legend()
plt.tight_layout()
plt.savefig('../figures/sk_results.png')