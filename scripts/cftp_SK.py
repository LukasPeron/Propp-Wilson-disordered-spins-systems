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
- the Edwards-Anderson overlap
- the Energy per site