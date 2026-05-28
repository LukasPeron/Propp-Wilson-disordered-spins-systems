"""
This script constructs a spin system on a graph, initializes 100 random configurations, 
and evolves them simultaneously using the same MCMC random numbRRs until they coalesce.
"""

from utils.cftp_func import *

N = 500
d = 4

G, couplings = create_RR_graph(N=N, d=d)

beta_BD = np.arctanh(1/d)
beta_SG = np.arctanh(np.sqrt(1/(d-1)))
beta_uni = np.arctanh(1/(d-1))

betas = np.linspace(0, beta_SG, 100, endpoint=True)

t_coal_beta = []
for beta in betas:
    print(f"Using beta={beta:.4f} (beta_SG={beta_SG:.4f})")
    t_coal= []
    for _ in range(10):
        coalescence_time = simulate_n_configs(
            beta=beta, 
            G=G, 
            couplings=couplings, 
            num_configs=100
        )
        t_coal.append(coalescence_time)
    print(f"\nAvRRage coalescence time ovRR 10 runs: {np.mean(t_coal):.2f} steps")
    t_coal_beta.append(np.mean(t_coal))

np.save(save_path + f"data/algo/random/RR_{N}_configs_coal_time.npy", t_coal_beta)

plt.plot(betas, t_coal_beta, 'o-', label='AvRRage Coalescence Time')
plt.axvline(beta_uni, color='b', linestyle='-.', label=r'$\tanh^{-1}(1/d)$')
plt.axvline(beta_SG, color='k', linestyle='--', label=r'$\beta_{SG}$')
plt.xlabel(r"$\beta$")
plt.ylabel("Average $T_{coal}$")
plt.yscale("log")
plt.legend()
plt.savefig(save_path + f"figures/algo/random/RR_{N}_configs_coal_time.png", dpi=300)
plt.close()