"""
This script constructs a spin system on a graph, initializes 100 random configurations, 
and evolves them simultaneously using the same MCMC random numbers until they coalesce.
"""

from utils.cftp_func import *

N = 100

G, couplings = create_complete_graph(N=N, model="SK")

beta_sg = 1

betas = np.linspace(0, beta_sg*1.2, 100, endpoint=True)

t_coal_beta = []
for beta in betas:
    print(f"Using beta={beta:.4f} (beta_sg={beta_sg:.4f})")
    t_coal= []
    for _ in range(10):
        coalescence_time = simulate_n_configs(
            beta=beta, 
            G=G, 
            couplings=couplings, 
            num_configs=100
        )
        t_coal.append(coalescence_time)
    print(f"\nAverage coalescence time over 10 runs: {np.mean(t_coal):.2f} steps")
    t_coal_beta.append(np.mean(t_coal))

np.save(save_path + f"data/algo/complet/SK_{N}_configs_coal_time.npy", t_coal_beta)

plt.plot(betas, t_coal_beta, 'o-', label='Average Coalescence Time')
plt.axvline(beta_sg, color='k', linestyle='--', label=r'$\beta_{sg}$')
plt.xlabel(r"$\beta$")
plt.ylabel("Average $T_{coal}$")
plt.yscale("log")
plt.legend()
plt.savefig(save_path + f"figures/algo/complet/SK_{N}_configs_coal_time.png", dpi=300)
plt.close()