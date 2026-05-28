"""
This script constructs a spin system on a graph, initializes 100 random configurations, 
and evolves them simultaneously using the same MCMC random numbers until they coalesce.
"""

from utils.cftp_func import *

# loop over all the dimensions and models

for d in [2, 3]:
    for model in ['RL_ferr', 'RL_sg']:
        if d==2:
            L = 32
            if model=="RL_ferr":
                beta_pm = 1/2 * (np.log(1 + np.sqrt(2)))
            else: # no spin glass transition in 2D
                beta_uni = np.arctanh(1/(2*d-1))
                beta_sg = 1
                beta_dm = 1/1.69
        elif d==3:
            L = 16
            if model=="RL_ferr":
                beta_pm = 0.221654626
            else:
                beta_sg = 1/1.3
                beta_uni = np.arctanh(1/(2*d-1))
                beta_dm = 1/3.89
        G, couplings = create_lattice_graph(L=L, d=d, model=model)
        if model=="RL_ferr":
            betas = np.linspace(0, beta_pm*1.2, 100, endpoint=True)
        else:
            betas = np.linspace(0, beta_sg, 100, endpoint=True)
        t_coal_beta = []
        for beta in betas:
            print(f"Using beta={beta:.4f}")
            if model=="RL_ferr":
                print(f"Critical beta (PM transition): {beta_pm:.4f}")
            else:
                if d==2:
                    print(f"Critical beta (uniqueness threshold): {beta_uni:.4f}")
                    print(f"Critical beta (DM transition): {beta_dm:.4f}")
                else:
                    print(f"Critical beta (SG transition): {beta_sg:.4f}")
                    print(f"Critical beta (uniqueness threshold): {beta_uni:.4f}")
                    print(f"Critical beta (DM transition): {beta_dm:.4f}")
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

        np.save(save_path + f"data/algo/latt/d{d}/{model.replace("RL_", "")}/{model}_{L**d}_configs_coal_time.npy", t_coal_beta)

        plt.plot(betas, t_coal_beta, 'o-', label='Average Coalescence Time')
        if model=="RL_ferr":
            plt.axvline(beta_pm, color='r', linestyle='--', label=r'$\beta_{pm}$')
        else:
            if d==2:
                plt.axvline(beta_uni, color='b', linestyle='-.', label=r'$\tanh^{-1}(1/d)$')
                plt.axvline(beta_dm, color='g', linestyle=':', label=r'$\beta_{DM}$')
            else:
                plt.axvline(beta_sg, color='r', linestyle='--', label=r'$\beta_{sg}$')
                plt.axvline(beta_uni, color='b', linestyle='-.', label=r'$\tanh^{-1}(1/d)$')
                plt.axvline(beta_dm, color='g', linestyle=':', label=r'$\beta_{DM}$')
        plt.xlabel(r"$\beta$")
        plt.ylabel("Average $T_{coal}$")
        plt.yscale("log")
        plt.legend()
        plt.savefig(save_path + f"figures/algo/latt/d{d}/{model.replace("RL_", "")}/{model}_{L**d}_configs_coal_time.png", dpi=300)
        plt.close()

# unique choice of dimension and model

# for d in [2, 3]:
#     for model in ['RL_ferr', 'RL_sg']:
#         if d==2:
#             L = 32
#             if model=="RL_ferr":
#                 beta_pm = 1/2 * (np.log(1 + np.sqrt(2)))
#             else: # no spin glass transition in 2D
#                 beta_uni = np.arctanh(1/(2*d-1))
#                 beta_sg = 1
#                 beta_dm = 1/1.69
#         elif d==3:
#             L = 16
#             if model=="RL_ferr":
#                 beta_pm = 0.221654626
#             else:
#                 beta_sg = 1/1.3
#                 beta_uni = np.arctanh(1/(2*d-1))
#                 beta_dm = 1/3.89

#         if model=="RL_ferr":
#             betas = np.linspace(0, beta_pm*1.2, 100, endpoint=True)
#         else:
#             betas = np.linspace(0, beta_sg, 100, endpoint=True)

#         t_coal_beta = np.load(save_path + f"data/algo/latt/d{d}/{model.replace("RL_", "")}/{model}_{L**d}_configs_coal_time.npy")

#         plt.plot(betas, t_coal_beta, 'o-', label='Average Coalescence Time')
#         if model=="RL_ferr":
#             plt.axvline(beta_pm, color='k', linestyle='--', label=r'$\beta_{pm}$')
#         else:
#             if d==2:
#                 plt.axvline(beta_uni, color='b', linestyle='-.', label=r'$\tanh^{-1}(1/d)$')
#                 plt.axvline(beta_dm, color='g', linestyle=':', label=r'$\beta_{DS}$')
#             else:
#                 plt.axvline(beta_sg, color='k', linestyle='--', label=r'$\beta_{sg}$')
#                 plt.axvline(beta_uni, color='b', linestyle='-.', label=r'$\tanh^{-1}(1/d)$')
#                 plt.axvline(beta_dm, color='g', linestyle=':', label=r'$\beta_{DS}$')
#         plt.xlabel(r"$\beta$")
#         plt.ylabel("Average $T_{coal}$")
#         plt.yscale("log")
#         plt.legend()
#         plt.savefig(save_path + f"figures/algo/latt/d{d}/{model.replace("RL_", "")}/{model}_{L**d}_configs_coal_time.png", dpi=300)
#         plt.close()