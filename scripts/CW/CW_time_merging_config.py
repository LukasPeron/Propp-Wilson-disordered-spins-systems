"""
This script constructs a spin system on a graph, initializes 100 random configurations, 
and evolves them simultaneously using the same MCMC random numbers until they coalesce.
"""

from utils.cftp_func import *

save_path_data = save_path+f"data/algo/complet/"
save_path_fig = save_path+f"figures/algo/complet/"

N_list = [100, 200, 500, 750, 1000, 1500, 2000, 2250, 2500, 2750, 3000]
beta_pm = 1
betas = np.linspace(0, beta_pm*1.2, 50, endpoint=True)
# with open(save_path_data + f"CW_damage_spreading.csv", 'a') as f:
#     f.write("N,beta,t_coal\n")
# for N in N_list:
#     G, couplings = create_complete_graph(N=N, model="CW")
#     for beta in betas:
#         t_coal = []
#         print(f"Using beta={beta:.4f} (beta_pm={beta_pm:.4f}), N={N})")
#         for _ in range(10):
#             coalescence_time = simulate_n_configs(
#                 beta=beta, 
#                 G=G,
#                 couplings=couplings, 
#                 num_configs=100
#             )
#             t_coal.append(coalescence_time)
#         t_coal_mean = np.mean(t_coal)
#         print(f"\nAverage coalescence time over 10 runs: {np.mean(t_coal):.2f} steps")
#         # save data for this beta and N
#         with open(save_path_data + f"CW_damage_spreading.csv", 'a') as f:
#             f.write(f"{N},{beta},{t_coal_mean}\n")

N_full = np.loadtxt(save_path_data + "CW_damage_spreading.csv", delimiter=',', skiprows=1, usecols=0)
betas_full = np.loadtxt(save_path_data + "CW_damage_spreading.csv", delimiter=',', skiprows=1, usecols=1)
t_coal_full = np.loadtxt(save_path_data + "CW_damage_spreading.csv", delimiter=',', skiprows=1, usecols=2)

beta_DS_N = []
popt_list = []
for N in N_list:
    print(f"\n--- Fitting for N = {N} ---")
    
    # 1. Isolate the data for the current N
    mask_N = (N_full == N)
    betas_N = betas_full[mask_N]
    t_coal_N = t_coal_full[mask_N]
    
    # 2. Filter out the -1 values
    valid_mask = (t_coal_N != -1)
    
    # 3. Apply valid mask and drop the last 10 elements
    final_betas = betas_N[valid_mask][:-1]
    final_t_coal = t_coal_N[valid_mask][:-1]
    # 4. Perform the fit
    try:
        popt, pcov = curve_fit(
            power_law_log_fit, 
            final_betas, 
            np.log(final_t_coal), 
            p0=[-2, 1.1, 7], 
            maxfev=10000, 
            absolute_sigma=False
        )
        a_fit, b_fit, c_fit = popt
        print(f"Fitted parameters: a={a_fit:.4f}, b={b_fit:.4f}, c={c_fit:.4f}")
        popt_list.append(popt)
        r2score = r2_score(np.log(final_t_coal), power_law_log_fit(final_betas, *popt))
        print(f"R^2 score: {r2score:.4f}")
        
        beta_DS_N.append(popt[1])
        
    except Exception as e:
        print(f"Fit failed for N = {N}. Reason: {e}")
        beta_DS_N.append(np.nan)

plt.figure(0)
plt.plot(N_list, beta_DS_N, 'o-', label=r'Estimated $\beta_{DS}$')
plt.xlabel("N")
plt.ylabel(r"Estimated $\beta_{DS}$")
plt.legend()
plt.savefig(save_path_fig + "CW_beta_damage_spreading.png", dpi=300)
plt.savefig(save_path_fig + "CW_beta_damage_spreading.pdf", dpi=300)
plt.close()

fig, ax = plt.subplots()
for i, N in enumerate(N_list):
    # Use the exact same masking technique for plotting
    mask_N = (N_full == N)
    betas_N = betas_full[mask_N]
    t_coal_N = t_coal_full[mask_N]
    popt = popt_list[N_list.index(N)]
    color = plt.cm.viridis(i / len(N_list))
    ax.plot(betas_N, np.log(t_coal_N), 'o-', color=color)
    ax.plot(betas_N, power_law_log_fit(betas_N, *popt), 'k-', linewidth=2, color=color)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(N_list), vmax=max(N_list)))
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('L')
ax.axvline(beta_pm, color='k', linestyle='-', label=r'$\beta_{PM}$')
ax.set_xlabel(r"$\beta$")
ax.set_ylabel("Average $T_{coal}$")
ax.legend()
fig.savefig(save_path_fig + f"CW_damage_spreading_coal.png", dpi=300)
fig.savefig(save_path_fig + f"CW_damage_spreading_coal.pdf", dpi=300)
plt.close()
