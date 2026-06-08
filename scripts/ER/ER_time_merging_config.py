"""
This script constructs a spin system on a graph, initializes 100 random configurations, 
and evolves them simultaneously using the same MCMC random numbers until they coalesce.
"""

from utils.cftp_func import *

save_path_data = save_path + "data/algo/random/"
save_path_fig = save_path + "figures/algo/random/"

N_list = np.array([100, 200, 500, 750, 1000, 1500, 2000, 2250, 2500, 2750, 3000])
d = 4
beta_BD_true_N = []

beta_uni = np.arctanh(1/d)
beta_SG = np.arctanh(np.sqrt(1/d))
betas = np.linspace(0, beta_SG, 50, endpoint=True)

# --- 1. Generation Phase ---
# with open(save_path_data + f"ER_damage_spreading.csv", 'a') as f:
#     f.write("N,beta,t_coal\n")
for N in N_list:
    G, couplings = create_ER_graph(N=N, d=d)
#     # Calculate and store beta_BD for this specific graph instance
    current_beta_BD = np.arctanh(1/np.max([G.degree(v) for v in range(N)]))
    beta_BD_true_N.append(current_beta_BD)
#     for beta in betas:
#         t_coal = []
#         print(f"Using beta={beta:.4f} (beta_SG={beta_SG:.4f}), N={N}")
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
#         with open(save_path_data + f"ER_damage_spreading.csv", 'a') as f:
#             f.write(f"{N},{beta},{t_coal_mean}\n")

# --- 3. Loading, Fitting, and Plotting ---
beta_DS_estimated = []
popt_list = []

# Load all three columns
N_full = np.loadtxt(save_path_data + "ER_damage_spreading.csv", delimiter=',', skiprows=1, usecols=0)
betas_full = np.loadtxt(save_path_data + "ER_damage_spreading.csv", delimiter=',', skiprows=1, usecols=1)
t_coal_full = np.loadtxt(save_path_data + "ER_damage_spreading.csv", delimiter=',', skiprows=1, usecols=2)

for i, N in enumerate(N_list):
    print(f"\n--- Fitting and Plotting for N = {N} ---")
    
    # Isolate the data for the current N using boolean masking
    mask_N = (N_full == N)
    betas_N = betas_full[mask_N]
    t_coal_N = t_coal_full[mask_N]
    
    # Filter out -1 values and drop the last 10 elements
    valid_mask = (t_coal_N != -1)
    final_betas = betas_N[valid_mask][:-5]
    final_t_coal = t_coal_N[valid_mask][:-5]
    
    # Perform the curve fit
    try:
        popt, pcov = curve_fit(
            power_law_log_fit, 
            final_betas, 
            np.log(final_t_coal), 
            p0=[-2, 0.5, 10], 
            maxfev=10000, 
            absolute_sigma=False
        )
        a_fit, b_fit, c_fit = popt
        print(f"Fitted parameters: a={a_fit:.4f}, b={b_fit:.4f}, c={c_fit:.4f}")
        
        r2score = r2_score(np.log(final_t_coal), power_law_log_fit(final_betas, *popt))
        print(f"R^2 score of the fit: {r2score:.4f}")
        
        beta_DS_estimated.append(popt[1])
        popt_list.append(popt)
        betas_fit = np.linspace(min(final_betas), max(final_betas), 100)
        t_coal_fit = power_law_log_fit(betas_fit, *popt)
        
    except Exception as e:
        print(f"Fit failed for N = {N}. Reason: {e}")
        beta_DS_estimated.append(np.nan)
        betas_fit, t_coal_fit = [], []


plt.figure(0)
plt.plot(N_list, beta_DS_estimated, 'o-', label=r'Estimated $\beta_{DS}$')

# powerlaw fit of the estimated beta_DS values
# popt_beta_DS, _ = curve_fit(power_law_fit, N_list[3:], beta_DS_estimated[3:], p0=[1, -2, 0.3], maxfev=10000, absolute_sigma=False)
# r2score_beta_DS = r2_score(beta_DS_estimated[3:], power_law_fit(N_list[3:], *popt_beta_DS))
# print(f"Fitted parameters for beta_DS vs N: a={popt_beta_DS[0]:.6f}, b={popt_beta_DS[1]:.4f}, c={popt_beta_DS[2]:.4f}")
# print(f"R^2 score for beta_DS fit: {r2score_beta_DS:.4f}")
# N_fit = np.linspace(min(N_list[3:]), max(N_list[3:]), 100)
# beta_DS_fit = power_law_fit(N_fit, *popt_beta_DS)
# plt.plot(N_fit, beta_DS_fit, 'k-', label=f'Power-law fit $\searrow$ {popt_beta_DS[2]:.4f}', linewidth=2)
plt.axhline(np.mean(beta_DS_estimated[3:]), 0, np.max(N_list)*1.1, color='k', linestyle='--', label=fr'Mean $\beta_{{DS}}={np.mean(beta_DS_estimated[3:]):.4f}$ ($L\geq${N_list[3]})')
plt.axhline(beta_uni, 0, np.max(N_list)*1.1, color='b', linestyle='-.', label=r'$\beta_{Uni}$')
plt.xlabel("N")
plt.ylabel(r"Estimated $\beta_{DS}$")
plt.legend()
plt.savefig(save_path_fig + "ER_beta_damage_spreading.png", dpi=300)
plt.close()

fig, ax = plt.subplots()
for i, N in enumerate(N_list):
    # Use the exact same masking technique for plotting
    mask_N = (N_full == N)
    betas_N = betas_full[mask_N]
    t_coal_N = t_coal_full[mask_N]
    popt = popt_list[i]
    color = plt.cm.viridis(i / len(N_list))
    ax.plot(betas_N, np.log(t_coal_N), 'o-', color=color)
    betas_fit = np.linspace(min(betas_N), max(betas_N), 100)
    t_coal_fit = power_law_log_fit(betas_fit, *popt)
    max_value_fit_index = np.argmax(t_coal_fit)
    ax.plot(betas_fit[:max_value_fit_index], t_coal_fit[:max_value_fit_index], 'k-', linewidth=2, color=color)

# create a colorbar for the N values
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(N_list), vmax=max(N_list)))
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('N')
ax.axvline(beta_SG, color='k', linestyle='-', label=r'$\beta_{SG}$')
ax.axvspan(np.min(beta_BD_true_N), np.max(beta_BD_true_N), color='red', alpha=0.25, label=r'$\beta_{BD/Dobr}$ region')
ax.axvline(beta_uni, color='b', linestyle='-.', label=r'$\beta_{Uni}$')
ax.set_xlabel(r"$\beta$")
ax.set_ylabel("$\log(T_{coal})$")
ax.legend()
fig.savefig(save_path_fig + f"ER_damage_spreading_coal.png", dpi=300)
plt.close()