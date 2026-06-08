"""
This script constructs a spin system on a Random Regular (RR) graph, initializes 100 random configurations, 
and evolves them simultaneously using the same MCMC random numbers until they coalesce.
"""

from utils.cftp_func import *

# --- Setup and Parameters ---
save_path_data = save_path + "data/algo/random/"
save_path_fig = save_path + "figures/algo/random/"

N_list = [100, 200, 500, 750, 1000, 1500, 2000, 2250, 2500, 2750, 3000]
d = 4

# Theoretical critical betas for RR graphs
beta_BD = np.arctanh(1/d)
beta_uni = np.arctanh(1/(d-1))
beta_SG = np.arctanh(np.sqrt(1/(d-1)))

# Generate betas up to the Spin Glass transition
betas = np.linspace(0, beta_SG, 50, endpoint=True)

# --- 1. Generation Phase ---
# Initialize the CSV file with headers
# with open(save_path_data + "RR_damage_spreading.csv", 'w') as f:
#     f.write("N,beta,t_coal\n")

# for N in N_list:
#     G, couplings = create_RR_graph(N=N, d=d)
    
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
#         print(f"Average coalescence time over 10 runs: {t_coal_mean:.2f} steps\n")
        
#         # Save data continuously for this beta and N
#         with open(save_path_data + "RR_damage_spreading.csv", 'a') as f:
#             f.write(f"{N},{beta},{t_coal_mean}\n")


# --- 2. Loading and Fitting Phase ---
beta_DS_estimated = []
popt_list = []

# Load all three columns using the flattened format
N_full = np.loadtxt(save_path_data + "RR_damage_spreading.csv", delimiter=',', skiprows=1, usecols=0)
betas_full = np.loadtxt(save_path_data + "RR_damage_spreading.csv", delimiter=',', skiprows=1, usecols=1)
t_coal_full = np.loadtxt(save_path_data + "RR_damage_spreading.csv", delimiter=',', skiprows=1, usecols=2)

for i, N in enumerate(N_list):
    print(f"\n--- Fitting and Plotting for N = {N} ---")
    
    # Isolate the data for the current N using boolean masking
    mask_N = (N_full == N)
    betas_N = betas_full[mask_N]
    t_coal_N = t_coal_full[mask_N]
    
    # Filter out -1 values and drop the last 5 elements (to avoid tail artifacts)
    valid_mask = (t_coal_N != -1)
    final_betas = betas_N[valid_mask][:-5]
    final_t_coal = t_coal_N[valid_mask][:-5]
    
    # Perform the curve fit
    try:
        # Note: Using the p0=[a, b, c] from your RR snippet
        popt, pcov = curve_fit(
            power_law_log_fit, 
            final_betas, 
            np.log(final_t_coal), 
            p0=[-2, 0.6, 10], 
            maxfev=10000, 
            absolute_sigma=False
        )
        a_fit, b_fit, c_fit = popt
        print(f"Fitted parameters: a={a_fit:.4f}, b={b_fit:.4f}, c={c_fit:.4f}")
        
        r2score = r2_score(np.log(final_t_coal), power_law_log_fit(final_betas, *popt))
        print(f"R^2 score of the fit: {r2score:.4f}")
        
        beta_DS_estimated.append(popt[1])
        popt_list.append(popt)
        
    except Exception as e:
        print(f"Fit failed for N = {N}. Reason: {e}")
        beta_DS_estimated.append(np.nan)
        popt_list.append(None)


# --- 3. Plotting Phase ---

# Plot 0: Estimated beta_DS vs N
plt.figure(0)
plt.plot(N_list, beta_DS_estimated, 'o-', label=r'Estimated $\beta_{DS}$')
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
plt.savefig(save_path_fig + "RR_beta_damage_spreading.png", dpi=300)
plt.close()

# Plot 1: Log(T_coal) vs beta for all N
fig, ax = plt.subplots()
for i, N in enumerate(N_list):
    mask_N = (N_full == N)
    betas_N = betas_full[mask_N]
    t_coal_N = t_coal_full[mask_N]
    color = plt.cm.viridis(i / len(N_list))

    ax.plot(betas_N, np.log(t_coal_N), 'o', color=color)
    
    popt = popt_list[i]
    if popt is not None:
        betas_fit = np.linspace(min(betas_N), max(betas_N), 100)
        t_coal_fit = power_law_log_fit(betas_fit, *popt)
        
        # Plot up to the argmax to avoid fitting the downward curve
        max_value_fit_index = np.argmax(t_coal_fit)
        ax.plot(betas_fit[:max_value_fit_index], t_coal_fit[:max_value_fit_index], 'k-', linewidth=2, color=color)

# Theoretical vertical lines (all static for RR graphs)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(N_list), vmax=max(N_list)))
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('N')
ax.axvline(beta_BD, color='r', linestyle='--', label=r'$\beta_{BD/Dobr}$')
ax.axvline(beta_uni, color='b', linestyle='-.', label=r'$\beta_{Uni}$')
ax.axvline(beta_SG, color='k', linestyle='-', label=r'$\beta_{SG}$')
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\log(T_{coal})$")
ax.legend()
fig.savefig(save_path_fig + "RR_damage_spreading_coal.png", dpi=300)
plt.close()