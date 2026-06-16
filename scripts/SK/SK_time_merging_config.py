"""
This script constructs a spin system on a complete graph for the SK model, 
initializes 100 random configurations, and evolves them simultaneously 
using the same MCMC random numbers until they coalesce.
"""

from utils.cftp_func import *

# --- Setup and Parameters ---
save_path_data = save_path + "data/algo/complet/"
save_path_fig = save_path + "figures/algo/complet/"

N_list = [100, 200, 500, 750, 1000, 1500]#, 2000, 2250, 2500, 2750, 3000]

# Theoretical critical beta for the SK Model
beta_sg = 1

# Generate betas up to slightly past the Spin Glass transition
betas = np.linspace(0, beta_sg * 1.2, 50, endpoint=True)

csv_filename = save_path_data + "SK_damage_spreading.csv"

# --- 1. Generation Phase ---
# Initialize the CSV file with headers
# with open(csv_filename, 'w') as f:
#     f.write("N,beta,t_coal\n")

# for N in N_list:
#     G, couplings = create_complete_graph(N=N, model="SK")
    
#     for beta in betas:
#         t_coal = []
#         print(f"Using beta={beta:.4f} (beta_SG={beta_sg:.4f}), N={N}")
        
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
#         with open(csv_filename, 'a') as f:
#             f.write(f"{N},{beta},{t_coal_mean}\n")


# --- 2. Loading and Fitting Phase ---
beta_DS_estimated = []
popt_list = []

# Load all three columns using the flattened format
N_full = np.loadtxt(csv_filename, delimiter=',', skiprows=1, usecols=0)
betas_full = np.loadtxt(csv_filename, delimiter=',', skiprows=1, usecols=1)
t_coal_full = np.loadtxt(csv_filename, delimiter=',', skiprows=1, usecols=2)

for i, N in enumerate(N_list):
    print(f"\n--- Fitting and Plotting for N = {N} ---")
    
    # Isolate the data for the current N using boolean masking
    mask_N = (N_full == N)
    betas_N = betas_full[mask_N]
    t_coal_N = t_coal_full[mask_N]
    
    # Filter out -1 values and drop the last 5 elements (to avoid tail artifacts)
    valid_mask = (t_coal_N != -1)
    if N<500:
        final_betas = betas_N[valid_mask][:-5]
        final_t_coal = t_coal_N[valid_mask][:-5]
    else:
        final_betas = betas_N[valid_mask]
        final_t_coal = t_coal_N[valid_mask]
    
    # Perform the curve fit
    try:
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
plt.xlabel("N")
plt.ylabel(r"Estimated $\beta_{DS}$")
plt.legend()
plt.savefig(save_path_fig + "SK_beta_damage_spreading.png", dpi=300)
plt.savefig(save_path_fig + "SK_beta_damage_spreading.pdf", dpi=300)
plt.close()

# Plot 1: Log(T_coal) vs beta for all N
fig, ax = plt.subplots()
for i, N in enumerate(N_list):
    mask_N = (N_full == N)
    betas_N = betas_full[mask_N]
    t_coal_N = t_coal_full[mask_N]
    color = plt.cm.viridis(i / len(N_list))
    
    ax.plot(betas_N, np.log(t_coal_N), 'o-', color=color)
    
    popt = popt_list[i]
    if popt is not None:
        betas_fit = np.linspace(min(betas_N), max(betas_N), 100)
        t_coal_fit = power_law_log_fit(betas_fit, *popt)
        
        # Plot up to the argmax to avoid fitting the downward curve
        max_value_fit_index = np.argmax(t_coal_fit)
        ax.plot(betas_fit[:max_value_fit_index], t_coal_fit[:max_value_fit_index], 'k-', linewidth=2, color=color)

# Theoretical vertical line for SK
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(N_list), vmax=max(N_list)))
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('L')
ax.axvline(beta_sg, color='k', linestyle='-', label=r'$\beta_{SG}$')
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\log(T_{coal})$")
# Place legend outside the plot to handle the large number of N values cleanly
ax.legend()
fig.tight_layout()
fig.savefig(save_path_fig + "SK_damage_spreading_coal.png", dpi=300)
fig.savefig(save_path_fig + "SK_damage_spreading_coal.pdf", dpi=300)
plt.close()