"""
This script processes models individually. For each model, it filters the data by 
specific beta values, performs a power law fit (a * N^b + c) for each beta, and 
plots the fit curves with a colorbar for beta. It also plots how the exponent 'b' 
and the R^2 score vary as a function of beta for ALL models on combined graphs.
"""

from utils.cftp_func import *


def power_law_fit(x, a, b, c):
    # Fits a power law to find the exponent b
    return a * np.power(x, b) + c

model_list = ["ER", "RR", "CW", "SK", "RL_ferr2", "RL_sg2", "RL_ferr3", "RL_sg3"]  # Add RL models with dimensions specified

# Define marker and color lists for the global exponent plot
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
marker_list = ['o', 's', 'P', 'D', 'X', '^', 'v', '<']

# ==========================================
# INITIALIZE GLOBAL FIGURES: Exponent vs Beta and R^2 vs Beta
# ==========================================
fig_exp_global, ax_exp_global = plt.subplots()
fig_r2_global, ax_r2_global = plt.subplots()

all_b_exponents = []  
for m_idx, model in enumerate(model_list):
    path_to_file = "/home/lperon/cftp_dis_spin/data/algo/"
    d=None
    if model in ["CW", "SK"]:
        path_to_file += f"complet/{model}_coal_time.csv"
    elif model in ["ER", "RR"]:
        path_to_file += f"random/{model}_coal_time.csv"
    elif model in ["RL_ferr2", "RL_sg2", "RL_ferr3", "RL_sg3"]:
        d = int(model[-1])
        model = model.replace(f'{d}', '')
        path_to_file += f"latt/d{d}/{model.replace('RL_', '')}/{model}_coal_time.csv"
    print(path_to_file)
    df = pd.read_csv(path_to_file)

    betas = sorted(df['beta'].unique())

    # Lists to store the results for the current model
    exponents_b = []
    r2_scores = []
    valid_betas = []

    print(f"\n========================================")
    print(f"--- Processing Model: {model} ---")
    print(f"========================================")

    # Initialize figure for this specific model's fits
    fig_fits, ax_fits = plt.subplots()

    # Create a colormap and a normalization based on the min/max of betas
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=min(betas), vmax=max(betas))

    for beta in betas:
        # Isolate data for the specific beta
        df_beta = df[df['beta'] == beta]
        
        # Group by N and calculate mean and variance for the current beta
        df_mean = df_beta.groupby('N')['mean_log_coal_N'].agg(['mean', 'var']).reset_index()
        
        N_list = df_mean['N'].values
        y_vals = df_mean['mean'].values
        # Fill NaNs with 0 for variance if there's only one point per N
        y_err = np.sqrt(df_mean['var'].fillna(0).values) 

        # ====================================================
        # Filter out NaN values from y_vals 
        # and corresponding elements in N_list and y_err
        # ====================================================
        valid_mask = ~np.isnan(y_vals)
        N_list = N_list[valid_mask]
        y_vals = y_vals[valid_mask]
        y_err = y_err[valid_mask]

        # We need at least 3 points to fit 3 parameters (a, b, c)
        if len(N_list) < 3:
            print(f"[{model} | beta={beta}] Skipped: Not enough valid data points ({len(N_list)}).")
            continue
        
        # Get the exact color for this beta from the normalized colormap
        color_val = cmap(norm(beta))

        # Plot the raw data points for this beta
        ax_fits.errorbar(N_list, y_vals, yerr=y_err, fmt='o', color=color_val, 
                         linestyle='None', alpha=0.6)

        # Perform the Power Law Fit
        try:
            # p0 guesses: a=1, b=-1 (assuming decay, change if growth is expected), c=0
            popt_power, pcov_power = curve_fit(power_law_fit, N_list, y_vals, p0=(1, -1, 0), maxfev=10000)
            a_fit, b_fit, c_fit = popt_power
            
            r2 = r2_score(y_vals, power_law_fit(N_list, *popt_power))
            
            # Store valid fits for the global Exponent vs Beta plot and R^2 plot
            exponents_b.append(b_fit)
            r2_scores.append(r2)
            valid_betas.append(beta)
            
            # Plot the continuous fit line
            N_fit_dense = np.linspace(min(N_list), max(N_list), 100)
            ax_fits.plot(N_fit_dense, power_law_fit(N_fit_dense, *popt_power), color=color_val, linestyle='-')
            
            print(f"[{model} | beta={beta}] Fit: a={a_fit:.2e}, b={b_fit:.4f}, c={c_fit:.2e} | R^2={r2:.3f}")
        
        except Exception as e:
            print(f"[{model} | beta={beta}] Power law fit failed: {e}")

    # ==========================================
    # FINALIZE FIGURE 1: All Fits for the Model
    # ==========================================
    ax_fits.set_xlabel('$N$')
    ax_fits.set_ylabel('$\\log(T_{coal})/N$')
    ax_fits.set_xscale('log')
    ax_fits.set_yscale('log')
    
    # Create the colorbar using the ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig_fits.colorbar(sm, ax=ax_fits)
    cbar.set_label(r'$\beta$')
    save_name = f"/home/lperon/cftp_dis_spin/figures/temp/power_law_fits_all_betas_{model}.png"
    if d!=None:
        save_name = save_name.replace(".png", f"_d{d}.png")
    fig_fits.savefig(save_name)
    plt.close(fig_fits)

    # ==========================================
    # ADD TO GLOBAL FIGURES: Exponent vs Beta and R^2 vs Beta
    # ==========================================
    if len(valid_betas) > 0:
        c = color_list[m_idx]
        m = marker_list[m_idx]
        label = f"{model}"
        if d!=None:
            label += f" (d={d})"
            
        # Plot Exponent b vs Beta
        ax_exp_global.plot(valid_betas, exponents_b, marker=m, linestyle='-', 
                           color=c, label=label, markersize=8, alpha=0.5)
        
        # Plot R^2 vs Beta
        ax_r2_global.plot(valid_betas, r2_scores, marker=m, linestyle='-', 
                          color=c, label=label, markersize=8, alpha=0.5)
    else:
        print(f"[{model}] No valid beta fits available to plot global metrics.")
    all_b_exponents.extend(exponents_b)  # Collect all exponents for global reference

# ==========================================
# FINALIZE GLOBAL FIGURE: Exponent vs Beta
# ==========================================
if len(all_b_exponents) > 0:
    mean_b_exponent = np.mean(all_b_exponents)
    ax_exp_global.axhline(mean_b_exponent, xmin=0, xmax=1, color='black', linestyle='--', label=f'Mean $b=${mean_b_exponent:.4f}')
ax_exp_global.set_xlabel(r'$\beta$')
ax_exp_global.set_ylabel('Power Law Exponent ($b$)')
ax_exp_global.legend()
ax_exp_global.set_title('Values of Exponent $b$ vs Beta for the fit\n$\\log(T_{coal})/N \sim a N^b + c$ across all models', fontsize=20)
fig_exp_global.savefig("/home/lperon/cftp_dis_spin/figures/temp/exponent_vs_beta_all_models.png")
plt.close(fig_exp_global)

# ==========================================
# FINALIZE GLOBAL FIGURE: R^2 vs Beta
# ==========================================
ax_r2_global.set_xlabel(r'$\beta$')
ax_r2_global.set_ylabel(r'$R^2$ Score')
ax_r2_global.legend()
ax_r2_global.set_title('Values of $R^2$ Score vs Beta for the fit\n$\\log(T_{coal})/N \sim a N^b + c$ across all models', fontsize=20)
fig_r2_global.savefig("/home/lperon/cftp_dis_spin/figures/temp/r2_vs_beta_all_models.png")
plt.close(fig_r2_global)

print("\nProcessing complete. Figures saved.")