"""
This script processes data from all models without averaging over beta. 
It performs a single global power-law fit (a * N^b + c) on all un-averaged 
data points and plots log(T_coal)/N vs N on a single graph. Markers distinguish 
the models, and a viridis colormap is used to represent the beta values.
"""

from utils.cftp_func import *

def power_law_fit(x, a, b, c):
    return a * np.power(x, b) + c

model_list = ["ER", "RR", "CW", "SK", "RL_ferr2", "RL_sg2", "RL_ferr3", "RL_sg3"]

# We only need markers now, as color will be determined by beta
marker_list = ['o', 's', 'P', 'D', 'X', '^', 'v', '<']

# Global lists to store all raw data points across all models for the global fit
all_N_data = []
all_y_data = []
all_beta_data = []

# Dictionary to store data per model for plotting purposes
model_data_dict = {}

min_beta = np.inf
max_beta = -np.inf

for i, model in enumerate(model_list):
    path_to_file = "/home/lperon/cftp_dis_spin/data/algo/"
    d = None
    if model in ["CW", "SK"]:
        path_to_file += f"complet/{model}_coal_time.csv"
    elif model in ["ER", "RR"]:
        path_to_file += f"random/{model}_coal_time.csv"
    elif model in ["RL_ferr2", "RL_sg2", "RL_ferr3", "RL_sg3"]:
        d = int(model[-1])
        model_base = model.replace(f'{d}', '')
        path_to_file += f"latt/d{d}/{model_base.replace('RL_', '')}/{model_base}_coal_time.csv"
    print(f"Reading: {path_to_file}")

    df = pd.read_csv(path_to_file)
    
    # Drop rows where our target values might be NaN
    valid_df = df.dropna(subset=['N', 'beta', 'mean_log_coal_N'])

    # Extract all un-averaged data
    N_vals = valid_df['N'].values
    beta_vals = valid_df['beta'].values
    y_vals = valid_df['mean_log_coal_N'].values
    
    # Append to global lists for fitting
    all_N_data.extend(N_vals)
    all_beta_data.extend(beta_vals)
    all_y_data.extend(y_vals)

    # Store for plotting
    model_data_dict[model] = {
        'N': N_vals,
        'beta': beta_vals,
        'y': y_vals,
        'marker': marker_list[i],
        'd': d
    }

    # Track min and max beta for the global colormap
    if len(beta_vals) > 0:
        min_beta = min(min_beta, min(beta_vals))
        max_beta = max(max_beta, max(beta_vals))

all_N_array = np.array(all_N_data)
all_y_array = np.array(all_y_data)

# ==========================================
# PERFORM GLOBAL POWER LAW FIT
# ==========================================
print("\n--- Performing Global Power Law Fit on Un-averaged Data ---")
try:
    popt_power, pcov_power = curve_fit(power_law_fit, all_N_array, all_y_array, p0=(1, -1, 0), maxfev=10000)
    a_fit, b_fit, c_fit = popt_power
    
    y_fit = power_law_fit(all_N_array, *popt_power)
    r2 = r2_score(all_y_array, y_fit)
    
    print(f"[GLOBAL] Power Law Fit: a={a_fit:.4f}, b={b_fit:.4f}, c={c_fit:.4f} | R^2 = {r2:.4f}")
except Exception as e:
    print(f"[GLOBAL] Power law fit failed: {e}")
    popt_power = None

# ==========================================
# PLOTTING
# ==========================================
fig, ax = plt.subplots(figsize=(10, 7))

# Set up the colormap and normalization based on global beta limits
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=min_beta, vmax=max_beta)

# Plot each model's data
for model in model_list:
    data = model_data_dict[model]
    if len(data['N']) == 0:
        continue
    
    # Scatter plot handles individual colors for every point via the 'c' argument
    ax.scatter(data['N'], data['y'], c=data['beta'], cmap=cmap, norm=norm, 
               marker=data['marker'], s=40, alpha=0.7, edgecolors='k', linewidth=0.5)

# Plot the global fit line if successful
if popt_power is not None:
    N_fit_dense = np.linspace(min(all_N_array), max(all_N_array), 200)
    power_label = r'Global Fit $aN^b+c$'+f'\n$R^2$={r2:.4f}\n$a=${a_fit:.4f}\n$b=${b_fit:.4f}\n$c=${c_fit:.4f}'
    ax.plot(N_fit_dense, power_law_fit(N_fit_dense, *popt_power), color='black', 
            linestyle='-', linewidth=2.5, label=power_label)

# Format axes
ax.set_xlabel('$N$')
ax.set_ylabel('$\\log(T_{coal})/N$')

# Add the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r'$\beta$')

# Create custom legend handles to keep the legend clean 
# (otherwise it will grab random colors from the colormap)
custom_handles = []
for model in model_list:
    data = model_data_dict[model]
    label = f"{model}"
    if data['d'] is not None:
        label += f" (d={data['d']})"
    
    h = mlines.Line2D([], [], color='w', marker=data['marker'], 
                      markerfacecolor='gray', markeredgecolor='k', 
                      markersize=8, label=label)
    custom_handles.append(h)

if popt_power is not None:
    fit_handle = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2.5, label=power_label)
    custom_handles.append(fit_handle)

ax.set_xscale('log')
ax.set_yscale('log')
# Place legend outside the plot so it doesn't obscure the dense data
ax.legend(handles=custom_handles)

save_path = "/home/lperon/cftp_dis_spin/figures/temp/log_T_coal_vs_N_global_power_law_all_betas.png"
fig.savefig(save_path)

print(f"\nProcessing complete. Figure saved to:\n{save_path}")