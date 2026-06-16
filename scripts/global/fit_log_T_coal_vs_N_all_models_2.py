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

# ==========================================
# PLOT WITH LOG(T_coal)/N vs N
# ==========================================

# for i, model in enumerate(model_list):
#     path_to_file = "/home/lperon/cftp_dis_spin/data/algo/"
#     d = None
#     if model in ["CW", "SK"]:
#         path_to_file += f"complet/{model}_coal_time_F_beta_Glauber.csv"
#     elif model in ["ER", "RR"]:
#         path_to_file += f"random/{model}_coal_time_F_beta_Glauber.csv"
#     elif model in ["RL_ferr2", "RL_sg2", "RL_ferr3", "RL_sg3"]:
#         d = int(model[-1])
#         model_base = model.replace(f'{d}', '')
#         path_to_file += f"latt/d{d}/{model_base.replace('RL_', '')}/{model_base}_coal_time_F_beta_Glauber.csv"
#     print(f"Reading: {path_to_file}")

#     df = pd.read_csv(path_to_file)
    
#     # Drop rows where our target values might be NaN
#     valid_df = df.dropna(subset=['N', 'beta', 'mean_log_coal_N'])

#     # Extract all un-averaged data
#     N_vals = valid_df['N'].values
#     beta_vals = valid_df['beta'].values
#     y_vals = valid_df['mean_log_coal_N'].values
    
#     # Append to global lists for fitting
#     all_N_data.extend(N_vals)
#     all_beta_data.extend(beta_vals)
#     all_y_data.extend(y_vals)

#     # Store for plotting
#     model_data_dict[model] = {
#         'N': N_vals,
#         'beta': beta_vals,
#         'y': y_vals,
#         'marker': marker_list[i],
#         'd': d
#     }

#     # Track min and max beta for the global colormap
#     if len(beta_vals) > 0:
#         min_beta = min(min_beta, min(beta_vals))
#         max_beta = max(max_beta, max(beta_vals))

# all_N_array = np.array(all_N_data)
# all_y_array = np.array(all_y_data)

# # ==========================================
# # PLOTTING
# # ==========================================
# fig, ax = plt.subplots()

# # Set up the colormap and normalization based on global beta limits
# cmap = mpl.cm.viridis
# norm = mpl.colors.Normalize(vmin=min_beta, vmax=max_beta)

# # Plot each model's data
# for model in model_list:
#     data = model_data_dict[model]
#     if len(data['N']) == 0:
#         continue
    
#     # Scatter plot handles individual colors for every point via the 'c' argument
#     ax.scatter(data['N'], data['y'], c=data['beta'], cmap=cmap, norm=norm, 
#                marker=data['marker'], s=40, alpha=0.7, edgecolors='k', linewidth=0.5)

# # Format axes
# ax.set_xlabel('$N$')
# ax.set_ylabel('$\\log(T_{coal})/N$')

# # Add the colorbar
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# cbar = fig.colorbar(sm, ax=ax)
# cbar.set_label(r'$\beta$')

# # Create custom legend handles to keep the legend clean 
# # (otherwise it will grab random colors from the colormap)
# custom_handles = []
# for model in model_list:
#     if model == "RL_ferr2":
#         label = "F2D"
#     elif model == "RL_sg2":
#         label = "EA2D"
#     elif model == "RL_ferr3":
#         label = "F3D"
#     elif model == "RL_sg3":
#         label = "EA3D"
#     else:
#         label = model
#     data = model_data_dict[model]
    
#     h = mlines.Line2D([], [], color='w', marker=data['marker'], 
#                       markerfacecolor='gray', markeredgecolor='k', 
#                       markersize=8, label=label)
#     custom_handles.append(h)

# ax.set_xscale('log')
# ax.set_yscale('log')
# # Place legend outside the plot so it doesn't obscure the dense data
# ax.legend(handles=custom_handles)

# save_path = "/home/lperon/cftp_dis_spin/figures/temp/log_T_coal_vs_N_global_power_law_all_betas.png"
# fig.savefig(save_path)
# plt.close(fig)

# print(f"\nProcessing complete. Figure saved to:\n{save_path}")

# ==========================================
# PLOT WITH LOG(T_coal)/LOG(N) vs N
# ==========================================

for i, model in enumerate(model_list):
    path_to_file = "/home/lperon/cftp_dis_spin/data/algo/"
    d = None
    if model in ["CW", "SK"]:
        path_to_file += f"complet/{model}_coal_time_F_beta_Glauber.csv"
    elif model in ["ER", "RR"]:
        path_to_file += f"random/{model}_coal_time_F_beta_Glauber.csv"
    elif model in ["RL_ferr2", "RL_sg2", "RL_ferr3", "RL_sg3"]:
        d = int(model[-1])
        model_base = model.replace(f'{d}', '')
        path_to_file += f"latt/d{d}/{model_base.replace('RL_', '')}/{model_base}_coal_time_F_beta_Glauber.csv"
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
        'y': y_vals * N_vals / np.log(N_vals),  # Transform y for the new plot
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
# PLOTTING
# ==========================================
fig, ax = plt.subplots()

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

# Format axes
ax.set_xlabel('$N$')
ax.set_ylabel('$\\log(T_{coal})/\\log(N)$')

# Add the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r'$\beta$')

# Create custom legend handles to keep the legend clean 
# (otherwise it will grab random colors from the colormap)
custom_handles = []
for model in model_list:
    if model == "RL_ferr2":
        label = "F2D"
    elif model == "RL_sg2":
        label = "EA2D"
    elif model == "RL_ferr3":
        label = "F3D"
    elif model == "RL_sg3":
        label = "EA3D"
    else:
        label = model
    data = model_data_dict[model]
    
    h = mlines.Line2D([], [], color='w', marker=data['marker'], 
                      markerfacecolor='gray', markeredgecolor='k', 
                      markersize=8, label=label)
    custom_handles.append(h)

# ax.set_xscale('log')
# ax.set_yscale('log')
# Place legend outside the plot so it doesn't obscure the dense data
ax.legend(handles=custom_handles)

save_path = "/home/lperon/cftp_dis_spin/figures/temp/log_T_coal_vs_log(N)_global_power_law_all_betas.png"
fig.savefig(save_path)

print(f"\nProcessing complete. Figure saved to:\n{save_path}")