"""
This script processes models individually. For each model, it filters the data by 
specific beta values, performs a 2-parameter fit (b/log(N) + c) for log(T)/log(N) 
across N. This corresponds to the polynomial scaling T = e^b * N^c.
It plots the fit curves, and generates global plots for 'c', the uncertainty of 'c', 
the prefactor term 'b', and the R^2 score as a function of beta across ALL models.
"""

from utils.cftp_func import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# ==========================================
# DEFINE INVERSE LOG FIT FUNCTION
# ==========================================
def log_inv_fit(N, b, c):
    """
    Fits b / log(N) + c.
    Equivalent to assuming T = exp(b) * N^c
    """
    return (b / np.log(N)) + c

model_list = ["ER", "RR", "CW", "SK", "RL_ferr2", "RL_sg2", "RL_ferr3", "RL_sg3"]
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
marker_list = ['o', 's', 'P', 'D', 'X', '^', 'v', '<']
model_beta_uni = [np.arctanh(1/4), np.arctanh(1/3), 1, 1, 1/2 * np.log(1 + np.sqrt(2)), np.arctanh(1/3), 0.221654626, np.arctanh(1/5)]
sampler = F_beta_Metropolis

# ==========================================
# INITIALIZE GLOBAL FIGURES
# ==========================================
fig_c_global, ax_c_global = plt.subplots()
fig_err_global, ax_err_global = plt.subplots()
fig_b_global, ax_b_global = plt.subplots()
fig_r2_global, ax_r2_global = plt.subplots()

all_c_vals = []  
for m_idx, model in enumerate(model_list):
    path_to_file = "/home/lperon/cftp_dis_spin/data/algo/"
    d = None
    if model in ["CW", "SK"]:
        path_to_file += f"complet/{model}_coal_time_{sampler.__name__}.csv"
    elif model in ["ER", "RR"]:
        path_to_file += f"random/{model}_coal_time_{sampler.__name__}.csv"
    elif model in ["RL_ferr2", "RL_sg2", "RL_ferr3", "RL_sg3"]:
        d = int(model[-1])
        model = model.replace(f'{d}', '')
        path_to_file += f"latt/d{d}/{model.replace('RL_', '')}/{model}_coal_time_{sampler.__name__}.csv"
    
    print(path_to_file)
    df = pd.read_csv(path_to_file)

    betas = sorted(df['beta'].unique())

    # Lists to store the results for the current model
    b_vals = []
    c_vals = []
    c_errs = []
    r2_scores = []
    valid_betas = []

    print(f"\n========================================")
    print(f"--- Processing Model: {model} ---")
    print(f"========================================")

    # Initialize figure for this specific model's fits
    fig_fits, ax_fits = plt.subplots()

    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=min(betas), vmax=max(betas))

    for beta in betas:
        df_beta = df[df['beta'] == beta]
        
        df_mean = df_beta.groupby('N')['mean_log_coal_N'].agg(['mean', 'var']).reset_index()
        
        N_list = df_mean['N'].values
        
        # Scale for log(T)/log(N)
        scale_factor = N_list / np.log(N_list)
        y_vals = df_mean['mean'].values * scale_factor
        y_err = np.sqrt(df_mean['var'].fillna(0).values) * scale_factor

        # Filter out NaN values
        valid_mask = ~np.isnan(y_vals)
        N_list = N_list[valid_mask]
        y_vals = y_vals[valid_mask]
        y_err = y_err[valid_mask]

        # Need at least 3 points for a 2-parameter fit to get meaningful variance/R^2
        if len(N_list) < 3:
            print(f"[{model} | beta={beta}] Skipped: Not enough valid data points ({len(N_list)}).")
            continue
        
        color_val = cmap(norm(beta))

        # Replace 0.0 error values with a small number to prevent ZeroDivisionError in weighting
        y_err_safe = np.where(y_err == 0, 1e-8, y_err)

        # Plot raw data
        ax_fits.errorbar(N_list, y_vals, yerr=y_err, fmt='o', color=color_val, 
                         linestyle='None', alpha=0.6)

        # Perform the b/log(N) + c fit
        try:
            # p0 guesses: b=0 (prefactor A=1), c=1. 
            popt, pcov = curve_fit(
                log_inv_fit, N_list, y_vals, 
                p0=(0.0, 1.0), 
                sigma=y_err_safe, 
                absolute_sigma=True, 
                maxfev=10000
            )
            b_fit, c_fit = popt
            
            # Uncertainty for c is the sqrt of the 2nd diagonal covariance matrix element
            c_err = np.sqrt(pcov[1][1])
            
            r2 = r2_score(y_vals, log_inv_fit(N_list, *popt))
            
            b_vals.append(b_fit)
            c_vals.append(c_fit)
            c_errs.append(c_err)
            r2_scores.append(r2)
            valid_betas.append(beta)
            
            N_fit_dense = np.linspace(min(N_list), max(N_list), 100)
            ax_fits.plot(N_fit_dense, log_inv_fit(N_fit_dense, *popt), color=color_val, linestyle='-')
            
            print(f"[{model} | beta={beta}] Fit: b={b_fit:.4f}, c={c_fit:.4f} ± {c_err:.4e} | R^2={r2:.3f}")
        
        except Exception as e:
            print(f"[{model} | beta={beta}] Fit failed: {e}")

    # ==========================================
    # FINALIZE FIGURE 1: All Fits for the Model
    # ==========================================
    ax_fits.set_xlabel('$N$')
    ax_fits.set_ylabel('$\\log(T_{coal})/\\log(N)$')
    ax_fits.set_xscale('log')
    # Leaving y-scale as linear since the curve b/log(N) + c is visually easier to parse here
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig_fits.colorbar(sm, ax=ax_fits)
    cbar.set_label(r'$\beta$')
    
    save_name = f"/home/lperon/cftp_dis_spin/figures/temp/poly_pref_fits_all_betas_{model}_{sampler.__name__}.png"
    if d != None:
        save_name = save_name.replace(".png", f"_d{d}.png")
    fig_fits.savefig(save_name)
    plt.close(fig_fits)

    # ==========================================
    # ADD TO GLOBAL FIGURES
    # ==========================================
    if len(valid_betas) > 0:
        col = color_list[m_idx]
        m = marker_list[m_idx]
        if model in ["RL_ferr", "RL_sg"]:
            if model == "RL_ferr" and d==2: label = "F2D"
            elif model == "RL_sg" and d==2: label = "EA2D"
            elif model == "RL_ferr" and d==3: label = "F3D"
            elif model == "RL_sg" and d==3: label = "EA3D"
        else:
            label = model
            
        # Plot Constant c vs Beta
        ax_c_global.plot(valid_betas, c_vals, marker=m, linestyle='-', 
                         color=col, label=label, markersize=7)
        if label not in ["SK", "EA2D"]:
            ax_c_global.axvline(model_beta_uni[m_idx], color=col, linestyle='--', alpha=0.75, linewidth=2)
        if model == "SK":
            ax_c_global.axvline(np.sqrt(np.pi/(200)), color=col, linestyle='--', alpha=0.75, linewidth=2)
            
                         
        # Plot Uncertainty of c vs Beta
        ax_err_global.plot(valid_betas, c_errs, marker=m, linestyle='-', 
                           color=col, label=label, markersize=7)
                           
        # Plot Prefactor b vs Beta
        ax_b_global.plot(valid_betas, b_vals, marker=m, linestyle='-', 
                         color=col, label=label, markersize=7)
        
        # Plot R^2 vs Beta
        ax_r2_global.plot(valid_betas, r2_scores, marker=m, linestyle='-', 
                          color=col, label=label, markersize=7)
    else:
        print(f"[{model}] No valid beta fits available to plot global metrics.")
        
    all_c_vals.extend(c_vals)

# ==========================================
# FINALIZE GLOBAL FIGURE: Exponent 'c' vs Beta
# ==========================================
if len(all_c_vals) > 0:
    mean_c_val = np.mean(all_c_vals)
ax_c_global.set_xlabel(r'$\beta$')
ax_c_global.set_ylabel('Polynomial Exponent ($c$)')
ax_c_global.legend()

fig_c_global.savefig(f"/home/lperon/cftp_dis_spin/figures/algo/all_models/exponent_c_vs_beta_all_models_{sampler.__name__}.png")
plt.close(fig_c_global)

# ==========================================
# FINALIZE GLOBAL FIGURE: Uncertainty of 'c' vs Beta
# ==========================================
ax_err_global.set_xlabel(r'$\beta$')
ax_err_global.set_ylabel('Uncertainty of $c$ (from Covariance)')
ax_err_global.set_yscale('log') 
ax_err_global.legend()

fig_err_global.savefig(f"/home/lperon/cftp_dis_spin/figures/algo/all_models/uncertainty_c_vs_beta_all_models_{sampler.__name__}.png")
plt.close(fig_err_global)

# ==========================================
# FINALIZE GLOBAL FIGURE: Prefactor term 'b' vs Beta
# ==========================================
ax_b_global.set_xlabel(r'$\beta$')
ax_b_global.set_ylabel('Prefactor Term ($b=\\log A$)')
ax_b_global.legend()

fig_b_global.savefig(f"/home/lperon/cftp_dis_spin/figures/algo/all_models/prefactor_b_vs_beta_all_models_{sampler.__name__}.png")
plt.close(fig_b_global)

# ==========================================
# FINALIZE GLOBAL FIGURE: R^2 vs Beta
# ==========================================
ax_r2_global.set_xlabel(r'$\beta$')
ax_r2_global.set_ylabel(r'$R^2$ Score')
ax_r2_global.legend()

fig_r2_global.savefig(f"/home/lperon/cftp_dis_spin/figures/algo/all_models/r2_vs_beta_poly_pref_all_models_{sampler.__name__}.png")
plt.close(fig_r2_global)

print("\nProcessing complete. Figures saved.")