"""
This script gets the data from the coal_time files and performs a global exponential 
fit of log(T_coal)/N vs N across all models. It also plots log(T_coal)/log(N) vs N 
to check if the growth is polynomial or exponential across all models combined.
"""

from utils.cftp_func import *
    
def polynomial_fit(x, a, b):
    # Note: Fits a logarithmic curve, preserving the original mathematical structure
    return a + b * np.log(x)

def exponential_fit(x, a, b, c):
    return a + b * np.exp(-c * x)

def power_law_fit(x, a, b, c):
    return a * np.power(x, b) + c

model_list = ["ER", "RR", "CW", "SK", "RL_ferr2", "RL_sg2", "RL_ferr3", "RL_sg3"]

# 1. Define marker and color lists
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
marker_list = ['o', 's', 'P', 'D', 'X', '^', 'v', '<']

# Global lists to store all data points across all models for the combined fits
all_N_data = []
all_y_data = []

for i, model in enumerate(model_list):
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

    N_list = sorted(df['N'].unique())
    N_list = np.array(N_list)
    betas = sorted(df['beta'].unique())

    beta_no_zero = None
    ceiling_mask = df['mean_coal_time'] == max_nb_iter_MCMC2_BC
    if ceiling_mask.any():
        beta_no_zero = df[ceiling_mask]['beta'].min()

    pivot_df = df.pivot(index='N', columns='beta', values='mean_coal_time')
    coalescence_times_matrix = pivot_df.values

    mean_log_T_coal_vs_N_lst = []
    var_log_T_coal_vs_N_lst = []

    print(f"\n--- Processing Model: {model} ---")
    for N in N_list:
        sub_df = df[df['N'] == N]
        beta = sub_df['beta'].values
        mean_log_T_coal_vs_N = sub_df['mean_log_coal_N'].values

        mean_log_T_coal_vs_N_mean = np.nanmean(mean_log_T_coal_vs_N)
        mean_log_T_coal_vs_N_lst.append(mean_log_T_coal_vs_N_mean)

        var_log_T_coal_vs_N = np.nanvar(mean_log_T_coal_vs_N)
        var_log_T_coal_vs_N_lst.append(var_log_T_coal_vs_N)
    
    all_N_data.extend(N_list)
    all_y_data.extend(mean_log_T_coal_vs_N_lst)

    c = color_list[i]
    m = marker_list[i]
    label = f"{model}"
    if d!=None:
        label += f" (d={d})"
    plt.errorbar(N_list, mean_log_T_coal_vs_N_lst, yerr=var_log_T_coal_vs_N_lst, 
                 fmt=m, color=c, label=label, linestyle='None', alpha=0.7)

all_N_array = np.array(all_N_data)
all_y_array = np.array(all_y_data)

N_fit_dense = np.linspace(min(all_N_array), max(all_N_array), 200)

print("\n--- Performing Global Fits ---")

try:
    popt_power, pcov_power = curve_fit(power_law_fit, all_N_array, all_y_array, p0=(1, -1, 0), maxfev=10000)
    a_fit_power, b_fit_power, c_fit_power = popt_power
    
    y_fit_power = power_law_fit(all_N_array, *popt_power)
    r2_power = r2_score(all_y_array, y_fit_power)
    
    print(f"[GLOBAL] Power Law Fit: a={a_fit_power:.4f}, b={b_fit_power:.4f}, c={c_fit_power:.4f} | R^2 = {r2_power:.4f}")
    
    power_label = r'Fit $aN^b+c$'+f'\n$R^2$={r2_power:.4f}\n$a=${a_fit_power:.4f}\n$b=${b_fit_power:.4f}\n$c=${c_fit_power:.4f}'
    plt.plot(N_fit_dense, power_law_fit(N_fit_dense, *popt_power), color='black', linestyle='-', linewidth=2, label=power_label)
except Exception as e:
    print(f"[GLOBAL] Power law fit failed: {e}")

plt.xlabel('$N$')
plt.ylabel('$\\log(T_{coal})/N$')
plt.xscale('log')
plt.yscale('log')
# Move legend outside to avoid obscuring data
plt.legend()
plt.tight_layout()
plt.savefig("/home/lperon/cftp_dis_spin/figures/temp/log_T_coal_vs_N_all_models_power_law.png")

print("\nProcessing complete. Figures saved.")