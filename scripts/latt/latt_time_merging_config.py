"""
This script constructs a spin system on a graph, initializes 100 random configurations, 
and evolves them simultaneously using the same MCMC random numbers until they coalesce.
"""

from utils.cftp_func import *

L_lists = {
    2: [10, 14, 23, 27, 32, 39, 45, 48, 50, 52, 55],
    3: [5, 6, 8, 9, 10, 12, 13, 14, 15]
}

for d in [2, 3]:
    for model in ['RL_ferr', 'RL_sg']:
        print(f"\n=========================================")
        print(f"Starting d={d}, model={model}")
        print(f"=========================================\n")
        
        # Determine critical betas based on dimension and model
        beta_BD = np.arctanh(1/(2*d))
        if d == 2:
            if model == "RL_ferr":
                beta_pm = 0.5 * (np.log(1 + np.sqrt(2)))
            else:
                beta_uni = np.arctanh(1/(2*d-1))
                beta_sg = 1  # No SG transition in 2D
                beta_ds = 1/1.69
        elif d == 3:
            if model == "RL_ferr":
                beta_pm = 0.221654626
            else:
                beta_sg = 1/1.3
                beta_uni = np.arctanh(1/(2*d-1))
                beta_ds = 1/3.89

        # Generate the list of betas to evaluate
        if model == "RL_ferr":
            betas = np.linspace(0, beta_pm*1.2, 50, endpoint=True)
        else:
            betas = np.linspace(0, beta_sg, 50, endpoint=True)

        model_dir = model.replace("RL_", "")
        current_save_path_data = save_path + f"data/algo/latt/d{d}/{model_dir}/"
        current_save_path_fig = save_path + f"figures/algo/latt/d{d}/{model_dir}/"
        
        csv_filename = current_save_path_data + f"{model}_damage_spreading.csv"

        # --- 1. Generation Phase ---
        # with open(csv_filename, 'w') as f:
        #     f.write("N,beta,t_coal\n")

        # for L in L_lists[d]:
        #     # Calculate L from N
        #     N = L**d
        #     G, couplings = create_lattice_graph(L=L, d=d, model=model)
            
        #     for beta in betas:
        #         t_coal = []
        #         print(f"Using beta={beta:.4f}, N={N} (L={L})")
                
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
                
        #         # Save data continuously
        #         with open(csv_filename, 'a') as f:
        #             f.write(f"{N},{beta},{t_coal_mean}\n")

        # --- 2. Loading and Fitting Phase ---
        beta_DS_estimated = []
        popt_list = []

        # Load the newly generated data
        N_full = np.loadtxt(csv_filename, delimiter=',', skiprows=1, usecols=0)
        betas_full = np.loadtxt(csv_filename, delimiter=',', skiprows=1, usecols=1)
        t_coal_full = np.loadtxt(csv_filename, delimiter=',', skiprows=1, usecols=2)

        for i, L in enumerate(L_lists[d]):
            N = L**d
            print(f"\n--- Fitting for L = {L} ---")
            mask_N = (N_full == N)
            betas_N = betas_full[mask_N]
            t_coal_N = t_coal_full[mask_N]
            
            # Filter out -1 values and drop the last 5 elements
            valid_mask = (t_coal_N != -1)
            final_betas = betas_N[valid_mask][:-5]
            final_t_coal = t_coal_N[valid_mask][:-5]
            
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
                print(f"R^2 score: {r2score:.4f}")
                
                beta_DS_estimated.append(popt[1])
                popt_list.append(popt)
                
            except Exception as e:
                print(f"Fit failed for N = {N}. Reason: {e}")
                beta_DS_estimated.append(np.nan)
                popt_list.append(None)

        # --- 3. Plotting Phase ---

        # Plot 0: Estimated beta_DS vs N
        plt.figure(0)
        plt.plot(L_lists[d], beta_DS_estimated, 'o-', label=r'Estimated $\beta_{DS}$')
        # popt_beta_DS, _ = curve_fit(power_law_fit, L_lists[d][3:], beta_DS_estimated[3:], p0=[1, -2, 0.3], maxfev=10000, absolute_sigma=False)
        # r2score_beta_DS = r2_score(beta_DS_estimated[3:], power_law_fit(L_lists[d][3:], *popt_beta_DS))
        # print(f"Fitted parameters for beta_DS vs N: a={popt_beta_DS[0]:.6f}, b={popt_beta_DS[1]:.4f}, c={popt_beta_DS[2]:.4f}")
        # print(f"R^2 score for beta_DS fit: {r2score_beta_DS:.4f}")
        # L_fit = np.linspace(min(L_lists[d][3:]), max(L_lists[d][3:]), 100)
        # beta_DS_fit = power_law_fit(L_fit, *popt_beta_DS)
        # plt.plot(L_fit, beta_DS_fit, 'k-', label=f'Power-law fit $\searrow$ {popt_beta_DS[2]:.4f}', linewidth=2)
        plt.axhline(np.mean(beta_DS_estimated[3:]), 0, np.max(L_lists[d])*1.1, color='k', linestyle='--', label=fr'Mean $\beta_{{DS}}={np.mean(beta_DS_estimated[3:]):.4f}$ ($L\geq${L_lists[d][3]})')
        if model == "RL_sg":
            plt.axhline(beta_ds, 0, np.max(L_lists[d])*1.1, color='g', linestyle=':', label=r'$\beta_{DS}$ from literature')
            plt.axhline(beta_uni, 0, np.max(L_lists[d])*1.1, color='b', linestyle='-.', label=r'$\beta_{Uni}$')
        plt.xlabel(f"L (d={d})")
        plt.ylabel(r"Estimated $\beta_{DS}$")
        plt.legend()
        plt.savefig(current_save_path_fig + f"{model}_beta_damage_spreading.png", dpi=300)
        plt.close()

        # Plot 1: Log(T_coal) vs beta for all N
        fig, ax = plt.subplots()
        for i, L in enumerate(L_lists[d]):
            N = L**d
            mask_N = (N_full == N)
            betas_N = betas_full[mask_N]
            t_coal_N = t_coal_full[mask_N]
            color = plt.cm.viridis(i / len(L_lists[d]))
            
            ax.plot(betas_N, np.log(t_coal_N), 'o-', color=color)
            
            popt = popt_list[i]
            if popt is not None:
                betas_fit = np.linspace(min(betas_N), max(betas_N), 100)
                t_coal_fit = power_law_log_fit(betas_fit, *popt)
                
                max_value_fit_index = np.argmax(t_coal_fit)
                ax.plot(betas_fit[:max_value_fit_index], t_coal_fit[:max_value_fit_index], 'k-', linewidth=2, color=color)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(L_lists[d]), vmax=max(L_lists[d])))
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('L')
        # Add vertical lines depending on the model and dimension
        if model == "RL_ferr":
            ax.axvline(beta_pm, color='k', linestyle='-', label=r'$\beta_{PM}$')
        else:
            ax.axvline(beta_BD, color='r', linestyle='--', label=r'$\beta_{BD/Dobr}$')
            ax.axvline(beta_uni, color='b', linestyle='-.', label=r'$\beta_{Uni}$')
            ax.axvline(beta_ds, color='g', linestyle=':', label=r'$\beta_{DS}$')
            if d == 3:
                ax.axvline(beta_sg, color='k', linestyle='-', label=r'$\beta_{SG}$')

        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\log(T_{coal})$")
        
        # Legend formatting to handle many N values gracefully
        ax.legend()
        fig.tight_layout()
        fig.savefig(current_save_path_fig + f"{model}_damage_spreading_coal.png", dpi=300)
        plt.close()
