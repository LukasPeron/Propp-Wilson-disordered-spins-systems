"""
This script constructs an ER graph, initializes 100 random configurations, 
and evolves them simultaneously using the same MCMC random numbers until they coalesce.
"""

from cftp_my_lib import *


if __name__ == "__main__":
#     # --- 1. Graph Construction ---
    N = 500
    d = 4
#     print(f"Constructing ER graph with N={N}, d={d}")
    
#     G = nx.erdos_renyi_graph(N, d/N)
    
#     # Generate random symmetric couplings scaled by the graph adjacency
#     couplings = np.triu(np.random.choice([-1, 1], size=(N, N)), k=1)
#     couplings += couplings.T  # Symmetric graph
#     couplings = couplings * nx.to_numpy_array(G)
    
#     # Define an appropriate beta (taking Bubley-Dyer beta slightly lowered to ensure fast observable coalescence)
    beta_uni = np.arctanh(1/d)
#     betas = np.linspace(0, beta_uni * 1.5, 30, endpoint=True)
#     t_coal_beta = []
#     for beta in betas:
#     # print(f"Using beta={test_beta:.4f} (beta_uni={beta_uni:.4f})")
#         t_coal= []
#         for _ in range(10):  # Run multiple times to get an average coalescence time
#             # --- 2. Build and Simulate Configurations ---
#             coalescence_time = simulate_100_configs(
#                 beta=beta, 
#                 G=G, 
#                 couplings=couplings, 
#                 num_configs=100
#             )
#             t_coal.append(coalescence_time)

#         print(f"\nAverage coalescence time over 10 runs: {np.mean(t_coal):.2f} steps")
#         t_coal_beta.append(np.mean(t_coal))

    # --- 3. Save Results ---
    # np.save("/home/lperon/cftp_dis_spin/data/t_coal_beta.npy", t_coal_beta)
    # np.save("/home/lperon/cftp_dis_spin/data/betas.npy", betas)    

    # --- 4. Plotting Results ---

    # perform fit to power law

    def exponential_fit(beta, a, b, c):
        return a * np.exp(b * beta) + c

    def power_law_fit(beta, a, b, c):
        return a * beta**b + c

    t_coal_beta = np.load("/home/lperon/cftp_dis_spin/data/t_coal_beta.npy")
    betas = np.load("/home/lperon/cftp_dis_spin/data/betas.npy")

    below_BD_mask = betas < beta_uni
    above_BD_mask = betas >= beta_uni

    popt_power, pcov_power = curve_fit(
    power_law_fit, betas[below_BD_mask], t_coal_beta[below_BD_mask], 
    p0=(1e3, 5, 2e4), sigma=t_coal_beta[below_BD_mask], absolute_sigma=False)
    popt_exp, pcov_exp = curve_fit(
    exponential_fit, betas[above_BD_mask], t_coal_beta[above_BD_mask],
    sigma=t_coal_beta[above_BD_mask], absolute_sigma=False)
    # Calculate R^2 for both fits

    print("Fitted parameters for Power Law Fit (below beta_uni):", popt_power)
    print("Fitted parameters for Exponential Fit (above beta_uni):", popt_exp)

    r2_power = r2_score(t_coal_beta[below_BD_mask], power_law_fit(betas[below_BD_mask], *popt_power))
    r2_exp = r2_score(t_coal_beta[above_BD_mask], exponential_fit(betas[above_BD_mask], *popt_exp)) 
    
    print(f"R^2 for Power Law Fit (below beta_uni): {r2_power:.4f}")
    print(f"R^2 for Exponential Fit (above beta_uni): {r2_exp:.4f}")

    # unique figure with both fits on the same plot

    plt.plot(betas, t_coal_beta, 'o-', label='Average Coalescence Time')
    plt.plot(betas[below_BD_mask], power_law_fit(betas[below_BD_mask], *popt_power), 'k-', label=f'Power Law Fit ($R^2=${r2_power:.4f})', linewidth=2)
    plt.plot(betas[above_BD_mask], exponential_fit(betas[above_BD_mask], *popt_exp), 'm-', label=f'Exponential Fit ($R^2=${r2_exp:.4f})', linewidth=2)
    plt.axvline(beta_uni, color='r', linestyle='--', label=r'$\tanh^{-1}(1/d)$')
    plt.xlabel(r"$\beta$")
    plt.yscale("log")
    plt.ylabel("Average $T_{coal}$")
    plt.legend()
    plt.savefig("/home/lperon/cftp_dis_spin/figures/temp/coalescence_time_vs_beta_with_fits.png")
    plt.close()

    # unique plot with unique exponential fit for all points

    popt_exp_all, pcov_exp_all = curve_fit(
    exponential_fit, betas, t_coal_beta, 
    sigma=t_coal_beta, absolute_sigma=False)
    r2_exp_all = r2_score(t_coal_beta, exponential_fit(betas, *popt_exp_all))
    print("Fitted parameters for Exponential Fit (all points):", popt_exp_all)
    print(f"R^2 for Exponential Fit (all points): {r2_exp_all:.4f}")
    plt.plot(betas, t_coal_beta, 'o-', label='Average Coalescence Time')
    plt.plot(betas, exponential_fit(betas, *popt_exp_all), 'k-', label=f'Exponential Fit ($R^2=${r2_exp_all:.4f})', linewidth=2)
    plt.axvline(beta_uni, color='r', linestyle='--', label=r'$\tanh^{-1}(1/d)$')
    plt.xlabel(r"$\beta$")
    plt.ylabel("Average $T_{coal}$")
    plt.yscale("log")
    plt.legend()
    plt.savefig("/home/lperon/cftp_dis_spin/figures/temp/coalescence_time_vs_beta_exponential_fit_all.png")
    plt.close()

    # unique plot with unique power law fit for all points

    popt_power_all, pcov_power_all = curve_fit(
    power_law_fit, betas, t_coal_beta, 
    p0=(1e3, 5, 2e4), sigma=t_coal_beta, absolute_sigma=False)
    r2_power_all = r2_score(t_coal_beta, power_law_fit(betas, *popt_power_all))
    print("Fitted parameters for Power Law Fit (all points):", popt_power_all)
    print(f"R^2 for Power Law Fit (all points): {r2_power_all:.4f}")
    plt.plot(betas, t_coal_beta, 'o-', label='Average Coalescence Time')
    plt.plot(betas, power_law_fit(betas, *popt_power_all), 'k-', label=f'Power Law Fit ($R^2=${r2_power_all:.4f})', linewidth=2)
    plt.axvline(beta_uni, color='r', linestyle='--', label=r'$\tanh^{-1}(1/d)$')
    plt.xlabel(r"$\beta$")
    plt.ylabel("Average $T_{coal}$")
    plt.yscale("log")
    plt.legend()
    plt.savefig("/home/lperon/cftp_dis_spin/figures/temp/coalescence_time_vs_beta_power_law_fit_all.png")
    plt.close()

    # unique plot with both fits but both are power law (one for below and one for above)
    popt_power_below, pcov_power_below = curve_fit(power_law_fit, betas[below_BD_mask], t_coal_beta[below_BD_mask], p0=(1e3, 5, 2e4), sigma=t_coal_beta[below_BD_mask], absolute_sigma=False)
    popt_power_above, pcov_power_above = curve_fit(
    power_law_fit, betas[above_BD_mask], t_coal_beta[above_BD_mask], 
    p0=(1e3, 5, 2e4), sigma=t_coal_beta[above_BD_mask], absolute_sigma=False)
    r2_power_below = r2_score(t_coal_beta[below_BD_mask], power_law_fit(betas[below_BD_mask], *popt_power_below))
    r2_power_above = r2_score(t_coal_beta[above_BD_mask], power_law_fit(betas[above_BD_mask], *popt_power_above))
    print("Fitted parameters for Power Law Fit (below beta_uni):", popt_power_below)
    print("Fitted parameters for Power Law Fit (above beta_uni):", popt_power_above)
    print(f"R^2 for Power Law Fit (below beta_uni): {r2_power_below:.4f}")
    print(f"R^2 for Power Law Fit (above beta_uni): {r2_power_above:.4f}")
    plt.plot(betas, t_coal_beta, 'o-', label='Average Coalescence Time')
    plt.plot(betas[below_BD_mask], power_law_fit(betas[below_BD_mask], *popt_power_below), 'k-', label=f'Power Law Fit Below ($R^2=${r2_power_below:.4f})', linewidth=2)
    plt.plot(betas[above_BD_mask], power_law_fit(betas[above_BD_mask], *popt_power_above), 'm-', label=f'Power Law Fit Above ($R^2=${r2_power_above:.4f})', linewidth=2)
    plt.axvline(beta_uni, color='r', linestyle='--', label=r'$\tanh^{-1}(1/d)$')
    plt.xlabel(r"$\beta$")
    plt.ylabel("Average $T_{coal}$")
    plt.yscale("log")
    plt.legend()
    plt.savefig("/home/lperon/cftp_dis_spin/figures/temp/coalescence_time_vs_beta_power_law_fit_split.png")
    plt.close()

    # unique plot with both fits but both are exponential (one for below and one for above)
    popt_exp_below, pcov_exp_below = curve_fit(exponential_fit, betas[below_BD_mask], t_coal_beta[below_BD_mask], sigma=t_coal_beta[below_BD_mask], absolute_sigma=False)
    popt_exp_above, pcov_exp_above = curve_fit(
    exponential_fit, betas[above_BD_mask], t_coal_beta[above_BD_mask], 
    sigma=t_coal_beta[above_BD_mask], absolute_sigma=False)
    r2_exp_below = r2_score(t_coal_beta[below_BD_mask], exponential_fit(betas[below_BD_mask], *popt_exp_below))
    r2_exp_above = r2_score(t_coal_beta[above_BD_mask], exponential_fit(betas[above_BD_mask], *popt_exp_above))
    print("Fitted parameters for Exponential Fit (below beta_uni):", popt_exp_below)
    print("Fitted parameters for Exponential Fit (above beta_uni):", popt_exp_above)
    print(f"R^2 for Exponential Fit (below beta_uni): {r2_exp_below:.4f}")
    print(f"R^2 for Exponential Fit (above beta_uni): {r2_exp_above:.4f}")
    plt.plot(betas, t_coal_beta, 'o-', label='Average Coalescence Time')
    plt.plot(betas[below_BD_mask], exponential_fit(betas[below_BD_mask], *popt_exp_below), 'k-', label=f'Exponential Fit Below ($R^2=${r2_exp_below:.4f})', linewidth=2)
    plt.plot(betas[above_BD_mask], exponential_fit(betas[above_BD_mask], *popt_exp_above), 'm-', label=f'Exponential Fit Above ($R^2=${r2_exp_above:.4f})', linewidth=2)
    plt.axvline(beta_uni, color='r', linestyle='--', label=r'$\tanh^{-1}(1/d)$')
    plt.xlabel(r"$\beta$")
    plt.ylabel("Average $T_{coal}$")
    plt.yscale("log")
    plt.legend()
    plt.savefig("/home/lperon/cftp_dis_spin/figures/temp/coalescence_time_vs_beta_exponential_fit_split.png")
    plt.close()

    # unique plot with just the data points and the vertical line at beta_uni
    plt.plot(betas, t_coal_beta, 'o-', label='Average Coalescence Time')
    plt.axvline(beta_uni, color='r', linestyle='--', label=r'$\tanh^{-1}(1/d)$')
    plt.xlabel(r"$\beta$")
    plt.ylabel("Average $T_{coal}$")
    plt.yscale("log")
    plt.legend()
    plt.savefig("/home/lperon/cftp_dis_spin/figures/temp/coalescence_time_vs_beta_data_only.png")
    plt.close()