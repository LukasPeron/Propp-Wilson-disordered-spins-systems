"""
This script is made to create a video of the simulation of a perfect simulation of a chosen model at a chosen beta using matplotlib animation.
"""

from utils.cftp_func import *
from matplotlib.animation import FuncAnimation

N = 100
d = 4
G, couplings = create_RR_graph(N, d)

beta_critical = np.arctanh(1/np.sqrt(d-1)) # ~ 0.6584789484624085

beta_simu = 0.3

config_eq, _, _ = CFTP_MCMC2_BC(beta_simu, G, couplings)

number_of_frames = 100

# plot first just the graph with the initial configuration

pos = nx.spring_layout(G)

fig_init, ax_init = plt.subplots()
node_color_init = ['red' if config_eq[i] == 1 else 'blue' for i in range(N)]
nx.draw(G, pos, node_color=node_color_init, node_size=100, ax=ax_init, edge_color='gray')
fig_init.tight_layout()
fig_init.savefig(save_path+f"figures/temp/simulation_init.png", dpi=100)
print(f"Initial configuration saved as simulation_init.png")

fig, ax = plt.subplots()

current_config = config_eq.copy()
node_list = list(G.nodes())

def update(frame):
    global current_config
    ax.clear()
    
    # Run one sweep of MCMC (N proposals) per frame
    current_config, mag = MCMC2_fwd(beta_simu, G, couplings, current_config, max_nb_iter_MCMC2=N)
    
    # Color mapping: +1 is red, -1 is blue
    node_color = ['red' if current_config[i] == 1 else 'blue' for i in range(N)]
    
    nx.draw(G, pos, node_color=node_color, node_size=100, ax=ax, edge_color='gray')
    # ax.set_title(fr"RR graph $N={N}$, $d={d}$, $\beta$: {beta_simu}")
    fig.tight_layout()

ani = FuncAnimation(fig, update, frames=number_of_frames, interval=50)
ani.save(save_path+f"figures/temp/simulation.gif", writer="pillow", dpi=100)
print(f"Video saved as simulation.gif")

# now we can do the same animation but starting from a random configuration instead of the equilibrium one

random_config = np.random.choice([-1, 1], size=N)

# plot first just the graph with the initial random configuration

fig_random_init, ax_random_init = plt.subplots()
node_color_random_init = ['red' if random_config[i] == 1 else 'blue' for i in range(N)]
nx.draw(G, pos, node_color=node_color_random_init, node_size=100, ax=ax_random_init, edge_color='gray')
fig_random_init.tight_layout()
fig_random_init.savefig(save_path+f"figures/temp/simulation_random_init.png", dpi=100)
print(f"Initial random configuration saved as simulation_random_init.png")


fig_random, ax_random = plt.subplots()

current_config_random = random_config.copy()

def update_random(frame):
    global current_config_random
    ax_random.clear()
    
    # Run one sweep of MCMC (N proposals) per frame
    current_config_random, mag = MCMC2_fwd(beta_simu, G, couplings, current_config_random, max_nb_iter_MCMC2=N)
    
    # Color mapping: +1 is red, -1 is blue
    node_color = ['red' if current_config_random[i] == 1 else 'blue' for i in range(N)]
    
    nx.draw(G, pos, node_color=node_color, node_size=100, ax=ax_random, edge_color='gray')
    # ax.set_title(fr"RR graph $N={N}$, $d={d}$, $\beta$: {beta_simu}")
    fig_random.tight_layout()

ani_random = FuncAnimation(fig_random, update_random, frames=number_of_frames, interval=50)
ani_random.save(save_path+f"figures/temp/simulation_random.gif", writer="pillow", dpi=100)
print(f"Video saved as simulation_random.gif")
