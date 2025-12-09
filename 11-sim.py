import numpy as np
from scipy.io import loadmat
from time import time
import matplotlib.pyplot as plt
import os

from base_station import BaseStation
from mobile_station import MobileStation
from network import Network

# SIMULATION PARAMETERS #

# Basic network parameters
x_max = 1000 # Definition of Area in meters (x-axes)
x_min = 0
y_max = 1000 # Definition of Area in meters (y-axes)
y_min = 0

Number_MS = 100 # Number of MS in the system level simulation
Number_BS = 4 # Number of BS in the system level simulation

freq = 2 # carrier frequency in GHz
BW = 50e6  # channel bandwidth

Pt_BS = 35  # transmission power of base station in dBm
Pt_MS = 23  # transmission power of MS (in dBm)

BS_height = 30 # height of BS in meters
MS_height = 1.5  # height of MS in meters

# Handover parameters
ttt = 10  # Time to Trigger in ms
delta_H = 2  # Handover Margin in dB

# Mobile station postitions
matlab_positions = loadmat("data/MSposition_NoMS_100_NoSimStep_1000.mat")
ms_positions = matlab_positions["PosMS"] # create nd array where row = sim step, column = x or y coordinate of MS
sim_steps = ms_positions.shape[0]

# Define base stations
BS_list = [
    BaseStation(250, 250, BS_height, Pt_BS, "SBS1"),
    BaseStation(750, 250, BS_height, Pt_BS, "SBS2"),
    BaseStation(250, 750, BS_height, Pt_BS, "SBS3"),
    BaseStation(750, 750, BS_height, Pt_BS, "SBS4")
]


# Load initial MS positions
MS_list = []
for j in range(int(ms_positions.shape[1]/2)):
    x = ms_positions[0][2*j]
    y = ms_positions[0][2*j+1]
    ms = MobileStation(x, y, MS_height, Pt_MS, f"MS{j+1}")
    MS_list.append(ms)

# Create network to simulate with current deltaH and TTT
network = Network(BS_list, MS_list, freq, BW)
# Set initial connections
network.update_distances()
network.update_path_losses()
network.update_RSS_downlink()
network.connect_ms_to_bs()

network.update_SINR_downlink() # update SINR values after possible handovers


capacities_rand_shannon = np.zeros(sim_steps)
network.assign_BW_random()
capacities_rand_shannon[0] = network.total_capacity_shannon()

capacities_fair_shannon = np.zeros(sim_steps)
network.assign_BW_fair()
capacities_fair_shannon[0] = network.total_capacity_shannon()

capacities_rand_5G = np.zeros(sim_steps)
network.assign_REs_random()
capacities_rand_5G[0] = network.total_capacity_5G()

capacities_fair_5G = np.zeros(sim_steps)
network.assign_REs_fair()
capacities_fair_5G[0] = network.total_capacity_5G()

capacities_equal_5G = np.zeros(sim_steps)
network.assign_REs_equal()
capacities_equal_5G[0] = network.total_capacity_5G()

# Iterate through different positions
num_handovers = 0
for i in range(sim_steps):
    # Load new MS positions
    MS_list = []            
    for j in range(int(ms_positions.shape[1]/2)):
        x = ms_positions[i][2*j]
        y = ms_positions[i][2*j+1]
        # Update MS position
        network.mobile_stations[j].move_to_pos(x,y)
    
    # Update distance related data (calc. for every ms-bs pair)
    network.update_distances()
    network.update_path_losses()
    network.update_RSS_downlink()
    
    # Update connections if handover conditions are met    
    network.check_handovers(delta_H, ttt, 1)   

    # Update distance related data (calc. for every ms-bs pair)
    network.update_distances()
    network.update_path_losses()
    network.update_RSS_downlink()

    # Calculate capacity with random BW assignment and Shannon formula
    network.update_SINR_downlink() # update SINR values after possible handovers

    network.assign_BW_random()
    capacities_rand_shannon[i] = network.total_capacity_shannon()
    network.assign_BW_fair()
    capacities_fair_shannon[i] = network.total_capacity_shannon()
    network.assign_REs_random()
    capacities_rand_5G[i] = network.total_capacity_5G()
    network.assign_REs_fair()
    capacities_fair_5G[i] = network.total_capacity_5G()
    network.assign_REs_equal()
    capacities_equal_5G[i] = network.total_capacity_5G()


# DISPLAY RESULTS #
out_folder = "11-out/"
os.makedirs(out_folder, exist_ok=True) # create folder if it does not exist

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(range(sim_steps), capacities_rand_shannon * 1e-6, label=f"Random BW, Shannon", linewidth=0.8)
ax.plot(range(sim_steps), capacities_fair_shannon * 1e-6, label=f"Fair BW, Shannon", linewidth=0.8)
ax.plot(range(sim_steps), capacities_rand_5G * 1e-6, label=f"Random REs, 5G", linewidth=0.8)
ax.plot(range(sim_steps), capacities_fair_5G * 1e-6, label=f"Fair REs, 5G", linewidth=0.8)

ax.set_xlabel("Simulation time [step]")
ax.set_ylabel("Channel capacity [Mbps]")
ax.legend()
ax.grid(visible=True, alpha=0.5)
fig.savefig(out_folder + "11-sim-channel.png", dpi=300)

fig2, ax2 = plt.subplots(figsize=(8,6))
ax2.plot(range(sim_steps), capacities_equal_5G * 1e-6, label=f"Equal REs, 5G", linewidth=0.8)
ax2.plot(range(sim_steps), capacities_fair_5G * 1e-6, label=f"Fair REs, 5G", linewidth=0.8)
ax2.plot(range(sim_steps), capacities_rand_5G * 1e-6, label=f"Random REs, 5G", linewidth=0.8)

ax2.set_xlabel("Simulation time [step]")
ax2.set_ylabel("Channel capacity [Mbps]")
ax2.legend()
ax2.grid(visible=True, alpha=0.5)
fig2.savefig(out_folder + "11-sim-channel-bonus.png", dpi=300)