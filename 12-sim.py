import numpy as np
from scipy.io import loadmat
from time import time
import matplotlib.pyplot as plt
import os

from base_station import BaseStation
from mobile_station import MobileStation
from network import Network

# SIMULATION PARAMETERS #

Number_MS = 100 # Number of MS in the system level simulation
Number_BS = 4 # Number of BS in the system level simulation

freq = 2 # carrier frequency in GHz
BW = 50e6  # channel bandwidth in Hz
BW_per_ms = BW / Number_MS  # bandwidth per mobile station

Pt_BS = 35  # transmission power of base station in dBm
Pt_MS = 23  # transmission power of MS (in dBm)

BS_height = 30 # height of BS in meters
MS_height = 1.5  # height of MS in meters

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

# Create network
network = Network(BS_list, MS_list, freq, BW_per_ms=BW_per_ms, d2d=True)

# Output arrays
capacity_CM = np.zeros(sim_steps)
capacity_CM_DM = np.zeros(sim_steps)
capacity_CM_opt = np.zeros(sim_steps)
capacity_CM_DM_opt = np.zeros(sim_steps)
capacity_CM_5G = np.zeros(sim_steps)
capacity_CM_DM_5G = np.zeros(sim_steps)

# Calculate distances
network.update_distances()
network.update_D2D_distances()

# Cellular parameters
network.update_path_losses()
network.update_SNR_downlink()
network.update_SNR_uplink()
network.update_RSS_downlink()
network.update_RSS_uplink()
network.connect_ms_to_bs()

# D2D parameters
network.update_D2D_path_losses()
network.update_SNR_D2D()
network.update_RSS_D2D()

# Initial capacities
cap = network.capacity_D2D_shannon()
cap_opt = network.capacity_D2D_shannon(optimize_BW=True)
cap_5G = network.capacity_D2D_5G()

capacity_CM[0] = cap[0]
capacity_CM_DM[0] = cap[1]
capacity_CM_opt[0] = cap_opt[0]
capacity_CM_DM_opt[0] = cap_opt[1]
capacity_CM_5G[0] = cap_5G[0]
capacity_CM_DM_5G[0] = cap_5G[1]


for i in range(sim_steps):
    # Load new MS positions
    MS_list = []            
    for j in range(int(ms_positions.shape[1]/2)):
        x = ms_positions[i][2*j]
        y = ms_positions[i][2*j+1]
        # Update MS position
        network.mobile_stations[j].move_to_pos(x,y)
    
    # Update distances
    network.update_distances()
    network.update_D2D_distances()

    # Cellular parameters
    network.update_path_losses()
    network.update_SNR_downlink()
    network.update_SNR_uplink()
    network.update_RSS_downlink()
    network.update_RSS_uplink()
    network.connect_ms_to_bs()

    # D2D parameters
    network.update_D2D_path_losses()
    network.update_SNR_D2D()
    network.update_RSS_D2D()

    cap = network.capacity_D2D_shannon(optimize_BW=False)
    capacity_CM[i] = cap[0]
    capacity_CM_DM[i] = cap[1]

    cap_opt = network.capacity_D2D_shannon(optimize_BW=True)
    capacity_CM_opt[i] = cap_opt[0]
    capacity_CM_DM_opt[i] = cap_opt[1]

    cap_5G = network.capacity_D2D_5G()
    capacity_CM_5G[i] = cap_5G[0]
    capacity_CM_DM_5G[i] = cap_5G[1]



# DISPLAY RESULTS #

out_folder = "12-out/"
os.makedirs(out_folder, exist_ok=True) # create folder if it does not exist

with open(out_folder + "12-pl-rss-snr.txt", "w") as file:
    network.print_to_file(file, RSS_uplink=True, SNR_uplink=True, D2D=True)

with open(out_folder + "12-debug.txt", "w") as file:
    network.print_to_file(file, D2D=True)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(range(sim_steps), capacity_CM_DM * 1e-6, label=f"Mode selection, Shannon", linewidth=0.8)
ax.plot(range(sim_steps), capacity_CM * 1e-6, label=f"Cellular mode, Shannon", linewidth=0.8)
ax.plot(range(sim_steps), capacity_CM_DM_5G * 1e-6, label=f"Mode selection, 5G", linewidth=0.8)
ax.plot(range(sim_steps), capacity_CM_5G * 1e-6, label=f"Cellular mode, 5G", linewidth=0.8)

ax.set_xlabel("Simulation time [step]")
ax.set_ylabel("Channel capacity [Mbps]")
ax.legend()
ax.grid(visible=True, alpha=0.5)
fig.savefig(out_folder + "12-sim.png", dpi=300)

fig2, ax2 = plt.subplots(figsize=(8,6))
ax2.plot(range(sim_steps), capacity_CM_DM_opt * 1e-6, label=f"Mode selection with optimized BW, Shannon", linewidth=0.8)
ax2.plot(range(sim_steps), capacity_CM_DM * 1e-6, label=f"Mode selection, Shannon", linewidth=0.8)
ax2.plot(range(sim_steps), capacity_CM_opt * 1e-6, label=f"Cellular mode with optimized BW, Shannon", linewidth=0.8)
ax2.plot(range(sim_steps), capacity_CM * 1e-6, label=f"Cellular mode, Shannon", linewidth=0.8)

ax2.set_xlabel("Simulation time [step]")
ax2.set_ylabel("Channel capacity [Mbps]")
ax2.legend()
ax2.grid(visible=True, alpha=0.5)
fig2.savefig(out_folder + "12-sim-bonus.png", dpi=300)
