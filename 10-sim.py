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
NoOfSteps=1000 # Number of simulation steps
Step=1 # duration of one step (assume 1 ms)
No_TTT_values=40  # Max value of TTT [ms]
Delta_H_values=[0, 1, 2, 3, 4] # DeltaH [dB]
# Delta_H_values=[0] # DeltaH [dB]

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

# SIMULATION #
t_start = time()

handovers = {} # dictionary to store handover results, key = DeltaH, value = array of handovers for each TTT

for delta_H in Delta_H_values:
    handovers_in_ttt = np.zeros(No_TTT_values)

    for TTT in range(1, No_TTT_values+1):
        
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
        
        # Iterate through different positions
        num_handovers = 0
        for i in range(1, sim_steps):
            # Load MS positions
            MS_list = []            
            for j in range(int(ms_positions.shape[1]/2)):
                x = ms_positions[i][2*j]
                y = ms_positions[i][2*j+1]
                # Update MS position
                network.mobile_stations[j].move_to_pos(x,y)
            # Are any handovers needed?
            network.update_distances()
            network.update_path_losses()
            network.update_RSS_downlink()
            num_handovers += network.check_handovers(delta_H, TTT, Step)            
        
        handovers_in_ttt[TTT-1] = num_handovers

    handovers[delta_H] = handovers_in_ttt

    t_end = time()
    print(f"Elapsed time: {t_end - t_start}")

# DISPLAY RESULTS #
out_folder = "10-out/"
os.makedirs(out_folder, exist_ok=True) # create folder if it does not exist
fig, ax = plt.subplots(figsize=(8,6))
cmap = plt.get_cmap("plasma")

for delta_H in Delta_H_values:
    handovers_in_ttt = handovers[delta_H]
    ax.plot(range(1, No_TTT_values+1), handovers_in_ttt, label=f"DeltaH={delta_H} dB")


ax.set_xlabel("Time to trigger [step]")
ax.set_ylabel("Number of handovers [-]")
ax.legend()
ax.grid(visible=True, alpha=0.5)
fig.savefig(out_folder + "10-sim-handovers-vs-TTT.png", dpi=300)