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
network.update_SINR_downlink()

network.assign_BW_random()
cap_rand_shannon_BS = network.total_capacity_shannon()

network.assign_BW_fair()
cap_fair_shannon_BS = network.total_capacity_shannon()

network.assign_REs_random()
cap_rand_5G_BS = network.total_capacity_5G()

network.assign_REs_fair()
cap_fair_5G_BS = network.total_capacity_5G()

# Switch BS for Flying BS
network.find_flying_bs_for_ms(4, 35)

# Set initial connections
network.update_distances()
network.update_path_losses()
network.update_RSS_downlink()
network.connect_ms_to_bs()
network.update_SINR_downlink()

network.assign_BW_random()
cap_rand_shannon_FBS = network.total_capacity_shannon()

network.assign_BW_fair()
cap_fair_shannon_FBS = network.total_capacity_shannon()

network.assign_REs_random()
cap_rand_5G_FBS = network.total_capacity_5G()

network.assign_REs_fair()
cap_fair_5G_FBS = network.total_capacity_5G()


# SAVE RESULTS #
out_folder = "11-out/"
os.makedirs(out_folder, exist_ok=True) # create folder if it does not exist
out_file = out_folder + "11-sim-bonus-results.txt"
with open(out_file, "w") as f:
    f.write("Capacities with Base Stations only:\n")
    f.write(f"Random BW allocation - Shannon: {cap_rand_shannon_BS/1e6:.2f} Mbps\n")
    f.write(f"Fair BW allocation - Shannon: {cap_fair_shannon_BS/1e6:.2f} Mbps\n")
    f.write(f"Random RE allocation - 5G: {cap_rand_5G_BS/1e6:.2f} Mbps\n")
    f.write(f"Fair RE allocation - 5G: {cap_fair_5G_BS/1e6:.2f} Mbps\n\n")

    f.write("Capacities with Flying Base Station:\n")
    f.write(f"Random BW allocation - Shannon: {cap_rand_shannon_FBS/1e6:.2f} Mbps\n")
    f.write(f"Fair BW allocation - Shannon: {cap_fair_shannon_FBS/1e6:.2f} Mbps\n")
    f.write(f"Random RE allocation - 5G: {cap_rand_5G_FBS/1e6:.2f} Mbps\n")
    f.write(f"Fair RE allocation - 5G: {cap_fair_5G_FBS/1e6:.2f} Mbps\n")
