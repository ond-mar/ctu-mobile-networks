import matplotlib.pyplot as plt
import numpy as np
import os

from mobile_station import MobileStation
from base_station import BaseStation
from network import Network

# SIMULATION PARAMETERS #

x_max = 1000 # Definition of Area in meters (x-axes)
x_min = 0
y_max = 1000 # Definition of Area in meters (y-axes)
y_min = 0
Number_MS = 100 # Number of MS in the system level simulation
Number_BS = 4 # Number of BS in the system level simulation
freq = 2 # carrier frequency in GHz
BW = 50e6  # channel bandwidth
Pt_BS=35  # transmission power of base station in dBm
Pt_MS=23  # transmission power of MS (in dBm)

BS_height = 30 # height of BS in meters
MS_height = 1.5  # height of MS in meters

# Define base stations
BS_list = [
    BaseStation(250, 250, BS_height, Pt_BS, "SBS1"),
    BaseStation(750, 250, BS_height, Pt_BS, "SBS2"),
    BaseStation(250, 750, BS_height, Pt_BS, "SBS3"),
    BaseStation(750, 750, BS_height, Pt_BS, "SBS4")
]

# Generate mobile stations at random locations
MS_list = []
for i in range(Number_MS):
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    ms = MobileStation(x, y, MS_height, Pt_MS, f"MS{i+1}")
    MS_list.append(ms)

# SIMULATION #

network = Network(BS_list, MS_list, freq, BW)
network.update_distances()
network.update_path_losses()
network.update_SNR_downlink()
network.update_RSS_downlink()
network.connect_ms_to_bs()
network.update_SINR_downlink()

# SAVE RESULTS #

folder = "09-out/"
os.makedirs(folder, exist_ok=True) # create folder if it does not exist

with open(os.path.join(folder, "09-sim-results.txt"), "w") as f:
    network.print_to_file(f)

# Plot the network layout
fig, ax = plt.subplots() # Create figure and axis objects
cmap = plt.get_cmap('plasma') # Define colormap

for i, bs in enumerate(network.base_stations):
    # Plot base stations
    color_idx = i / len(network.base_stations)
    ax.plot(bs.x, bs.y, 'o', color="k", markersize=5, markeredgewidth=3)
    # Plot connections
    for (ms, connected_bs) in network.connections.items():
        if connected_bs == bs:
            ax.plot([ms.x, bs.x], [ms.y, bs.y], '-', color=cmap(color_idx), alpha=0.6)

# Plot mobile stations
for ms in network.mobile_stations:
    ax.plot(ms.x, ms.y, 'bx')

# Set axis limits and save figure
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
fig.savefig(os.path.join(folder, "09-sim-figure.png"), dpi=300)

# BONUS #
network_flyingBS = Network([], MS_list, freq, BW) # same network but without BSs
network_flyingBS.find_flying_bs_for_ms(fbs_number=Number_BS, fbs_height=30)
# Update connections
network_flyingBS.update_distances()
network_flyingBS.update_path_losses()
network_flyingBS.update_SNR_downlink()
network_flyingBS.update_RSS_downlink()
network_flyingBS.connect_ms_to_bs()

# SAVE BONUS RESULTS #

folder = "09-out/"
os.makedirs(folder, exist_ok=True) # create folder if it does not exist

with open(os.path.join(folder, "09-sim-FBS-results.txt"), "w") as f:
    network_flyingBS.print_to_file(f, PL=False, RSS=False, SNR=False, CONN=True, SINR=False)

# Plot the bonus network layout
fig2, ax2 = plt.subplots() # Create figure and axis objects
cmap = plt.get_cmap('plasma')

for i, bs in enumerate(network_flyingBS.base_stations):
    color_idx = i / len(network_flyingBS.base_stations)
    ax2.plot(bs.x, bs.y, 'o', color="k", markersize=5, markeredgewidth=3)
    # Plot connections
    for (ms, connected_bs) in network_flyingBS.connections.items():
        if connected_bs == bs:
            plt.plot([ms.x, bs.x], [ms.y, bs.y], '-', color=cmap(color_idx), alpha=0.6)


for ms in network_flyingBS.mobile_stations:
    ax2.plot(ms.x, ms.y, 'bx')

ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
fig2.savefig(os.path.join(folder, "09-sim-FBS-figure.png"), dpi=300)

