import numpy as np

def node_distance(node1, node2):
    dx = node1.x - node2.x
    dy = node1.y - node2.y
    dz = node1.z - node2.z
    return (dx**2 + dy**2 + dz**2) ** 0.5

def path_loss(distance, frequency_GHz):       
    return (35.2 + 35 * np.log10(distance) + 26 * np.log10(frequency_GHz/2))

def watt_to_dBm(power_W):
    return 10 * np.log10(power_W * 1000)

def dBm_to_watt(power_dBm):
    return 10 ** (power_dBm / 10) / 1000