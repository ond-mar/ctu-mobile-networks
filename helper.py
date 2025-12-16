import numpy as np

def node_distance(node1, node2):
    dx = node1.x - node2.x
    dy = node1.y - node2.y
    dz = node1.z - node2.z
    return (dx**2 + dy**2 + dz**2) ** 0.5

def node_distance_xy(node1, node2):
    dx = node1.x - node2.x
    dy = node1.y - node2.y
    return (dx**2 + dy**2) ** 0.5

def path_loss(distance, frequency_GHz):       
    return (35.2 + 35 * np.log10(distance) + 26 * np.log10(frequency_GHz/2))

def watt_to_dBm(power_W):
    return 10 * np.log10(power_W * 1000)

def dBm_to_watt(power_dBm):
    return 10 ** (power_dBm / 10) / 1000

def db_to_linear(db):
    return 10 ** (db / 10)

def efficiency_5G(SINR_dB): 
    if -9.478 <= SINR_dB < -6.658:
        return 2 * 78 / 1024
    elif -6.658 <= SINR_dB < -4.098:
        return 2 * 120 / 1024
    elif -4.098 <= SINR_dB < -1.798:
        return 2 * 193 / 1024
    elif -1.798 <= SINR_dB < 0.399:
        return 2 * 308 / 1024
    elif 0.399 <= SINR_dB < 2.424:
        return 2 * 449 / 1024
    elif 2.424 <= SINR_dB < 4.489:
        return 2 * 602 / 1024
    elif 4.489 <= SINR_dB < 6.367:
        return 4 * 378 / 1024
    elif 6.367 <= SINR_dB < 8.456:
        return 4 * 490 / 1024
    elif 8.456 <= SINR_dB < 10.266:
        return 4 * 616 / 1024
    elif 10.266 <= SINR_dB < 12.218:
        return 6 * 466 / 1024
    elif 12.218 <= SINR_dB < 14.122:
        return 6 * 567 / 1024
    elif 14.122 <= SINR_dB < 15.849:
        return 6 * 666 / 1024
    elif 15.849 <= SINR_dB < 17.786:
        return 6 * 772 / 1024
    elif 17.786 <= SINR_dB < 19.809:
        return 6 * 873 / 1024
    elif 19.809 <= SINR_dB < 21.809:
        return 8 * 711 / 1024
    elif 21.809 <= SINR_dB < 23.809:
        return 8 * 797 / 1024
    elif 23.809 <= SINR_dB < 25.809:
        return 8 * 885 / 1024
    elif SINR_dB >= 25.809:
        return 8 * 948 / 1024
    else:
        return 0