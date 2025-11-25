from sklearn.cluster import KMeans
import numpy as np

from mobile_station import MobileStation
from base_station import BaseStation
from helper import node_distance, path_loss, watt_to_dBm, dBm_to_watt

class Network:
    def __init__(self, base_stations = None, mobile_stations = None, carrier_frequency_GHz = 1, BW_Hz = 1e6):
        self.base_stations = base_stations if base_stations is not None else []
        self.mobile_stations = mobile_stations if mobile_stations is not None else []
        self.carrier_frequency_GHz = carrier_frequency_GHz

        self.distances = {}
        self.path_losses = {}
        self.RSS_downlink = {}
        self.SNR_downlink = {}      
        self.connections = {} # key = ms, value = bs
        self.SINR_downlink = {} # key = ms, value = SINR from connected bs

        self.noise = BW_Hz / 4e21  # thermal noise in Watts
        self.noise_dBm = watt_to_dBm(self.noise) # thermal noise in dBm


    def update_distances(self):        
        for ms in self.mobile_stations:
            for bs in self.base_stations:
                self.distances[(ms, bs)] = node_distance(ms, bs)
        return self.distances
    
    def update_path_losses(self):
        for ms in self.mobile_stations:
            for bs in self.base_stations:
                self.path_losses[(ms, bs)] = path_loss(self.distances[(ms, bs)], self.carrier_frequency_GHz)

        return self.path_losses

    def update_SNR_downlink(self):
        for ms in self.mobile_stations:
            for bs in self.base_stations:                
                self.SNR_downlink[(ms, bs)] = bs.power - self.path_losses[(ms, bs)] - self.noise_dBm

        return self.SNR_downlink
    
    def update_RSS_downlink(self):        
        for ms in self.mobile_stations:
            for bs in self.base_stations:                
                self.RSS_downlink[(ms, bs)] = bs.power - self.path_losses[(ms, bs)]
                
        return self.RSS_downlink
    
    def connect_ms_to_bs(self):
        for ms in self.mobile_stations:
            max_RSS = float("-inf")   
                     
            for bs in self.base_stations:
                if self.RSS_downlink[(ms, bs)] > max_RSS:                    
                    max_RSS = self.RSS_downlink[(ms, bs)]
                    selected_BS = bs

            self.connections.update({ms: selected_BS})
        return
    
    def update_SINR_downlink(self):
        for ms in self.mobile_stations:
            bs_connected = self.connections[ms]
            
            sum_RSSi_W = 0
            for bs in self.base_stations:
                if bs is not bs_connected:                    
                    sum_RSSi_W += dBm_to_watt(self.RSS_downlink[(ms, bs)])

            NI_W = sum_RSSi_W + self.noise

            SINR = bs_connected.power - self.path_losses[(ms, bs_connected)] - watt_to_dBm(NI_W)
            self.SINR_downlink[ms] = SINR
            
        return self.SINR_downlink
    
    def find_flying_bs_for_ms(self, fbs_number, fbs_height):
        # Prepare data for clustering
        ms_positions = np.array([[ms.x, ms.y] for ms in self.mobile_stations])
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=fbs_number, random_state=0, n_init="auto")
        kmeans.fit(ms_positions)
        C = kmeans.cluster_centers_
        # Create flying base stations at cluster centers
        flying_BS_list = []
        for i, center in enumerate(C):
            fbs = BaseStation(center[0], center[1], fbs_height, 35, f"FBS{i+1}")
            flying_BS_list.append(fbs)

        self.base_stations = flying_BS_list  # Update network's base stations to flying BSs
        return flying_BS_list


    
    def print_to_file(self, file, PL = True, RSS = True, SNR = True, CONN = True, SINR = True):        
        if PL:
            file.write("PATH LOSSES\n")
            file.write("MS name \tPL BS1  \tPL BS2  \tPL BS3  \tPL BS4\n")
            for ms in self.mobile_stations:
                file.write(f"{ms.name} \t \t")
                for bs in self.base_stations:
                    file.write(f"{self.path_losses[(ms, bs)]:.2f}\t \t")
                file.write("\n")
            file.write("--------------------------------\n\n")
        if RSS:
            file.write("RSS DOWNLINK\n")
            file.write("MS name \tRSS BS1  \tRSS BS2  \tRSS BS3  \tRSS BS4\n")
            for ms in self.mobile_stations:
                file.write(f"{ms.name} \t \t")
                for bs in self.base_stations:
                    file.write(f"{self.RSS_downlink[(ms, bs)]:.2f}\t \t")
                file.write("\n")
            file.write("--------------------------------\n\n")
        if SNR:
            file.write("SNR DOWNLINK\n")
            file.write("MS name \tSNR BS1  \tSNR BS2  \tSNR BS3  \tSNR BS4\n")
            for ms in self.mobile_stations:
                file.write(f"{ms.name} \t \t")
                for bs in self.base_stations:
                    file.write(f"{self.SNR_downlink[(ms, bs)]:.2f}\t \t")
                file.write("\n")
            file.write("--------------------------------\n\n")
        if CONN:
            file.write("CONNECTIONS\n")
            file.write("MS name \tConnected BS \n")
            for ms in self.mobile_stations:
                connected_bs = self.connections[ms]
                file.write(f"{ms.name} \t \t {connected_bs.name}\n")
            file.write("--------------------------------\n\n")            
        if SINR:
            file.write("SINR DOWNLINK\n")
            file.write("MS name \tSINR from connected BS \n")
            for ms in self.mobile_stations:
                file.write(f"{ms.name} \t \t {self.SINR_downlink[ms]:.2f}\n")
            file.write("--------------------------------\n\n")
        return



