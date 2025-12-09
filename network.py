from sklearn.cluster import KMeans
import numpy as np

from mobile_station import MobileStation
from base_station import BaseStation
from helper import node_distance, path_loss, watt_to_dBm, dBm_to_watt, efficiency_5G, db_to_linear

class Network:
    def __init__(self, base_stations = None, mobile_stations = None, carrier_frequency_GHz = 1, BW_Hz = 1e6):
        self.base_stations = base_stations if base_stations is not None else []
        self.mobile_stations = mobile_stations if mobile_stations is not None else []
        self.carrier_frequency_GHz = carrier_frequency_GHz
        self.BW_Hz = BW_Hz

        self.distances = {}
        self.path_losses = {}
        self.RSS_downlink = {}
        self.SNR_downlink = {}      
        self.connections = {} # key = ms, value = bs
        self.SINR_downlink = {} # key = ms, value = SINR from connected bs

        self.BW = {} # key = ms, value = BW
        self.REs_per_ms = {} # key = ms, value = REs/s

        self.noise = BW_Hz / 4e21  # thermal noise in Watts
        self.noise_dBm = watt_to_dBm(self.noise) # thermal noise in dBm

        # Handover related data        
        self.HO_counters = {}  # key = ms, value = counter for TTT
        self.HO_targets = {}  # key = ms, value = target bs for handover
        for ms in self.mobile_stations:
            self.HO_targets[ms] = None

        # Capacity related data
        self.signaling_overhead = 0.25  # amount of signaling overhead (25%)
        self.total_REs = 7*12*100*20*275*(1-self.signaling_overhead)  # total amount of REs/s available for data transmission (i.e., exluding overhead)


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
    
    def assign_BW_random(self):
        for bs in self.base_stations:
            coefficients = []
            connected_ms = []

            for ms in self.mobile_stations:
                if self.connections[ms] == bs:
                    coefficients.append(np.random.rand())
                    connected_ms.append(ms)

            coeff_sum = np.sum(coefficients)
            BWs = (self.BW_Hz / coeff_sum) * np.array(coefficients)

            for i, ms in enumerate(connected_ms):
                self.BW[ms] = BWs[i]   

    def assign_BW_fair(self):
        for bs in self.base_stations:
            connected_ms = []
            SINR_list = []

            for ms in self.mobile_stations:
                if self.connections[ms] == bs:
                    SINR_list.append(self.SINR_downlink[ms] + 10)  # shift to avoid zero SINR
                    connected_ms.append(ms)

            SINR_sum = np.sum(SINR_list)
            BWs = (self.BW_Hz / SINR_sum) * np.array(SINR_list) 

            for i, ms in enumerate(connected_ms):
                self.BW[ms] = BWs[i]
                # self.BW[ms] = self.BW_Hz / len(connected_ms)  # equal BW for all connected MSs

    def assign_REs_random(self):
        for bs in self.base_stations:
            coefficients = []
            connected_ms = []

            for ms in self.mobile_stations:
                if self.connections[ms] == bs:
                    coefficients.append(np.random.rand())
                    connected_ms.append(ms)

            coeff_sum = np.sum(coefficients)
            REs_list = (self.total_REs / coeff_sum) * np.array(coefficients)

            for i, ms in enumerate(connected_ms):
                self.REs_per_ms[ms] = REs_list[i]

    def assign_REs_fair(self):
        for bs in self.base_stations:
            connected_ms = []
            SINR_list = []

            for ms in self.mobile_stations:
                if self.connections[ms] == bs:
                    SINR_list.append(self.SINR_downlink[ms] + 10)  # shift to avoid zero SINR
                    connected_ms.append(ms)            

            SINR_sum = np.sum(SINR_list)
            REs_list = (self.total_REs / SINR_sum) * np.array(SINR_list) 

            for i, ms in enumerate(connected_ms):
                self.REs_per_ms[ms] = REs_list[i]

    def assign_REs_equal(self):
        for bs in self.base_stations:
            connected_ms = []

            for ms in self.mobile_stations:
                if self.connections[ms] == bs:
                    connected_ms.append(ms)            

            REs_per_ms = self.total_REs / len(connected_ms) if len(connected_ms) > 0 else 0

            for ms in connected_ms:
                self.REs_per_ms[ms] = REs_per_ms

    def total_capacity_shannon(self):
        capacity = 0
        for ms in self.mobile_stations:
            BW_ms = self.BW[ms]
            SINR_ms_dB = self.SINR_downlink[ms]            
            SINR_ms_W = db_to_linear(SINR_ms_dB)
            capacity += BW_ms * np.log2(1 + SINR_ms_W)

        return capacity  # in bps
    
    def total_capacity_5G(self):
        capacity = 0
        for ms in self.mobile_stations:
            REs_ms = self.REs_per_ms[ms]
            efficiency = efficiency_5G(self.SINR_downlink[ms])  # in bits/RE
            capacity += REs_ms * efficiency

        return capacity  # in bps
    

    def check_handovers(self, delta_H, TTT, Step):
        num_handovers = 0

        # Check handover for every mobile station
        for ms in self.mobile_stations:              
            # Find which BS has MS strongest signal from
            RSS_max = float("-inf")
            BS_best = None
            for bs in self.base_stations:
                RSS_current = self.RSS_downlink[(ms, bs)]
                if RSS_current > RSS_max:
                    RSS_max = RSS_current
                    BS_best = bs # BS with highest RSS (candidate or current)
            
            bs_connected = self.connections[ms]
            if BS_best is bs_connected:
                # No handover needed, reset any ongoing handover data
                self.HO_counters[ms] = 0
                self.HO_targets[ms] = None
                continue           
            

            # BS_best is not bs_connected            
            RSS_connected = self.RSS_downlink[(ms, bs_connected)]
            if (RSS_max - RSS_connected) >= delta_H:
                if BS_best is self.HO_targets[ms]: # BS_best wins once again
                    self.HO_counters[ms] += Step
                else: # we've got new candidate
                    self.HO_targets[ms] = BS_best
                    self.HO_counters[ms] = 0
            else: # there's better BS, but not good enough -> connected BS takes over
                # No handover needed, reset any ongoing handover data
                self.HO_counters[ms] = 0
                self.HO_targets[ms] = None
                continue

            if self.HO_counters[ms] >= TTT:
                # Perform handover
                self.perform_handover(ms, self.HO_targets[ms])
                num_handovers += 1
                # Reset ongoing handover data
                self.HO_counters[ms] = 0
                self.HO_targets[ms] = None
        
        return num_handovers  
    
    def perform_handover(self, ms, target_bs):        
        self.connections[ms] = target_bs
        # Data like distances, PL etc. became invalid for this MS.
        # However it doesn't matter as they will be recalculated in the next simulation iteration.
    
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



