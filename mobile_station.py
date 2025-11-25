from network_node import NetworkNode

class MobileStation(NetworkNode):
    def __init__(self, x=0, y=0, z = 0, power=10, name=""):
        super().__init__(x, y, power, name)
        self.z = z
        self.name = name