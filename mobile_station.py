from network_node import NetworkNode

class MobileStation(NetworkNode):
    def __init__(self, x=0, y=0, z = 0, power=10, name=""):
        super().__init__(x, y, power, name)
        self.z = z
        self.name = name

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

    def move_to_pos(self, x, y, z = None):
        self.x = x
        self.y = y
        if z is not None:
            self.z = z