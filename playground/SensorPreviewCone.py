import numpy as np

class Cone:
    def __init__(self, x: float, y: float, z: float, alpha: float ):
        self.x = x
        self.y = y
        self.z = z
        self.alpha = alpha

    def preview(self, distance):


camera = Cone(0, 0, 0, np.deg2rad(80))
lidar = Cone(30, 20, -25, np.deg2rad(45))
