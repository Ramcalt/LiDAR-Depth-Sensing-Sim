import numpy as np

class Material:
    roughness: float
    specular: float
    albedo: float
    colour: np.ndarray

    def __init__(self, roughness, specular, albedo, colour):
        self.roughness = roughness
        self.specular = specular
        self.albedo = albedo
        self.colour = colour

    def __post_init__(self):
        if not (0.0 <= self.roughness <= 1.0):
            raise ValueError("Roughness must be in [0, 1].")
        if not (0.0 <= self.specular <= 1.0):
            raise ValueError("Specular must be in [0, 1].")
        if not (0.0 <= self.albedo <= 1.0):
            raise ValueError("Albedo must be in [0, 1].")
        r, g, b = self.colour
        for c in (r, g, b):
            if not (0.0 <= c <= 1.0):
                raise ValueError("Colour components must be in [0, 1].")