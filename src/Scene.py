import SceneObject
import numpy as np
from typing import List

class Scene:
    """Container for scene object instances and environment properties."""
    ambient_intensity: float
    ambient_colour: np.ndarray
    objects: List[SceneObject]

    def __init__(self, ambient_intensity, ambient_colour):
        self.ambient_intensity = ambient_intensity
        self.ambient_colour = ambient_colour
        objects = []

    def addObj(self, obj: SceneObject):
        self.objects.append(obj)

    def getObj(self, name: str):
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None