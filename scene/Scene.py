from scene.SceneObject import SceneObject
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
        self.objects = []

    def add_obj(self, obj: SceneObject):
        if self.get_obj(obj.name) is not None:
            raise Exception("Object with same name already exists")
        self.objects.append(obj)

    def get_obj(self, name: str):
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None