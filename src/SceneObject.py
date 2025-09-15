import Material
import HTransform

class SceneObject:
    """Renderable component with material, geometry, and transform"""
    name: str
    mesh_file: str
    material: Material
    transform: HTransform

    def __init__(self, name, mesh_file, material, transform):
        self.name = name
        self.mesh_file = mesh_file
        self.material = material
        self.transform = transform

