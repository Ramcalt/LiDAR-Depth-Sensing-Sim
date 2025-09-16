from scene.Scene import Scene
from scene.SceneObject import SceneObject
from scene.Material import Material
from scene.HTransform import HTransform
import open3d as o3d


class Simulation:
    """Singleton for running simulation."""
    scene: Scene

    def __init__(self):
        self.scene = Scene(0.1, [0.3, 0.3, 0.6])
        self.scene.add_obj(SceneObject("cavity", "res/cavity.stl",
                                       Material(0.2, 0.8, 1, [0.9, 0.2, 0.1]),
                                       HTransform()))
        self.scene.add_obj(SceneObject("cavity", "res/cavity.stl",
                                       Material(0.2, 0.8, 1, [0.9, 0.2, 0.1]),
                                       HTransform().rotation_x(0.01) @ HTransform.scaling(1, 0.25, 0.5) @ HTransform.translation(1, 1, 1)))

    def run(self):
        self.viewScene()

    def viewScene(self):
        meshes = [obj.to_o3d_geometry() for obj in self.scene.objects]
        o3d.visualization.draw(meshes, raw_mode=True)

sim = Simulation()
sim.run()