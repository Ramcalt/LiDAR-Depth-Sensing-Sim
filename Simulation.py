from scene.Scene import Scene
from scene.SceneObject import SceneObject
from scene.Material import Material
from scene.HTransform import HTransform
from lidar.Emitter import Emitter
from lidar.Detector import Detector
import open3d as o3d


class Simulation:
    """Singleton for managing simulation."""
    scene: Scene
    emitter: Emitter
    detector: Detector

    def __init__(self):
        # create scene and add objects
        self.scene = Scene(1.0, [1.0, 1.0, 1.0])
        self.Emitter = Emitter("emitter",
                               "res/sensor.stl",
                               Material(1.0, 1.0, 1.0, [0.8, 0.2, 0.2]),
                               HTransform().translation(-0.25, 0, 1),
                               940e-9,
                               (2 * 0.74e-3 * 0.90e-3),
                               1e-9,
                               0.79,
                               0.79
                               )
        self.Detector = Detector("detector",
                                 "res/sensor.stl",
                                 Material(1.0, 1.0, 1.0, [0.2, 0.2, 0.8]),
                                 HTransform().translation(0.25, 0, 1),
                                 8,
                                 8,
                                 0.79,
                                 0.79,
                                 100,
                                 125e-12
                                 )
        self.scene.add_obj(
            SceneObject("cavity",
                        "res/cavity.stl",
                        Material(1.0, 1.0, 1.0, [0.9, 0.2, 0.1]),
                        HTransform()
                        )
        )
        self.scene.add_obj(self.Detector)
        self.scene.add_obj(self.Emitter)

    def run(self):
        self.view_scene()

    def view_scene(self):
        meshes = [obj.to_o3d_geometry() for obj in self.scene.objects]
        o3d.visualization.draw(meshes, raw_mode=True)


sim = Simulation()
sim.run()
