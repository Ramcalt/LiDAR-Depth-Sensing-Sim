from scene.Scene import Scene
from scene.SceneObject import SceneObject
from scene.Material import Material
from scene.HTransform import HTransform
from lidar.Emitter import Emitter
from lidar.Detector import Detector
from Plotter import Plotter
import open3d as o3d

class Simulation:
    """Singleton for managing simulation."""
    scene: Scene
    emitter: Emitter
    detector: Detector

    def __init__(self):
        """Initialise scene and add emitter, detector, and scene objects"""
        self.scene = Scene(1.0, [1.0, 1.0, 1.0])
        self.emitter = Emitter("emitter",
                               "res/sensor.stl",
                               Material(1.0, 1.0, 1.0, [0.8, 0.2, 0.2]),
                               HTransform().translation(-0.25, 0, 1),
                               940e-9,
                               (2 * 0.74e-3 * 0.90e-3),
                               1e-9,
                               0.79,
                               0.79
                               )
        self.detector = Detector("detector",
                                 "res/sensor.stl",
                                 Material(1.0, 1.0, 1.0, [0.2, 0.2, 0.8]),
                                 HTransform().translation(0.25, 0, 1),
                                 8,
                                 8,
                                 0.79,
                                 0.79,
                                 40,
                                 125e-12
                                 )
        self.scene.add_obj(
            SceneObject("cavity",
                        "res/cavity.stl",
                        Material(1.0, 1.0, 1.0, [0.9, 0.2, 0.1]),
                        HTransform()
                        )
        )
        self.scene.add_obj(self.detector)
        self.scene.add_obj(self.emitter)

    def run(self):
        """Run the simulation"""
        self.detector.fill_hist_with_noise()
        self.view_plots()
        self.view_scene()

    def view_scene(self):
        """View the scene using Open3D"""
        meshes = [obj.to_o3d_mesh() for obj in self.scene.objects]
        o3d.visualization.draw(meshes, raw_mode=True)

    def view_plots(self):
        """Runs matplotlib plots in separate processes"""
        Plotter.new_process(Plotter.plot_hist, self.detector.histograms[0][0])
        Plotter.new_process(Plotter.plot_hist_arr, self.detector.histograms, self.detector.zone_rows, self.detector.zone_cols)
        Plotter.new_process(Plotter.plot_points, self.detector.histograms, self.detector.zone_rows, self.detector.zone_cols)


sim = Simulation()
sim.run()
