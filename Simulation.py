from lidar.RayTracer import RayTracer
from scene.Scene import Scene
from scene.SceneObject import SceneObject
from scene.Material import Material
from scene.HTransform import HTransform
from lidar.Emitter import Emitter
from lidar.Detector import Detector
from Plotter import Plotter
import open3d as o3d
import trimesh
import numpy as np

class Simulation:
    """Singleton for managing simulation."""
    scene: Scene
    emitter: Emitter
    detector: Detector

    def __init__(self):
        """Initialise scene and add emitter, detector, and scene objects"""
        self.scene = Scene(1.0, [1.0, 1.0, 1.0])
        self.emitter = Emitter("emitter",
                               "res/sensor_toscale.stl",
                               Material(1.0, 1.0, 1.0, [0.8, 0.2, 0.2]),
                               HTransform().translation(-0.002, 0, 0),
                               940e-9,
                               7.7e-8 * (10**(0.002*(940-700))), # 200 * 0.74e-3 * 0.90e-3,
                               1e-9,
                               1.010546, # 57.9deg for 10% signal from max
                               1.010546,
                               4
                               )
        self.detector = Detector("detector",
                                 "res/sensor_toscale.stl",
                                 Material(1.0, 1.0, 1.0, [0.2, 0.2, 0.8]),
                                 HTransform().translation(0.002, 0, 0),
                                 8,
                                 8,
                                 0.79,
                                 0.79,
                                 100,
                                 1.25e-10
                                 )
        self.scene.add_obj(
            SceneObject("cavity",
                        "res/flat_W105.stl",
                        Material(1.0, 1.0, 1.0, [0.9, 0.2, 0.1]),
                        HTransform().translation(0,0, +0.1) @ HTransform().rotation_x(np.pi)
                        )
        )
        self.scene.add_obj(self.detector)
        self.scene.add_obj(self.emitter)

    def test_plotting(self):
        self.detector.fill_hist_with_noise()
        self.view_plots()

    def run(self):
        """Run the simulation"""
        # self.detector.fill_hist_with_noise()
        RayTracer.run_trimesh(self.scene, self.scene.get_obj("cavity"), self.emitter, self.detector, 100_000)
        self.view_plots()

    def view_scene(self):
        """View the scene using Open3D"""
        meshes = [obj.to_o3d_mesh() for obj in self.scene.objects]
        o3d.visualization.draw(meshes, raw_mode=True)

    def view_scene_trimesh(self):
        """View the scene using Open3D"""
        meshes = [obj.to_trimesh_mesh() for obj in self.scene.objects]
        scene = trimesh.Scene()
        [scene.add_geometry(mesh) for mesh in meshes]
        scene.show()

    def view_plots(self):
        """Runs matplotlib plots in separate processes"""
        Plotter.new_process(Plotter.plot_hist, self.detector.histograms[0][0])
        Plotter.new_process(Plotter.plot_hist_arr, self.detector.histograms, self.detector.zone_rows, self.detector.zone_cols)
        Plotter.new_process(Plotter.plot_points, self.detector.histograms, self.detector.zone_rows, self.detector.zone_cols)


sim = Simulation()
sim.view_scene_trimesh()
sim.run()
