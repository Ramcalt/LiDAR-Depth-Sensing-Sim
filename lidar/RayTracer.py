import numpy as np
from raysect.optical.observer import RGBPipeline2D, PowerPipeline2D, RadiancePipeline2D
from raysect.optical.scenegraph.world import World
from scene.SceneObject import SceneObject
import matplotlib.pyplot as plt

class RayTracer:
    def __init__(self):
        pass

    @staticmethod
    def run(scene, emitter, detector):
        pipeline = RadiancePipeline2D()
        world = World()
        rs_meshes = [obj.to_raysect_mesh(world) for obj in scene.objects if type(obj) is SceneObject]
        rs_emitter = emitter.to_raysect_emitter(world)
        rs_detector = detector.to_raysect_detector(world, pipeline)
        plt.ion()
        rs_detector.observe()
        plt.ioff()
        pipeline.display()
        plt.show()

    @staticmethod
    def run_tof(scene, emitter, detector):
        world = World()
        rs_meshes = [obj.to_raysect_mesh(world) for obj in scene.objects if type(obj) is SceneObject]
        rs_emitter = emitter.to_raysect_emitter(world)
        rs_tof_detector, tof_pipeline = detector.to_raysect_tof_detector(world)
        rs_tof_detector.observe()

        acc = tof_pipeline._on_bin
        print(f"[RayTracer] pipeline_submitted={getattr(tof_pipeline, '_n_submitted', '?')}, "
              f"binned={getattr(acc, 'binned', '?')}, "
              f"out_of_range={getattr(acc, 'out_of_range', '?')}",
              flush=True)

