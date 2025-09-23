from raysect.optical.observer import RGBPipeline2D
from raysect.optical.scenegraph.world import World
from scene.SceneObject import SceneObject
import matplotlib.pyplot as plt

class RayTracer:
    def __init__(self):
        pass

    @staticmethod
    def run(scene, emitter, detector):
        rgb = RGBPipeline2D()
        world = World()
        rs_meshes = [obj.to_raysect_mesh(world) for obj in scene.objects if type(obj) is SceneObject]
        rs_emitter = emitter.to_raysect_emitter(world)
        rs_detector = detector.to_raysect_detector(world, rgb)
        plt.ion()
        rs_detector.observe()
        plt.ioff()
        rgb.display()
        plt.show()

