import numpy as np
from raysect.optical.observer import RGBPipeline2D, PowerPipeline2D, RadiancePipeline2D
from raysect.optical.scenegraph.world import World
from scene.SceneObject import SceneObject
import matplotlib.pyplot as plt
import trimesh
from scene.HTransform import HTransform

class RayTracer:
    def __init__(self):
        pass

    @staticmethod
    def run_trimesh(scene, target_mesh, emitter, detector, N):
        meshes = [obj.to_trimesh_mesh() for obj in scene.objects]
        target = target_mesh.to_trimesh_mesh()
        scene = trimesh.Scene()
        [scene.add_geometry(mesh) for mesh in meshes]

        # generate rays
        ray_directions = np.zeros((N, 3))
        ray_origins = np.array([emitter.transform.matrix[:3, 3]] * N)
        theta_x, theta_y = emitter.sample_gaussian_beam_angles(N)
        for i in range(N):
            R = HTransform().rotation_x(theta_x[i]) @ HTransform().rotation_y(theta_y[i])
            direction = R @ np.array([0.0, 0.0, 1.0, 0.0])
            ray_directions[i] = direction[0:3]

        locations, index_ray, _ = target.ray.intersects_location(ray_origins, ray_directions)
        ray_distances = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
        noisy_distances = emitter.apply_vcsel_pulse_broadening(ray_distances)
        detector.apply_binning(noisy_distances)


        ray_lines = []
        for origin, direction in zip(ray_origins[:20], ray_directions[:20]):
            line = trimesh.load_path(np.vstack((origin, origin + direction * 1.0)))
            line.colors = np.tile([255, 0, 0, 100], (line.entities.shape[0], 1))
            ray_lines.append(line)

        for ray in ray_lines:
            scene.add_geometry(ray)

        scene.show()

    @staticmethod
    def run_rs(scene, emitter, detector):
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
    def run_tof_rs(scene, emitter, detector):
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

