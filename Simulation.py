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
        self.scene.add_obj(SceneObject("cavity", "res/cavity.obj",
                                       Material(0.2, 0.8, 1, [0.9, 0.2, 0.1]),
                                       HTransform()))

    def run(self):
        self.viewScene()

    def viewScene(self):
        meshes = []
        for obj in self.scene.objects:
            meshes.append(obj.to_o3d_geometry())
        o3d.visualization.draw_geometries(meshes)

print(o3d.__version__)
mesh = o3d.geometry.TriangleMesh.create_sphere()
mesh.compute_vertex_normals()
o3d.visualization.draw(mesh, raw_mode=True)
sim = Simulation()
sim.run()