from scene.Material import Material
from scene.HTransform import HTransform
import open3d as o3d
import trimesh

class SceneObject:
    """Renderable component with material, geometry, and transform"""
    name: str
    mesh_path: str
    material: Material
    transform: HTransform

    def __init__(self, name, mesh_path, material, transform):
        self.name = name
        self.mesh_path = mesh_path
        self.material = material
        self.transform = transform

    def to_trimesh_mesh(self):
        mesh = trimesh.load(self.mesh_path)
        mesh.apply_transform(self.transform.matrix)
        mesh.visual.face_colors = self.material.colour
        mesh.visual.material = self.material.to_trimesh_material()
        return mesh

    def to_o3d_mesh(self):
        """Convert mesh into a renderable Open3D object with applied colour and transform"""
        mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        if mesh.is_empty() or len(mesh.triangles) == 0:
            raise ValueError("Mesh is empty or does not contain any triangles")
        mesh.transform(self.transform.matrix)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(self.material.colour)
        return mesh

