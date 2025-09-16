from scene.SceneObject import SceneObject

class Detector(SceneObject):
    zone_rows: int
    zone_cols: int
    fov_x_rad: float
    fov_y_rad: float
    bin_count: int
    bin_width_ps: float

    def __init__(self, name, mesh_path, material, transform, zone_rows, zone_cols, fov_x_rad, fov_y_rad, bin_count, bin_width_ps):
        super().__init__(name, mesh_path, material, transform)
        self.zone_rows = zone_rows
        self.zone_cols = zone_cols
        self.fov_x_rad = fov_x_rad
        self.fov_y_rad = fov_y_rad
        self.bin_count = bin_count
        self.bin_width_ps = bin_width_ps



