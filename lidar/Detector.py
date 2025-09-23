from raysect.core import AffineMatrix3D
from raysect.optical.observer import PinholeCamera

from scene.SceneObject import SceneObject
from lidar.Histogram import Histogram
from typing import List
import numpy as np
import random


class Detector(SceneObject):
    zone_rows: int
    zone_cols: int
    fov_x_rad: float
    fov_y_rad: float
    bin_count: int
    bin_width: float
    histograms: np.ndarray

    def __init__(self, name, mesh_path, material, transform, zone_rows, zone_cols, fov_x_rad, fov_y_rad, bin_count,
                 bin_width):
        super().__init__(name, mesh_path, material, transform)
        self.zone_rows = zone_rows
        self.zone_cols = zone_cols
        self.fov_x_rad = fov_x_rad
        self.fov_y_rad = fov_y_rad
        self.bin_count = bin_count
        self.bin_width = bin_width
        self.histograms = np.array(
            [[Histogram(0, 0, bin_count, bin_width) for _ in range(zone_cols)]
             for _ in range(zone_rows)],
            dtype=object
        )

    def fill_hist_with_noise(self):
        self.histograms = np.array([
            [Histogram(0,
                       0,
                       self.bin_count,
                       self.bin_width,
                       data=np.array([random.uniform(0, 1) for _ in range(self.bin_count)], dtype=float)
                       ) for _ in range(self.zone_cols)
             ] for _ in range(self.zone_rows)
        ], dtype=object)

    def to_raysect_detector(self, world, pipelines):
        detector = PinholeCamera((512, 512), pipelines=[pipelines], transform=AffineMatrix3D(self.transform.matrix))
        detector.fov = np.rad2deg(self.fov_x_rad)
        detector.spectral_rays = 1
        detector.spectral_bins = 20
        detector.ray_max_depth = 100
        detector.ray_extinction_prob = 0.1
        detector.min_wavelength = 375.0
        detector.max_wavelength = 740.0
        detector.parent = world
        return detector
