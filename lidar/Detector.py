from raysect.core import AffineMatrix3D
from raysect.optical.observer import PinholeCamera
from raysect.optical.observer.base.observer import MulticoreEngine

from lidar.HistogramAccumulator import HistogramAccumulatorToF, HistogramAccumulatorToF
from lidar.ToFPinholeCamera import ToFPinholeCamera
from lidar.ToFPipeline2D import ToFPipeline2D
from scene.SceneObject import SceneObject
from lidar.Histogram import Histogram
from typing import List
import numpy as np
import random

C = 299_792_458.0  # m/s

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

    def get_tof_edges_s(self) -> np.ndarray:
        """Uniform bin edges in seconds for the detector histograms (assumes all zones share layout)."""
        h = self.histograms[0, 0]
        t0 = h.time_start
        return np.linspace(t0, t0 + h.bin_count * h.bin_width, h.bin_count + 1)

    def get_range_edges_m(self, *, round_trip: bool = True) -> np.ndarray:
        """Distance edges derived from the time edges (for plotting); round-trip by default."""
        t_edges = self.get_tof_edges_s()
        if round_trip:
            return 0.5 * C * t_edges
        return C * t_edges

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
        detector.min_wavelength = 100.0
        detector.max_wavelength = 1100.0
        detector.parent = world
        return detector

    def to_raysect_tof_detector(self, world):
        """
        Build a ToF pipeline + ToF camera that will fill self.histograms.
        The third argument in RayTracer.run(...) is kept for API compatibility,
        but ignored now that the pipeline bins in time.
        """
        # Bridge: distance -> time-of-flight -> histogram bin
        on_bin = HistogramAccumulatorToF(self.histograms, round_trip=True)
        tof_pipeline = ToFPipeline2D(on_bin=on_bin)

        cam = ToFPinholeCamera(pixels=(self.zone_cols, self.zone_rows), tof_pipeline=tof_pipeline)
        cam.parent = world
        cam.transform = AffineMatrix3D(self.transform.matrix)

        cam.fov = np.rad2deg(self.fov_x_rad)
        cam.pixel_samples = 1

        # *** CRITICAL: set waveband to match your emitter line ***
        cam.max_wavelength = 930.0
        cam.min_wavelength = 880.0
        cam.spectral_bins = 1

        cam.ray_max_depth = 50
        cam.ray_extinction_prob = 0.0

        # sanity: assert pipeline is attached
        assert tof_pipeline in cam.pipelines, "ToF pipeline not attached to camera"

        return cam, tof_pipeline
