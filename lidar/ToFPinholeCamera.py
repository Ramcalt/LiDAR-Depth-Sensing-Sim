# lidar/tof_pinhole.py
from __future__ import annotations

from typing import Tuple
import numpy as np

from raysect.core import AffineMatrix3D, Point3D, Vector3D
from raysect.optical.observer import PinholeCamera
from raysect.optical.loggingray import LoggingRay  # used in upstream demos
from raysect.optical.spectrum import Spectrum

class ToFPinholeCamera(PinholeCamera):
    """
    Drop-in replacement for PinholeCamera that launches LoggingRay per sample,
    computes total path length, integrates spectral power, and forwards the
    sample to a ToFPipeline2D.
    """

    def __init__(self, pixels: Tuple[int, int], tof_pipeline, **kwargs):
        # enforce we also mount the tof_pipeline in the camera's pipelines list
        pipelines = list(kwargs.pop('pipelines', []))
        pipelines.append(tof_pipeline)
        # super().__init__(pixels=pixels, pipelines=pipelines, **kwargs)
        # self._tof = tof_pipeline
        # Extract Raysect PinholeCamera-safe args
        pixels_arg = pixels
        pipelines_arg = pipelines

        # Remove keys that PinholeCamera.__init__() does not accept
        invalid_keys = [
            "spectral_rays", "spectral_bins",
            "ray_max_depth", "ray_extinction_prob",
            "ray_extinction_min_depth", "ray_max_distance",
            "importance_sampling", "important_path_weight",
            "min_wavelength", "max_wavelength",
            "fov", "parent", "transform"
        ]
        safe_kwargs = {k: v for k, v in kwargs.items() if k not in invalid_keys}

        super().__init__(pixels=pixels_arg, pipelines=pipelines_arg, **safe_kwargs)

        # Now set the extra attributes manually
        self._tof = tof_pipeline

        # Optional camera attributes
        for k in invalid_keys:
            if k in kwargs:
                setattr(self, k, kwargs[k])

        self._debug_first = 5  # print the first few nonzero samples

    # --- helpers ------------------------------------------------------

    @staticmethod
    def _total_path_length_m(ray_log) -> float:
        """
        Sum Euclidean distances between consecutive intersection points, including
        origin->first_hit and last_hit->emitter when applicable (for ToF we care
        about total geometric distance from detector to light, i.e., observer->source
        in backwards path tracing).
        """
        pts = []
        # the LoggingRay log entries expose intersection.hit_point in world space (see demo)
        # https://github.com/raysect/source/blob/master/demos/optics/logging_trajectories.py
        for entry in ray_log:
            p = entry.hit_point
            pts.append((p.x, p.y, p.z))

        if not pts:
            return np.inf  # no light path found

        # The LoggingRay starts at the observer. Add the start point:
        # NB: logging demo uses 'start' separately; here we rely on ray.origin.
        total = 0.0
        prev = None
        for cur in pts:
            if prev is None:
                # The first segment: from ray origin to first hit
                prev = cur
                # We *could* include origin->first_hit explicitly if the log excludes it;
                # To be robust, insert origin at front below in _trace_and_log().
            else:
                dx = cur[0] - prev[0]
                dy = cur[1] - prev[1]
                dz = cur[2] - prev[2]
                total += (dx*dx + dy*dy + dz*dz) ** 0.5
                prev = cur

        return total

    def _trace_and_log(self, origin, direction):
        """
        Launch a LoggingRay and return (spectrum, total_path_length_m).
        Only pass constructor kwargs that are broadly supported.
        """
        # Build a conservative kwargs set. Add optional ones if present on self.
        ray_kwargs = {
            "min_wavelength": getattr(self, "min_wavelength", 375.0),
            "max_wavelength": getattr(self, "max_wavelength", 740.0),
            "bins": getattr(self, "spectral_bins", 1),
            "max_depth": getattr(self, "ray_max_depth", 50),
        }
        # Optional features (only include if defined on this camera)
        if hasattr(self, "ray_extinction_prob"):
            ray_kwargs["extinction_prob"] = getattr(self, "ray_extinction_prob")
        if hasattr(self, "ray_extinction_min_depth"):
            ray_kwargs["extinction_min_depth"] = getattr(self, "ray_extinction_min_depth")
        if hasattr(self, "importance_sampling"):
            ray_kwargs["importance_sampling"] = getattr(self, "importance_sampling")
        if hasattr(self, "important_path_weight"):
            ray_kwargs["important_path_weight"] = getattr(self, "important_path_weight")

        # DO NOT pass ray_max_distance – it is not present in your build.
        log_ray = LoggingRay(origin, direction, **ray_kwargs)

        spectrum = log_ray.trace(self.parent)  # world is the camera's parent

        # Reconstruct points = [origin] + intersections and sum segment lengths
        points = [(origin.x, origin.y, origin.z)]
        for entry in log_ray.log:
            p = entry.hit_point
            points.append((p.x, p.y, p.z))

        total = 0.0
        for a, b in zip(points[:-1], points[1:]):
            dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            total += (dx * dx + dy * dy + dz * dz) ** 0.5

        return spectrum, total

    # --- core: override a single-sample path and funnel into the ToF pipeline

    def _ray_through_pixel(self, x: int, y: int):
        """
        Compute a world-space ray for pixel (x, y) using a standard pinhole model.

        Camera-space conventions:
          - Camera looks along -Z
          - Image (sensor) plane at Z = -1
          - Horizontal FOV = self.fov (degrees)
        """
        # --- image geometry
        W, H = self.pixels  # (cols, rows)
        fx = np.deg2rad(getattr(self, "fov", 40.0))  # horizontal FOV in radians
        half_w = np.tan(0.5 * fx)
        aspect = H / float(W)
        half_h = half_w * aspect

        # pixel center in NDC [-1, +1] (x right, y up). If your image is upside-down, flip v.
        u = ( (x + 0.5) / float(W) ) * 2.0 - 1.0
        v = ( (y + 0.5) / float(H) ) * 2.0 - 1.0
        v = -v # ADDED?
        # Raysect’s image origin is typically bottom-left for observers; if yours is top-left,
        # you can invert v: v = -v

        # point on the image plane at z = -1 in camera space
        px = u * half_w
        py = v * half_h
        p_cam = Point3D(px, py, -1.0)

        # pinhole is at origin in camera space
        o_cam = Point3D(0.0, 0.0, 0.0)
        # d_cam = Vector3D(p_cam.x - o_cam.x, p_cam.y - o_cam.y, p_cam.z - o_cam.z).normalise()
        d_cam = Vector3D(p_cam.x, p_cam.y, p_cam.z).normalise() # ADDED CHANGE?

        # transform to world space using the camera's current transform (relative to parent/world)
        # AffineMatrix3D supports multiplying Points/Vectors directly.
        T = self.transform  # AffineMatrix3D
        o_world = T * o_cam
        d_world = (T * d_cam).normalise()

        return o_world, d_world

    def _observe_single_sample(self, x: int, y: int) -> None:
        origin, direction = self._ray_through_pixel(x, y)
        spectrum, distance_m = self._trace_and_log(origin, direction)

        if np.isfinite(distance_m) and spectrum is not None:
            power_W = float(spectrum.integrate(self.min_wavelength, self.max_wavelength))
            if power_W > 0 and self._debug_first > 0:
                print(f"[ToFPinholeCamera] sample @({x},{y}) d={distance_m:.3f} m, P={power_W:.3e} W",
                      flush=True)
                self._debug_first -= 1
            self._tof.submit_ray_sample((x, y), distance_m, power_W)

    def observe(self) -> None:
        for p in self.pipelines:
            if hasattr(p, "configure"):
                p.configure(self.pixels)

        printed = False
        for s in range(self.pixel_samples):
            for (x, y) in self.frame_sampler.generate_tasks(self.pixels):
                if not printed:
                    print(f"[ToFPinholeCamera] begin sampling s={s}, pixels={self.pixels}", flush=True)
                    printed = True
                self._observe_single_sample(x, y)

        for p in self.pipelines:
            if hasattr(p, "finalise"):
                p.finalise()