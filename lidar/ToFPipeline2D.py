from __future__ import annotations
from typing import Callable, Optional, Tuple
import numpy as np
from raysect.optical.observer.base.pipeline import Pipeline2D

class ToFPipeline2D(Pipeline2D):
    def __init__(self,
                 on_bin: Callable[[Tuple[int, int], float, float], None],
                 *,
                 name: str = "ToF Histogram Accumulator"):
        self.name = name
        self._on_bin = on_bin
        self._pixels: Optional[Tuple[int, int]] = None
        self._n_submitted = 0           # how many submit calls we received
        self._n_submitted_nonzero = 0   # how many had power>0

    def configure(self, pixels: Tuple[int, int]) -> None:
        self._pixels = pixels
        print(f"[ToFPipeline2D] configure pixels={pixels}", flush=True)

    def reset(self) -> None:
        print("[ToFPipeline2D] reset()", flush=True)

    def finalise(self) -> None:
        # print a summary EVERY frame
        # if the camera's observe() ran, you WILL see this line
        summary = f"[ToFPipeline2D] finalise: submitted={self._n_submitted}, nonzero={self._n_submitted_nonzero}"
        # If the accumulator exposes counters, print them:
        acc = getattr(self, "_on_bin", None)
        binned = getattr(acc, "binned", None)
        oor = getattr(acc, "out_of_range", None)
        if binned is not None or oor is not None:
            summary += f", binned={binned}, out_of_range={oor}"
        print(summary, flush=True)

    def frame_start(self) -> None:
        pass

    def frame_finish(self) -> None:
        pass

    def submit_ray_sample(self, xy: Tuple[int, int], distance_m: float, power_W: float) -> None:
        self._n_submitted += 1
        if power_W <= 0.0 or distance_m < 0.0:
            return
        self._n_submitted_nonzero += 1
        self._on_bin(xy, distance_m, power_W)