from __future__ import annotations
import numpy as np
from typing import Tuple
from lidar.Histogram import Histogram

C = 299_792_458.0  # m/s

class HistogramAccumulatorToF:
    def __init__(self, histograms_2d: np.ndarray, round_trip: bool = True):
        self.hists = histograms_2d
        self.round_trip = round_trip
        self.submitted = 0
        self.binned = 0
        self.out_of_range = 0

    def __call__(self, xy: Tuple[int, int], distance_m: float, power_W: float) -> None:
        x, y = xy
        h: Histogram = self.hists[y, x]
        tof_s = (2.0 * distance_m / C) if self.round_trip else (distance_m / C)

        self.submitted += 1
        idx = int((tof_s - h.time_start) / h.bin_width)
        if 0 <= idx < h.bin_count:
            h.data[idx] += power_W
            self.binned += 1
        else:
            self.out_of_range += 1