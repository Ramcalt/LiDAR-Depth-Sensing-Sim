from dataclasses import dataclass
from typing import List
import numpy as np

class Histogram:
    data: np.ndarray
    time_start: float
    time_end: float
    bin_count: int
    bin_width: float

    def __init__(self, time_start, time_end, bin_count, bin_width, data=None):
        if data is None:
            data = np.array([0.0 for _ in range(bin_count)])
        self.data = data
        self.time_start = time_start
        self.time_end = time_end
        self.bin_count = bin_count
        self.bin_width = bin_width

    @staticmethod
    def get_points_echo_detection():
        return 1.0

    @staticmethod
    def get_points_wav_decomp():
        return 1.0

    @staticmethod
    def get_points_deconv():
        return 1.0

    @staticmethod
    def get_points_ai_model():
        return 1.0

