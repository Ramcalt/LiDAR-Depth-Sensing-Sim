from scipy.signal.windows import gaussian
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
    bin_width_m: float
    histograms: np.ndarray

    def __init__(self, name, mesh_path, material, transform, zone_rows, zone_cols, fov_x_rad, fov_y_rad, bin_count,
                 bin_width_m):
        super().__init__(name, mesh_path, material, transform)
        self.zone_rows = zone_rows
        self.zone_cols = zone_cols
        self.fov_x_rad = fov_x_rad
        self.fov_y_rad = fov_y_rad
        self.bin_count = bin_count
        self.bin_width_m = bin_width_m
        self.histograms = np.array(
            [[Histogram(0, 0, bin_count, bin_width_m) for _ in range(zone_cols)]
             for _ in range(zone_rows)],
            dtype=object
        )

    def gaussian_kernel(self, size, sigma):
        offsets = np.arange(size) - (size - 1) / 2.0
        gridx, gridy = np.meshgrid(offsets, offsets, indexing='xy')
        K = np.exp(-(gridx**2 + gridy**2) / (2.0 * sigma**2)) # kernel
        K /= K.sum() # normalise
        return K

    def apply_binning(self, distances, row_idx, col_idx, bleed = True):
        """ Apply SPAD finite time-resolution binning. """
        if self.zone_rows == 4:
            psf_size = 2
            psf_sigma = 0.1
        elif self.zone_rows == 8:
            psf_size = 4
            psf_sigma = 0.2

        binned_distances = np.floor(distances / self.bin_width_m).astype(int)

        # Valid hits: in-range time bin and pixel indices
        valid = (
                (binned_distances >= 0) & (binned_distances < self.bin_count) &
                (row_idx >= 0) & (row_idx < self.zone_rows) &
                (col_idx >= 0) & (col_idx < self.zone_cols)
        )
        if not np.any(valid):
            return

        rows = row_idx[valid]
        cols = col_idx[valid]
        counts = binned_distances[valid]

        # accumulate counts
        H = np.zeros((self.zone_rows, self.zone_cols, self.bin_count), dtype=float)
        np.add.at(H, (rows, cols, counts), 1.0)

        # bleed across zones with a gaussian kernel
        if (bleed):
            # Build PSF kernel and pad once in space
            K = self.gaussian_kernel(psf_size, psf_sigma)
            pad = psf_size // 2

            # Pad only spatial dimensions; reflect padding conserves energy visually at borders
            H_padded = np.pad(H, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

            # 2D convolution per time bin
            Hp = np.zeros_like(H)
            for b in range(self.bin_count):
                # slide the kernel over the (padded) spatial plane for this bin
                for r in range(self.zone_rows):
                    for c in range(self.zone_cols):
                        # extract a psf_size x psf_size window and dot with kernel
                        window = H_padded[r:r + psf_size, c:c + psf_size, b]
                        Hp[r, c, b] = np.sum(window * K)

            H = Hp  # spatially mixed histograms

        # normalise
        max_per = H.max(axis=2, keepdims=True)
        np.divide(H, max_per, out=H, where=max_per > 0)

        for r in range(self.zone_rows):
            for c in range(self.zone_cols):
                self.histograms[r, c].data = H[r, c]

    def fill_hist_with_noise(self):
        self.histograms = np.array([
            [Histogram(0,
                       0,
                       self.bin_count,
                       self.bin_width_m,
                       data=np.array([random.uniform(0, 1) for _ in range(self.bin_count)], dtype=float)
                       ) for _ in range(self.zone_cols)
             ] for _ in range(self.zone_rows)
        ], dtype=object)