from dataclasses import dataclass
from typing import List
import numpy as np
import matplotlib.pyplot as plt


class Histogram:
    data: np.ndarray
    time_start: float
    time_end: float
    bin_count: int
    bin_width_m: float

    def __init__(self, time_start, time_end, bin_count, bin_width_m, data=None):
        if data is None:
            data = np.array([0.0 for _ in range(bin_count)])
        self.data = data
        self.time_start = time_start
        self.bin_count = bin_count
        self.bin_width_m = bin_width_m
        self.time_end = float(time_end) if time_end else (self.time_start + self.bin_count * self.bin_width_m)

    def get_points_echo_detection(self,
                                  min_deriv_diff=0.15,
                                  min_peak_height=0.0,
                                  thresh = 0.95,
                                  normalise=True):
        """
        Echo detection from a LiDAR histogram.

        Steps:
          1) Normalise data to [0, 1] (optional).
          2) Compute first-order central-difference derivative.
          3) Interpolate derivative zero-crossings (+ to -) to locate local maxima.
          4) Filter maxima by requiring a positive-to-negative crossing with
             a minimum derivative magnitude change and minimum peak height.
          5) For each accepted maximum, linearly interpolate the rising-edge
             position at 70% of the peak height (i.e., '−30%' point).

        Parameters
        ----------
        min_deriv_diff : float
            Minimum required magnitude of the derivative sign-change across the
            zero-crossing: (deriv[i-1] - deriv[i]) >= min_deriv_diff.
            Helps suppress noise-only peaks.
        min_peak_height : float
            Minimum peak height after normalisation. Set >0 to ignore tiny peaks.
        normalise : bool
            If True, divide by max(data) before processing.

        Returns
        -------
        points : list of float
            Distances (metres) at the −30% point on the rising edge for each
            accepted local maximum.
        """

        data = np.asarray(self.data, dtype=float).copy()
        if normalise:
            max_val = np.max(data)
            if max_val > 0:
                data /= max_val

        n = data.size
        if n < 3:
            return []

        # 2) First-order central difference (forward/backward at the ends)
        deriv = np.empty_like(data)
        deriv[0] = data[1] - data[0]
        deriv[-1] = data[-1] - data[-2]
        if n > 2:
            deriv[1:-1] = 0.5 * (data[2:] - data[:-2])

        # 3) Find derivative zero-crossings from positive to negative (local maxima).
        # Crossing occurs between i-1 and i where deriv[i-1] > 0 and deriv[i] < 0.
        pos_to_neg = np.where((deriv[:-1] > 0.0) & (deriv[1:] < 0.0))[0] + 1  # indices 'i'

        points = []
        for i in pos_to_neg:
            # 4) Filter by derivative magnitude change and (later) peak height
            d_prev, d_curr = deriv[i - 1], deriv[i]
            if (d_prev - d_curr) < min_deriv_diff:
                continue  # insufficient slope change

            # Interpolate the derivative zero-crossing for a sub-bin peak location.
            # Linear interpolation on derivative between (i-1, d_prev) and (i, d_curr):
            # x_zero = (i-1) + d_prev / (d_prev - d_curr)
            denom = (d_prev - d_curr)
            if np.isclose(denom, 0.0):
                x_zero = float(i)  # fallback
            else:
                x_zero = (i - 1) + (d_prev / denom)

            # Choose nearest integer sample around the peak for quadratic refinement
            ip = int(np.clip(round(x_zero), 1, n - 2))

            # Quadratic interpolation of the local maximum (parabolic fit)
            y1, y2, y3 = data[ip - 1], data[ip], data[ip + 1]
            denom_par = (y1 - 2.0 * y2 + y3)
            if np.isclose(denom_par, 0.0):
                sub_offset = 0.0
                peak_val = y2
                peak_pos = float(ip)
            else:
                sub_offset = 0.5 * (y1 - y3) / denom_par  # sub-bin shift from ip
                peak_pos = ip + sub_offset
                # Interpolated peak value: y(ip + sub_offset)
                peak_val = y2 - 0.25 * (y1 - y3) * sub_offset

            if peak_val < min_peak_height:
                continue

            # 5) Find the −30% point (i.e., 70% of peak) on the rising edge.
            threshold = thresh * peak_val

            # Search leftwards from the integer part of the peak until we bracket the threshold.
            js = int(np.clip(np.floor(peak_pos), 1, n - 1))
            j = js
            # Move left while we are still above the threshold (on/near the peak)
            while j > 0 and data[j] > threshold:
                j -= 1

            # Now j is the first index at/below threshold on the left.
            # Ensure we have a valid bracket [j, j+1] with a rising edge (data[j] <= thr <= data[j+1])
            if j < n - 1:
                y_lo, y_hi = data[j], data[j + 1]
                dy = y_hi - y_lo
                if dy > 0:
                    # Linear interpolation to find fractional index at 'threshold'
                    frac = (threshold - y_lo) / dy
                    x_thr = j + np.clip(frac, 0.0, 1.0)
                else:
                    # Degenerate or flat; fall back to integer j
                    x_thr = float(j)
            else:
                x_thr = float(j)

            # Convert sample position to distance (metres)
            dist_m = x_thr * self.bin_width_m + self.time_start
            points.append(dist_m)

        return points

    def get_points_wav_decomp(self):
        return 1.0

    def get_points_deconv(self):
        return 1.0

    def get_points_ai_model(self):
        return 1.0

