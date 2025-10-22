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
    pulse_width_m: float

    def __init__(self, time_start, time_end, bin_count, bin_width_m, data=None):
        if data is None:
            data = np.array([0.0 for _ in range(bin_count)])
        self.data = data
        self.time_start = time_start
        self.bin_count = bin_count
        self.bin_width_m = bin_width_m
        self.time_end = float(time_end) if time_end else (self.time_start + self.bin_count * self.bin_width_m)

    # central difference
    def compute_derivative(self):
        data = self.data
        N = self.bin_count
        deriv = np.zeros_like(data)

        deriv[0] = data[1] - data[0] # fwd diff for first point
        deriv[-1] = data[-1] - data[-2] # bwd diff for last point
        for i in range(1, N-1): # central diff for interior points
            deriv[i] = (data[i+1] - data[i-1])/2.0
        return deriv

    # second order central difference
    def compute_second_derivative(self):
        data = self.data
        N = self.bin_count
        dderiv = np.zeros_like(data)

        dderiv[0] = data[2] - 2 * data[1] + data[0]
        dderiv[-1] = data[-1] - 2 * data[-2] + data[-3]
        for i in range(1, N-1):
            dderiv[i] = data[i+1] - 2 * data[i] + data[i-1]
        return dderiv

    # find zero crossings of deriv
    def find_local_maxima(self, deriv: np.ndarray, dderiv: np.ndarray):
        N = self.bin_count
        maxima = []
        for i in range(1, N-1):
            if (deriv[i - 1] > 0.0 >= deriv[i]) and (dderiv[i-1] < 0.0 or dderiv[i] < 0.0):
                t = deriv[i-1] / (deriv[i-1] - deriv[i])
                maxima.append((i-1) + t)
        return np.array(maxima)

    # interpolation
    def lerp(self, v0, v1, t):
        return (1 - t) * v0 + t * v1

    # find fwhm
    def find_fwhm(self, data: np.ndarray, maxima: np.ndarray):
        N = self.bin_count
        fwhm = []
        for maximum in maxima:
            # Compute indices of points that pass 50% threshold
            i_left = np.floor(maximum)
            i_right = np.ceil(maximum)
            half_maximum_val = 0.5 * self.lerp(data[i_left], data[i_right], maximum - i_left)

            while i_left >= 0 and data[i_left] >= half_maximum_val:
                i_left -= 1
            while i_right < N and data[i_right] >= half_maximum_val:
                i_right += 1

            # Compute left side crossing
            if i_left < 0:
                # TODO: estimate based on derivatives
                # Half-maximum lies before index 0 -> extrapolate using (0,1)
                if N >= 2 and data[1] != data[0]:
                    x_left = 0.0 + (half_maximum_val - data[0]) / (data[1] - data[0])
                else:
                    x_left = 0.0  # degenerate fallback
            else:
                # Crossing is between i_left and i_left+1
                j0, j1 = i_left, i_left + 1
                y0, y1 = data[j0], data[j1]
                if y1 != y0:
                    x_left = j0 + (half_maximum_val - y0) / (y1 - y0)
                else:
                    x_left = j0 + 0.5  # flat segment fallback

            # Compute right side crossing
            if i_right >= N:
                # TODO: estimate based on derivatives
                # Half-maximum lies beyond index N-1 -> extrapolate using (N-2, N-1)
                if N >= 2 and data[N - 1] != data[N - 2]:
                    x_right = (N - 1) + (half_maximum_val - data[N - 1]) / (data[N - 1] - data[N - 2])
                else:
                    x_right = float(N - 1)  # degenerate fallback
            else:
                # Crossing is between i_right-1 and i_right
                j0, j1 = i_right - 1, i_right
                y0, y1 = data[j0], data[j1]
                if y1 != y0:
                    x_right = j0 + (half_maximum_val - y0) / (y1 - y0)
                else:
                    x_right = j0 + 0.5  # flat segment fallback

            # Interpolate to find xl and xr TODO: incorrect
            # x_left = -0.5 * maximum_val * ((maximum - i_left)/(maximum_val - data[i_left])) + i_left
            # x_right = -0.5 * maximum_val * ((maximum - i_left) / (maximum_val - data[i_left])) + i_left

            fwhm.append(x_right - x_left)

        return np.array(fwhm)


    def get_points_echo_detection(self):
        # data = self.data
        # deriv = self.compute_derivative()
        # dderiv = self.compute_second_derivative()
        # maxima = self.find_local_maxima(deriv, dderiv)
        # fwhm = self.find_fwhm(data, maxima)
        self.visualize(title="Histogram, Derivatives, Local Maxima, and FWHM")
        return [1.0] # TODO: ALGORITHM

    # helper to get FWHM segments for plotting (left/right/half height for each peak)
    def _fwhm_segments(self, data: np.ndarray, maxima: np.ndarray):
        N = self.bin_count
        segments = []  # list of dicts: {x_left, x_right, half, x_peak, y_peak}
        for maximum in maxima:
            iL = int(np.floor(maximum))
            iR = int(np.ceil(maximum))
            iL = max(0, min(N - 1, iL))
            iR = max(0, min(N - 1, iR))
            y_peak = self.lerp(data[iL], data[iR], maximum - iL)
            half = 0.5 * y_peak

            # Walk to find brackets
            jL = int(np.floor(maximum))
            while jL >= 0 and data[jL] >= half:
                jL -= 1
            jR = int(np.ceil(maximum))
            while jR < N and data[jR] >= half:
                jR += 1

            # Left crossing
            if jL < 0:
                if N >= 2 and data[1] != data[0]:
                    x_left = 0.0 + (half - data[0]) / (data[1] - data[0])
                else:
                    x_left = 0.0
            else:
                y0, y1 = data[jL], data[jL + 1]
                x_left = jL + (half - y0) / (y1 - y0) if y1 != y0 else jL + 0.5

            # Right crossing
            if jR >= N:
                if N >= 2 and data[N - 1] != data[N - 2]:
                    x_right = (N - 1) + (half - data[N - 1]) / (data[N - 1] - data[N - 2])
                else:
                    x_right = float(N - 1)
            else:
                y0, y1 = data[jR - 1], data[jR]
                x_right = (jR - 1) + (half - y0) / (y1 - y0) if y1 != y0 else (jR - 1) + 0.5

            segments.append({
                "x_left": x_left,
                "x_right": x_right,
                "half": half,
                "x_peak": float(maximum),
                "y_peak": float(y_peak),
            })
        return segments

    # Utility to convert sample index to time coordinate (meters here)
    def _idx_to_time(self, x_idx: np.ndarray):
        return self.time_start + self.bin_width_m * np.asarray(x_idx, dtype=float)

    # Main visualization
    def visualize(self, figsize=(10, 8), title=None):
        data = self.data
        N = self.bin_count
        x_idx = np.arange(N, dtype=float)
        t = self._idx_to_time(x_idx)

        # Compute derivatives and features
        deriv = self.compute_derivative()
        dderiv = self.compute_second_derivative()
        maxima_idx = self.find_local_maxima(deriv, dderiv)  # fractional indices
        segments = self._fwhm_segments(data, maxima_idx)

        # Prepare figure with shared x-axis
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize, constrained_layout=True)
        ax0, ax1, ax2 = axes

        # (a) Histogram
        ax0.plot(t, data, marker='o', linestyle='-', label='Histogram')
        ax0.set_ylabel('Amplitude (normalized)')
        if title:
            ax0.set_title(title)

        # (d) Local maxima overlay
        if len(maxima_idx) > 0:
            t_peaks = self._idx_to_time(maxima_idx)
            # Interpolate peak values for markers
            y_peaks = []
            for m in maxima_idx:
                iL = int(np.floor(m))
                iR = min(N - 1, iL + 1)
                y_peaks.append(self.lerp(data[iL], data[iR], m - iL))
            y_peaks = np.asarray(y_peaks, dtype=float)
            ax0.scatter(t_peaks, y_peaks, s=50, marker='^', label='Local maxima')

        # (e) FWHM bars
        fwhm_labeled = False  # track if we’ve already added one legend label
        for seg in segments:
            # horizontal half-maximum bar
            tl = self._idx_to_time(seg["x_left"])
            tr = self._idx_to_time(seg["x_right"])
            label = 'FWHM' if not fwhm_labeled else None
            ax0.hlines(seg["half"], tl, tr, linewidth=2, linestyles='--', label=label)
            fwhm_labeled = True  # only label the first one

            # small vertical ticks at crossings
            ax0.vlines([tl, tr], ymin=0.0, ymax=seg["half"], linewidth=1, linestyles=':')
            # optional width annotation
            width_m = tr - tl
            ax0.text((tl + tr) / 2.0, seg["half"], f'{width_m:.3g}', ha='center', va='bottom', fontsize=8)

        ax0.legend(loc='best')
        ax0.grid(True, alpha=0.3)

        # (b) First derivative (linear interpolation between points)
        ax1.plot(t, deriv, marker='o', linestyle='-', label='First derivative')
        ax1.axhline(0.0, linewidth=1)
        ax1.set_ylabel('d(data)/d(bin)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # (c) Second derivative (linear interpolation between points)
        ax2.plot(t, dderiv, marker='o', linestyle='-', label='Second derivative')
        ax2.axhline(0.0, linewidth=1)
        ax2.set_xlabel('Time (m)')
        ax2.set_ylabel('d²(data)/d(bin)²')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        plt.show()




    def get_points_wav_decomp(self):
        return 1.0

    def get_points_deconv(self):
        return 1.0

    def get_points_ai_model(self):
        return 1.0


# =========================
# Example usage (synthetic)
# =========================
if __name__ == "__main__":
    # Synthetic example: two Gaussian-like peaks sampled on unit steps
    np.random.seed(0)
    N = 200
    bin_width_m = 0.02  # 2 cm per bin (example)
    x = np.arange(N)
    # Create a couple of peaks
    y = (
        1.0 * np.exp(-0.5 * ((x - 60) / 5.0) ** 2) +
        0.7 * np.exp(-0.5 * ((x - 130) / 8.0) ** 2)
    )
    y += 0.02 * np.random.randn(N)  # small noise
    # Normalize if desired
    y /= y.max()

    h = Histogram(time_start=0.0, time_end=None, bin_count=N, bin_width_m=bin_width_m, data=y)
    h.visualize(title="Histogram, Derivatives, Local Maxima, and FWHM")