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

    # interpolation
    def lerp(self, v0, v1, t):
        return (1 - t) * v0 + t * v1

    # find zero crossings of deriv
    def find_local_maxima(self, deriv: np.ndarray, dderiv: np.ndarray):
        N = self.bin_count
        maxima = []
        for i in range(1, N-1):
            if deriv[i - 1] > 0.0 >= deriv[i]: # and (dderiv[i-1] < 0.0 or dderiv[i] < 0.0):
                t = deriv[i-1] / (deriv[i-1] - deriv[i])
                if self.lerp(dderiv[i-1], dderiv[i], t) < 0.0:
                    maxima.append((i-1) + t)
        return np.array(maxima)

    # find fwhm
    def find_fwhm(self, data: np.ndarray, maxima: np.ndarray):
        N = self.bin_count
        fwhm = []
        p_left = []
        p_right = []
        for maximum in maxima:
            # Compute indices of points that pass 50% threshold
            i_left = int(np.floor(maximum))
            i_right = int(np.ceil(maximum))
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
            p_left.append(x_left)
            p_right.append(x_right)

        return np.array(fwhm), np.array(p_left), np.array(p_right)

    def compute_points(self, pulse_width_m, maxima, fwhm, fwhm_left, fwhm_right):
        points = []
        fwhm * self.bin_width_m
        fwhm_left * self.bin_width_m
        fwhm_right * self.bin_width_m
        for i in range(len(maxima)):
            if fwhm[i] > pulse_width_m/2:
                if fwhm[i] < pulse_width_m:
                    points.append(maxima[i])
                else:
                    mid = (fwhm_right[i] + fwhm_left[i]) / 2.0
                    p_left = mid - pulse_width_m/2.0
                    p_right = mid + pulse_width_m/2.0
                    points.append(maxima[i] - (p_left - fwhm_left[i]))
                    points.append(maxima[i] + (p_right - fwhm_right[i]))
        return np.array(points)


    def get_points_echo_detection(self, pulse_width_m):
        data = np.asarray(self.data, dtype=float).copy()
        max_val = np.max(data)
        if max_val > 0:
            data /= max_val

        n = data.size
        if n < 3:
            return []

        data = self.data
        deriv = self.compute_derivative()
        dderiv = self.compute_second_derivative()
        maxima = self.find_local_maxima(deriv, dderiv)
        fwhm, p_left, p_right = self.find_fwhm(data, maxima)
        # self.visualize(title="Histogram, Derivatives, Local Maxima, and FWHM")
        return self.compute_points(pulse_width_m, maxima, fwhm, p_left, p_right)




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