from dataclasses import dataclass
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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

    # - - - - - - - - - - UTIL - - - - - - - - - - #
    # central difference
    def compute_derivative(self, data: np.ndarray) -> np.ndarray:
        N = len(data)
        deriv = np.zeros_like(data)
        deriv[0] = data[1] - data[0] # fwd diff for first point
        deriv[-1] = data[-1] - data[-2] # bwd diff for last point
        for i in range(1, N-1): # central diff for interior points
            deriv[i] = (data[i+1] - data[i-1])/2.0
        return deriv

    # second order central difference
    def compute_second_derivative(self, data: np.ndarray) -> np.ndarray:
        N = len(data)
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
    def find_local_maxima(self, data: np.ndarray, deriv: np.ndarray, dderiv: np.ndarray, min_amp = 0.5):
        N = len(data)
        maxima = []
        for i in range(1, N-1):
            if deriv[i - 1] > 0.0 >= deriv[i]: # ensure zero crossing # and (dderiv[i-1] < 0.0 or dderiv[i] < 0.0):
                t = deriv[i-1] / (deriv[i-1] - deriv[i])
                if self.lerp(dderiv[i-1], dderiv[i], t) < 0.0: # ensure concave down
                    if self.lerp(data[i - 1], data[i], t) > min_amp: # minimum prominence
                        maxima.append((i-1) + t)
        return np.array(maxima)

    # - - - - - - - - - - ECHO DETECTION - - - - - - - - - - #
    # find fwhm
    def find_fwhm(self, data: np.ndarray, maxima: np.ndarray, threshold = 0.50, sp_threshold = 0.98):
        N = self.bin_count
        fwhm = []
        p_left = []
        p_right = []
        single_points = []
        for maximum in maxima:
            # Compute indices of points that pass threshold (default 50%)
            i_left = int(np.floor(maximum))
            i_right = int(np.ceil(maximum))
            half_maximum_val = threshold * self.lerp(data[i_left], data[i_right], maximum - i_left)

            while i_left >= 0 and data[i_left] >= half_maximum_val:
                i_left -= 1
            while i_right < N and data[i_right] >= half_maximum_val:
                i_right += 1

            # Compute left side crossing
            if i_left < 0:
                # TODO: estimate based on derivatives
                # Half-maximum lies before index 0 -> extrapolate using (0,1)
                # if N >= 2 and data[1] != data[0]:
                #     x_left = 0.0 + (half_maximum_val - data[0]) / (data[1] - data[0])
                # else:
                #    x_left = 0.0  # degenerate fallback
                x_left = 0.0
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
                # if N >= 2 and data[N - 1] != data[N - 2]:
                #     x_right = (N - 1) + (half_maximum_val - data[N - 1]) / (data[N - 1] - data[N - 2])
                # else:
                #     x_right = float(N - 1)  # degenerate fallback
                x_right = float(N - 1)
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

        for maximum in maxima:
            # Compute indices of points that pass sp_threshold (default 95%)
            i_left = int(np.floor(maximum))
            i_right = int(np.ceil(maximum))
            half_maximum_val = sp_threshold * self.lerp(data[i_left], data[i_right], maximum - i_left)

            while i_left >= 0 and data[i_left] >= half_maximum_val:
                i_left -= 1

            # Compute left side crossing
            if i_left < 0:
                # TODO: estimate based on derivatives
                # Half-maximum lies before index 0 -> extrapolate using (0,1)
                # if N >= 2 and data[1] != data[0]:
                #     x_left = 0.0 + (half_maximum_val - data[0]) / (data[1] - data[0])
                # else:
                #    x_left = 0.0  # degenerate fallback
                x_left = 0.0
            else:
                # Crossing is between i_left and i_left+1
                j0, j1 = i_left, i_left + 1
                y0, y1 = data[j0], data[j1]
                if y1 != y0:
                    x_left = j0 + (half_maximum_val - y0) / (y1 - y0)
                else:
                    x_left = j0 + 0.5  # flat segment fallback

            single_points.append(x_left)

        return np.array(fwhm), np.array(p_left), np.array(p_right), np.array(single_points)

    def compute_points(self, pulse_width_m, maxima, fwhm, fwhm_left, fwhm_right, single_points):
        points = []
        pulse_width_bins = pulse_width_m / self.bin_width_m
        for i in range(len(maxima)):
            if fwhm[i] > pulse_width_bins/2:
                if fwhm[i] < pulse_width_bins:
                    points.append(single_points[i])
                else:
                    mid = (fwhm_right[i] + fwhm_left[i]) / 2.0
                    p_left = mid - pulse_width_bins/2.0
                    p_right = mid + pulse_width_bins/2.0
                    points.append(maxima[i] - (p_left - fwhm_left[i]))
                    points.append(maxima[i] + (fwhm_right[i] - p_right))
        return np.array(points)

    def get_points_echo_detection(self, pulse_width_m, theta_x=0, theta_y =0, offset=-0.0375):
        data = np.asarray(self.data, dtype=float).copy()
        if np.max(data) > 0:
            data /= np.max(data)
        if data.size < 3:
            return []

        deriv = self.compute_derivative(data)
        dderiv = self.compute_second_derivative(data)
        maxima = self.find_local_maxima(data, deriv, dderiv)
        fwhm, p_left, p_right, single_points = self.find_fwhm(data, maxima)
        points = self.compute_points(pulse_width_m, maxima, fwhm, p_left, p_right, single_points)
        correction = np.cos(np.sqrt(theta_x**2 + theta_y**2))
        #correction = 1
        return points * self.bin_width_m * correction + offset

    # - - - - - - - - - - DEPTH - - - - - - - - - - #
    @staticmethod
    def compute_depth(points, filtered=True):
        points = np.array(points)
        if filtered:
            lower, upper = np.percentile(points, [5, 95])
            filtered_points = points[(points >= lower) & (points <= upper)]
            return filtered_points.max() - filtered_points.min()
        else:
            return points.max() - points.min()

    # - - - - - - - - - - DECONVOLUTION - - - - - - - - - - #
    @staticmethod
    def wiener_deconv(signal, kernel, regularization=1):
        # Ensure kernel is centered and same length as signal
        kernel = np.asarray(kernel, dtype=float)
        signal = np.asarray(signal, dtype=float)
        N = len(signal)

        # Pad or truncate kernel to match signal length
        if len(kernel) < N:
            pad = (N - len(kernel)) // 2
            kernel = np.pad(kernel, (pad, N - len(kernel) - pad))
        elif len(kernel) > N:
            kernel = kernel[:N]

        # Optionally center the kernel (use fftshift if needed)
        kernel = np.fft.ifftshift(kernel)

        # FFT-based Wiener deconvolution
        signal_f = np.fft.rfft(signal)
        kernel_f = np.fft.rfft(kernel)
        deconv_f = (signal_f * np.conj(kernel_f)) / (np.abs(kernel_f) ** 2 + regularization)
        deconv = np.fft.irfft(deconv_f, N)

        return deconv

    def get_points_deconv(self, theta_x=0, theta_y =0, offset=-0.0375):
        data = np.asarray(self.data, dtype=float).copy()
        if np.max(data) > 0:
            data /= np.max(data)
        if data.size < 3:
            return []
        # KERNEL IS HISTOGRAM [0, 2] FROM SIM.RUN_FIND_KERNEL
        # kernel = np.array([0.005,0.016,0.066,0.211,0.519,0.882,1.000,0.873,0.556,0.243,0.079,0.018,0.003])
        # KERNEL IS HISTOGRAM [3, 7] FROM SIM.RUN_FIND_KERNEL
        # [0.009,0.018,0.037,0.078,0.136,0.237,0.395,0.557,0.733,0.908,0.985,1.000,0.949,0.821,0.655,0.451,0.318,0.170,0.093,0.057,0.034,0.008,0.005,0.001,0.001,0.000,0.000,0.000,0.000,0.000,0.000,0.000]
        # [0 7] [0.006,0.011,0.040,0.062,0.112,0.303,0.679,0.863,0.931,0.977,1.000,0.906,0.682,0.482,0.353,0.194,0.112,0.062,0.040,0.011,0.006]
        kernel = np.array([0.006,0.011,0.040,0.062,0.112,0.303,0.679,0.863,0.931,0.977,1.000,0.906,0.682,0.482,0.353,0.194,0.112,0.062,0.040,0.011,0.006])
        deconv = self.wiener_deconv(data, kernel/np.sum(kernel))
        deconv /= np.max(deconv)
        deriv = self.compute_derivative(deconv)
        dderiv = self.compute_second_derivative(deconv)
        maxima = self.find_local_maxima(deconv, deriv, dderiv)
        correction = np.cos(np.sqrt(theta_x**2 + theta_y**2))
        #correction = 1
        return maxima * self.bin_width_m * correction + offset

    # - - - - - - - - - - WAVFORM DECOMPOSITION - - - - - - - - - - #
    def get_decomp_estimate_maxima(self, data):
        deriv = self.compute_derivative(data)
        dderiv = self.compute_second_derivative(data)
        maxima = self.find_local_maxima(data, deriv, dderiv, min_amp=0.1)
        if len(maxima) <= 0:
            idx = int(np.argmax(data))
            A1 = float(data[idx])
            mu1 = idx * self.bin_width_m
            initial_guess =  [A1, mu1, 0.1 * A1, mu1 + self.bin_width_m * 5]
        elif len(maxima) == 1:
            max1_left = np.floor(maxima[0])
            max1_right = np.ceil(maxima[0])
            A1 = self.lerp(data[int(max1_left)], data[int(max1_right)], maxima[0] - max1_left)
            mu1 = maxima[0] * self.bin_width_m
            initial_guess = [A1, mu1, 0, 0]
        else:
            amps = [np.interp(m, np.arange(len(data)), data) for m in maxima]
            maxima = list(zip(maxima, amps))
            maxima_sorted = sorted(maxima, key=lambda m: m[1], reverse=True)
            max1_left = np.floor(maxima_sorted[0][0])
            max1_right = np.ceil(maxima_sorted[0][0])
            max2_left = np.floor(maxima_sorted[1][0])
            max2_right = np.ceil(maxima_sorted[1][0])
            A1 = self.lerp(data[int(max1_left)], data[int(max1_right)], maxima_sorted[0][0] - max1_left)
            A2 = self.lerp(data[int(max2_left)], data[int(max2_right)], maxima_sorted[1][0] - max2_left)
            mu1 = maxima_sorted[0][0] * self.bin_width_m
            mu2 = maxima_sorted[1][0] * self.bin_width_m
            initial_guess = [A1, mu1, A2, mu2]
        return initial_guess

    def get_decomp_estimate_deconv(self, data):
        estimated_points = self.get_points_deconv()
        if len(estimated_points) <= 0:
            A1 = np.argmax(data)
            mu1 = np.argmax(data) * self.bin_width_m + self.time_start
            initial_guess =  [A1, mu1, 0.1 * A1, mu1 + self.bin_width_m * 5]
        elif len(estimated_points) == 1:
            initial_guess = [1, estimated_points[0], 0, 0]
        else:
            initial_guess = [0.5, estimated_points[0], 0.5, estimated_points[1]]
        return initial_guess

    def get_points_wav_decomp(self, pulse_width_m, theta_x=0, theta_y=0, offset=-0.0375):
        data = np.asarray(self.data, dtype=float).copy()
        if np.max(data) > 0:
            data /= np.max(data)
        if data.size < 3:
            return []

        # setup model and estimate
        N = len(data)
        x_m = np.arange(N) * self.bin_width_m
        initial_guess = self.get_decomp_estimate_maxima(data)

        sigma = pulse_width_m / 2.355
        def model(x, A1, mu1, A2, mu2):
            return (
                A1 * np.exp(-0.5 * ((x - mu1) / sigma) ** 2) +
                A2 * np.exp(-0.5 * ((x - mu2) / sigma) ** 2)
            )

        # curve fit
        lower_bounds = [0, 0, 0, 0]
        upper_bounds = [1, np.inf, 1, np.inf]
        try:
            popt, pcov = curve_fit(model, x_m, data, p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=2000)
        except RuntimeError:
            return np.array([])

        # Quick amplitude based filtering
        correction = np.cos(np.sqrt(theta_x ** 2 + theta_y ** 2))

        A1, mu1, A2, mu2 = popt
        correction = np.cos(np.sqrt(theta_x ** 2 + theta_y ** 2))
        peaks = []

        # SORT
        if mu1 > mu2:
            mu1, mu2 = mu2, mu1
            A1, A2 = A2, A1
        # COLLECT PEAKS
        if A1 > 0.50 and mu1 > 0 and A2 > 0.50 and mu2 > 0: # if we have two strong peaks append both
            if np.abs(mu2 - mu1) > 0.5: # unreasonable, discard mu2
                peaks.append(mu1 * correction + offset)
            else:
                peaks.append(mu1 * correction + offset)
                peaks.append(mu2 * correction + offset)
        elif mu1 > 0 and A1 > 0.25: # otherwise append one
            peaks.append(mu1 * correction + offset)
        elif mu2 > 0 and A2 > 0.25:
            peaks.append(mu2 * correction + offset)
        return np.array(sorted(peaks))


    # ========================== VISUALISATION ======================= #
    def visualise_get_points_wav_decomp(self, pulse_width_m):
        """Visualise the result of the waveform decomposition method."""
        data = np.asarray(self.data, dtype=float).copy()
        if np.max(data) > 0:
            data /= np.max(data)
        if data.size < 3:
            return []

        # setup model and estimate
        N = len(data)
        x_m = np.arange(N) * self.bin_width_m
        initial_guess = self.get_decomp_estimate_maxima(data)

        sigma = pulse_width_m / 2.355

        def model(x, A1, mu1, A2, mu2):
            return (
                    A1 * np.exp(-0.5 * ((x - mu1) / sigma) ** 2) +
                    A2 * np.exp(-0.5 * ((x - mu2) / sigma) ** 2)
            )

        # curve fit
        lower_bounds = [0, 0, 0, 0]
        upper_bounds = [1, np.inf, 1, np.inf]
        try:
            popt, pcov = curve_fit(model, x_m, data, p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=2000)
        except RuntimeError:
            print("Curve fitting failed — plotting raw histogram only.")
            plt.figure(figsize=(10, 4))
            plt.plot(x_m, data, label="Histogram (normalised)")
            plt.xlabel("Range (m)")
            plt.ylabel("Amplitude")
            plt.title("Waveform Decomposition (fit failed)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()
            return

        A1, mu1, A2, mu2 = popt
        fit = model(x_m, *popt)
        comp1 = A1 * np.exp(-0.5 * ((x_m - mu1) / sigma) ** 2)
        comp2 = A2 * np.exp(-0.5 * ((x_m - mu2) / sigma) ** 2)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(x_m, data, 'k-', label="Measured waveform")
        ax.plot(x_m, fit, 'r-', label="Fitted sum (2-Gaussian)")
        ax.plot(x_m, comp1, 'b--', label=f"Component 1 (μ={mu1:.3f} m)")
        ax.plot(x_m, comp2, 'g--', label=f"Component 2 (μ={mu2:.3f} m)")

        # Mark peak centers
        ax.axvline(mu1, color='b', linestyle=':', alpha=0.6)
        ax.axvline(mu2, color='g', linestyle=':', alpha=0.6)
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Amplitude (norm.)")
        ax.set_title("Waveform Decomposition — Two Fixed-Width Gaussians")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def visualise_get_points_deconv(self, pulse_width_m):
        data = np.asarray(self.data, dtype=float).copy()
        if np.max(data) > 0:
            data /= np.max(data)
        if data.size < 3:
            return []
        kernel = np.array([0.006,0.011,0.040,0.062,0.112,0.303,0.679,0.863,0.931,0.977,1.000,0.906,0.682,0.482,0.353,0.194,0.112,0.062,0.040,0.011,0.006])
        deconv = self.wiener_deconv(data, kernel/np.sum(kernel))
        deconv /= np.max(deconv)
        deriv = self.compute_derivative(deconv)
        dderiv = self.compute_second_derivative(deconv)
        maxima = self.find_local_maxima(deconv, deriv, dderiv)
        points = maxima * self.bin_width_m
        # TODO visualise_get_points_deconv plot
        # TODO: visualisation with matplotlib
        # a) plot histogram and the deconvolved histogram and the detected maxima
        # b) plot deconvolved derivative (separate plot under)
        # c) plot deconvolved dderivative (separate plot under)
        # X-axis in meters (offset by time_start which is in meters here)
        # Convert x-axis to physical distance (meters)
        # X-axis in meters
        N = data.size
        x_bins = np.arange(N, dtype=float)
        x_m = self.time_start + x_bins * self.bin_width_m
        maxima_m = self.time_start + np.asarray(maxima) * self.bin_width_m if maxima.size else np.array([])

        # === PLOTS ===
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        ax_h, ax_d1, ax_d2 = axs

        # (a) Original + Deconvolved histograms
        ax_h.plot(x_m, data, label="Original Histogram", alpha=0.6)
        ax_h.plot(x_m, deconv, label="Deconvolved Signal", linewidth=1.2)
        ax_h.set_ylabel("Amplitude (norm.)")
        ax_h.set_title("Deconvolution-Based Peak Detection")

        # Mark detected maxima
        if maxima_m.size:
            ax_h.scatter(maxima_m, np.interp(maxima_m, x_m, deconv),
                         color="red", marker="x", s=60, label="Detected Maxima")

            # Optional pulse width visualization band
            pw = float(pulse_width_m)
            for xm in maxima_m:
                ax_h.axvspan(xm - 0.5 * pw, xm + 0.5 * pw, alpha=0.05, color="gray")

        ax_h.grid(True, alpha=0.3)
        ax_h.legend(loc="upper right")

        # (b) First derivative
        ax_d1.plot(x_m, deriv, label="First Derivative", color="tab:orange")
        ax_d1.set_ylabel("d/dx (shape)")
        ax_d1.grid(True, alpha=0.3)
        ax_d1.legend(loc="upper right")

        # (c) Second derivative
        ax_d2.plot(x_m, dderiv, label="Second Derivative", color="tab:green")
        ax_d2.set_ylabel("d^2/dx^2 (shape)")
        ax_d2.set_xlabel("Range (m)")
        ax_d2.grid(True, alpha=0.3)
        ax_d2.legend(loc="upper right")

        plt.tight_layout()
        plt.show()


    def visualise_get_points_echo_detection(self, pulse_width_m):
        data = np.asarray(self.data, dtype=float).copy()
        if np.max(data) > 0:
            data /= np.max(data)
        if data.size < 3:
            return []
        deriv = self.compute_derivative(data)
        dderiv = self.compute_second_derivative(data)
        maxima = self.find_local_maxima(data, deriv, dderiv)
        fwhm, p_left, p_right, single_points = self.find_fwhm(data, maxima)
        points = self.compute_points(pulse_width_m, maxima, fwhm, p_left, p_right, single_points)
        N = data.size
        x_bins = np.arange(N, dtype=float)
        x_m = self.time_start + x_bins * self.bin_width_m

        # Convert features to meters
        maxima_m = self.time_start + np.asarray(maxima) * self.bin_width_m if maxima.size else np.array([])
        p_left_m = self.time_start + np.asarray(p_left) * self.bin_width_m if p_left.size else np.array([])
        p_right_m = self.time_start + np.asarray(p_right) * self.bin_width_m if p_right.size else np.array([])
        points_m = self.time_start + np.asarray(points) * self.bin_width_m if points.size else np.array([])

        # Recompute half-maximum values for drawing FWHM bars (per-peak)
        # Use the normalised histogram 'data' and the fractional maxima indices.
        half_vals = []
        if maxima.size:
            for m in maxima:
                i0 = int(np.floor(m))
                i1 = min(i0 + 1, N - 1)
                t = m - i0
                peak_val = (1.0 - t) * data[i0] + t * data[i1]
                half_vals.append(0.5 * peak_val)
        half_vals = np.asarray(half_vals) if maxima.size else np.array([])

        # Create figure with three stacked axes
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        ax_h, ax_d1, ax_d2 = axs

        # (a) Histogram with annotations
        ax_h.plot(x_m, data, label="Histogram (normalised)")
        ax_h.set_ylabel("Amplitude (norm.)")
        ax_h.set_title("Echo Detection Visualisation")

        # Mark maxima
        if maxima_m.size:
            ax_h.scatter(maxima_m, np.interp(maxima_m, x_m, data), marker="x", s=60, label="Maxima")

        # Mark FWHM left/right bounds and draw half-max bars
        if p_left_m.size and p_right_m.size and half_vals.size:
            # Vertical lines at FWHM bounds
            ax_h.vlines(p_left_m, ymin=0, ymax=np.interp(p_left_m, x_m, data), linestyles="dotted", label="FWHM left")
            ax_h.vlines(p_right_m, ymin=0, ymax=np.interp(p_right_m, x_m, data), linestyles="dotted",
                        label="FWHM right")

            # Horizontal half-maximum segments between p_left and p_right for each peak
            for xl, xr, hv in zip(p_left_m, p_right_m, half_vals):
                if np.isfinite(xl) and np.isfinite(xr):
                    ax_h.hlines(hv, xmin=xl, xmax=xr, linestyles="dashdot", linewidth=1.0)

        # Mark final points
        if points_m.size:
            ax_h.scatter(points_m, np.interp(points_m, x_m, data), marker="o", facecolors="none", edgecolors="green",
                         s=70, label="Output points")

        # Show pulse width band around each peak center (optional visual cue)
        if maxima_m.size:
            pw = float(pulse_width_m)
            for xm in maxima_m:
                ax_h.axvspan(xm - 0.5 * pw, xm + 0.5 * pw, alpha=0.05)

        ax_h.grid(True, alpha=0.3)
        ax_h.legend(loc="upper right")

        # (b) First derivative (shape only)
        ax_d1.plot(x_m, deriv, label="First derivative")
        ax_d1.set_ylabel("d/dx (shape)")
        ax_d1.grid(True, alpha=0.3)
        ax_d1.legend(loc="upper right")

        # (c) Second derivative (shape only)
        ax_d2.plot(x_m, dderiv, label="Second derivative")
        ax_d2.set_ylabel("d²/dx² (shape)")
        ax_d2.set_xlabel("Range (m)")
        ax_d2.grid(True, alpha=0.3)
        ax_d2.legend(loc="upper right")

        plt.tight_layout()
        plt.show()


# =========================
# Example usage (synthetic)
# =========================
if __name__ == "__main__":
    # Synthetic example: two Gaussian-like peaks sampled on unit steps
    np.random.seed(0)
    N = 200
    bin_width_m = 0.0375  # 2 cm per bin (example)
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
    print(h.get_points_echo_detection(0.15))
    h.visualise_get_points_echo_detection(0.15)
    h.visualise_get_points_deconv(0.15)
    h.visualise_get_points_wav_decomp(0.15)
