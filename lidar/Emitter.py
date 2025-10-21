from raysect.core import AffineMatrix3D
from raysect.optical import UniformSurfaceEmitter, InterpolatedSF
from raysect.primitive import import_stl

from scene.SceneObject import SceneObject
import numpy as np
from scipy.special import gamma

def gaussian_line_sf(center_m: float,
                     total_power: float,
                     fwhm_m: float = 5e-9,
                     wl_min_m: float = 100e-9,
                     wl_max_m: float = 1100e-9,
                     step_m: float = 1e-9) -> InterpolatedSF:
    """Wavelength grid in meters; profile integrates (sum * step_m) to total_power [W]."""
    wavelengths = np.arange(wl_min_m, wl_max_m + step_m, step_m, dtype=float)
    sigma = float(fwhm_m) / 2.354820045
    profile = np.exp(-0.5 * ((wavelengths - center_m) / sigma) ** 2)
    area = profile.sum() * step_m
    if area > 0.0:
        profile *= (total_power / area)
    return InterpolatedSF(wavelengths, profile)  # wavelengths now in meters


class Emitter(SceneObject):
    wavelength_m: float
    pulse_energy_J: float
    pulse_width_s: float
    pulse_length_m: float
    pulse_average_power_W: float
    emission_angle_x_rad: float # defined as 1/e^2 gaussian beam
    emission_angle_y_rad: float # defined as 1/e^2 gaussian beam
    gaussian_exponent: float # exponent for super gaussian function
    c: float = 3e8


    def __init__(self, name, mesh_path, material, transform, wavelength_m, pulse_energy_J, pulse_width_s,
                 emission_angle_x_rad, emission_angle_y_rad, gaussian_exponent):
        super().__init__(name, mesh_path, material, transform)
        self.wavelength_m = wavelength_m
        self.pulse_energy_J = pulse_energy_J
        self.pulse_width_s = pulse_width_s
        self.pulse_length_m = (self.c * self.pulse_width_s)
        self.pulse_average_power_W = pulse_energy_J / pulse_width_s
        self.emission_angle_x_rad = emission_angle_x_rad
        self.emission_angle_y_rad = emission_angle_y_rad
        self.gaussian_exponent = gaussian_exponent


    def apply_vcsel_pulse_broadening(self, distances):
        """ Apply a random broadening due to VCSEL pulse shape. """
        sigma = self.pulse_length_m / 2.355 # Convert FWHM to standard devication
        gaussian_noise = np.random.normal(loc=0.0, scale=sigma, size=distances.shape)
        broadened = distances + gaussian_noise
        return np.clip(broadened, a_min=0.0, a_max=None)

    def gaussian_beam(self, z, rx, ry):
        """ Equation for power normalised gaussian beam f(x,y,z) """
        w0x = self.wavelength_m / (np.pi * self.emission_angle_x_rad)
        w0y = self.wavelength_m / (np.pi * self.emission_angle_y_rad)
        wx = w0x * np.sqrt(1 + ((self.wavelength_m * z) / (np.pi * w0x**2))**2)
        wy = w0y * np.sqrt(1 + ((self.wavelength_m * z) / (np.pi * w0y**2))**2)
        P = self.pulse_average_power_W
        p = self.gaussian_exponent
        A = (P * p ** 2) / ((2 ** (2 - 2 / p)) * (gamma(1 / p) ** 2) * wx * wy) # normalised by total power
        return A * np.exp(-2*((np.abs(rx)**p/wx**p) + (np.abs(ry)**p/wy**p)))

    def gaussian_beam_angular(self, theta_x, theta_y):
        """ Equation for power normalised gaussian beam f(theta_x, theta_y) """
        theta_0x = self.emission_angle_x_rad
        theta_0y = self.emission_angle_y_rad
        p = self.gaussian_exponent
        C = (p ** 2) / ((2 ** (2 - 2 / p)) * (gamma(1 / p) ** 2) * theta_0x * theta_0y) # normalised by total power
        return C * np.exp(-2*(np.abs(theta_x / theta_0x)**p + np.abs(theta_y / theta_0y)**p))

    def sample_gaussian_beam_angles(self, n_samples=1):
        """ Sampling random angles based on gaussian beam """
        theta_0x = self.emission_angle_x_rad
        theta_0y = self.emission_angle_y_rad
        p = self.gaussian_exponent

        # shape and scale for equivalent Gamma distribution
        shape = 1.0 / p
        scale = 0.5  # because exp(-2x)

        # sample absolute values using gamma distribution
        Xx = np.random.gamma(shape=shape, scale=scale, size=n_samples)
        Xy = np.random.gamma(shape=shape, scale=scale, size=n_samples)

        abs_theta_x = theta_0x * (Xx ** (1.0 / p))
        abs_theta_y = theta_0y * (Xy ** (1.0 / p))

        # assign random sign (symmetric)
        theta_x = abs_theta_x * np.random.choice([-1, 1], size=n_samples)
        theta_y = abs_theta_y * np.random.choice([-1, 1], size=n_samples)

        return theta_x, theta_y

    def check_integral(self):
        """ Check gaussian beam totals """
        x = np.linspace(-0.1, 0.1, 1000)
        y = np.linspace(-0.1, 0.1, 1000)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        z = 0.1
        sum = 0.0
        for xi in x:
            for yi in y:
                sum += dx * dy * self.gaussian_beam(z, xi, yi)
        print("pulse power ", self.pulse_average_power_W)
        print("sum ", sum)
        print("ratio ", sum/self.pulse_average_power_W)

        x = np.linspace(-self.emission_angle_x_rad, self.emission_angle_x_rad, 1000)
        y = np.linspace(-self.emission_angle_y_rad, self.emission_angle_y_rad, 1000)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        sum = 0.0
        for xi in x:
            for yi in y:
                sum += dx * dy * self.gaussian_beam_angular(xi, yi)
        print("sum ", sum)
        print("ratio ", sum)

    def to_raysect_emitter(self, world):
        spectral_radiance_sf = gaussian_line_sf(
            center_m=self.wavelength_m,
            total_power=self.pulse_average_power_W,
            fwhm_m=100e-9,  # was 100 nm
            wl_min_m=100e-9,
            wl_max_m=1100e-9,
            step_m=1e-9
        )

        emitter_material = UniformSurfaceEmitter(spectral_radiance_sf)
        emitter = import_stl(
            self.mesh_path,
            scaling=1,
            mode='binary',
            parent=world,
            transform=AffineMatrix3D(self.transform.matrix),
            material=emitter_material
        )
        return emitter
