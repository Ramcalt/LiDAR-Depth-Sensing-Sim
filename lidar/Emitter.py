from raysect.core import AffineMatrix3D
from raysect.optical import UniformSurfaceEmitter, InterpolatedSF
from raysect.primitive import import_stl

from scene.SceneObject import SceneObject
import numpy as np
from scipy.special import gamma

def gaussian_line_sf(center_nm: float,
                     total_power: float,
                     fwhm_nm: float = 5.0,
                     wl_min: int = 100,
                     wl_max: int = 1100,
                     step_nm: int = 1) -> InterpolatedSF:
    """
    Build a narrowband spectral function with Gaussian shape centered at `center_nm`.
    The discrete area (sum * step_nm) is normalised to `total_power`.

    Returns an InterpolatedSF suitable for UniformSurfaceEmitter.
    """
    wavelengths = np.arange(wl_min, wl_max + step_nm, step_nm, dtype=float)
    sigma = float(fwhm_nm) / 2.354820045  # FWHM -> sigma
    profile = np.exp(-0.5 * ((wavelengths - center_nm) / sigma) ** 2)

    # Normalise so that the discrete integral equals total_power
    area = profile.sum() * step_nm
    if area > 0.0:
        profile *= (total_power / area)

    # No extrapolation outside the defined grid
    return InterpolatedSF(wavelengths, profile)


class Emitter(SceneObject):
    wavelength: float
    pulse_energy: float
    pulse_width: float
    pulse_length: float
    pulse_average_power: float
    emission_angle_x_rad: float # defined as 1/e^2 gaussian beam
    emission_angle_y_rad: float # defined as 1/e^2 gaussian beam
    gaussian_exponent: float # exponent for super gaussian function
    c: float = 3e8


    def __init__(self, name, mesh_path, material, transform, wavelength, pulse_energy, pulse_width,
                 emission_angle_x_rad, emission_angle_y_rad, gaussian_exponent):
        super().__init__(name, mesh_path, material, transform)
        self.wavelength = wavelength
        self.pulse_energy = pulse_energy
        self.pulse_width = pulse_width
        self.pulse_length = (self.c * self.pulse_width)
        self.pulse_average_power = pulse_energy / pulse_width
        self.emission_angle_x_rad = emission_angle_x_rad
        self.emission_angle_y_rad = emission_angle_y_rad
        self.gaussian_exponent = gaussian_exponent


    def apply_vcsel_pulse_broadening(self, distances):
        """ Apply a random broadening due to VCSEL pulse shape. """
        random_pulse_delay = np.random.uniform(0, self.pulse_length, size=distances.shape)
        return distances + random_pulse_delay

    def gaussian_beam(self, z, rx, ry):
        """ Equation for power normalised gaussian beam f(x,y,z) """
        w0x = self.wavelength / (np.pi * self.emission_angle_x_rad)
        w0y = self.wavelength / (np.pi * self.emission_angle_y_rad)
        wx = w0x * np.sqrt(1 + ((self.wavelength * z) / (np.pi * w0x**2))**2)
        wy = w0y * np.sqrt(1 + ((self.wavelength * z) / (np.pi * w0y**2))**2)
        P = self.pulse_average_power
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
        print("pulse power ", self.pulse_average_power)
        print("sum ", sum)
        print("ratio ", sum/self.pulse_average_power)

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
        power = self.pulse_energy / self.pulse_width  # units depend on your scene calibration

        spectral_radiance_sf = gaussian_line_sf(
            center_nm=self.wavelength,
            total_power=power,
            fwhm_nm=100
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
