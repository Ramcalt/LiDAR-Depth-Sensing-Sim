from raysect.core import AffineMatrix3D
from raysect.optical import UniformSurfaceEmitter, InterpolatedSF
from raysect.primitive import import_stl

from scene.SceneObject import SceneObject
import numpy as np

def gaussian_line_sf(center_nm: float,
                     total_power: float,
                     fwhm_nm: float = 5.0,
                     wl_min: int = 380,
                     wl_max: int = 780,
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
    emission_angle_x_rad: float
    emission_angle_y_rad: float

    def __init__(self, name, mesh_path, material, transform, wavelength, pulse_energy, pulse_width, emission_angle_x_rad, emission_angle_y_rad):
        super().__init__(name, mesh_path, material, transform)
        self.wavelength = wavelength
        self.pulse_energy = pulse_energy
        self.pulse_width = pulse_width
        self.emission_angle_x_rad = emission_angle_x_rad
        self.emission_angle_y_rad = emission_angle_y_rad

    def to_raysect_emitter(self, world):
        power = self.pulse_energy / self.pulse_width  # units depend on your scene calibration

        spectral_radiance_sf = gaussian_line_sf(
            center_nm=self.wavelength,
            total_power=power,
            fwhm_nm=0.5
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
