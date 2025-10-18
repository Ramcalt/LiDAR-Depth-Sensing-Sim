import numpy as np
from raysect.optical import Lambert, srgb_to_ciexyz, InterpolatedSF
import trimesh

def _srgb_to_linear(c: float) -> float:
    """Convert one sRGB component in [0,1] to linear RGB."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def srgb_to_reflectance_sf(r: float, g: float, b: float,
                           wl_min: int = 380, wl_max: int = 780, step: int = 5) -> InterpolatedSF:
    """
    Approximate a reflectance spectrum from an sRGB colour using three Gaussian bases.
    Returns an InterpolatedSF suitable for Lambert().
    """
    # 1) Linearise sRGB (you pass r,g,b in [0,1])
    R = _srgb_to_linear(r)
    G = _srgb_to_linear(g)
    B = _srgb_to_linear(b)

    # 2) Wavelength grid (nm)
    wavelengths = np.arange(wl_min, wl_max + step, step, dtype=float)

    # 3) Simple basis functions (centres/widths chosen for a plausible diffuse look)
    def gaussian(center, width):
        return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

    basis_r = gaussian(610.0, 40.0)  # red lobe
    basis_g = gaussian(545.0, 35.0)  # green lobe
    basis_b = gaussian(460.0, 30.0)  # blue lobe

    # Optional broadband “body” term to avoid overly saturated, spiky spectra
    body = gaussian(560.0, 120.0)

    # 4) Mix and clamp to [0,1]
    reflectance = (R * basis_r + G * basis_g + B * basis_b + 0.05 * body)
    reflectance = np.clip(reflectance, 0.0, 1.0)

    # 5) Build spectral function (no extrapolation outside grid)
    return InterpolatedSF(wavelengths, reflectance)

class Material:
    roughness: float
    specular: float
    albedo: float
    colour: np.ndarray

    def __init__(self, roughness, specular, albedo, colour):
        self.roughness = roughness
        self.specular = specular
        self.albedo = albedo
        self.colour = colour

    def __post_init__(self):
        if not (0.0 <= self.roughness <= 1.0):
            raise ValueError("Roughness must be in [0, 1].")
        if not (0.0 <= self.specular <= 1.0):
            raise ValueError("Specular must be in [0, 1].")
        if not (0.0 <= self.albedo <= 1.0):
            raise ValueError("Albedo must be in [0, 1].")
        r, g, b = self.colour
        for c in (r, g, b):
            if not (0.0 <= c <= 1.0):
                raise ValueError("Colour components must be in [0, 1].")

    def to_trimesh_material(self):
        return trimesh.visual.material.PBRMaterial(
            basicColorFactor=self.colour,
            specular=self.specular,
            roughness=self.roughness
        )

    def to_raysect_material(self):
        r, g, b = float(self.colour[0]), float(self.colour[1]), float(self.colour[2])
        spectral_reflectance = srgb_to_reflectance_sf(r, g, b)
        return Lambert(spectral_reflectance)