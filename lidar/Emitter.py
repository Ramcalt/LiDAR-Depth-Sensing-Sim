from scene.SceneObject import SceneObject

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


