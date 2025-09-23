import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

# Gaussian spectral output function
peak_wavelength = 520e-9 # m
spectral_bandwidth = 35e-9 # m
sigma = spectral_bandwidth/(2*np.sqrt(2*np.log(2)))
def gaussian_spectral_distribution(wl):
    return np.exp(-((wl - peak_wavelength)**2)/(2*sigma**2))

# IV function (diode equation with offset)
q = 1.602e-19 # C
n = 1
k = 1.3806488e-23 # J/K
T = 300 # K
I0 = 0.03186018
n = q/(k*T*1.58581591)
offset = 3.06868791
# old: 0.00845294 2.09264807 3.80954629
# new: 0.03186018 1.58581591 3.06868791]
def iv_relationship(v):
    return np.clip(I0 * (np.exp((q*v)/(k*n*T) - offset) - 1), 0, None)

# Total luminous flux vs forward current equation
def luminous_flux(i):
    return 180 * i**0.71

# Normalised luminous efficiency vs wavelength (photopic = well lit conditions)
photopic_luminous_sensitivity = 683 # lm/W
def photopic_luminous_efficiency(wl):
    peak_wavelength_photopic = 555e-9 # m
    # spectral_bandwidth_photopic = 187e-9  # m
    # sigma_photopic = spectral_bandwidth_photopic / (2 * np.sqrt(2 * np.log(2)))
    # return np.exp(-((wl - peak_wavelength_photopic) ** 2) / (2 * sigma_photopic ** 2))
    sigma_photopic = 60e-9
    return np.exp(-0.5 * ((wl - peak_wavelength_photopic)/sigma_photopic)**2)

data = np.loadtxt("CIE_sle_photopic.csv", delimiter=",", skiprows=1)
wl_cie_data = data[:, 0]      # in nanometers
eff_cie_data = data[:, 1]     # relative efficiency in percent
def photopic_luminous_efficiency_empiracle(wl):
    return np.interp(
        wl*1e9,
        wl_cie_data,
        eff_cie_data,
        left=eff_cie_data[0],  # clamp below range to first data value
        right=eff_cie_data[-1]  # clamp above range to last data value
    )

# Normalised luminous efficiency vs wavelength (scotopic = poorly lit conditions)
scotopic_luminous_sensitivity = 1700 # lm/W
def scotopic_luminous_efficiency(wl):
    peak_wavelength_scotopic = 507e-9  # m
    spectral_bandwidth_scotopic = 173e-9  # m
    sigma_scotopic = spectral_bandwidth_scotopic / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-((wl - peak_wavelength_scotopic) ** 2) / (2 * sigma_scotopic ** 2))

def spectral_luminous_flux(i_f, wl):
    return (luminous_flux(i_f) * gaussian_spectral_distribution(wl))/(sigma * np.sqrt(2 * np.pi))

def spectral_radiant_flux(i_f, wl):
    return spectral_luminous_flux(i_f, wl) / (photopic_luminous_sensitivity * photopic_luminous_efficiency_empiracle(wl))

def luminous_flux_recalc(i_f, wl_start, wl_end):
    wl = np.linspace(wl_start, wl_end, 1000)
    d_wl = wl[1] - wl[0]
    sum = 0
    for w in wl:
        sum += spectral_luminous_flux(i_f, w) * d_wl
    return sum

def radiant_flux(i_f, wl_start, wl_end):
    wl = np.linspace(wl_start, wl_end, 1000)
    d_wl = wl[1] - wl[0]
    sum = 0
    for w in wl:
        sum += spectral_radiant_flux(i_f, w) * d_wl
    return sum

def plot_equations():
    v = np.linspace(0.1, 4.2, 100)
    i = iv_relationship(v)
    fig, axs = plt.subplots(2, 2)
    ax1, ax2, ax3, ax4 = axs[0,0], axs[0,1], axs[1,0], axs[1,1]
    ax1.plot(v, i)
    ax1.set_title('Plot of If vs Vd')
    ax1.set_xlabel('Voltage Drop (V)')
    ax1.set_ylabel('Forward Current (A)')
    ax1.yaxis.set_major_formatter(EngFormatter(unit='A'))
    ax1.xaxis.set_major_formatter(EngFormatter(unit='V'))
    ax1.grid(True)
    ax1.plot()

    phi_v = luminous_flux(i)
    ax2.plot(i, phi_v)
    ax2.set_title('Luminous Flux vs Forward Current')
    ax2.set_xlabel('Forward Current (A)')
    ax2.set_ylabel('Luminous Flux (lm)')
    ax2.yaxis.set_major_formatter(EngFormatter(unit='lm'))
    ax2.xaxis.set_major_formatter(EngFormatter(unit='A'))
    ax2.grid(True)
    ax2.plot()

    wl = np.linspace(400e-9, 650e-9, 100)
    intensity = 100*gaussian_spectral_distribution(wl)
    ax3.plot(wl, intensity)
    ax3.set_title('Plot of Relative Intensity vs Wavelength')
    ax3.set_xlabel('Wavelength (m)')
    ax3.set_ylabel('Relative Intensity (%)')
    ax3.yaxis.set_major_formatter(EngFormatter(unit='%'))
    ax3.xaxis.set_major_formatter(EngFormatter(unit='m'))
    ax3.grid(True)
    ax3.plot()

    wl = np.linspace(350e-9, 750e-9, 100)
    intensity = 100*photopic_luminous_efficiency(wl)
    ax4.plot(wl, intensity, label="Gaussian Approximation")
    intensity_tabulated = 100*photopic_luminous_efficiency_empiracle(wl)
    ax4.plot(wl, intensity_tabulated, label="Tabulated Interpolation", linestyle='--')
    ax4.set_title('Plot of Normalised Photopic Luminous Efficiency vs Wavelength')
    ax4.set_xlabel('Wavelength (m)')
    ax4.set_ylabel('Relative Intensity (%)')
    ax4.yaxis.set_major_formatter(EngFormatter(unit='%'))
    ax4.xaxis.set_major_formatter(EngFormatter(unit='m'))
    ax4.legend()
    ax4.grid(True)
    ax4.plot()

    plt.tight_layout()
    plt.show()

def plot_spectral_flux():
    i_f = 350e-3
    wl = np.linspace(350e-9, 750e-9, 200)
    phi_v = spectral_luminous_flux(i_f, wl)
    phi_e = spectral_radiant_flux(i_f, wl)

    fig, ax1 = plt.subplots()
    # Primary y-axis (luminous flux)
    ax1.plot(wl, phi_v, color='tab:blue', label='Luminous Flux')
    ax1.set_xlabel('Wavelength (m)')
    ax1.set_ylabel('Luminous Flux (lm)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.xaxis.set_major_formatter(EngFormatter(unit='m'))
    ax1.yaxis.set_major_formatter(EngFormatter(unit='lm'))
    # Secondary y-axis (radiant flux)
    ax2 = ax1.twinx()
    ax2.plot(wl, phi_e, color='tab:red', label='Radiant Flux')
    ax2.set_ylabel('Radiant Flux (W)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.yaxis.set_major_formatter(EngFormatter(unit='W'))
    # Title and grid
    plt.title('Spectral Luminous and Radiant Flux vs Wavelength at i_f = 350mA')
    ax1.grid(True)
    plt.show()

# Cosine law: this underestimates the spreading which overestimates the maximum power viewed
def viewing_angle_relative_intensity(theta):
    return np.cos(theta)

def viewing_radiant_flux(total_radiant_flux, axial_distance, aperture_diameter):
    max_theta = np.atan2(aperture_diameter, 2*axial_distance)
    theta_range = np.linspace(-max_theta, max_theta, 1000)
    d_theta = theta_range[1] - theta_range[0]
    sum = 0
    for theta in theta_range:
        sum += total_radiant_flux * viewing_angle_relative_intensity(theta) * d_theta
    return sum

# Assumes continuous wave t >= 0.25s
def angular_subtense_factor(axial_distance, emitter_diameter):
    alpha = np.atan2(emitter_diameter, 2*axial_distance) * 2
    alpha_min = 1.5e-3
    alpha_max = 100e-3
    alpha = np.min([alpha_max, alpha])
    alpha = np.max([alpha_min, alpha])
    return alpha / alpha_min

# Assumes t = 0.25s
class_I_radiant_flux_limit = 7e-4 * (0.25**0.75)
def required_class_I_PWM(radiant_flux, C6):
    limit = class_I_radiant_flux_limit * C6
    return class_I_radiant_flux_limit * C6 / radiant_flux

def print_power_values():
    v = 4.2
    i_f = iv_relationship(v)
    theta_v = luminous_flux(i_f)
    theta_e = radiant_flux(i_f, 200e-9, 800e-9)
    aperture_theta_e = viewing_radiant_flux(theta_e, 100e-3, 7e-3)
    C6 = angular_subtense_factor(100e-3, 2.6e-3)
    PWM = required_class_I_PWM(aperture_theta_e, C6)
    print(" - - - CDY11 LED - - - - - - - - - - - - - ")
    print("voltage drop: ", v,  " V | forward current: ", i_f, " A | power: ", v * i_f, " W")
    print("total luminous flux: ", theta_v, " lm")
    print("total radiant flux: ", theta_e*1000, " mW")
    print("radiant flux into aperture 7mm at distance 100mm: ", aperture_theta_e*1000, " mW")
    print("angular subtense factor (C6) of emitting aperture 2.6mm at distance 100mm: ", C6)
    print("required class I PWM: ", PWM*100, " %")
    print(" - - - - - - - - - - - - - - - - - - - - - ")

print_power_values()
plot_equations()
plot_spectral_flux()
