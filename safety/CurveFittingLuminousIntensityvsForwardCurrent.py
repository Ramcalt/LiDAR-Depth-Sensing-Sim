import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

phi_v_data = np.array([0, 50, 100, 160, 180])
i_data = np.array([0, 200e-3, 400e-3, 800e-3, 1000e-3])
lower_bounds = [1, 0.5]
upper_bounds = [500, 1.5]
def func(i, a, n):
    return a * i**n

params, covariance = curve_fit(func, i_data, phi_v_data, bounds=(lower_bounds, upper_bounds))
print("Fitted parameters:", params)
i_fit = np.linspace(min(i_data), max(i_data), 100)
phi_v_fit = func(i_fit, *params)
plt.scatter(i_data, phi_v_data, label='Data')
plt.plot(i_fit, phi_v_fit, label='Fitted Curve', color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve Fitting Example')
plt.show()

