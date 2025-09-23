import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

v_data = np.array([0, 2.5, 3, 3.4, 4])
i_data = np.array([0, 10e-3, 100e-3, 350e-3, 800e-3])

def diode_IV(V, a, b, c):
    return a * (np.exp(V*b - c) - 1)

params, covariance = curve_fit(diode_IV, v_data, i_data)
print("Fitted parameters:", params)
v_fit = np.linspace(min(v_data), max(v_data), 100)
i_fit = diode_IV(v_fit, *params)
plt.scatter(v_data, i_data, label='Data')
plt.plot(v_fit, i_fit, label='Fitted Curve', color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve Fitting Example')
plt.show()

