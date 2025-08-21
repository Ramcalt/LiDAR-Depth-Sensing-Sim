import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# How to combine 2D camera and LiDAR depth data
f_x0 = 0.1
f_y0 = 0.1
o_x0 = 200.0
o_y0 = 200.0
K = np.array([[f_x0, 0, o_x0], [0, f_y0, o_y0], [0, 0, 1]], dtype=float)
M_cam = np.array([[x, y, x**2 - y**2, 1] for x in range(256) for y in range(256)])
m_img = [K @ M_cam]


# -------- Figure & axes --------
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.28)  # space for sliders

# Fixed 100x100 viewing window
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.3)
ax.set_title("Parametric 2D curve with sliders (fixed 100×100)")

# Initial curve
(line,) = ax.plot(m_img[0], m_img[1], lw=2)

# -------- Slider controls --------
ax_f_x = plt.axes([0.15, 0.16, 0.70, 0.03])
ax_f_y = plt.axes([0.15, 0.11, 0.70, 0.03])
ax_o_x = plt.axes([0.15, 0.06, 0.70, 0.03])
ax_o_y = plt.axes([0.15, 0.01, 0.70, 0.03])

s_f_x = Slider(ax_f_x, "f_x", 1, 10.0, valinit=f_x0, valstep=0.1)
s_f_y = Slider(ax_f_y, "f_y", -1, 10.0, valinit=f_y0, valstep=0.1)
s_o_x = Slider(ax_o_x, "o_x", -256, 256, valinit=o_x0)
s_o_y = Slider(ax_o_y, "o_y", -256, 256, valinit=o_y0)
def update(_):
    f_x, f_y, o_x, o_y = s_f_x.val, s_f_y.val, s_o_x.val, s_o_y.val
    K = np.array([[f_x, 0, o_x], [0, f_y, o_y], [0, 0, 1]])
    m_img = K @ M_cam
    line.set_data(m_img[0], m_img[1])
    fig.canvas.draw_idle()

for s in (s_f_x, s_f_y, s_o_x, s_o_y):
    s.on_changed(update)

plt.show()