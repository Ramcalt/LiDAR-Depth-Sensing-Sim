from __future__ import annotations
from math import sin, cos, sqrt, radians, tan
import matplotlib.pyplot as plt
from typing import Iterable, Tuple, Optional, Sequence

def draw_circles(
    circles: Iterable[CircleArgs],
    *,
    ax: Optional[plt.Axes] = None,
    fill: bool = False,
    colors: Optional[Sequence[str]] = None,
    linewidth: float = 2.0,
    alpha: float = 0.35,
    grid: bool = True,
):
    circles = list(circles)
    if not circles:
        raise ValueError("No circles provided.")

    # Prepare axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Draw circles
    for i, (r, ox, oy) in enumerate(circles):
        kw = dict(fill=fill, linewidth=linewidth)
        if colors:
            kw["edgecolor"] = colors[i % len(colors)]
            if fill:
                kw["facecolor"] = colors[i % len(colors)]
                kw["alpha"] = alpha
        else:
            if fill:
                kw["alpha"] = alpha
        ax.add_patch(plt.Circle((ox, oy), r, **kw))

    # Compute global bounds so overlaps are fully visible
    min_x = min(ox - r for r, ox, oy in circles)
    max_x = max(ox + r for r, ox, oy in circles)
    min_y = min(oy - r for r, ox, oy in circles)
    max_y = max(oy + r for r, ox, oy in circles)

    # Add a small padding proportional to the largest radius
    max_r = max(r for r, _, _ in circles)
    pad = 0.1 * max_r + 2.0
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)

    ax.set_aspect("equal", adjustable="box")
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.5)

    return fig, ax


# Assumptions: field of view is a cone, not oblique or ovular
class Cone:
    def __init__(self, x: float, y: float, z: float, theta: float):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta

    def circleArgs(self, z):
        zi = z - self.z

        return (z-self.z)*tan(self.theta), self.x, self.y

    @staticmethod
    def intersectionCircleArgs(a: Cone, b: Cone, z: float):
        # reference a to the origin
        xi = b.x - a.x
        yi = b.y - a.y
        zi = b.z - a.z
        alpha = a.theta
        beta = b.theta
        offset_x = -((xi*cos(beta)**2)/(cos(alpha)**2 - cos(beta)**2))
        offset_y = -((yi*cos(beta)**2)/(cos(alpha)**2 - cos(beta)**2))
        radius = sqrt(
            z**2 * sin(alpha)**2
            -(z-zi)**2 * sin(beta)**2
            -xi**2 * cos(beta)**2
            -yi**2 * cos(beta)**2
            -(
                (xi**2 * cos(beta)**4 + yi**2 * cos(beta)**4)
                / (cos(alpha)**2 - cos(beta)**2)**3
            )
        )
        return (radius, offset_x, offset_y)

camera = Cone(0, 0, 0, radians(40))
lidar = Cone(100, 10, 20, 22.5)
draw_circles([
    Cone.intersectionCircleArgs(camera, lidar, 100),
    camera.circleArgs(100),
    lidar.circleArgs(100)],
    fill=True,
    colors=["tab:blue", "tab:orange", "tab:green"]
)
plt.show()