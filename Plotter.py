import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import tight_layout
from lidar.Histogram import Histogram

class Plotter:
    def __init__(self):
        pass

    @staticmethod
    def plot_hist(hist):
        edges = np.linspace(0, hist.bin_count * hist.bin_width, hist.bin_count + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)

        plt.figure()
        plt.bar(centers, hist.data, width=widths, align='center')
        plt.xlabel('bin')
        plt.ylabel('value')
        plt.title('Pre-binned histogram')
        plt.show()

    @staticmethod
    def plot_hist_arr(hists, rows, cols):
        # Create shared-axis subplot grid
        fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True,
                                figsize=(3.2 * cols, 2.6 * rows))
        # Normalize axs to a 2D array for consistent indexing
        if rows == 1 and cols == 1:
            axs = np.array([[axs]])
        elif rows == 1 or cols == 1:
            axs = np.array(axs).reshape(rows, cols)

        for r in range(rows):
            for c in range(cols):
                ax = axs[r][c]
                # Support both list-of-lists and numpy object arrays
                hist = hists[r][c] if isinstance(hists, list) else hists[r, c]

                # Build edges/centers/widths from bin_count and bin_width
                edges = np.linspace(0, hist.bin_count * hist.bin_width, hist.bin_count + 1)
                centers = 0.5 * (edges[:-1] + edges[1:])
                widths = np.diff(edges)

                ax.bar(centers, hist.data, width=widths, align='center')
                ax.set_title(f'[{r}, {c}]', fontsize=9)

        # Common axis labels
        fig.supxlabel('bin')
        fig.supylabel('value')
        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_points(hists, rows, cols):
        # Create a 3D axis (bar3d is a method on Axes3D, not pyplot)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ys, xs = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        x = xs.ravel().astype(float)
        y = ys.ravel().astype(float)
        z = np.zeros_like(x)
        dx = np.ones_like(x)
        dy = np.ones_like(x)
        dz = np.array([hists[yy][xx].get_points_echo_detection()
                       for yy in range(rows) for xx in range(cols)], dtype=float)
        # 3D bar plot
        ax.bar3d(x, y, z, dx, dy, dz, shade=True)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('value')
        plt.show()