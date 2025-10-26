import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from multiprocessing import Process
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

class Plotter:
    def __init__(self):
        pass

    @staticmethod
    def new_process(target, *args, daemon=False, **kwargs):
        if isinstance(target, str):
            func = getattr(Plotter, target, None)
            if func is None or not callable(func):
                raise ValueError(f"Plotter has no callable '{target}'")
        elif callable(target):
            func = target
        else:
            raise TypeError("target must be a function name (str) or a callable")

        # Launch the plotting function in a new process
        p = Process(target=func, args=args, kwargs=kwargs)
        p.daemon = daemon  # usually keep False for GUI stability
        p.start()
        return p

    @staticmethod
    def plot_hist(hist):
        edges = np.linspace(0, hist.bin_count * hist.bin_width_m, hist.bin_count + 1)
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
                edges = np.linspace(0, hist.bin_count * hist.bin_width_m, hist.bin_count + 1)
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
    def plot_points(algo, hists, rows, cols, pulse_width_m, detector):
        # use browser renderer to avoid terminal output
        pio.renderers.default = "browser"

        xs, ys, zs = [], [], []
        for yy in range(rows):
            for xx in range(cols):
                theta_x = (detector.fov_x_rad / detector.zone_cols) * (
                            xx + 0.5 - (detector.zone_cols / 2))
                theta_y = (detector.fov_y_rad / detector.zone_rows) * (
                            yy + 0.5 - (detector.zone_rows / 2))
                if algo == "echo":
                    pts = hists[yy][xx].get_points_echo_detection(pulse_width_m, theta_x, theta_y)
                elif algo == "deconv":
                    pts = hists[yy][xx].get_points_deconv(theta_x, theta_y)
                else:
                    pts = hists[yy][xx].get_points_wav_decomp(pulse_width_m)

                if pts is None or len(pts) == 0:
                    continue
                for p in pts:
                    xs.append(xx)
                    ys.append(yy)
                    zs.append(p)

        fig = go.Figure(
            data=[go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='markers',
                marker=dict(size=3, color=zs, colorscale='Viridis', opacity=0.8)
            )]
        )
        fig.update_layout(
            scene=dict(
                xaxis_title='X (cols)',
                yaxis_title='Y (rows)',
                zaxis_title='Range (m)',
            ),
            title="Echo Detections (3D)",
            template="plotly_white",
            height=700
        )

        fig.show()  # opens in browser, cleanly

    @staticmethod
    def plot_points_matplot(hists, rows, cols, pulse_width_m):
        # Create a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ys, xs = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        x = xs.ravel().astype(float)
        y = ys.ravel().astype(float)

        # Clear axes
        ax.cla()

        # Iterate over each histogram cell
        for yy in range(rows):
            for xx in range(cols):
                points = hists[yy][xx].get_points_echo_detection(pulse_width_m)
                if points is None or len(points) == 0:
                    continue

                # Draw a point for each echo detection
                for p in points:
                    ax.scatter(xx, yy, p, color='b', s=20)  # point marker
                    # Draw a vertical line down to z=0
                    ax.plot([xx, xx], [yy, yy], [0, p], color='gray', linewidth=1)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('value')
        plt.show()

    @staticmethod
    def plot_emitter_beam_heatmap(obj, z_min=0.001, z_max=0.100, xy_half_range=0.100, resolution=400, keep_scale_across_z=True):
        """
            Visualise obj.gaussian_beam(z, r) as relative emitted power (% of peak intensity).

            Parameters
            ----------
            obj : object
                An object implementing `gaussian_beam(self, z, r)`.
            z_min, z_max : float
                Slider range for z (same units as used by gaussian_beam).
            xy_half_range : float
                Half-width of the [x,y] window (in same units as r).
            resolution : int, optional
                Number of grid points per axis.
            keep_scale_across_z : bool, optional
                If True, fixes colour scale to 0–100 % for easy comparison.
            """

        # Grid in x, y, and radial distance r
        x = np.linspace(-xy_half_range, xy_half_range, resolution)
        X, Y = np.meshgrid(x, x, indexing="xy")
        R = np.sqrt(X ** 2 + Y ** 2)

        # Initial z
        z0 = 0.5 * (z_min + z_max)
        I0 = obj.gaussian_beam(z0, X, Y)
        I0_rel = 100.0 * I0 / np.max(I0)

        # Set up figure and plot
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.18)
        im = ax.imshow(I0_rel,
                       extent=[-xy_half_range, xy_half_range, -xy_half_range, xy_half_range],
                       origin="lower",
                       cmap="inferno",
                       aspect="equal",
                       vmin=0,
                       vmax=100 if keep_scale_across_z else None)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Relative Emitted Power (%)")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Gaussian beam (z = {z0:g})")

        # Slider for z
        ax_z = fig.add_axes([0.15, 0.07, 0.7, 0.04])
        z_slider = Slider(ax=ax_z, label="z", valmin=z_min, valmax=z_max, valinit=z0)

        # Update function
        def _update(_):
            z = z_slider.val
            I = obj.gaussian_beam(z, X, Y)
            I_rel = 100.0 * I / np.max(I)
            im.set_data(I_rel)

            if not keep_scale_across_z:
                im.set_clim(vmin=0, vmax=np.max(I_rel))

            ax.set_title(f"Gaussian beam (z = {z:g})")
            fig.canvas.draw_idle()

        z_slider.on_changed(_update)
        plt.show()

    # = = = = = = = = = = = = = = = = = = = = EMITTER = = = = = = = = = = = = = = = = = = = = = #
    @staticmethod
    def plot_emitter_beam_heatmap_angular(obj, zspan_factor=1.5, n=601, units="rad", show_checks=True):
        """
            Visualize I(theta_x, theta_y) on a 2D grid and 1D cuts through the axes.

            Parameters
            ----------
            zspan_factor : float
                Grid half-span in units of the corresponding theta_0* (e.g., 5 -> ±5*theta_0x, ±5*theta_0y).
                Increase if tails are not negligible for your p.
            n : int
                Number of grid samples per axis (odd preferred so the origin is a grid point).
            units : {"rad","mrad","deg"}
                Axis labeling units for the plots only (computations stay in radians).
            show_checks : bool
                If True, prints a Riemann-sum normalization check over the chosen window.
            """
        # Angular scales (radians)
        th0x = float(obj.emission_angle_x_rad)
        th0y = float(obj.emission_angle_y_rad)

        # Grid in radians, sized independently along x/y to respect anisotropy
        tx = np.linspace(-zspan_factor * th0x, zspan_factor * th0x, n)
        ty = np.linspace(-zspan_factor * th0y, zspan_factor * th0y, n)
        dtx = tx[1] - tx[0]
        dty = ty[1] - ty[0]
        TX, TY = np.meshgrid(tx, ty, indexing="xy")

        # Evaluate intensity
        I = obj.gaussian_beam_angular(TX, TY)

        # Optional: normalization check over the finite window
        if show_checks:
            integral = np.sum(I) * dtx * dty  # ≈ 1.0 if window is large enough
            peak = I[n // 2, n // 2]
            print(f"[check] window ±{zspan_factor}·theta0: integral≈{integral:.8f}, peak={peak:.6g}")

        # Unit conversion for axis labels only
        if units == "mrad":
            scl, ulabel = 1e3, "mrad"
        elif units == "deg":
            scl, ulabel = 180.0 / np.pi, "deg"
        else:
            scl, ulabel = 1.0, "rad"

        # 2D map
        plt.figure(figsize=(6, 5))
        extent = [scl * tx.min(), scl * tx.max(), scl * ty.min(), scl * ty.max()]
        plt.imshow(I, origin="lower", cmap="inferno", extent=extent, aspect="auto")
        plt.colorbar(label="Normalized intensity")
        plt.xlabel(rf"$\theta_x$ [{ulabel}]")
        plt.ylabel(rf"$\theta_y$ [{ulabel}]")
        plt.title("Angular intensity map")

        # 1D cuts through the axes
        mid_y = n // 2  # θy = 0 cut
        mid_x = n // 2  # θx = 0 cut

        plt.figure(figsize=(6, 4))
        plt.plot(scl * tx, I[mid_y, :], label=r"$\theta_y=0$")
        plt.plot(scl * ty, I[:, mid_x], linestyle="--", label=r"$\theta_x=0$")
        plt.xlabel(rf"Angle [{ulabel}]")
        plt.ylabel("Normalized intensity")
        plt.title("On-axis angular cuts")
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def check_emitter_sampler(obj):
        # Sample beam angles
        tx, ty = obj.sample_gaussian_beam_angles(100_000)

        # Create figure and 3D axis
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Create 2D histogram data for 3D bars
        hist, xedges, yedges = np.histogram2d(tx, ty, bins=20, density=True)

        # Compute bar positions
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.5 * (xedges[1] - xedges[0]),
                                 yedges[:-1] + 0.5 * (yedges[1] - yedges[0]),
                                 indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos)

        # Bar dimensions
        dx = dy = 0.02 * np.ones_like(zpos)
        dz = hist.ravel()

        # Plot 3D histogram
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', alpha=0.6)

        # Create 3D line plot for Gaussian beam (tx vs intensity at ty=0)
        t_vals = np.linspace(-1, 1, 200)
        beam_vals = [obj.gaussian_beam_angular(t, 0) for t in t_vals]
        ax.plot(t_vals, np.zeros_like(t_vals), beam_vals, 'r--', linewidth=2, label='Gaussian beam (ty=0)')

        # Labels and legend
        ax.set_xlabel('tx')
        ax.set_ylabel('ty')
        ax.set_zlabel('Density / Intensity')
        ax.legend()
        plt.title('3D Histogram of Gaussian Beam Angles')
        plt.show()