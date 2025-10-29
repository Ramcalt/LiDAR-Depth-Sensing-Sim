from xml.etree.ElementTree import tostring

from lidar.RayTracer import RayTracer
from scene.Scene import Scene
from scene.SceneObject import SceneObject
from scene.Material import Material
from scene.HTransform import HTransform
from lidar.Emitter import Emitter
from lidar.Detector import Detector
from Plotter import Plotter
import open3d as o3d
import trimesh
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.ndimage import zoom
from scipy.special import rel_entr
import matplotlib.pyplot as plt
from lidar.Histogram import Histogram

vl53l8ch_emitter = Emitter("emitter",
                       "res/sensor_toscale.stl",
                       Material(1.0, 1.0, 1.0, [0.8, 0.2, 0.2]),
                       HTransform().translation(-0.002, 0, 0),
                       940e-9,
                       7.7e-8 * (10 ** (0.002 * (940 - 700))),  # 200 * 0.74e-3 * 0.90e-3,
                       1e-9,
                       1.010546,  # 57.9deg for 10% signal from max
                       1.010546,
                       4
                       )
vl53l8ch_detector = Detector("detector",
                         "res/sensor_toscale.stl",
                         Material(1.0, 1.0, 1.0, [0.2, 0.2, 0.8]),
                         HTransform().translation(0.002, 0, 0),
                         8,
                         8,
                         0.79,
                         0.79,
                         18,
                         0.0375
                         )

class Simulation:
    """Singleton for managing simulation."""
    scene: Scene
    emitter: Emitter
    detector: Detector

    def __init__(self):
        pass

    def test_plotting(self):
        self.detector.fill_hist_with_noise()
        self.view_plots()

    def run_find_kerenel(self):
        """Initialise scene and add emitter, detector, and scene objects"""
        self.scene = Scene(1.0, [1.0, 1.0, 1.0])
        self.emitter = vl53l8ch_emitter
        self.detector = vl53l8ch_detector
        self.scene.add_obj(
            SceneObject("cavity",
                        "res/flat_W105000.stl",
                        Material(1.0, 1.0, 1.0, [0.9, 0.2, 0.1]),
                        HTransform().translation(0,0, +0.15) @ HTransform().rotation_x(np.pi)
                        )
        )
        self.scene.add_obj(self.detector)
        self.scene.add_obj(self.emitter)
        RayTracer.run_trimesh(self.scene, self.scene.get_obj("cavity"), self.emitter, self.detector, 1_000_000)
        for r in range(8):
            for c in range (8):
                print(f"Histogram[{r}, {c}] = [{','.join(map(lambda x: f'{x:.3f}', self.detector.histograms[r, c].data))}]")

        self.view_plots()

    def run(self):
        """Run the simulation"""
        """Initialise scene and add emitter, detector, and scene objects"""
        self.scene = Scene(1.0, [1.0, 1.0, 1.0])
        self.emitter = vl53l8ch_emitter
        self.detector = vl53l8ch_detector
        self.scene.add_obj(
            SceneObject("cavity",
                        "res/square_W105_L40_D40.stl",
                        Material(1.0, 1.0, 1.0, [0.9, 0.2, 0.1]),
                        HTransform().translation(0,0, +0.1) @ HTransform().rotation_x(np.pi)
                        )
        )
        self.scene.add_obj(self.detector)
        self.scene.add_obj(self.emitter)
        RayTracer.run_trimesh(self.scene, self.scene.get_obj("cavity"), self.emitter, self.detector, 250_000)
        self.view_histograms()
        self.view_plots(algo="echo")
        self.view_plots(algo="deconv")
        self.view_plots(algo="decomp")

    def view_scene(self):
        """View the scene using Open3D"""
        meshes = [obj.to_o3d_mesh() for obj in self.scene.objects]
        o3d.visualization.draw(meshes, raw_mode=True)

    def view_scene_trimesh(self):
        """View the scene using Open3D"""
        meshes = [obj.to_trimesh_mesh() for obj in self.scene.objects]
        scene = trimesh.Scene()
        [scene.add_geometry(mesh) for mesh in meshes]
        scene.show()

    def view_histograms(self):
        Plotter.new_process(Plotter.plot_hist_arr, self.detector.histograms, self.detector.zone_rows,
                            self.detector.zone_cols)

    def view_plots(self, algo="echo"):
        """Runs matplotlib plots in separate processes"""
        # COMPUTE DEPTH
        xs, ys, zs = [], [], []  # zone-x, zone-y, range[m]
        for r in range(self.detector.zone_rows):
            for c in range(self.detector.zone_cols):
                theta_x = (self.detector.fov_x_rad / self.detector.zone_cols) * (
                            c + 0.5 - (self.detector.zone_cols / 2)) * 0.75
                theta_y = (self.detector.fov_y_rad / self.detector.zone_rows) * (
                            r + 0.5 - (self.detector.zone_rows / 2)) * 0.75
                if algo == "echo":
                    pts = self.detector.histograms[r, c].get_points_echo_detection(self.emitter.pulse_length_m, theta_x = theta_x, theta_y = theta_y)
                elif algo == "deconv":
                    pts = self.detector.histograms[r, c].get_points_deconv(theta_x=theta_x, theta_y=theta_y)
                else:
                    pts = self.detector.histograms[r, c].get_points_wav_decomp(self.emitter.pulse_length_m, theta_x=theta_x, theta_y=theta_y, )
                if pts is not None and len(pts) > 0:
                    for p in pts:
                        xs.append(c)
                        ys.append(r)
                        zs.append(float(p))
        print(f"depth = {Histogram.compute_depth(zs, filtered=False)}")
        print(f"depth filtered = {Histogram.compute_depth(zs, filtered=True)}")
        # SHOW POINTS
        Plotter.plot_points((xs, ys, zs), algo, self.detector.zone_rows, self.detector.zone_cols)
        # SHOW HISTOGRAM PROCESSING
        self.select_and_visualize(algo)

    def select_and_visualize(self, algo):
        """Allow user to select which histogram cell(s) to visualize via terminal input."""
        rows = self.detector.zone_rows
        cols = self.detector.zone_cols
        pulse_width_m = self.emitter.pulse_length_m

        print(f"\nDetector grid: {rows} rows × {cols} columns")
        print("Enter indices to visualize specific cells (e.g. '1 3'), or 'all' to view all cells.")
        print("Type 'q' or 'quit' to exit.\n")

        while True:
            user_input = input("Select cell (row col): ").strip().lower()
            if user_input in ("q", "quit", "exit"):
                print("Exiting visualization.")
                break

            if user_input == "all":
                for r in range(rows):
                    for c in range(cols):
                        if algo == "echo":
                            Plotter.new_process(self.detector.histograms[r, c].visualise_get_points_echo_detection,pulse_width_m)
                        elif algo == "deconv":
                            Plotter.new_process(self.detector.histograms[r, c].visualise_get_points_deconv,pulse_width_m)
                        else:
                            Plotter.new_process(self.detector.histograms[r, c].visualise_get_points_wav_decomp,pulse_width_m)
                continue

            parts = user_input.split()
            if len(parts) != 2 or not all(p.isdigit() for p in parts):
                print("Invalid input. Please enter two integers (row col), 'all', or 'q' to quit.")
                continue

            r, c = map(int, parts)
            if not (0 <= r < rows and 0 <= c < cols):
                print(f"Indices out of range. Must be within 0–{rows - 1} and 0–{cols - 1}.")
                continue

            print(f"Launching visualization for cell ({r}, {c})...")
            if algo == "echo":
                Plotter.new_process(self.detector.histograms[r, c].visualise_get_points_echo_detection, pulse_width_m)
            elif algo == "deconv":
                Plotter.new_process(self.detector.histograms[r, c].visualise_get_points_deconv, pulse_width_m)
            else:
                Plotter.new_process(self.detector.histograms[r, c].visualise_get_points_wav_decomp, pulse_width_m)

    def init_kl_test(self, mm_range, mesh_path):
        self.scene = Scene(1.0, [1.0, 1.0, 1.0])
        self.emitter = vl53l8ch_emitter
        self.detectors = [Detector(f"detector{i}",
                                 "res/sensor_toscale.stl",
                                 Material(1.0, 1.0, 1.0, [0.2, 0.2, 0.8]),
                                 HTransform().translation(0.002, 0, 0),
                                 8,
                                 8,
                                 0.79,
                                 0.79,
                                 18,
                                 0.0375
                                 )
                            for i, _ in enumerate(mm_range)
                          ]
        self.scene.add_obj(
            SceneObject("cavity",
                        mesh_path,
                        Material(1.0, 1.0, 1.0, [0.9, 0.2, 0.1]),
                        HTransform().translation(0,0, +0.3) @ HTransform().rotation_x(np.pi)
                        )
        )
        self.scene.add_obj(self.emitter)

    def run_kl_test(self, mm_range, mesh_path, csv_path, N):
        # # # RUN SIMULATION # # #
        self.init_kl_test(mm_range, mesh_path)
        target = self.scene.get_obj("cavity")
        for i, mm in enumerate(mm_range):
            z_offset = mm * 1e-3 # move target rfurther each step
            det = self.detectors[i]
            self.scene.add_obj(det)
            target.transform = HTransform().translation(0, 0, z_offset) @ HTransform().rotation_x(np.pi)
            RayTracer.run_trimesh(self.scene, target, self.emitter, det, N)

        # Collect results into dict
        simresults = {}  # dict: {'100mm': np.ndarray(shape=(8,8,32)), ...}
        for i, mm in enumerate(mm_range):
            det = self.detectors[i]
            H = np.stack(
                [np.stack([det.histograms[r, c].data for c in range(det.zone_cols)], axis=0)
                 for r in range(det.zone_rows)],
                axis=0
            )
            simresults[f"{mm}mm"] = H

        # # # IMPORT CSV # # #
        df = pd.read_csv(csv_path)
        results = {}
        for _, csv_row in df.iterrows():
            histograms = np.zeros((8, 8, 18), dtype=float)
            distance = csv_row.iloc[0]
            for r in range(8):
                for c in range(8):
                    for b in range(18):
                        histograms[r, c, b] = csv_row.iloc[6 + ((r * 8 + c) * 21) + b]
            results[f"{distance}mm"] = histograms

        # # # PLOT SIM VS REAL # # #
        BIN_WIDTH_M = self.detectors[0].bin_width_m
        R_SEL, C_SEL = 2, 3
        fig, axes = plt.subplots(4, 4, figsize=(12, 8))
        axes = axes.flatten()

        for idx, mm in enumerate(mm_range):
            dict_name = f"{mm}mm"

            # raw counts
            sim_raw = simresults[dict_name][R_SEL, C_SEL].copy()
            exp_raw = results[dict_name][R_SEL, C_SEL].copy()

            # normalised counts
            sim_counts = sim_raw / sim_raw.sum() if sim_raw.sum() > 0 else np.zeros_like(sim_raw)
            exp_counts = exp_raw / exp_raw.sum() if exp_raw.sum() > 0 else np.zeros_like(exp_raw)

            bin_edges = np.arange(18) * BIN_WIDTH_M
            bin_centers = bin_edges + BIN_WIDTH_M / 2

            ax = axes[idx]
            ax.bar(bin_centers, exp_counts, width=BIN_WIDTH_M * 0.9, alpha=0.6, label="Experimental", color="red")
            ax.step(bin_centers, sim_counts, where='mid', label="Simulated", color="blue", linewidth=2)

            # --- DETECTION POINTS (metres) using Histogram.get_points_echo_detection ---
            h_exp = Histogram(time_start=0.0, time_end=None, bin_count=18, bin_width_m=BIN_WIDTH_M, data=exp_raw)
            h_sim = Histogram(time_start=0.0, time_end=None, bin_count=18, bin_width_m=BIN_WIDTH_M, data=sim_raw)

            theta_x = (self.detectors[0].fov_x_rad / self.detectors[0].zone_cols) * (
                    C_SEL + 0.5 - (self.detectors[0].zone_cols / 2))
            theta_y = (self.detectors[0].fov_y_rad / self.detectors[0].zone_rows) * (
                    R_SEL + 0.5 - (self.detectors[0].zone_rows / 2))
            exp_pts_m = h_exp.get_points_echo_detection(self.emitter.pulse_length_m, theta_x, theta_y)
            sim_pts_m = h_sim.get_points_echo_detection(self.emitter.pulse_length_m, theta_x, theta_y)

            # draw vertical lines at detected points
            for k, x in enumerate(exp_pts_m):
                ax.axvline(x, linestyle=':', linewidth=1.6,
                           label=("Exp detections" if (idx == 0 and k == 0) else None))
            for k, x in enumerate(sim_pts_m):
                ax.axvline(x, linestyle='--', linewidth=1.6,
                           label=("Sim detections" if (idx == 0 and k == 0) else None))

            ax.set_title(f"{mm} mm")
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Normalized Counts")
            ax.grid(True, linestyle='--', alpha=0.5)
            if idx == 0:
                ax.legend()

        # Hide any unused axes (if any)
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

        # # # KL TEST # # #

        kl_results = {}
        alpha = 1e-6  # Laplace smoothing (pseudocount per bin)

        for mm in mm_range:
            key = f"{mm}mm"
            S = np.asarray(simresults[key], dtype=np.float64)
            E = np.asarray(results[key], dtype=np.float64)

            # Sanitize any NaN/Inf in inputs to zeros (raw counts should be >= 0)
            S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
            E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)

            # Per-histogram totals
            S_sum = S.sum(axis=2, keepdims=True)
            E_sum = E.sum(axis=2, keepdims=True)

            # Zones with nonzero totals on both sides
            valid_mask = ((S_sum > 0.0) & (E_sum > 0.0)).squeeze(-1)

            # Laplace smoothing BEFORE normalisation to prevent zero probabilities
            # (add alpha pseudocount to each bin)
            Sn = (S + alpha) / (S_sum + alpha * S.shape[2])
            En = (E + alpha) / (E_sum + alpha * E.shape[2])

            # KL(exp || sim)
            with np.errstate(divide='ignore', invalid='ignore'):
                KL = np.sum(rel_entr(En, Sn), axis=2)

            # Invalidate zones where totals were zero on either side
            KL[~valid_mask] = np.nan

            # Also treat non-finite as NaN (we will ignore them in the mean)
            KL[~np.isfinite(KL)] = np.nan

            kl_results[key] = KL

        # Diagnostic: count finite values (excludes NaN and +/- Inf)
        for mm in mm_range:
            key = f"{mm}mm"
            finite_count = int(np.sum(np.isfinite(kl_results[key])))
            print(f"{key}: finite KL = {finite_count}")

        # Aggregate across distances: ignore NaN/Inf
        KL_stack = np.stack([kl_results[f"{mm}mm"] for mm in mm_range], axis=0)
        KL_mean = np.nanmean(KL_stack, axis=0)

        # Plot (robust to NaNs)
        M = np.ma.masked_invalid(KL_mean)
        plt.figure(figsize=(6, 5))
        if np.all(M.mask):
            plt.text(0.5, 0.5, "KL heatmap is all invalid (check inputs / CSV stride).",
                     ha='center', va='center')
        else:
            vmax = np.nanpercentile(KL_mean, 99)  # robust upper bound
            plt.imshow(M, cmap='viridis', aspect='equal', interpolation='nearest', vmin=0.0, vmax=vmax)
            plt.colorbar(label="Mean KL (exp || sim)")
            plt.xticks(range(8), [f"C{c}" for c in range(8)])
            plt.yticks(range(8), [f"R{r}" for r in range(8)])
            plt.title("Mean KL Divergence per Pixel (30–150 mm)")
        plt.tight_layout()
        plt.show()

        ##### DEPTH ######
        # # # PLOT SIM VS REAL # # #
        # Helper to compute depth (in mm) from a full 8x8xB histogram cube
        def _depth_from_hist_cube(hist_cube, algo, det, emitter, use_angle_scale=0.75):
            rows, cols, bins = hist_cube.shape
            BIN_WIDTH_M = det.bin_width_m

            zs = []  # collected ranges [m] from all valid pixels
            for r in range(rows):
                for c in range(cols):
                    # Build a Histogram object for this pixel
                    h = Histogram(
                        time_start=0.0,
                        time_end=None,
                        bin_count=bins,
                        bin_width_m=BIN_WIDTH_M,
                        data=hist_cube[r, c].astype(float)
                    )

                    # Per-pixel angles (match your earlier convention and 0.75 scaling)
                    theta_x = (det.fov_x_rad / det.zone_cols) * (c + 0.5 - (det.zone_cols / 2)) * use_angle_scale
                    theta_y = (det.fov_y_rad / det.zone_rows) * (r + 0.5 - (det.zone_rows / 2)) * use_angle_scale

                    # Extract detection points for the chosen algorithm
                    if algo == "echo":
                        pts = h.get_points_echo_detection(emitter.pulse_length_m, theta_x=theta_x, theta_y=theta_y)
                    elif algo == "deconv":
                        pts = h.get_points_deconv(theta_x=theta_x, theta_y=theta_y)
                    elif algo == "decomp":
                        pts = h.get_points_wav_decomp(emitter.pulse_length_m, theta_x=theta_x, theta_y=theta_y)
                    else:
                        pts = None

                    if pts is not None and len(pts) > 0:
                        zs.extend([float(p) for p in pts])

            # Use the filtered estimator (requested)
            return Histogram.compute_depth(zs, filtered=True) if len(zs) > 0 else np.nan

        # Compute depth-vs-distance for each algorithm for both sim and experiment
        algos = ("echo", "deconv", "decomp")
        true_depth_mm = 40  # cavity depth

        # Keep per-algorithm series
        sim_series = {a: [] for a in algos}
        exp_series = {a: [] for a in algos}

        for idx, mm in enumerate(mm_range):
            key = f"{mm}mm"
            S = simresults[key]  # shape (8,8,B)
            E = results[key]  # shape (8,8,B)
            det = self.detectors[idx]  # detector used at this distance (for FOV, bin width, etc.)

            for algo in algos:
                sim_d_mm = _depth_from_hist_cube(S, algo, det, self.emitter)
                exp_d_mm = _depth_from_hist_cube(E, algo, det, self.emitter)
                sim_series[algo].append(sim_d_mm*1000)
                exp_series[algo].append(exp_d_mm*1000)

        # --- Plot: three subplots, one per algorithm ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
        for ax, algo in zip(axes, algos):
            ax.plot(mm_range, exp_series[algo], 'o-', label='Experimental')
            ax.plot(mm_range, sim_series[algo], 's--', label='Simulated')
            ax.axhline(y=true_depth_mm, linestyle=':', linewidth=2, label='Ground Truth' if algo == algos[0] else None)

            ax.set_title(f"Depth vs Distance — {algo}")
            ax.set_xlabel("Test Distance (mm)")
            ax.grid(True, linestyle='--', alpha=0.6)

        axes[0].set_ylabel("Depth (mm)")
        axes[0].legend()
        fig.suptitle("Computed Depth (filtered) — Experimental vs Simulation")
        plt.tight_layout()
        plt.show()


# sim = Simulation()
# sim.run_kl_test(mm_range = range(30, 121, 10), mesh_path = "res/square_W105_L40_D40.stl", csv_path = "res/square_W105_L40_D40.csv", N=10_000)

sim = Simulation()
sim.run()
sim.view_scene_trimesh()

#sim = Simulation()
#sim.run_find_kerenel()
#sim.view_scene_trimesh()


