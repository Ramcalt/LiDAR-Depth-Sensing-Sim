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
import re
from pathlib import Path
from scipy.special import rel_entr
import matplotlib.pyplot as plt
from lidar.Histogram import Histogram

class Simulation:
    """Singleton for managing simulation."""
    scene: Scene
    emitter: Emitter
    detector: Detector

    def __init__(self):
        """Initialise scene and add emitter, detector, and scene objects"""
        self.scene = Scene(1.0, [1.0, 1.0, 1.0])
        self.emitter = Emitter("emitter",
                               "res/sensor_toscale.stl",
                               Material(1.0, 1.0, 1.0, [0.8, 0.2, 0.2]),
                               HTransform().translation(-0.002, 0, 0),
                               940e-9,
                               7.7e-8 * (10**(0.002*(940-700))), # 200 * 0.74e-3 * 0.90e-3,
                               0.5e-9,
                               1.010546, # 57.9deg for 10% signal from max
                               1.010546,
                               4
                               )
        self.detector = Detector("detector",
                                 "res/sensor_toscale.stl",
                                 Material(1.0, 1.0, 1.0, [0.2, 0.2, 0.8]),
                                 HTransform().translation(0.002, 0, 0),
                                 8,
                                 8,
                                 0.79,
                                 0.79,
                                 100,
                                 0.0375
                                 )
        self.scene.add_obj(
            SceneObject("cavity",
                        "res/flat_W105.stl",
                        Material(1.0, 1.0, 1.0, [0.9, 0.2, 0.1]),
                        HTransform().translation(0,0, +0.1) @ HTransform().rotation_x(np.pi)
                        )
        )
        self.scene.add_obj(self.detector)
        self.scene.add_obj(self.emitter)

    def test_plotting(self):
        self.detector.fill_hist_with_noise()
        self.view_plots()

    def run(self):
        """Run the simulation"""
        RayTracer.run_trimesh(self.scene, self.scene.get_obj("cavity"), self.emitter, self.detector, 100_000)
        self.view_plots()

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

    def view_plots(self):
        """Runs matplotlib plots in separate processes"""
        # PLOT HISTOGRAMS
        Plotter.new_process(Plotter.plot_hist_arr, self.detector.histograms, self.detector.zone_rows, self.detector.zone_cols)
        # COMPUTE DEPTH
        points = []
        for r in range(self.detector.zone_rows):
            for c in range(self.detector.zone_cols):
                pts = self.detector.histograms[r, c].get_points_echo_detection(self.emitter.pulse_length_m)
                if pts is not None and len(pts) > 0:
                    points.extend(pts)
        points = np.array(points, dtype=float)
        points = points[points > 0]
        print(f"depth = {np.max(points) - np.min(points)}")
        # SHOW POINTS
        Plotter.plot_points(self.detector.histograms, self.detector.zone_rows, self.detector.zone_cols, self.emitter.pulse_length_m)
        # SHOW HISTOGRAM PROCESSING
        self.select_and_visualize()

    def select_and_visualize(self):
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
                        Plotter.new_process(
                            self.detector.histograms[r, c].visualise_get_points_echo_detection,pulse_width_m)
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
            Plotter.new_process(
                self.detector.histograms[r, c].visualise_get_points_echo_detection,pulse_width_m)

    def init_kl_test(self):
        self.scene = Scene(1.0, [1.0, 1.0, 1.0])
        self.emitter = Emitter("emitter",
                               "res/sensor_toscale.stl",
                               Material(1.0, 1.0, 1.0, [0.8, 0.2, 0.2]),
                               HTransform().translation(-0.002, 0, 0),
                               940e-9,
                               7.7e-8 * (10**(0.002*(940-700))), # 200 * 0.74e-3 * 0.90e-3,
                               0.5e-9,
                               1.010546, # 57.9deg for 10% signal from max
                               1.010546,
                               4
                               )
        self.detectors = [Detector(f"detector{i}",
                                 "res/sensor_toscale.stl",
                                 Material(1.0, 1.0, 1.0, [0.2, 0.2, 0.8]),
                                 HTransform().translation(0.002, 0, 0),
                                 4,
                                 4,
                                 0.79,
                                 0.79,
                                 32,
                                 0.0375
                                 )
                            for i, _ in enumerate(range(100, 201, 10))
                          ]
        self.scene.add_obj(
            SceneObject("cavity",
                        "res/square_W105_D25.stl",
                        Material(1.0, 1.0, 1.0, [0.9, 0.2, 0.1]),
                        HTransform().translation(0,0, +0.1) @ HTransform().rotation_x(np.pi)
                        )
        )
        self.scene.add_obj(self.emitter)

    def run_kl_test(self):
        # # # RUN SIMULATION # # #
        self.init_kl_test()
        target = self.scene.get_obj("cavity")
        target_diameter = 80e-3  # m
        offset = target_diameter / (2 * np.tan(np.radians(43.4 / 2)))
        target.transform = HTransform().translation(0,0, offset) @ HTransform().rotation_x(np.pi)
        print("offset = ", offset)
        for i, mm in enumerate(range(100, 201, 10)):
            z_offset = mm * 1e-3 # move target rfurther each step
            det = self.detectors[i]
            self.scene.add_obj(det)
            target.transform = HTransform().translation(0, 0, z_offset) @ HTransform().rotation_x(np.pi)
            RayTracer.run_trimesh(self.scene, target, self.emitter, det, 100_000)

        # Collect results into dict
        simresults = {}  # dict: {'100mm': np.ndarray(shape=(8,8,32)), ...}
        for i, mm in enumerate(range(100, 201, 10)):
            det = self.detectors[i]
            H = np.stack(
                [np.stack([det.histograms[r, c].data for c in range(det.zone_cols)], axis=0)
                 for r in range(det.zone_rows)],
                axis=0
            )
            simresults[f"{mm}mm"] = H

        # # # IMPORT CSV # # #
        # --- CSV LOAD (32 bins × 16 angular groups) into 4×4×32 ---
        directory = Path('res/CNHexpcsv')
        results = {}  # { '100mm': np.ndarray shape (4,4,32) }

        N_ROW = 1  # read only first row
        N_SHIFT_CONST = 0  # optional fixed bin shift (usually 0)

        def a_to_rc(a: int) -> tuple[int, int]:
            """Map angular group index a∈[0..15] to 4×4 (row, col), row-major."""
            r, c = divmod(a, 4)
            return r, c

        for mm in range(100, 201, 10):
            file_path = directory / f"{mm}mm.csv"
            try:
                df = pd.read_csv(file_path, nrows=N_ROW)
                # target array
                G = np.zeros((4, 4, 32), dtype=float)

                # 1) fill raw counts
                for col in df.columns:
                    m = re.match(r'^cnh__hist_bin_(\d{1,2})_a(\d{1,2})$', col)
                    if not m:
                        continue
                    b = int(m.group(1))
                    a = int(m.group(2))
                    if 0 <= b < 32 and 0 <= a < 16:
                        r, c = a_to_rc(a)
                        bi = b + N_SHIFT_CONST
                        if 0 <= bi < 32:
                            G[r, c, bi] = float(df[col].iloc[N_ROW - 1])

                # 2) (optional) clip negatives from CSV
                G = np.clip(G, 0.0, None)

                # 3) dynamic alignment of the experimental histogram using distance_mm_za*
                #    – this shifts each (r,c) trace so that its peak roughly matches the measured range
                #    – turn off if your CSV already starts bins at the correct physical zero.
                BIN_WIDTH_M = self.detectors[0].bin_width_m  # meters per bin (one-way)
                for a in range(16):
                    r, c = a_to_rc(a)
                    dz_col = f'distance_mm_z{a}'
                    if dz_col in df.columns:
                        d_m = float(df[dz_col].iloc[N_ROW - 1]) * 1e-3  # mm -> m
                        # Predicted bin of the main return (assuming one-way distance bins)
                        # If your device histogram is round-trip, use: b_pred = int(round((2*d_m)/BIN_WIDTH_M))
                        b_pred = int(round(d_m / BIN_WIDTH_M))
                        # Current peak bin from data
                        b_peak = int(np.argmax(G[r, c, :])) if G[r, c, :].sum() > 0 else 0
                        # Shift to align peak ~ expected bin
                        delta = b_pred - b_peak
                        if delta != 0:
                            G[r, c, :] = np.roll(G[r, c, :], delta)

                results[f"{mm}mm"] = G

            except FileNotFoundError:
                print(f"File not found: {file_path.name}. Skipping.")
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

        # # # PLOT SIM VS REAL # # #
        R_SEL, C_SEL = 0, 0
        fig, axes = plt.subplots(4, 3, figsize=(12, 8))
        axes = axes.flatten()

        for idx, mm in enumerate(range(100, 201, 10)):
            dict_name = f"{mm}mm"

            # --- RAW counts (do not normalise in-place) ---
            sim_raw = simresults[dict_name][R_SEL, C_SEL].copy()  # shape (32,)
            exp_raw = results[dict_name][R_SEL, C_SEL].copy()  # shape (32,)

            # --- Normalised for plotting ---
            sim_counts = sim_raw / sim_raw.sum() if sim_raw.sum() > 0 else np.zeros_like(sim_raw)
            exp_counts = exp_raw / exp_raw.sum() if exp_raw.sum() > 0 else np.zeros_like(exp_raw)

            bin_edges = np.arange(32) * BIN_WIDTH_M
            bin_centers = bin_edges + BIN_WIDTH_M / 2

            ax = axes[idx]
            ax.bar(bin_centers, exp_counts, width=BIN_WIDTH_M * 0.9, alpha=0.6, label="Experimental", color="red")
            ax.step(bin_centers, sim_counts, where='mid', label="Simulated", color="blue", linewidth=2)

            # --- DETECTION POINTS (metres) using Histogram.get_points_echo_detection ---
            h_exp = Histogram(time_start=0.0, time_end=None, bin_count=32, bin_width_m=BIN_WIDTH_M, data=exp_raw)
            h_sim = Histogram(time_start=0.0, time_end=None, bin_count=32, bin_width_m=BIN_WIDTH_M, data=sim_raw)

            exp_pts_m = h_exp.get_points_echo_detection(self.emitter.pulse_length_m)
            sim_pts_m = h_sim.get_points_echo_detection(self.emitter.pulse_length_m)

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

        kl_results = {}  # dict[str -> (8,8)]
        epsilon = 1e-8

        for mm in range(100, 201, 10):
            key = f"{mm}mm"
            S = simresults[key]  # (8,8,32)
            E = results[key]  # (8,8,32)

            # normalise with epsilon
            S = S + epsilon
            S /= S.sum(axis=2, keepdims=True)

            E = E + epsilon
            E /= E.sum(axis=2, keepdims=True)

            # KL(exp || sim) per pixel
            KL = np.sum(rel_entr(E, S), axis=2)  # (8,8)
            kl_results[key] = KL

        # Aggregate: mean KL over distances
        KL_stack = np.stack([kl_results[f"{mm}mm"] for mm in range(100, 201, 10)], axis=0)  # (11,8,8)
        KL_mean = np.nanmean(KL_stack, axis=0)  # (8,8)

        plt.figure(figsize=(6, 5))
        plt.imshow(KL_mean, cmap='viridis', aspect='equal', interpolation='nearest')
        plt.colorbar(label="Mean KL (exp || sim)")
        plt.xticks(range(4), [f"C{c}" for c in range(4)])
        plt.yticks(range(4), [f"R{r}" for r in range(4)])
        plt.title("Mean KL Divergence per Pixel (100–200 mm)")
        plt.tight_layout()
        plt.show()

        ##### DEPTH ######
        # # # PLOT SIM VS REAL # # #
        sim_depths = []
        exp_depths = []
        true_depth = 25
        for idx, mm in enumerate(range(100, 201, 10)):
            sim_max = exp_max = -1.0 * 10e10
            sim_min = exp_min = 1.0 * 10e10
            for r, c in zip(range(4), range(4)):
                dict_name = f"{mm}mm"
                sim_raw = simresults[dict_name][r, c].copy()  # shape (32,)
                exp_raw = results[dict_name][r, c].copy()  # shape (32,)
                h_exp = Histogram(time_start=0.0, time_end=None, bin_count=32, bin_width_m=BIN_WIDTH_M, data=exp_raw)
                h_sim = Histogram(time_start=0.0, time_end=None, bin_count=32, bin_width_m=BIN_WIDTH_M, data=sim_raw)
                exp_pts_m = h_exp.get_points_echo_detection(self.emitter.pulse_length_m)
                sim_pts_m = h_sim.get_points_echo_detection(self.emitter.pulse_length_m)
                sim_max = max(sim_max, sim_pts_m.max())
                sim_min = min(sim_min, sim_pts_m.min())
                exp_max = max(exp_max, exp_pts_m.max())
                exp_min = min(exp_min, exp_pts_m.min())

            sim_depths.append((sim_max - sim_min) * 1000)
            exp_depths.append((exp_max - exp_min) * 1000)

        test_distances = np.arange(100, 201, 10)

        # # # PLOT # # #
        plt.figure(figsize=(8, 5))
        plt.plot(test_distances, exp_depths, 'o-', label='Experimental Depth')
        plt.plot(test_distances, sim_depths, 's--', label='Simulated Depth')
        plt.axhline(y=true_depth, color='r', linestyle=':', linewidth=2, label='Ground Truth')

        plt.title("Depth Comparison: Experimental vs Simulation vs Ground Truth")
        plt.xlabel("Test Distance (mm)")
        plt.ylabel("Depth (mm)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()


sim = Simulation()
sim.run_kl_test()
# sim = Simulation()
# sim.view_scene_trimesh()
# sim.run()

