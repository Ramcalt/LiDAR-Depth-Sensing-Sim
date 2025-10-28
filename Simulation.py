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
                         32,
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
                        HTransform().translation(0,0, +0.4) @ HTransform().rotation_x(np.pi)
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
                        "res/cavity_H24_D30.stl",
                        Material(1.0, 1.0, 1.0, [0.9, 0.2, 0.1]),
                        HTransform().translation(0,0, +0.1) @ HTransform().rotation_x(np.pi)
                        )
        )
        self.scene.add_obj(self.detector)
        self.scene.add_obj(self.emitter)
        RayTracer.run_trimesh(self.scene, self.scene.get_obj("cavity"), self.emitter, self.detector, 500_000)
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
        points = []
        for r in range(self.detector.zone_rows):
            for c in range(self.detector.zone_cols):
                theta_x = (self.detector.fov_x_rad / self.detector.zone_cols) * (
                            c + 0.5 - (self.detector.zone_cols / 2))
                theta_y = (self.detector.fov_y_rad / self.detector.zone_rows) * (
                            r + 0.5 - (self.detector.zone_rows / 2))
                if algo == "echo":
                    pts = self.detector.histograms[r, c].get_points_echo_detection(self.emitter.pulse_length_m, theta_x = theta_x, theta_y = theta_y)
                elif algo == "deconv":
                    pts = self.detector.histograms[r, c].get_points_deconv(theta_x=theta_x, theta_y=theta_y)
                else:
                    pts = self.detector.histograms[r, c].get_points_wav_decomp(self.emitter.pulse_length_m)
                if pts is not None and len(pts) > 0:
                    points.extend(pts)
        points = np.array(points, dtype=float)
        points = points[points > 0]
        print(f"depth = {Histogram.compute_depth(points, filtered=False)}")
        print(f"depth filtered = {Histogram.compute_depth(points, filtered=True)}")
        # SHOW POINTS
        Plotter.plot_points(algo, self.detector.histograms, self.detector.zone_rows, self.detector.zone_cols, self.emitter.pulse_length_m, self.detector)
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

        kl_results = {}  # dict[str -> (8,8)]
        epsilon = 1e-8

        for mm in mm_range:
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
        KL_stack = np.stack([kl_results[f"{mm}mm"] for mm in mm_range], axis=0)  # (11,8,8)
        KL_mean = np.nanmean(KL_stack, axis=0)  # (8,8)

        plt.figure(figsize=(6, 5))
        plt.imshow(KL_mean, cmap='viridis', aspect='equal', interpolation='nearest')
        plt.colorbar(label="Mean KL (exp || sim)")
        plt.xticks(range(8), [f"C{c}" for c in range(8)])
        plt.yticks(range(8), [f"R{r}" for r in range(8)])
        plt.title("Mean KL Divergence per Pixel (30–150 mm)")
        plt.tight_layout()
        plt.show()

        ##### DEPTH ######
        # # # PLOT SIM VS REAL # # #
        sim_depths = []
        exp_depths = []
        true_depth = 25
        for idx, mm in enumerate(mm_range):
            sim_max = exp_max = -1.0 * 10e10
            sim_min = exp_min = 1.0 * 10e10
            for r in range(8):
                for c in range(8):
                    dict_name = f"{mm}mm"
                    sim_raw = simresults[dict_name][r, c].copy()  # shape (32,)
                    exp_raw = results[dict_name][r, c].copy()  # shape (32,)
                    h_exp = Histogram(time_start=0.0, time_end=None, bin_count=18, bin_width_m=BIN_WIDTH_M, data=exp_raw)
                    h_sim = Histogram(time_start=0.0, time_end=None, bin_count=18, bin_width_m=BIN_WIDTH_M, data=sim_raw)
                    exp_pts_m = h_exp.get_points_echo_detection(self.emitter.pulse_length_m)
                    sim_pts_m = h_sim.get_points_echo_detection(self.emitter.pulse_length_m)
                    if len(sim_pts_m) > 0:
                        sim_max = max(sim_max, sim_pts_m.max())
                        sim_min = min(sim_min, sim_pts_m.min())
                    if len(exp_pts_m) > 0:
                        exp_max = max(exp_max, exp_pts_m.max())
                        exp_min = min(exp_min, exp_pts_m.min())

            if sim_max > (-1.0 * 10e10) and sim_min < (1.0 * 10e10):
                sim_depths.append((sim_max - sim_min) * 1000)
            else:
                sim_depths.append(-1)
            if exp_max > (-1.0 * 10e10) and exp_min < (1.0 * 10e10):
                exp_depths.append((exp_max - exp_min) * 1000)
            else:
                exp_depths.append(-1)

        # # # PLOT # # #
        plt.figure(figsize=(8, 5))
        plt.plot(mm_range, exp_depths, 'o-', label='Experimental Depth')
        plt.plot(mm_range, sim_depths, 's--', label='Simulated Depth')
        plt.axhline(y=true_depth, color='r', linestyle=':', linewidth=2, label='Ground Truth')

        plt.title("Depth Comparison: Experimental vs Simulation vs Ground Truth")
        plt.xlabel("Test Distance (mm)")
        plt.ylabel("Depth (mm)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()


sim = Simulation()
sim.run_kl_test(mm_range = range(30, 151, 10), mesh_path = "res/square_W105_L40_D40.stl", csv_path = "res/square_W105_L40_D40.csv", N=250_000)

# sim = Simulation()
# sim.run()
# sim.view_scene_trimesh()

# sim = Simulation()
# sim.run_find_kerenel()
# sim.view_scene_trimesh()


