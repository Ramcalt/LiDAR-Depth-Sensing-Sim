import numpy as np
from scene.SceneObject import SceneObject
import matplotlib.pyplot as plt
import trimesh
from scene.HTransform import HTransform

class RayTracer:
    def __init__(self):
        pass

    @staticmethod
    def run_trimesh(scene, target, emitter, detector, N, visualise=False):
        # construct trimesh scene
        target_mesh = target.to_trimesh_mesh()
        emitter_mesh = emitter.to_trimesh_mesh()
        detector_mesh = detector.to_trimesh_mesh()

        tri_scene = trimesh.Scene()
        tri_scene.add_geometry(target_mesh)
        tri_scene.add_geometry(emitter_mesh)
        tri_scene.add_geometry(detector_mesh)

        # = = = = = = = = = = = = = RAY TRACING = = = = = = = = = = = = = =
        # generate rays
        primary_ray_directions = np.zeros((N, 3))
        primary_ray_origins = np.array([emitter.transform.matrix[:3, 3]] * N)
        theta_x, theta_y = emitter.sample_gaussian_beam_angles(N)
        for i in range(N):
            R = HTransform().rotation_x(theta_x[i]) @ HTransform().rotation_y(theta_y[i])
            direction = R @ np.array([0.0, 0.0, 1.0, 0.0])
            primary_ray_directions[i] = direction[0:3]

        EPS = 1e-6

        # perform ray tracing to target
        _, primary_idx, primary_hit_locations = target_mesh.ray.intersects_id(primary_ray_origins, primary_ray_directions, multiple_hits=False, max_hits=1, return_locations=True)
        primary_hit_dirs = primary_ray_directions[primary_idx]
        primary_hit_locations = primary_hit_locations - EPS * primary_hit_dirs
        primary_hit_origins = primary_ray_origins[primary_idx]
        primary_hit_distances = np.linalg.norm(primary_hit_locations - primary_hit_origins, axis=1)

        # perform ray tracing to destination
        destination = detector.transform.matrix[:3,3]
        v_hd = (destination - primary_hit_locations)
        seg_lengths = np.linalg.norm(v_hd, axis=1)
        sec_dirs = np.divide(v_hd, seg_lengths[:, None], out=np.zeros_like(v_hd), where=seg_lengths[:, None] > 0)
        sec_origins = primary_hit_locations + sec_dirs * EPS
        _, sec_idx_ray, sec_locations = target_mesh.ray.intersects_id(sec_origins, sec_dirs, multiple_hits=False, max_hits=1, return_locations=True)
        # normalise to (K,3) even when empty
        if sec_locations is None or np.size(sec_locations) == 0:
            sec_locations = np.empty((0, 3), dtype=float)
        else:
            sec_locations = np.asarray(sec_locations, dtype=float).reshape(-1, 3)

        # filter rays that have intersections with the target_mesh before hitting the detector mesh
        blocker_dists = np.linalg.norm(sec_locations - sec_origins[sec_idx_ray], axis=1)
        los_mask = np.ones(len(seg_lengths), dtype=bool)
        los_mask[sec_idx_ray] = blocker_dists >= seg_lengths[sec_idx_ray]


        # = = = = = = = = = = = = = DETECTION = = = = = = = = = = = = = =
        # TODO: BIN distances into the correct detector bin based on their incident angle
        # Assume the detector optical axis is [0, 0, 1] and has fov from (+fov_x, +fov_y) to (-fov_x, -fov_y)
        los_dists = primary_hit_distances[los_mask] + seg_lengths[los_mask]
        los_dirs = -sec_dirs[los_mask]

        # Rotate into the detector's local frame so +Z is the detector's optical axis.
        M = detector.transform.matrix[:3, :3]  # world->detector rotation is R^T
        d_local = (M.T @ los_dirs.T).T  # shape (K,3)
        ux, uy, uz = d_local[:, 0], d_local[:, 1], d_local[:, 2]

        # Get angles (don't require dot product since we assume optical axis is [0, 0, 1])
        theta_x = np.arctan2(ux, uz)
        theta_y = np.arctan2(uy, uz)

        # Mask outside FOV
        in_fov = (
                (np.abs(theta_x) <= detector.fov_x_rad/2.0) &
                (np.abs(theta_y) <= detector.fov_y_rad/2.0)
        )
        theta_x = theta_x[in_fov]
        theta_y = theta_y[in_fov]
        los_dists = los_dists[in_fov]

        # Map angles to [0, 1)
        tx = (theta_x + detector.fov_x_rad/2.0) / (detector.fov_x_rad)
        ty = (theta_y + detector.fov_y_rad/2.0) / (detector.fov_y_rad)
        sx = np.clip(tx, 0.0, 1.0 - np.finfo(float).eps)
        sy = np.clip(ty, 0.0, 1.0 - np.finfo(float).eps)

        # Convert to zoning indices
        row_idx = np.floor(sy * detector.zone_rows).astype(int)
        col_idx = np.floor(sx * detector.zone_cols).astype(int)

        if theta_x.size:
            print(
                f"[coverage] θx∈[{theta_x.min():+.3f}, {theta_x.max():+.3f}] rad, "
                f"θy∈[{theta_y.min():+.3f}, {theta_y.max():+.3f}] rad; "
                f"rows hit: {row_idx.min()}..{row_idx.max()}, cols hit: {col_idx.min()}..{col_idx.max()}",
                flush=True
            )

        # apply pulse broadening and binning
        one_way_tof = 0.5 * los_dists
        noisy_distances = emitter.apply_vcsel_pulse_broadening(one_way_tof)
        detector.apply_binning(noisy_distances, row_idx, col_idx)

        # = = = = = = = = = = = = = = = = VIZUALISER = = = = = = = = = = = = =
        if (visualise):
            # Visualize rays: always one secondary segment per displayed primary hit
            ray_lines = []
            k = min(20, len(primary_hit_locations))

            # Map: secondary ray index -> its first blocker point (since max_hits=1)
            blocker_for_ray = {rid: loc for rid, loc in zip(sec_idx_ray, sec_locations)}

            for i in range(k):
                o = primary_hit_origins[i]
                p = primary_hit_locations[i]

                # Primary: emitter -> target hit (red)
                seg_primary = trimesh.load_path(np.vstack((o, p)))
                seg_primary.colors = np.tile([255, 0, 0, 140], (seg_primary.entities.shape[0], 1))
                ray_lines.append(seg_primary)

                # Secondary: choose endpoint & color
                has_los = (i < len(los_mask)) and bool(los_mask[i])
                if has_los:
                    q = destination
                    color = [0, 255, 0, 140]  # green
                else:
                    q = blocker_for_ray.get(i, destination)  # if no blocker recorded, still draw to detector
                    color = [255, 165, 0, 140]  # orange

                # Ensure non-degenerate segment (very rare but avoids zero-length path)
                if np.allclose(p, q):
                    q = p + (destination - p) * 1e-6

                seg_secondary = trimesh.load_path(np.vstack((p, q)))
                seg_secondary.colors = np.tile(color, (seg_secondary.entities.shape[0], 1))
                ray_lines.append(seg_secondary)

            for ray in ray_lines:
                tri_scene.add_geometry(ray)

            tri_scene.show()

