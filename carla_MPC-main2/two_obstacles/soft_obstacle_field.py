import math
import numpy as np


def apply_soft_obstacle_field(
    waypoints,
    ego_x,
    ego_y,
    ego_yaw,
    obstacles,
    influence_dist=7.5,
    safe_dist=1.4,
    side_clearance=2.8,
    side_cost_weight=0.35,
    pass_side=1.0,
    front_cost_distance=17.0,
    back_release_distance=0.8,
):
    """Apply a CILQR-style soft obstacle field to reference waypoints.

    Args:
        waypoints: np.ndarray shaped [4, N], rows are [x, y, v, yaw].
        obstacles: array-like shaped [M, >=2], each row [x, y, yaw(optional)].

    Returns:
        np.ndarray with same shape as waypoints.
    """
    if waypoints is None or len(waypoints.shape) != 2 or waypoints.shape[1] == 0:
        return waypoints

    if obstacles is None:
        return waypoints

    obs = np.array(obstacles, dtype=float)
    if obs.size == 0:
        return waypoints
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)

    out = waypoints.copy()
    pts = out[:2, :].T

    # Reference heading from the current ego yaw (rad)
    forward = np.array([math.cos(ego_yaw), math.sin(ego_yaw)], dtype=float)
    lateral = np.array([-math.sin(ego_yaw), math.cos(ego_yaw)], dtype=float)
    side_sign = 1.0 if pass_side >= 0.0 else -1.0

    for i in range(len(pts)):
        p = pts[i]
        total_push = np.zeros(2, dtype=float)

        for ob in obs:
            ox, oy = ob[0], ob[1]
            rel = np.array([p[0] - ox, p[1] - oy], dtype=float)
            dist = float(np.linalg.norm(rel))
            if dist < 1e-6:
                rel = np.array([1.0, 0.0], dtype=float)
                dist = 1e-6

            # obstacle_ahead > 0 means obstacle ahead in ego frame
            obstacle_ahead = -float(rel @ forward)
            if obstacle_ahead < -back_release_distance or obstacle_ahead > front_cost_distance:
                continue

            # front_gate in [0, 1], nearer front obstacles produce stronger push
            den = max(front_cost_distance - influence_dist, 1e-6)
            front_gate = np.clip((front_cost_distance - obstacle_ahead) / den, 0.0, 1.0)

            clearance = dist - 0.8  # obstacle radius approximation
            unit = rel / dist

            # Radial repulsion term
            if clearance < influence_dist:
                margin = influence_dist - clearance
                rep_gain = 0.08 * margin * front_gate
                total_push += rep_gain * unit

            # Strong close-range safety term
            if clearance < safe_dist:
                vio = safe_dist - clearance
                total_push += 0.20 * vio * unit

            # Weak side preference term: prefer passing left of path by default
            signed_lateral = side_sign * float(rel @ lateral)
            side_error = side_clearance - signed_lateral
            if side_error > 0.0 and clearance < influence_dist:
                side_gate = np.clip((influence_dist - clearance) / max(influence_dist, 1e-6), 0.0, 1.0)
                total_push += (
                    side_cost_weight
                    * 0.025
                    * side_error
                    * side_gate
                    * front_gate
                    * side_sign
                    * lateral
                )

        pts[i] = p + total_push

    out[0, :] = pts[:, 0]
    out[1, :] = pts[:, 1]
    return out
