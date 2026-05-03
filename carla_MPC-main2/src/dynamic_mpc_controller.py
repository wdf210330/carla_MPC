"""
dynamic_mpc_controller.py

Dynamic-obstacle MPC controller extension. It inherits the original Vehicle
class and overrides only solver_add_soft_obs(). Static obstacle behavior remains
compatible, but rows with velocity columns are predicted over the MPC horizon.

Obstacle row formats:
    static:  [x, y, yaw]
    static + side: [x, y, yaw, preferred_side]
    dynamic: [x, y, yaw, preferred_side, vx, vy, yaw_rate]
"""

import numpy as np
import casadi as ca

from src import mcp_controller as base


class DynamicVehicle(base.Vehicle):
    def solver_add_soft_obs(self, obs=None, ratio=500, expn=1):
        log_file = base._debug_log_path()

        def log(msg):
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

        if obs is None:
            log("[DYNAMIC solver_add_soft_obs] obs=None, skip")
            return
        obs = np.array(obs, dtype=float)
        if obs.size == 0:
            log("[DYNAMIC solver_add_soft_obs] empty obs, skip")
            return
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        n_obs = obs.shape[0]
        log(f"[DYNAMIC solver_add_soft_obs] n_obs={n_obs}, shape={obs.shape}")

        vehicle_radius = 1.5
        obst_radius = 1.68

        # Keep these close to the original controller's tuned static-obstacle values.
        R_max = 8.0
        safe_dist = 1.80
        k_u = 220.0
        k_close = 450.0
        front_cost_dist = 10.0
        back_release_dist = 1.2

        side_clearance = 2.20
        k_side = 80.0

        v_min_near = 1.0
        k_vnear = 20.0
        k_brake_near = 10.0
        lateral_consider = 3.2
        eps = 1e-3
        front_gate_den = max(front_cost_dist - R_max, 1e-3)
        dt = float(base.dt if base.dt is not None else 0.05)

        for i in range(self.horizon):
            state_i = self.X[:, i]
            (vcx1, vcy1), (vcx2, vcy2) = self.get_self_centers(state_i, vehicle_radius)

            for j in range(n_obs):
                obs_x0 = float(obs[j, 0])
                obs_y0 = float(obs[j, 1])
                obs_yaw0 = float(obs[j, 2]) if obs.shape[1] > 2 else 0.0
                preferred_side = float(obs[j, 3]) if obs.shape[1] > 3 else 0.0
                obs_vx = float(obs[j, 4]) if obs.shape[1] > 4 else 0.0
                obs_vy = float(obs[j, 5]) if obs.shape[1] > 5 else 0.0
                obs_yaw_rate = float(obs[j, 6]) if obs.shape[1] > 6 else 0.0

                # Constant-velocity obstacle prediction across the horizon.
                t_pred = i * dt
                obs_x = obs_x0 + obs_vx * t_pred
                obs_y = obs_y0 + obs_vy * t_pred
                obs_yaw = obs_yaw0 + obs_yaw_rate * t_pred

                (ocx1, ocy1), (ocx2, ocy2) = self.get_obs_centers_simple(obs_x, obs_y, obs_yaw, obst_radius)

                d1 = ca.sqrt((vcx1 - ocx1) ** 2 + (vcy1 - ocy1) ** 2)
                d2 = ca.sqrt((vcx1 - ocx2) ** 2 + (vcy1 - ocy2) ** 2)
                d3 = ca.sqrt((vcx2 - ocx1) ** 2 + (vcy2 - ocy1) ** 2)
                d4 = ca.sqrt((vcx2 - ocx2) ** 2 + (vcy2 - ocy2) ** 2)
                dist = ca.mmin(ca.vertcat(d1, d2, d3, d4))

                ego_x = state_i[0]
                ego_y = state_i[1]
                ego_yaw = state_i[2]
                rel_x = obs_x - ego_x
                rel_y = obs_y - ego_y
                obstacle_ahead = rel_x * ca.cos(ego_yaw) + rel_y * ca.sin(ego_yaw)
                signed_lateral = -rel_x * ca.sin(ego_yaw) + rel_y * ca.cos(ego_yaw)

                gate_front = (front_cost_dist - obstacle_ahead) / front_gate_den
                gate_front = ca.fmax(0.0, ca.fmin(1.0, gate_front))
                rear_gate = ca.if_else(
                    obstacle_ahead >= 0.0,
                    1.0,
                    ca.fmax(0.0, 1.0 + obstacle_ahead / back_release_dist),
                )
                release_gate = ca.if_else(obstacle_ahead < -back_release_dist, 0.0, rear_gate)
                gate_front = gate_front * release_gate
                gate_front = gate_front * ca.if_else(obstacle_ahead > front_cost_dist, 0.0, 1.0)

                rep_term = ca.fmax(0.0, 1.0 / (dist + eps) - 1.0 / (R_max + eps))
                rep_potential = 0.5 * k_u * rep_term ** 2
                close_margin = ca.fmax(0.0, safe_dist - dist)
                close_potential = 0.5 * k_close * close_margin ** 2

                side_error = ca.if_else(
                    ca.fabs(preferred_side) > 0.5,
                    ca.fmax(0.0, side_clearance - preferred_side * signed_lateral),
                    ca.fmax(0.0, side_clearance - ca.fabs(signed_lateral)),
                )
                side_cost = 0.5 * k_side * side_error ** 2

                pass_lateral = ca.if_else(
                    ca.fabs(preferred_side) > 0.5,
                    preferred_side * signed_lateral,
                    ca.fabs(signed_lateral),
                )
                pass_progress = ca.fmax(0.0, ca.fmin(1.0, (pass_lateral - (side_clearance - 0.20)) / 0.55))
                side_by_side_gate = ca.fmax(0.0, ca.fmin(1.0, (2.0 - ca.fabs(obstacle_ahead)) / 2.0))
                pass_commit_gate = pass_progress * side_by_side_gate
                just_passed_gate = ca.fmax(0.0, ca.fmin(1.0, (0.15 - obstacle_ahead) / 0.60))
                pass_release_gate = pass_progress * just_passed_gate

                rep_potential = rep_potential * (1.0 - 0.40 * pass_commit_gate - 0.55 * pass_release_gate)
                close_potential = close_potential * (1.0 - 0.65 * pass_commit_gate - 0.92 * pass_release_gate)
                side_cost = side_cost * (1.0 - 0.75 * pass_release_gate)

                v_near_error = ca.fmax(0.0, v_min_near - state_i[3])
                v_near_cost = 0.5 * k_vnear * v_near_error ** 2
                near_gate = ca.fmax(0.0, (R_max - dist) / (R_max + eps))
                lateral_gate = ca.if_else(ca.fabs(signed_lateral) > lateral_consider, 0.0, 1.0)
                brake_near = ca.fmax(0.0, -self.U[0, i])
                brake_release_scale = 1.0 - 0.90 * pass_release_gate
                brake_near_cost = 0.5 * k_brake_near * near_gate * brake_release_scale * brake_near ** 2

                # Closing-speed boost: a faster approaching obstacle should be treated earlier.
                obs_speed_along_ego = obs_vx * ca.cos(ego_yaw) + obs_vy * ca.sin(ego_yaw)
                closing_speed = ca.fmax(0.0, state_i[3] - obs_speed_along_ego)
                dynamic_scale = 1.0 + 0.08 * ca.fmin(8.0, closing_speed)

                post_pass_cost_scale = 1.0 - 0.70 * pass_release_gate
                self.obj += dynamic_scale * gate_front * lateral_gate * post_pass_cost_scale * (
                        rep_potential + close_potential + near_gate * side_cost + v_near_cost + brake_near_cost
                ) / (i + 1)

        log("[DYNAMIC solver_add_soft_obs] dynamic obstacle potential added")
