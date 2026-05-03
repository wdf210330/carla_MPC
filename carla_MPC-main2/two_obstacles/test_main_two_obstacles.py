"""Two-obstacle MPC test using pure MPC soft potential-field obstacle avoidance."""

import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib
import time
import pygame
import carla

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.mcp_controller import Vehicle
from env import Env, draw_waypoints
from src.global_route_planner import GlobalRoutePlanner
from src.x_v2x_agent import Xagent
import os


simu_step = 0.05
target_v = 40
sample_res = 2.0
display_mode = "spec"
start_idx, end_idx = 87, 70

BASE_Q = np.diag([10.0, 10.0, 9.0, 9.0, 4.0])
BASE_R = np.diag([0.45, 0.90])
BASE_Rd = np.diag([0.15, 12.0])

# During the pass we care less about snapping back to the centerline immediately
# and more about maintaining forward progress with enough steering freedom.
AVOID_Q = np.diag([8.5, 8.5, 6.5, 20.0, 2.0])
AVOID_R = np.diag([0.08, 0.32])
AVOID_Rd = np.diag([0.015, 0.70])

# Once the obstacle is already beside / just behind the ego nose, bias even more
# toward finishing the maneuver cleanly instead of braking into a standstill.
RELEASE_Q = np.diag([7.0, 7.0, 5.5, 22.0, 1.5])
RELEASE_R = np.diag([0.05, 0.20])
RELEASE_Rd = np.diag([0.008, 0.30])

# obstacle placement indexes on route
obs_indices = [55, 115]


env = Env(display_method=display_mode, dt=simu_step)

for actor in env.world.get_actors().filter('vehicle.*'):
    actor.destroy()

spawn_points = env.map.get_spawn_points()
grp = GlobalRoutePlanner(env.map, sample_res)
route = grp.trace_route(spawn_points[start_idx].location, spawn_points[end_idx].location)

draw_waypoints(env.world, [wp for wp, _ in route], z=0.5, color=(0, 255, 0))
env.reset(spawn_point=spawn_points[start_idx])

collision_state = {
    "count": 0,
    "last_intensity": 0.0,
    "last_actor": "",
}


def _on_collision(event):
    impulse = event.normal_impulse
    intensity = float(np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2))
    other_actor = event.other_actor.type_id if event.other_actor is not None else "unknown"
    collision_state["count"] += 1
    collision_state["last_intensity"] = intensity
    collision_state["last_actor"] = other_actor


collision_bp = env.world.get_blueprint_library().find("sensor.other.collision")
collision_sensor = env.world.spawn_actor(
    collision_bp,
    carla.Transform(),
    attach_to=env.ego_vehicle,
)
collision_sensor.listen(_on_collision)
env.actor_list.append(collision_sensor)

obs_positions = []
for ridx in obs_indices:
    if len(route) > ridx:
        obs_wp, _ = route[ridx]
        spawn_loc = carla.Location(
            x=obs_wp.transform.location.x,
            y=obs_wp.transform.location.y,
            z=obs_wp.transform.location.z + 0.5,
        )
        obs_transform = carla.Transform(location=spawn_loc, rotation=obs_wp.transform.rotation)
        env.spawn_obstacle_vehicle(obs_transform)
        yaw_rad = np.deg2rad(obs_wp.transform.rotation.yaw)
        obs_positions.append([spawn_loc.x, spawn_loc.y, yaw_rad])

obs_points = np.array(obs_positions) if obs_positions else None
route_points = np.array([[wp.transform.location.x, wp.transform.location.y] for wp, _ in route])

dynamic_model = Vehicle(
    actor=env.ego_vehicle,
    horizon=16,
    target_v=target_v,
    delta_t=simu_step,
    max_iter=30,
)

agent = Xagent(env, dynamic_model, dt=simu_step)
agent._model.solver_basis(Q=BASE_Q, R=BASE_R, Rd=BASE_Rd)
agent.set_start_end_transforms(start_idx, end_idx)
agent.plan_route(agent._start_transform, agent._end_transform)
if obs_points is not None:
    agent.set_obstacles(obs_points)

# Log output for tuning
log_dir = ROOT / "two_obstacles" / "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = log_dir / "test_main_two_obstacles.log"
with open(log_file, "w", encoding="utf-8") as f:
    if obs_points is not None and len(obs_points) > 0:
        for i, ob in enumerate(obs_points):
            f.write(f"# obstacle_{i},x={ob[0]:.4f},y={ob[1]:.4f},yaw_rad={ob[2]:.6f}\n")
    f.write(
        "step,time_s,solve_ms,a_opt,delta_opt,"
        "pred_x,pred_y,pred_yaw,pred_vx,actual_x,actual_y,actual_yaw,actual_vx,lat_err,"
        "min_obs_dist,min_obs_idx,near_obs,"
        "obs0_dist,obs0_ahead_m,obs0_lat_m,obs1_dist,obs1_ahead_m,obs1_lat_m,"
        "current_yaw_rh,ref_yaw_0,ref_yaw_1,pred_yaw_1,yaw_error_1,vel_error_0,target_ind,"
        "nearest_ahead_m,nearest_lat_m,speed_mode,cost_mode,target_v_kmh,dist_to_goal,stopped_flag,"
        "collision_flag,new_collision_flag,collision_count,last_collision_intensity,last_collision_actor\n"
    )

# Let controller internal debug log go to the same folder
os.environ["MPC_DEBUG_LOG_DIR"] = str(log_dir)

max_sim_steps = 2000
trajectory = []
velocities = []
accelerations = []
steerings = []
times = []
solve_times = []
lateral_errors = []
last_logged_collision_count = 0
try:
    for step in range(max_sim_steps):
        try:
            a_opt, delta_opt, next_state, solve_time_ms, lateral_error = agent.run_step()
            env.step([a_opt, delta_opt])

            pred_x, pred_y, pred_yaw, pred_vx, pred_vy, pred_omega = next_state[0]
            actual_state, _ = dynamic_model.get_state_carla()
            actual_x, actual_y, actual_yaw, actual_vx = actual_state[:4]
            actual_yaw_rad = np.deg2rad(actual_yaw)

            trajectory.append([actual_x, actual_y])
            velocities.append(actual_vx)
            accelerations.append(a_opt)
            steerings.append(delta_opt)
            times.append(step * simu_step)
            solve_times.append(solve_time_ms)
            lateral_errors.append(lateral_error)

            min_obs_dist = float("inf")
            min_obs_idx = -1
            near_obs = 0
            obs_metrics = [(float("nan"), float("nan"), float("nan")) for _ in range(2)]
            if obs_points is not None and len(obs_points) > 0:
                dists = np.linalg.norm(obs_points[:, :2] - np.array([actual_x, actual_y]), axis=1)
                min_obs_idx = int(np.argmin(dists))
                min_obs_dist = float(dists[min_obs_idx])
                near_obs = 1 if min_obs_dist < 6.0 else 0
                cos_yaw = np.cos(actual_yaw_rad)
                sin_yaw = np.sin(actual_yaw_rad)
                obs_metrics = []
                for ob in obs_points[:2]:
                    rel_x = float(ob[0] - actual_x)
                    rel_y = float(ob[1] - actual_y)
                    obs_dist = float(np.hypot(rel_x, rel_y))
                    obs_ahead = float(rel_x * cos_yaw + rel_y * sin_yaw)
                    obs_lat = float(-rel_x * sin_yaw + rel_y * cos_yaw)
                    obs_metrics.append((obs_dist, obs_ahead, obs_lat))
                while len(obs_metrics) < 2:
                    obs_metrics.append((float("nan"), float("nan"), float("nan")))
            obs0_dist, obs0_ahead, obs0_lat = obs_metrics[0]
            obs1_dist, obs1_ahead, obs1_lat = obs_metrics[1]

            # Generic obstacle-aware speed policy:
            # use both obstacle distance and current lateral clearance to avoid
            # falling into a low-speed local minimum once a pass maneuver is underway.
            if min_obs_idx >= 0:
                nearest_ahead = [obs0_ahead, obs1_ahead][min_obs_idx]
                nearest_lat = abs([obs0_lat, obs1_lat][min_obs_idx])
                if nearest_ahead < -2.0 and nearest_lat > 2.4:
                    speed_mode = "clear"
                    agent._model.set_target_velocity(target_v)
                elif nearest_ahead < -0.3 and nearest_lat > 2.8:
                    speed_mode = "release_finish"
                    agent._model.set_target_velocity(30)
                elif nearest_ahead < -0.8 and nearest_lat > 2.5:
                    speed_mode = "release_hold"
                    agent._model.set_target_velocity(26)
                elif nearest_ahead < 1.2 and nearest_lat > 2.8:
                    speed_mode = "pass_commit"
                    agent._model.set_target_velocity(24)
                elif nearest_ahead < 4.0 and nearest_lat > 2.5:
                    speed_mode = "pass_window"
                    agent._model.set_target_velocity(20)
                elif nearest_ahead < 0.0:
                    speed_mode = "release_hold"
                    agent._model.set_target_velocity(22)
                elif nearest_ahead < 8.0 and nearest_lat < 2.8:
                    speed_mode = "approach_tight"
                    agent._model.set_target_velocity(16)
                elif nearest_ahead < 10.0 and nearest_lat > 2.4:
                    speed_mode = "pass_setup"
                    agent._model.set_target_velocity(20)
                elif min_obs_dist < 7.5 and nearest_lat < 2.8:
                    speed_mode = "near_tight"
                    agent._model.set_target_velocity(14)
                elif min_obs_dist < 12.0 and nearest_lat < 2.6:
                    speed_mode = "mid_tight"
                    agent._model.set_target_velocity(18)
                elif nearest_ahead < 16.0 and nearest_lat < 2.8:
                    speed_mode = "prepare_ahead"
                    agent._model.set_target_velocity(24)
                elif min_obs_dist < 20.0 and nearest_lat < 2.0:
                    speed_mode = "prepare"
                    agent._model.set_target_velocity(26)
                else:
                    speed_mode = "free"
                    agent._model.set_target_velocity(target_v)
            else:
                nearest_ahead = float("nan")
                nearest_lat = float("nan")
                speed_mode = "no_obs"
                agent._model.set_target_velocity(target_v)

            if speed_mode in {"pass_setup", "pass_window", "pass_commit"}:
                cost_mode = "avoid"
                agent._model.Q = AVOID_Q.copy()
                agent._model.R = AVOID_R.copy()
                agent._model.Rd = AVOID_Rd.copy()
            elif speed_mode in {"release_hold", "release_finish"}:
                cost_mode = "release"
                agent._model.Q = RELEASE_Q.copy()
                agent._model.R = RELEASE_R.copy()
                agent._model.Rd = RELEASE_Rd.copy()
            else:
                cost_mode = "base"
                agent._model.Q = BASE_Q.copy()
                agent._model.R = BASE_R.copy()
                agent._model.Rd = BASE_Rd.copy()

            dist_to_goal = np.linalg.norm([
                next_state[0][0] - agent._end_transform.location.x,
                next_state[0][1] - agent._end_transform.location.y,
            ])
            stopped_flag = int(abs(actual_vx) < 0.6 and min_obs_dist < 15.0)
            collision_flag = int(collision_state["count"] > 0)
            new_collision_flag = int(collision_state["count"] > last_logged_collision_count)
            last_collision_actor = collision_state["last_actor"].replace(",", "|")
            debug_info = getattr(agent, "_last_debug", {})
            current_yaw_rh = float(debug_info.get("current_yaw", float("nan")))
            ref_yaw_0 = float(debug_info.get("ref_yaw_0", float("nan")))
            ref_yaw_1 = float(debug_info.get("ref_yaw_1", float("nan")))
            pred_yaw_1 = float(debug_info.get("pred_yaw_1", float("nan")))
            yaw_error_1 = float(debug_info.get("yaw_error_1", float("nan")))
            vel_error_0 = float(debug_info.get("vel_error_0", float("nan")))
            target_ind = int(debug_info.get("target_ind", -1))

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"{step},{step * simu_step:.3f},{solve_time_ms:.3f},"
                    f"{a_opt:.4f},{delta_opt:.4f},"
                    f"{pred_x:.4f},{pred_y:.4f},{pred_yaw:.4f},{pred_vx:.4f},"
                    f"{actual_x:.4f},{actual_y:.4f},{actual_yaw:.4f},{actual_vx:.4f},{lateral_error:.4f},"
                    f"{min_obs_dist:.4f},{min_obs_idx},{near_obs},"
                    f"{obs0_dist:.4f},{obs0_ahead:.4f},{obs0_lat:.4f},"
                    f"{obs1_dist:.4f},{obs1_ahead:.4f},{obs1_lat:.4f},"
                    f"{current_yaw_rh:.4f},{ref_yaw_0:.4f},{ref_yaw_1:.4f},{pred_yaw_1:.4f},{yaw_error_1:.4f},{vel_error_0:.4f},{target_ind},"
                    f"{nearest_ahead:.4f},{nearest_lat:.4f},{speed_mode},{cost_mode},{agent._model.target_v * 3.6:.2f},{dist_to_goal:.4f},{stopped_flag},"
                    f"{collision_flag},{new_collision_flag},{collision_state['count']},{collision_state['last_intensity']:.4f},{last_collision_actor}\n"
                )
            last_logged_collision_count = collision_state["count"]
            if dist_to_goal < 1.0:
                if env.display_method == "pygame":
                    pygame.quit()
                break

            if env.display_method == "pygame":
                time.sleep(simu_step)

        except Exception as e:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"ERROR,step={step},msg={e}\n")
            print(f"Error at step {step}: {e}")
            break

except KeyboardInterrupt:
    pass

trajectory = np.array(trajectory)
velocities = np.array(velocities)
accelerations = np.array(accelerations)
steerings = np.array(steerings)
times = np.array(times)
solve_times = np.array(solve_times)
lateral_errors = np.array(lateral_errors)

fig, axs = plt.subplots(3, 2, figsize=(16, 14))

if len(trajectory) > 0:
    axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1], label="Vehicle Path", color='darkorange', linewidth=2)
else:
    axs[0, 0].text(0.5, 0.5, "No trajectory data", transform=axs[0, 0].transAxes, ha='center')

axs[0, 0].scatter(agent._start_transform.location.x, agent._start_transform.location.y, color='green', label="Start", zorder=5)
axs[0, 0].scatter(agent._end_transform.location.x, agent._end_transform.location.y, color='red', label="End", zorder=5)
axs[0, 0].plot(route_points[:, 0], route_points[:, 1], '--', color='blue', label="Planned Route", alpha=0.6)

if obs_points is not None:
    axs[0, 0].scatter(obs_points[:, 0], obs_points[:, 1], color='red', s=220, marker='X', label="Obstacles", zorder=6)

axs[0, 0].set_title("Two-Obstacle MPC: Vehicle Path and Planned Route")
axs[0, 0].set_xlabel("X Position")
axs[0, 0].set_ylabel("Y Position")
axs[0, 0].legend(loc='upper left')
axs[0, 0].grid(True)

axs[0, 1].plot(times, velocities, label="Velocity (m/s)", color='royalblue', linewidth=2)
axs[0, 1].set_title("Velocity over Time")
axs[0, 1].grid(True)

axs[1, 0].plot(times, accelerations, label="Acceleration", color='orange', linewidth=2)
axs[1, 0].set_title("Acceleration over Time")
axs[1, 0].grid(True)

axs[1, 1].plot(times, steerings, label="Steering", color='green', linewidth=2)
axs[1, 1].set_title("Steering over Time")
axs[1, 1].grid(True)

axs[2, 0].plot(times, solve_times, label="Solve Time (ms)", color='purple', linewidth=1.5)
axs[2, 0].axhline(y=50, color='red', linestyle='--', label="50ms")
axs[2, 0].set_title("MPC Solve Time")
axs[2, 0].grid(True)

axs[2, 1].plot(times, lateral_errors, label="Lateral Error", color='crimson', linewidth=1.5)
axs[2, 1].set_title("Lateral Tracking Error")
axs[2, 1].grid(True)

plt.subplots_adjust(hspace=0.45, wspace=0.3)
plt.show()
