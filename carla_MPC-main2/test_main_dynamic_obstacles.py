"""
test_main_dynamic_obstacles.py

Dynamic obstacle MPC demo.
This file is additive: it does not change the original test_main.py,
x_v2x_agent.py, or mcp_controller.py.

Run from carla_MPC-main2:
    python test_main_dynamic_obstacles.py

Before running:
    1. Start CARLA server.
    2. Make sure carla, casadi, pygame, numpy, scipy, matplotlib are installed.
"""

import os
import sys
import pathlib
import time
import numpy as np
import matplotlib.pyplot as plt
import carla

sys.path.insert(0, str(pathlib.Path(__file__).with_name("src")))

from env import Env, draw_waypoints
from src.global_route_planner import GlobalRoutePlanner
from src.dynamic_mpc_controller import DynamicVehicle
from src.dynamic_xagent import DynamicObstacleXAgent
from src.dynamic_obstacles import RouteDynamicObstacleManager


# =============================
# Simulation parameters
# =============================
simu_step = 0.05
target_v = 40          # ego target speed, km/h
sample_res = 2.0
display_mode = "spec"  # "spec" or "pygame"
max_sim_steps = 2000

# Use a local log directory instead of /debug_logs or a hard-coded Windows path.
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_logs_dynamic")
os.makedirs(LOG_DIR, exist_ok=True)
os.environ["MPC_DEBUG_LOG_DIR"] = LOG_DIR
open(os.path.join(LOG_DIR, "run_debug.log"), "w", encoding="utf-8").close()


# =============================
# Environment setup
# =============================
env = Env(display_method=display_mode, dt=simu_step)

# Clean old vehicles from previous runs.
for actor in env.world.get_actors().filter("vehicle.*"):
    actor.destroy()

spawn_points = env.map.get_spawn_points()
start_idx, end_idx = 87, 70

grp = GlobalRoutePlanner(env.map, sample_res)
route = grp.trace_route(spawn_points[start_idx].location, spawn_points[end_idx].location)
draw_waypoints(env.world, [wp for wp, _ in route], z=0.5, color=(0, 255, 0))

env.reset(spawn_point=spawn_points[start_idx])


# =============================
# Dynamic obstacles
# =============================
obstacle_manager = RouteDynamicObstacleManager(env, route)

# Lead vehicle: moving slowly in the same lane ahead of ego.
# Increase/decrease route_index if it spawns too near/far on your map.
if len(route) > 110:
    obstacle_manager.spawn_on_route(route_index=110, speed_mps=3.0, lateral_offset=0.0)

# 先注释掉第二个
# if len(route) > 170:
#     obstacle_manager.spawn_on_route(route_index=170, speed_mps=2.0, lateral_offset=0.35)

# Move once before the first MPC solve so obstacle velocities are initialized.
obstacle_manager.tick(simu_step)
env.world.tick()


# =============================
# MPC and agent
# =============================
dynamic_model = DynamicVehicle(
    actor=env.ego_vehicle,
    horizon=10,
    target_v=target_v,
    delta_t=simu_step,
    max_iter=30,
)

agent = DynamicObstacleXAgent(env, dynamic_model, obstacle_manager=obstacle_manager, dt=simu_step)
agent.set_start_end_transforms(start_idx, end_idx)
agent.plan_route(agent._start_transform, agent._end_transform)


# =============================
# Data logging
# =============================
trajectory = []
velocities = []
accelerations = []
steerings = []
times = []
solve_times = []
lateral_errors = []
obstacle_history = []

try:
    for step in range(max_sim_steps):
        try:
            # Update obstacle positions before MPC. The MPC receives current
            # obstacle position + velocity, then predicts them over the horizon.
            obstacle_manager.tick(simu_step)
            obstacle_history.append(obstacle_manager.get_obstacle_array()[:, :2].copy())

            a_opt, delta_opt, next_state, solve_time_ms, lateral_error = agent.run_step()

            vel = env.ego_vehicle.get_velocity()
            speed = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5
            throttle, steer_norm, brake, reverse = env.get_cmd([a_opt, delta_opt])

            print(
                f"step={step:04d}, "
                f"a={a_opt:.3f}, steer={delta_opt:.3f}, "
                f"throttle={throttle:.3f}, brake={brake:.3f}, "
                f"speed={speed:.3f} m/s, solve={solve_time_ms:.1f} ms"
            )

            env.step([a_opt, delta_opt])

            x, y, yaw, vx, vy, omega = next_state[0]
            trajectory.append([x, y])
            velocities.append(vx)
            accelerations.append(a_opt)
            steerings.append(delta_opt)
            times.append(step * simu_step)
            solve_times.append(solve_time_ms)
            lateral_errors.append(lateral_error)

            dist_to_goal = np.linalg.norm([
                next_state[0][0] - agent._end_transform.location.x,
                next_state[0][1] - agent._end_transform.location.y,
            ])
            if dist_to_goal < 1.0:
                print("Destination reached.")
                break

            if env.display_method == "pygame":
                time.sleep(simu_step)

        except Exception as e:
            import traceback
            with open(os.path.join(LOG_DIR, "error.log"), "w", encoding="utf-8") as f:
                f.write(f"Error at step {step}: {e}\n")
                f.write(traceback.format_exc())
            print(f"Error at step {step}: {e}")
            break

except KeyboardInterrupt:
    pass


# =============================
# Plot results
# =============================
trajectory = np.array(trajectory)
if len(trajectory) > 0:
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="ego trajectory")

    # Plot sampled obstacle positions.
    if obstacle_history:
        max_obs = max((arr.shape[0] for arr in obstacle_history), default=0)
        for obs_idx in range(max_obs):
            pts = []
            for arr in obstacle_history:
                if obs_idx < arr.shape[0]:
                    pts.append(arr[obs_idx])
            if pts:
                pts = np.array(pts)
                plt.plot(pts[:, 0], pts[:, 1], "--", label=f"dynamic obstacle {obs_idx}")

    plt.axis("equal")
    plt.xlabel("x / m")
    plt.ylabel("y / m")
    plt.legend()
    plt.title("MPC dynamic obstacle avoidance")
    plt.savefig(os.path.join(LOG_DIR, "dynamic_obstacle_trajectory.png"), dpi=150)

if len(times) > 0:
    plt.figure()
    plt.plot(times, velocities)
    plt.xlabel("time / s")
    plt.ylabel("vx / m/s")
    plt.title("ego velocity")
    plt.savefig(os.path.join(LOG_DIR, "dynamic_obstacle_velocity.png"), dpi=150)

print(f"Logs and figures saved to: {LOG_DIR}")
