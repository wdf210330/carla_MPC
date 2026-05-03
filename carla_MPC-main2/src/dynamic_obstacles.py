"""
dynamic_obstacles.py

Non-invasive dynamic obstacle helpers for CARLA MPC tests.
This file does not modify the original Env/Xagent/MPC files.

Main idea:
    - Spawn NPC vehicles as kinematic actors on the planned route.
    - Move them every simulation tick using a simple constant-speed route follower.
    - Export obstacle states as [x, y, yaw, pass_side, vx, vy, yaw_rate].
      The DynamicVehicle MPC controller uses vx/vy/yaw_rate to predict obstacle
      positions across the MPC horizon.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence

import numpy as np
import carla


@dataclass
class RouteObstacleState:
    actor: carla.Actor
    s: float
    speed: float
    lateral_offset: float = 0.0
    yaw_offset_deg: float = 0.0
    prev_x: Optional[float] = None
    prev_y: Optional[float] = None
    prev_yaw: Optional[float] = None
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0


class RouteDynamicObstacleManager:
    """Move dynamic obstacle vehicles along a CARLA route without touching Env."""

    def __init__(self, env, route, z_offset: float = 0.50):
        self.env = env
        self.world = env.world
        self.route = route
        self.z_offset = z_offset
        self.obstacles: List[RouteObstacleState] = []
        self._points = self._build_route_points(route)

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _build_route_points(route):
        pts = []
        s = 0.0
        last_xy = None
        for wp, _ in route:
            t = wp.transform
            x = float(t.location.x)
            y = float(t.location.y)
            z = float(t.location.z)
            yaw_deg = float(t.rotation.yaw)
            if last_xy is not None:
                s += math.hypot(x - last_xy[0], y - last_xy[1])
            pts.append((s, x, y, z, yaw_deg))
            last_xy = (x, y)
        if len(pts) < 2:
            raise ValueError("route must contain at least two waypoints")
        return pts

    def _s_from_route_index(self, index: int) -> float:
        index = max(0, min(index, len(self._points) - 1))
        return self._points[index][0]

    def _sample_route(self, s_query: float, lateral_offset: float = 0.0, yaw_offset_deg: float = 0.0):
        # Clamp at route end. The actor will stop at the final point.
        s_query = max(self._points[0][0], min(s_query, self._points[-1][0]))

        j = 1
        while j < len(self._points) and self._points[j][0] < s_query:
            j += 1
        j = min(j, len(self._points) - 1)
        s0, x0, y0, z0, yaw0 = self._points[j - 1]
        s1, x1, y1, z1, yaw1 = self._points[j]
        denom = max(s1 - s0, 1e-6)
        r = (s_query - s0) / denom

        x = x0 + r * (x1 - x0)
        y = y0 + r * (y1 - y0)
        z = z0 + r * (z1 - z0)
        yaw_deg = math.degrees(math.atan2(y1 - y0, x1 - x0)) + yaw_offset_deg
        yaw_rad = math.radians(yaw_deg)

        # Positive offset means left side of the route heading.
        x += -math.sin(yaw_rad) * lateral_offset
        y += math.cos(yaw_rad) * lateral_offset
        return x, y, z, yaw_deg

    def spawn_on_route(
        self,
        route_index: int,
        speed_mps: float,
        lateral_offset: float = 0.0,
        yaw_offset_deg: float = 0.0,
        blueprint_filter: str = "vehicle.*",
    ):
        """Spawn a kinematic vehicle obstacle on the route.

        Args:
            route_index: waypoint index on the already-planned route.
            speed_mps: constant route-following speed in m/s.
            lateral_offset: lateral route offset in meters.
            yaw_offset_deg: add 180 for an oncoming vehicle on the same route.
        """
        bp_candidates = self.world.get_blueprint_library().filter(blueprint_filter)
        if not bp_candidates:
            raise RuntimeError(f"No blueprint matched {blueprint_filter}")
        bp = bp_candidates[0]
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "dynamic_obstacle")

        s0 = self._s_from_route_index(route_index)
        x, y, z, yaw_deg = self._sample_route(s0, lateral_offset, yaw_offset_deg)
        transform = carla.Transform(
            carla.Location(x=x, y=y, z=z + self.z_offset),
            carla.Rotation(yaw=yaw_deg),
        )
        actor = self.world.spawn_actor(bp, transform)
        actor.set_simulate_physics(False)
        self.env.actor_list.append(actor)

        state = RouteObstacleState(
            actor=actor,
            s=s0,
            speed=float(speed_mps),
            lateral_offset=float(lateral_offset),
            yaw_offset_deg=float(yaw_offset_deg),
            prev_x=x,
            prev_y=y,
            prev_yaw=math.radians(yaw_deg),
        )
        self.obstacles.append(state)
        return actor

    def tick(self, dt: float):
        """Advance all dynamic obstacles by one simulation step."""
        for obs in self.obstacles:
            obs.s = min(obs.s + obs.speed * dt, self._points[-1][0])
            x, y, z, yaw_deg = self._sample_route(obs.s, obs.lateral_offset, obs.yaw_offset_deg)
            yaw = math.radians(yaw_deg)

            if obs.prev_x is not None and dt > 0:
                obs.vx = (x - obs.prev_x) / dt
                obs.vy = (y - obs.prev_y) / dt
                obs.yaw_rate = self._wrap_pi(yaw - obs.prev_yaw) / dt
            obs.prev_x, obs.prev_y, obs.prev_yaw = x, y, yaw

            obs.actor.set_transform(carla.Transform(
                carla.Location(x=x, y=y, z=z + self.z_offset),
                carla.Rotation(yaw=yaw_deg),
            ))

    def get_obstacle_array(self) -> np.ndarray:
        """Return [x, y, yaw, pass_side, vx, vy, yaw_rate] for MPC."""
        rows = []
        for obs in self.obstacles:
            t = obs.actor.get_transform()
            rows.append([
                float(t.location.x),
                float(t.location.y),
                math.radians(float(t.rotation.yaw)),
                0.0,                # Xagent will fill preferred passing side.
                float(obs.vx),
                float(obs.vy),
                float(obs.yaw_rate),
            ])
        return np.array(rows, dtype=float) if rows else np.empty((0, 7), dtype=float)
