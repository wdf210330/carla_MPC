import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.x_v2x_agent import Xagent
from two_obstacles.soft_obstacle_field import apply_soft_obstacle_field


class SoftFieldMPCXAgent(Xagent):
    """Xagent variant that offsets reference waypoints using a soft obstacle field."""

    def __init__(self, env, model, dt=0.1):
        super().__init__(env, model, dt=dt)
        self.enable_soft_field = True
        self.field_influence_dist = 7.5
        self.field_safe_dist = 1.4
        self.field_side_clearance = 2.8
        self.field_side_cost_weight = 0.35
        self.field_pass_side = 1.0
        self.field_front_cost_distance = 17.0
        self.field_back_release_distance = 0.8

    def apply_obs_avoidance_offset(self, waypoints, ego_x, ego_y, ego_yaw):
        if not self.enable_soft_field or self._obstacles is None:
            return waypoints

        return apply_soft_obstacle_field(
            waypoints=waypoints,
            ego_x=ego_x,
            ego_y=ego_y,
            ego_yaw=ego_yaw,
            obstacles=np.array(self._obstacles, dtype=float),
            influence_dist=self.field_influence_dist,
            safe_dist=self.field_safe_dist,
            side_clearance=self.field_side_clearance,
            side_cost_weight=self.field_side_cost_weight,
            pass_side=self.field_pass_side,
            front_cost_distance=self.field_front_cost_distance,
            back_release_distance=self.field_back_release_distance,
        )
