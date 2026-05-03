"""
dynamic_xagent.py

A small wrapper around the original Xagent.
It updates obstacle states from a RouteDynamicObstacleManager before each MPC solve.

Original x_v2x_agent.py is not modified.
"""

import numpy as np

from src.x_v2x_agent import Xagent


class DynamicObstacleXAgent(Xagent):
    def __init__(self, env, model, obstacle_manager=None, dt=0.1):
        super().__init__(env, model, dt=dt)

        self.obstacle_manager = obstacle_manager

        # 兼容新版 x_v2x_agent.py 的 yaw 连续化逻辑
        self._prev_unwrapped_yaw = getattr(self, "_prev_unwrapped_yaw", None)

        # 兼容新版 x_v2x_agent.py 的目标点索引缓存逻辑
        self._last_target_ind = getattr(self, "_last_target_ind", 0)
        self._last_ref_path = getattr(self, "_last_ref_path", None)
        self._last_control = getattr(self, "_last_control", None)

    def set_obstacles(self, obstacles):
        """
        动态避障专用：更新障碍物，但不每帧清空绕行方向。
        这样不会影响原始 x_v2x_agent.py，也不会影响 test_main.py。
        """
        self._obstacles = obstacles

        if not hasattr(self, "_obs_pass_side"):
            self._obs_pass_side = {}

    def run_step(self, lv=None):
        if self.obstacle_manager is not None:
            self.set_obstacles(self.obstacle_manager.get_obstacle_array())

        return super().run_step(lv=lv)

    def _align_angle_to_ref(self, angle, ref_angle):
        """
        将 angle 调整到 ref_angle 附近，避免 yaw 从 +pi 到 -pi 跳变。
        """
        while angle - ref_angle > np.pi:
            angle -= 2.0 * np.pi

        while angle - ref_angle < -np.pi:
            angle += 2.0 * np.pi

        return angle
