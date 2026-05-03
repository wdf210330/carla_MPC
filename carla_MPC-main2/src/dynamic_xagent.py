"""
dynamic_xagent.py

A small wrapper around the original Xagent. It updates obstacle states from a
RouteDynamicObstacleManager before each MPC solve. Original x_v2x_agent.py is
not modified.
"""

from src.x_v2x_agent import Xagent
import numpy as np

class DynamicObstacleXAgent(Xagent):
    def __init__(self, env, model, obstacle_manager=None, dt=0.1):
        super().__init__(env, model, dt=dt)
        self.obstacle_manager = obstacle_manager

        # 兼容新版 x_v2x_agent.py 里的 yaw 连续化逻辑
        # 防止 run_step() 第一次调用时报:
        # AttributeError: 'DynamicObstacleXAgent' object has no attribute '_prev_unwrapped_yaw'
        if not hasattr(self, "_prev_unwrapped_yaw"):
            self._prev_unwrapped_yaw = None

        if not hasattr(self, "_last_target_ind"):
            self._last_target_ind = 0

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