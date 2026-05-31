"""Trajectory planning: discrete iLQR with optional SCP collision avoidance."""

from .config import DEFAULT_CLEARANCE_MARGIN, DEFAULT_HORIZON, DEFAULT_RESAMPLE_TO
from .lqr_planner import plan_lqr_trajectory

__all__ = [
    "DEFAULT_CLEARANCE_MARGIN",
    "DEFAULT_HORIZON",
    "DEFAULT_RESAMPLE_TO",
    "plan_lqr_trajectory",
]
