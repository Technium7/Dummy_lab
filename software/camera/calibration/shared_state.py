from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class JointSnapshot:
    """Robot pose written by the control thread, read by the CV thread."""

    transformed_joint_positions_deg: list[float] = field(default_factory=list)
    robot_target_deg: list[float] = field(default_factory=list)
    active_keys: list[str] = field(default_factory=list)
    twist: list[float] = field(default_factory=lambda: [0.0] * 6)
    timestamp: float = 0.0


@dataclass
class CaptureSnapshot:
    """FK + joint pose captured on the control thread when a shot is requested."""

    joints_deg: list[float]
    t_base_target: np.ndarray
    transformed_joint_positions_deg: list[float]
    timestamp: float


class SharedCalibrationBus:
    """Queues and latest joint state between control and CV threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._joint_snapshot = JointSnapshot()
        self.stop = threading.Event()
        self.key_codes: queue.SimpleQueue[int] = queue.SimpleQueue()
        self.capture_signals: queue.SimpleQueue[bool] = queue.SimpleQueue()
        self.capture_snapshots: queue.SimpleQueue[CaptureSnapshot] = queue.SimpleQueue()

    def publish_joint_snapshot(
        self,
        *,
        transformed_joint_positions_deg: list[float],
        robot_target_deg: list[float],
        active_keys: list[str],
        twist: list[float],
        timestamp: float,
    ) -> None:
        with self._lock:
            self._joint_snapshot = JointSnapshot(
                transformed_joint_positions_deg=list(transformed_joint_positions_deg),
                robot_target_deg=list(robot_target_deg),
                active_keys=list(active_keys),
                twist=list(twist),
                timestamp=timestamp,
            )

    def read_joint_snapshot(self) -> JointSnapshot:
        with self._lock:
            s = self._joint_snapshot
            return JointSnapshot(
                transformed_joint_positions_deg=list(s.transformed_joint_positions_deg),
                robot_target_deg=list(s.robot_target_deg),
                active_keys=list(s.active_keys),
                twist=list(s.twist),
                timestamp=s.timestamp,
            )
