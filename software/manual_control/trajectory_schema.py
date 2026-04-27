from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass
class TrajectoryPoint:
    t: float
    joints_deg: List[float]
    joint_vel_deg_s: List[float]
    active_keys: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": float(self.t),
            "joints_deg": [float(v) for v in self.joints_deg],
            "joint_vel_deg_s": [float(v) for v in self.joint_vel_deg_s],
            "active_keys": self.active_keys,
        }


@dataclass
class TrajectoryData:
    version: int
    dt: float
    joint_names: List[str]
    points: List[TrajectoryPoint]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "dt": float(self.dt),
            "joint_names": list(self.joint_names),
            "points": [point.to_dict() for point in self.points],
        }


def save_trajectory(path: Path, data: TrajectoryData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data.to_dict(), indent=2), encoding="utf-8")


def _as_float_list(values: Sequence[Any], expected_len: int, field_name: str) -> List[float]:
    out = [float(v) for v in values]
    if len(out) != expected_len:
        raise ValueError(f"Field {field_name!r} must have exactly {expected_len} values")
    return out


def load_trajectory(path: Path) -> TrajectoryData:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = int(payload.get("version", 1))
    dt = float(payload["dt"])
    joint_names = [str(name) for name in payload["joint_names"]]
    if len(joint_names) != 6:
        raise ValueError("Trajectory expects 6 joints")

    points: List[TrajectoryPoint] = []
    for raw_point in payload["points"]:
        points.append(
            TrajectoryPoint(
                t=float(raw_point["t"]),
                joints_deg=_as_float_list(raw_point["joints_deg"], 6, "joints_deg"),
                joint_vel_deg_s=_as_float_list(raw_point["joint_vel_deg_s"], 6, "joint_vel_deg_s"),
                active_keys=str(raw_point.get("active_keys", "")),
            )
        )

    return TrajectoryData(version=version, dt=dt, joint_names=joint_names, points=points)
