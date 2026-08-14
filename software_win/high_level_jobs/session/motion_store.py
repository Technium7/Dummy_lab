"""Persist / reverse motion segments for put_sensor and return."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .paths import LIBRARY_DIR, RECORDS_DIR

_SEGMENT_KEYS = ("pre_grasp", "grasp", "lift", "home")


def reverse_segment(seg: np.ndarray | None) -> np.ndarray | None:
    if seg is None:
        return None
    arr = np.asarray(seg, dtype=np.float64)
    if arr.ndim != 2 or len(arr) < 2:
        return None
    return np.ascontiguousarray(arr[::-1])


def _seg_to_list(seg: np.ndarray | None) -> list[list[float]] | None:
    if seg is None:
        return None
    arr = np.asarray(seg, dtype=np.float64)
    if arr.ndim != 2 or len(arr) < 2:
        return None
    return arr.tolist()


def _list_to_seg(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or len(arr) < 2:
        return None
    return arr


@dataclass
class MotionRecord:
    """One recorded skill motion (segment waypoints in planner joint rad)."""

    name: str
    kind: str  # grab_sensor | power_measure | standby | ...
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    start_joints_deg: list[float] = field(default_factory=list)
    segments: dict[str, np.ndarray | None] = field(default_factory=dict)
    plan_meta: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def segment(self, key: str) -> np.ndarray | None:
        return self.segments.get(key)

    def reversed_put_order(self) -> list[tuple[str, str, np.ndarray]]:
        """WORK→lift→grasp→pre→WORK put-back order as (key, label, traj)."""
        out: list[tuple[str, str, np.ndarray]] = []
        mapping = [
            ("home", "WORK -> lift (rev home)"),
            ("lift", "lift -> grasp (rev lift)"),
            ("grasp", "grasp -> pre (rev grasp)"),
            ("pre_grasp", "pre -> WORK (rev pre)"),
        ]
        for key, label in mapping:
            rev = reverse_segment(self.segment(key))
            if rev is not None:
                out.append((key, label, rev))
        return out

    def to_insert_segments(self) -> list[tuple[str, np.ndarray]]:
        """Outbound place path: WORK → pre → insert (pre_grasp + grasp)."""
        out: list[tuple[str, np.ndarray]] = []
        for key, label in (
            ("pre_grasp", "WORK -> pre"),
            ("grasp", "pre -> insert"),
        ):
            seg = self.segment(key)
            if seg is not None:
                out.append((label, np.asarray(seg, dtype=np.float64)))
        return out

    def return_to_work_segments(self) -> list[tuple[str, np.ndarray]]:
        """Prefer planned lift+home; else reverse placement (grasp+pre)."""
        out: list[tuple[str, np.ndarray]] = []
        lift = self.segment("lift")
        home = self.segment("home")
        if lift is not None:
            out.append(("post-insert lift", np.asarray(lift, dtype=np.float64)))
        if home is not None:
            out.append(("return WORK", np.asarray(home, dtype=np.float64)))
        if out:
            return out
        # Placement-only record without lift/home: reverse grasp then pre.
        for key, label in (
            ("grasp", "insert -> pre (rev)"),
            ("pre_grasp", "pre -> WORK (rev)"),
        ):
            rev = reverse_segment(self.segment(key))
            if rev is not None:
                out.append((label, rev))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "created_utc": self.created_utc,
            "start_joints_deg": list(self.start_joints_deg),
            "segments": {k: _seg_to_list(self.segments.get(k)) for k in _SEGMENT_KEYS},
            "plan_meta": _jsonable_meta(self.plan_meta),
            "extras": _jsonable_meta(self.extras),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MotionRecord:
        segs_raw = data.get("segments") or {}
        segments = {k: _list_to_seg(segs_raw.get(k)) for k in _SEGMENT_KEYS}
        return cls(
            name=str(data.get("name") or "unnamed"),
            kind=str(data.get("kind") or ""),
            created_utc=str(data.get("created_utc") or ""),
            start_joints_deg=[float(x) for x in (data.get("start_joints_deg") or [])],
            segments=segments,
            plan_meta=dict(data.get("plan_meta") or {}),
            extras=dict(data.get("extras") or {}),
        )


def _jsonable_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if k == "segments":
            continue
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = str(v)
    return out


def record_from_plan(
    *,
    name: str,
    kind: str,
    plan_meta: Mapping[str, Any],
    start_joints_deg: list[float] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> MotionRecord:
    raw = plan_meta.get("segments") or {}
    segments = {k: _list_to_seg(raw.get(k)) for k in _SEGMENT_KEYS}
    return MotionRecord(
        name=name,
        kind=kind,
        start_joints_deg=list(start_joints_deg or []),
        segments=segments,
        plan_meta=_jsonable_meta(plan_meta),
        extras=dict(extras or {}),
    )


def records_dir(base: Path | None = None) -> Path:
    path = Path(base) if base is not None else RECORDS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_path(name: str, base: Path | None = None) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip()) or "record"
    return records_dir(base) / f"{safe}.json"


def save_record(record: MotionRecord, base: Path | None = None) -> Path:
    path = record_path(record.name, base)
    path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    print(f"[session] motion record saved: {path}")
    return path


def load_record(name: str, base: Path | None = None) -> MotionRecord | None:
    path = record_path(name, base)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return MotionRecord.from_dict(data)


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip()) or "traj"


def library_dir(base: Path | None = None) -> Path:
    """Named trajectory library under records/library (or base/library)."""
    if base is None:
        path = LIBRARY_DIR
    else:
        path = Path(base) / "library"
    path.mkdir(parents=True, exist_ok=True)
    return path


def library_path(name: str, base: Path | None = None) -> Path:
    return library_dir(base) / f"{_safe_name(name)}.json"


def list_library(base: Path | None = None) -> list[str]:
    root = library_dir(base)
    names = sorted(p.stem for p in root.glob("*.json") if p.is_file())
    return names


def save_library(record: MotionRecord, name: str, base: Path | None = None) -> Path:
    """Copy a motion record into the named trajectory library."""
    named = MotionRecord(
        name=_safe_name(name),
        kind=record.kind,
        created_utc=record.created_utc,
        start_joints_deg=list(record.start_joints_deg),
        segments=dict(record.segments),
        plan_meta=dict(record.plan_meta),
        extras=dict(record.extras),
    )
    path = library_path(named.name, base)
    path.write_text(json.dumps(named.to_dict(), indent=2), encoding="utf-8")
    print(f"[session] traj library saved: {path}")
    return path


def load_library(name: str, base: Path | None = None) -> MotionRecord | None:
    path = library_path(name, base)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return MotionRecord.from_dict(data)
