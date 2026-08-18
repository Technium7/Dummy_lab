"""Hold pose and record an averaged optical-power reading (PM100)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from high_level_jobs.power_measure.config_loader import (
    apply_runtime_calibrations,
    load_power_measure_config,
)
from high_level_jobs.power_measure.meter import open_power_meter_from_config


def wait_until_joints_quiet(
    robot: Any,
    *,
    tol_deg: float = 0.12,
    stable_reads: int = 4,
    poll_s: float = 0.12,
    timeout_s: float = 10.0,
    label: str = "[record_power] ",
) -> bool:
    """Poll joints until consecutive GETJPOS reads differ by ≤ tol_deg."""
    from trajectory_run.run_planned_trajectory import (
        wait_until_joints_quiet as _wait_quiet,
    )

    return _wait_quiet(
        robot,
        tol_deg=tol_deg,
        stable_reads=stable_reads,
        poll_s=poll_s,
        timeout_s=timeout_s,
        label=label,
    )


def _pin_current_pose(robot: Any, *, label: str = "") -> None:
    """One-shot: set stream target to current joints (not a timed wait)."""
    from trajectory_run.run_planned_trajectory import _hold_stream_speed, read_physical_joints
    from manual_control.mapping import clamp_firmware_joints_deg

    q = read_physical_joints(robot)
    if q is None:
        return
    try:
        robot.hold_stream_pose(clamp_firmware_joints_deg(q), _hold_stream_speed())
        print(f"{label}pinned stream target to current joints.")
    except Exception as exc:  # noqa: BLE001
        print(f"{label}warning: hold_stream_pose failed ({exc})")


def record_averaged_power(
    *,
    settle_s: float = 1.0,
    n_reads: int = 5,
    inter_read_s: float = 0.05,
    label: str = "[record_power] ",
    extras: Mapping[str, Any] | None = None,
    log_path: Path | None = None,
    dry_run: bool = False,
    robot: Any | None = None,
    quiet_tol_deg: float = 0.12,
    quiet_timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Wait until joints stop, sleep ``settle_s``, then average meter samples.

    ``hold_stream_pose`` only pins the target once (not a timed hold). The timed
    wait before sampling is ``settle_s``.
    """
    settle_s = max(0.0, float(settle_s))
    n_reads = max(1, int(n_reads))
    ts = datetime.now(timezone.utc).isoformat()
    print(
        f"{label}pipeline: joints-quiet → pin pose → sleep {settle_s:g}s → "
        f"average {n_reads} reads..."
    )

    if dry_run:
        entry = {
            "t_utc": ts,
            "source": "record_power",
            "settle_s": settle_s,
            "n_reads": n_reads,
            "power_w": None,
            "samples_w": [],
            "dry_run": True,
            "extras": dict(extras or {}),
        }
        append_power_log(log_path, entry)
        print(f"{label}dry-run: skip meter")
        return entry

    if robot is not None:
        wait_until_joints_quiet(
            robot,
            tol_deg=quiet_tol_deg,
            timeout_s=quiet_timeout_s,
            label=label,
        )
        _pin_current_pose(robot, label=label)

    if settle_s > 0:
        time.sleep(settle_s)

    cfg = apply_runtime_calibrations(load_power_measure_config(), {})
    meter = open_power_meter_from_config(cfg.power_meter)
    meter.settle_s = 0.0
    samples: list[float] = []
    try:
        for i in range(n_reads):
            sample = meter.read_watts(label=f"{label}r{i}")
            samples.append(float(sample.watts))
            if i + 1 < n_reads and inter_read_s > 0:
                time.sleep(float(inter_read_s))
    finally:
        meter.close()

    power_w = float(sum(samples) / len(samples))
    entry = {
        "t_utc": ts,
        "source": "record_power",
        "settle_s": settle_s,
        "n_reads": n_reads,
        "power_w": power_w,
        "samples_w": samples,
        "dry_run": False,
        "extras": {
            k: extras[k]
            for k in ("click_uv", "traj_name", "name")
            if extras and k in extras
        },
    }
    append_power_log(log_path, entry)
    print(f"{label}averaged power = {power_w:.6e} W  (n={n_reads}, settle={settle_s:g}s)")
    if log_path is not None:
        print(f"{label}logged -> {log_path}")
    return entry


def log_optimization_power(
    scan: Mapping[str, Any] | None,
    *,
    log_path: Path | None,
    extras: Mapping[str, Any] | None = None,
    settle_s: float = 0.7,
    n_reads: int = 5,
    dry_run: bool = False,
    label: str = "[power_optimization] ",
) -> dict[str, Any] | None:
    """Append the optimizer's final peak reading to the session power JSONL."""
    ts = datetime.now(timezone.utc).isoformat()
    extra = dict(extras or {})
    best = dict((scan or {}).get("best") or {}) if not dry_run else {}
    power_w = best.get("power_w") if not dry_run else None
    if not dry_run and power_w is None:
        print(f"{label}no final peak to log (scan best missing).")
        return None

    for k in ("a_mm", "b_mm", "tcp_m"):
        if k in best:
            extra[k] = best[k]
    extra = {
        k: extra[k]
        for k in ("click_uv", "traj_name", "name", "a_mm", "b_mm", "tcp_m")
        if k in extra
    }
    samples: list[float] = []
    if power_w is not None:
        samples = [float(power_w)]
    entry = {
        "t_utc": ts,
        "source": "power_optimization",
        "settle_s": float(settle_s),
        "n_reads": int(n_reads),
        "power_w": float(power_w) if power_w is not None else None,
        "samples_w": samples,
        "dry_run": bool(dry_run),
        "stop_reason": "" if dry_run else str((scan or {}).get("stop_reason") or ""),
        "extras": extra,
    }
    append_power_log(log_path, entry)
    if power_w is not None:
        print(f"{label}final power = {float(power_w):.6e} W  (logged peak)")
    if log_path is not None:
        print(f"{label}logged -> {log_path}")
    return entry


def append_power_log(log_path: Path | None, entry: Mapping[str, Any]) -> None:
    if log_path is None:
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
