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
    """Poll joints until consecutive reads differ by ≤ tol_deg (motion stopped)."""
    from trajectory_run.run_planned_trajectory import read_physical_joints

    tol = max(1e-3, float(tol_deg))
    need = max(1, int(stable_reads))
    poll = max(0.05, float(poll_s))
    deadline = time.perf_counter() + max(0.5, float(timeout_s))
    prev: list[float] | None = None
    stable = 0
    print(
        f"{label}waiting for joints quiet "
        f"(tol≤{tol:.3f} deg, {need} stable reads, timeout {timeout_s:g}s)..."
    )
    while time.perf_counter() < deadline:
        cur = read_physical_joints(robot)
        if cur is None:
            time.sleep(poll)
            continue
        if prev is not None:
            err = float(np.max(np.abs(np.asarray(cur[:6]) - np.asarray(prev[:6]))))
            if err <= tol:
                stable += 1
                if stable >= need:
                    print(f"{label}joints quiet (Δ≤{err:.4f} deg).")
                    return True
            else:
                stable = 0
        prev = list(cur[:6])
        time.sleep(poll)
    print(f"{label}warning: joints not quiet within {timeout_s:g}s; continuing.")
    return False


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
            "settle_s": settle_s,
            "n_reads": n_reads,
            "power_w": None,
            "samples_w": [],
            "dry_run": True,
            "extras": dict(extras or {}),
        }
        _append_log(log_path, entry)
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
    _append_log(log_path, entry)
    print(f"{label}averaged power = {power_w:.6e} W  (n={n_reads}, settle={settle_s:g}s)")
    if log_path is not None:
        print(f"{label}logged -> {log_path}")
    return entry


def _append_log(log_path: Path | None, entry: Mapping[str, Any]) -> None:
    if log_path is None:
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
