"""Lateral power search / maximize in the plane ⊥ robot→target (radial).

Search plane basis at insert depth:
  e_left  — left of robot→target in table XY (CCW 90° from radial)
  e_z     — world +Z

Offsets (a, b) in meters along (e_left, e_z). Trial region is a
``side_mm`` × ``side_mm`` square. Slow serpentine scan streams between
grid points with no post-move settle wait.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from high_level_jobs.grab_sensor.execute import execute_grasp_trajectory
from high_level_jobs.grab_sensor.plan_motion import build_scene, start_q_from_robot_joints_deg
from high_level_jobs.grab_sensor.tcp_ik import (
    TcpOffset,
    measure_tcp_target_pose_error,
    solve_tcp_ik,
    tcp_world_rpy_from_q,
)
from trajectory_planner.collision import is_state_valid
from trajectory_planner.scene import disconnect_planning_scene, set_joint_positions
from trajectory_run.run_planned_trajectory import (
    _hold_stream_speed,
    read_physical_joints,
    sim_trajectory_to_robot_deg,
)
from manual_control.mapping import clamp_firmware_joints_deg

from .meter import PowerMeter
from .models import PowerMeasureConfig, PowerScanConfig
from .target import InsertTarget


@dataclass
class PlanePoint:
    """Point in the ⊥-radial search plane."""

    a_m: float  # along left
    b_m: float  # along +Z
    tcp_m: np.ndarray
    power_w: float | None = None
    label: str = ""


@dataclass
class PowerScanResult:
    insert_samples: list[dict[str, Any]] = field(default_factory=list)
    along_max_w: float = 0.0
    insert_power_w: float = 0.0
    obvious_at_insert: bool = False
    search_samples: list[dict[str, Any]] = field(default_factory=list)
    optimize_samples: list[dict[str, Any]] = field(default_factory=list)
    first_detect: dict[str, Any] | None = None
    best: dict[str, Any] | None = None
    stop_reason: str = ""
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "insert_samples": self.insert_samples,
            "along_max_w": self.along_max_w,
            "insert_power_w": self.insert_power_w,
            "obvious_at_insert": self.obvious_at_insert,
            "search_samples": self.search_samples,
            "optimize_samples": self.optimize_samples,
            "first_detect": self.first_detect,
            "best": self.best,
            "stop_reason": self.stop_reason,
            "log": self.log,
        }


def plane_basis(insert: InsertTarget) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (origin_tcp, e_left, e_z) in planner/table world meters."""
    origin = np.asarray(insert.tcp_robot_xyz_m, dtype=np.float64)[:3].copy()
    lx, ly = insert.left_unit_xy
    e_left = np.asarray([float(lx), float(ly), 0.0], dtype=np.float64)
    n = float(np.linalg.norm(e_left))
    if n < 1e-9:
        e_left = np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
    else:
        e_left = e_left / n
    e_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return origin, e_left, e_z


def tcp_from_plane(
    origin: np.ndarray,
    e_left: np.ndarray,
    e_z: np.ndarray,
    a_m: float,
    b_m: float,
) -> np.ndarray:
    return origin + float(a_m) * e_left + float(b_m) * e_z


def _in_square(
    a_m: float,
    b_m: float,
    center_a: float,
    center_b: float,
    half_m: float,
) -> bool:
    return abs(a_m - center_a) <= half_m + 1e-12 and abs(b_m - center_b) <= half_m + 1e-12


def _square_serpentine(
    side_mm: float,
    step_mm: float,
    *,
    center_a: float = 0.0,
    center_b: float = 0.0,
) -> list[tuple[float, float]]:
    """Serpentine raster covering a side_mm × side_mm square."""
    half_m = float(side_mm) * 0.5 * 1e-3
    step = max(float(step_mm) * 1e-3, 1e-4)
    n = max(1, int(math.ceil(half_m / step - 1e-12)))
    offs = [float(i) * step for i in range(-n, n + 1)]
    # Clamp exact edges to ±half
    offs = [max(-half_m, min(half_m, v)) for v in offs]
    # Deduplicate near-equal edges
    cleaned: list[float] = []
    for v in offs:
        if not cleaned or abs(v - cleaned[-1]) > 1e-9:
            cleaned.append(v)
    pts: list[tuple[float, float]] = []
    for i, db in enumerate(cleaned):
        row = cleaned if (i % 2 == 0) else list(reversed(cleaned))
        for da in row:
            pts.append((center_a + da, center_b + db))
    return pts


def obvious_power_at_insert(
    insert_w: float,
    along_max_w: float,
    scan: PowerScanConfig,
) -> bool:
    """True when insert reading clearly exceeds the insertion-path baseline."""
    baseline = max(float(along_max_w), 0.0)
    delta = float(insert_w) - baseline
    if delta >= float(scan.contrast_delta_w):
        return True
    if baseline > 0.0 and float(insert_w) >= baseline * float(scan.contrast_ratio):
        return True
    if baseline <= 0.0 and float(insert_w) >= float(scan.detect_threshold_w):
        return True
    return False


def power_detected(watts: float, scan: PowerScanConfig) -> bool:
    return float(watts) >= float(scan.detect_threshold_w)


class PlaneMover:
    """IK + stream TCP moves in the search plane, holding insert orientation."""

    def __init__(
        self,
        cfg: PowerMeasureConfig,
        insert: InsertTarget,
        robot,
        *,
        grasp_q_sim: Sequence[float],
        label: str = "",
        speed_scale: float = 0.16,
        send_interval_ms: float = 220.0,
        settle_timeout_sec: float = 0.0,
    ) -> None:
        self.cfg = cfg
        self.insert = insert
        self.robot = robot
        self.label = label
        self.speed_scale = float(speed_scale)
        self._cruise_speed = float(speed_scale)  # last executed cruise
        self._desired_cruise = float(speed_scale)  # next-move cruise target
        self.send_interval_ms = float(send_interval_ms)
        self.settle_timeout_sec = float(settle_timeout_sec)
        self.origin, self.e_left, self.e_z = plane_basis(insert)
        self.grab_cfg = cfg.to_grab_sensor_config()
        self.scene, self._temp = build_scene(self.grab_cfg, use_gui=False)
        self.tcp = TcpOffset()
        self.q = [float(v) for v in grasp_q_sim]
        set_joint_positions(self.scene, self.q)
        self.hold_rpy = tcp_world_rpy_from_q(self.scene, self.q, self.tcp)
        import pybullet as p

        self.hold_orn = p.getQuaternionFromEuler([float(v) for v in self.hold_rpy])
        self.a_m = 0.0
        self.b_m = 0.0
        self.power_seen = False

    def set_cruise_speed(self, speed_scale: float, *, reason: str = "") -> None:
        """Queue a new cruise; next move_to ramps from the last executed speed."""
        new = max(0.05, float(speed_scale))
        if abs(new - self._desired_cruise) < 1e-6:
            return
        msg = (
            f"[power_measure] cruise speed {self._cruise_speed:.2f} → {new:.2f} "
            f"(ramps on next move)"
        )
        if reason:
            msg += f" ({reason})"
        print(msg)
        self._desired_cruise = new
        self.speed_scale = new

    def close(self) -> None:
        try:
            disconnect_planning_scene(self.scene)
        except Exception:
            pass
        if self._temp is not None:
            try:
                Path(self._temp).unlink(missing_ok=True)
            except Exception:
                pass

    def sync_q_from_robot(self) -> None:
        cur = read_physical_joints(self.robot)
        if cur is None:
            return
        self.q = start_q_from_robot_joints_deg(self.scene, cur)
        set_joint_positions(self.scene, self.q)

    def move_to(
        self,
        a_m: float,
        b_m: float,
        *,
        name: str = "",
        speed_scale: float | None = None,
        send_interval_ms: float | None = None,
        settle_timeout_sec: float | None = None,
        hold_end: bool = True,
        tip_step_mm: float = 2.0,
        tip_speed_mm_s: float = 8.0,
    ) -> bool:
        tip = tcp_from_plane(self.origin, self.e_left, self.e_z, a_m, b_m)
        tip0 = tcp_from_plane(self.origin, self.e_left, self.e_z, self.a_m, self.b_m)
        hop_mm = float(np.linalg.norm(tip - tip0)) * 1e3
        if hop_mm < 0.05:
            self.a_m = float(a_m)
            self.b_m = float(b_m)
            return True

        # Sync before planning so short hops start from the real arm pose.
        self.sync_q_from_robot()

        # Cartesian densify (~tip_step_mm): avoids 2-point smooth blow-ups that
        # stream a single 160 ms jump and miss the commanded TCP.
        n = max(2, int(math.ceil(hop_mm / max(float(tip_step_mm), 0.5))) + 1)
        n = min(n, 40)
        qs: list[list[float]] = []
        q = [float(v) for v in self.q]
        for alpha in np.linspace(0.0, 1.0, n):
            tip_i = tip0 * (1.0 - float(alpha)) + tip * float(alpha)
            set_joint_positions(self.scene, q)
            try:
                q = [float(v) for v in solve_tcp_ik(self.scene, tip_i, self.hold_orn, q)]
            except (ValueError, TypeError) as exc:
                print(
                    f"{self.label}IK failed mid-path a={a_m*1e3:.2f} b={b_m*1e3:.2f} "
                    f"mm (@{alpha:.2f}): {exc}"
                )
                return False
            if not is_state_valid(self.scene, q):
                print(
                    f"{self.label}IK invalid mid-path a={a_m*1e3:.2f} b={b_m*1e3:.2f} "
                    f"mm (@{alpha:.2f})"
                )
                return False
            qs.append(q)

        q_goal = qs[-1]
        pos_err, _ori_err = measure_tcp_target_pose_error(
            self.scene, tip, self.hold_rpy, q_goal, self.tcp
        )
        if pos_err > 0.003:
            print(
                f"{self.label}IK tip err {pos_err*1e3:.2f} mm at "
                f"a={a_m*1e3:.2f} b={b_m*1e3:.2f} — skip"
            )
            return False

        traj = np.asarray(qs, dtype=np.float64)
        target_speed = float(
            self._desired_cruise if speed_scale is None else speed_scale
        )
        target_speed = max(0.05, min(1.0, target_speed))
        prior = float(self._cruise_speed)
        ramp_from = prior if abs(prior - target_speed) > 1e-6 else None

        # Wall-clock pace from tip distance so fallback raw streams are not snaps.
        base_interval = float(
            self.send_interval_ms if send_interval_ms is None else send_interval_ms
        )
        duration_s = max(hop_mm / max(float(tip_speed_mm_s), 1.0), 0.35) / target_speed
        paced_interval = max(base_interval, 1000.0 * duration_s / max(n - 1, 1))
        paced_interval = min(paced_interval, 450.0)

        ok = execute_grasp_trajectory(
            self.robot,
            traj,
            smooth=True,
            send_interval_ms=float(paced_interval),
            speed_scale=target_speed,
            ramp_speed_scale=ramp_from,
            label=self.label,
            segment=name or f"plane a={a_m*1e3:.1f},b={b_m*1e3:.1f}",
            arm=False,
            seed_from_robot=True,
            sync_start_to_robot=True,
            settle_timeout_sec=float(
                self.settle_timeout_sec if settle_timeout_sec is None else settle_timeout_sec
            ),
            hold_end=bool(hold_end),
            min_speed_scale=0.05,
        )
        if not ok:
            return False
        self.sync_q_from_robot()
        self.a_m = float(a_m)
        self.b_m = float(b_m)
        self._cruise_speed = target_speed
        self._desired_cruise = target_speed
        self.speed_scale = target_speed
        hold = clamp_firmware_joints_deg(sim_trajectory_to_robot_deg(np.asarray([self.q]))[0])
        self.robot.hold_stream_pose(hold, _hold_stream_speed())
        return True

    def sample_at(
        self,
        meter: PowerMeter,
        a_m: float,
        b_m: float,
        *,
        label: str = "",
        meter_pause_s: float = 0.08,
        speed_scale: float | None = None,
    ) -> PlanePoint | None:
        if not self.move_to(a_m, b_m, name=label or "scan", speed_scale=speed_scale):
            return None
        if meter_pause_s > 0.0:
            time.sleep(float(meter_pause_s))
        prev_n = meter.samples_per_read
        prev_settle = meter.settle_s
        meter.samples_per_read = 1
        meter.settle_s = 0.0
        try:
            sample = meter.read_watts(label=label)
        finally:
            meter.samples_per_read = prev_n
            meter.settle_s = prev_settle
        tip = tcp_from_plane(self.origin, self.e_left, self.e_z, a_m, b_m)
        return PlanePoint(
            a_m=a_m,
            b_m=b_m,
            tcp_m=tip,
            power_w=sample.watts,
            label=label,
        )


def _point_dict(pt: PlanePoint) -> dict[str, Any]:
    return {
        "a_mm": float(pt.a_m) * 1e3,
        "b_mm": float(pt.b_m) * 1e3,
        "tcp_m": np.asarray(pt.tcp_m).round(5).tolist(),
        "power_w": float(pt.power_w) if pt.power_w is not None else None,
        "label": pt.label,
    }


def slow_scan_square(
    mover: PlaneMover,
    meter: PowerMeter,
    scan: PowerScanConfig,
    *,
    center_a: float = 0.0,
    center_b: float = 0.0,
    step_mm: float | None = None,
    label: str = "scan",
    stop_on_first_detect: bool = False,
) -> tuple[list[PlanePoint], PlanePoint | None, PlanePoint | None]:
    """Slow serpentine over side×side square; no robot settle waits.

    Returns (all_samples, first_detect_or_None, best_or_None).
    """
    step = float(step_mm) if step_mm is not None else scan.sample_step_mm()
    path = _square_serpentine(
        scan.side_mm,
        step,
        center_a=center_a,
        center_b=center_b,
    )
    print(
        f"[power_measure] slow square scan {scan.side_mm:.0f}×{scan.side_mm:.0f} mm  "
        f"step={step:.1f} mm  ({len(path)} pts, no settle wait)"
    )
    samples: list[PlanePoint] = []
    first: PlanePoint | None = None
    best: PlanePoint | None = None
    pause = float(scan.meter_pause_s)
    for a_m, b_m in path:
        pt = mover.sample_at(meter, a_m, b_m, label=label, meter_pause_s=pause)
        if pt is None or pt.power_w is None:
            continue
        samples.append(pt)
        print(
            f"[power_measure] {label} a={pt.a_m*1e3:+.2f} b={pt.b_m*1e3:+.2f} mm  "
            f"P={pt.power_w:.4e} W"
        )
        if best is None or float(pt.power_w) > float(best.power_w or 0.0):
            best = pt
        if first is None and power_detected(pt.power_w, scan):
            first = pt
            if stop_on_first_detect:
                print(
                    f"[power_measure] first power detect at "
                    f"a={pt.a_m*1e3:+.2f} b={pt.b_m*1e3:+.2f} mm  P={pt.power_w:.4e} W"
                )
                break
    return samples, first, best


def run_grid_search(
    mover: PlaneMover,
    meter: PowerMeter,
    scan: PowerScanConfig,
    *,
    center_a: float = 0.0,
    center_b: float = 0.0,
) -> tuple[list[PlanePoint], PlanePoint | None]:
    """Gentle expanding ring from center until power is detected."""
    half_m = float(scan.side_mm) * 0.5 * 1e-3
    step = scan.sample_step_mm() * 1e-3
    pause = float(scan.meter_pause_s)
    samples: list[PlanePoint] = []
    search_speed = float(getattr(scan, "search_speed_scale", scan.move_speed_scale))
    climb_speed = float(getattr(scan, "climb_speed_scale", scan.final_speed_scale))
    mover.set_cruise_speed(search_speed, reason="search")

    def _note_power(pt: PlanePoint) -> None:
        if not mover.power_seen and power_detected(float(pt.power_w or 0.0), scan):
            mover.power_seen = True
            mover.set_cruise_speed(climb_speed, reason="power detected — slow climb")

    origin_pt = mover.sample_at(
        meter, center_a, center_b, label="grid", meter_pause_s=pause
    )
    if origin_pt is not None and origin_pt.power_w is not None:
        samples.append(origin_pt)
        print(
            f"[power_measure] grid a={origin_pt.a_m*1e3:+.2f} b={origin_pt.b_m*1e3:+.2f} mm  "
            f"P={origin_pt.power_w:.4e} W"
        )
        _note_power(origin_pt)
        if power_detected(origin_pt.power_w, scan):
            return samples, origin_pt

    print(
        f"[power_measure] expanding search in {scan.side_mm:.0f}x{scan.side_mm:.0f} mm  "
        f"step={scan.sample_step_mm():.1f} mm (toward signal)"
    )
    radius = step
    while radius <= half_m + 1e-12:
        ring = [
            (center_a + radius, center_b),
            (center_a - radius, center_b),
            (center_a, center_b + radius),
            (center_a, center_b - radius),
            (center_a + radius, center_b + radius),
            (center_a + radius, center_b - radius),
            (center_a - radius, center_b + radius),
            (center_a - radius, center_b - radius),
        ]
        for a_m, b_m in ring:
            if not _in_square(a_m, b_m, center_a, center_b, half_m):
                continue
            pt = mover.sample_at(meter, a_m, b_m, label="grid", meter_pause_s=pause)
            if pt is None or pt.power_w is None:
                continue
            samples.append(pt)
            print(
                f"[power_measure] grid a={pt.a_m*1e3:+.2f} b={pt.b_m*1e3:+.2f} mm  "
                f"P={pt.power_w:.4e} W"
            )
            _note_power(pt)
            if power_detected(pt.power_w, scan):
                print(
                    f"[power_measure] first power detect at "
                    f"a={pt.a_m*1e3:+.2f} b={pt.b_m*1e3:+.2f} mm  P={pt.power_w:.4e} W"
                )
                return samples, pt
        radius += step
    return samples, None


def _cardinal_dirs() -> list[tuple[float, float]]:
    return [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)]


def _read_fast(meter: PowerMeter, *, label: str = "") -> float:
    prev_n = meter.samples_per_read
    prev_settle = meter.settle_s
    meter.samples_per_read = 1
    meter.settle_s = 0.0
    try:
        return float(meter.read_watts(label=label).watts)
    finally:
        meter.samples_per_read = prev_n
        meter.settle_s = prev_settle


def _verify_at(
    mover: PlaneMover,
    meter: PowerMeter,
    a_m: float,
    b_m: float,
    *,
    label: str,
    pause_s: float,
) -> PlanePoint | None:
    """Move carefully to (a,b), settle briefly, read power."""
    if not mover.move_to(a_m, b_m, name=label, hold_end=True):
        return None
    time.sleep(max(float(pause_s), 0.25))
    watts = _read_fast(meter, label=label)
    tip = tcp_from_plane(mover.origin, mover.e_left, mover.e_z, a_m, b_m)
    return PlanePoint(a_m=a_m, b_m=b_m, tcp_m=tip, power_w=watts, label=label)


def run_maximize(
    mover: PlaneMover,
    meter: PowerMeter,
    scan: PowerScanConfig,
    *,
    center_a: float,
    center_b: float,
    start_a: float,
    start_b: float,
    start_power: float,
) -> tuple[list[PlanePoint], PlanePoint, str]:
    """Directed climb with pose verify — never discard a higher measured peak."""
    half_m = float(scan.side_mm) * 0.5 * 1e-3
    step = scan.opt_step_mm() * 1e-3
    min_step = scan.opt_min_step_mm() * 1e-3
    pause = max(float(scan.meter_pause_s), 0.2)
    a, b = float(start_a), float(start_b)
    samples: list[PlanePoint] = []
    reason = "local_max"
    dirs: list[tuple[float, float]] = _cardinal_dirs()
    dir_i = 0
    failed_dirs = 0

    print(
        f"[power_measure] directed climb in {scan.side_mm:.0f}x{scan.side_mm:.0f} mm  "
        f"step={scan.opt_step_mm():.1f}->{scan.opt_min_step_mm():.1f} mm"
    )
    climb_speed = float(getattr(scan, "climb_speed_scale", scan.final_speed_scale))
    mover.power_seen = True
    mover.set_cruise_speed(climb_speed, reason="climb to max")

    # Fresh measurement at the climb origin (do not trust insert-path reading alone).
    origin = _verify_at(
        mover, meter, a, b, label="opt_start", pause_s=pause
    )
    if origin is None or origin.power_w is None:
        origin = PlanePoint(
            a_m=a,
            b_m=b,
            tcp_m=tcp_from_plane(mover.origin, mover.e_left, mover.e_z, a, b),
            power_w=float(start_power),
            label="opt_start",
        )
    samples.append(origin)
    best = origin
    print(
        f"[power_measure] climb start a={best.a_m*1e3:+.2f} b={best.b_m*1e3:+.2f} mm  "
        f"P={float(best.power_w):.4e} W (insert reported {float(start_power):.4e} W)"
    )

    for it in range(int(scan.optimize_max_iters)):
        ua, ub = dirs[dir_i % len(dirs)]
        ca = a + ua * step
        cb = b + ub * step
        if not _in_square(ca, cb, center_a, center_b, half_m):
            failed_dirs += 1
            dir_i += 1
            if failed_dirs >= len(dirs):
                if step <= min_step + 1e-12:
                    reason = "trial_range_edge"
                    break
                step *= 0.5
                failed_dirs = 0
                print(f"[power_measure] optimize step -> {step*1e3:.2f} mm")
            continue

        pt = mover.sample_at(meter, ca, cb, label=f"opt_{it}", meter_pause_s=pause)
        if pt is None or pt.power_w is None:
            failed_dirs += 1
            dir_i += 1
            _verify_at(mover, meter, a, b, label=f"opt_{it}_base", pause_s=pause)
            continue

        samples.append(pt)
        print(
            f"[power_measure] opt[{it}] a={pt.a_m*1e3:+.2f} b={pt.b_m*1e3:+.2f} mm  "
            f"P={pt.power_w:.4e} W"
        )
        if float(pt.power_w) > float(best.power_w or 0.0) * (
            1.0 + float(scan.optimize_min_rel_improve)
        ):
            best = pt
            a, b = best.a_m, best.b_m
            dirs = [(ua, ub)] + [d for d in dirs if d != (ua, ub)]
            dir_i = 0
            failed_dirs = 0
            continue

        # Worse — return to best and re-read so the next probe starts on-peak.
        verified = _verify_at(
            mover, meter, best.a_m, best.b_m, label=f"opt_{it}_base", pause_s=pause
        )
        if verified is not None and verified.power_w is not None:
            samples.append(verified)
            # Keep the higher of stored peak vs re-read (re-read can be noisy/low).
            if float(verified.power_w) > float(best.power_w or 0.0):
                best = verified
            elif float(verified.power_w) < float(best.power_w or 0.0) * 0.5:
                print(
                    f"[power_measure] warn: re-read at best only "
                    f"{verified.power_w:.4e} W (peak was {best.power_w:.4e} W) — "
                    f"pose may have drifted; keeping peak coords"
                )
            a, b = best.a_m, best.b_m

        failed_dirs += 1
        dir_i += 1
        if failed_dirs >= len(dirs):
            if step <= min_step + 1e-12:
                reason = "local_max"
                break
            step *= 0.5
            failed_dirs = 0
            dir_i = 0
            print(f"[power_measure] optimize step -> {step*1e3:.2f} mm")
    else:
        reason = "max_iters"

    peak = best
    print(
        f"[power_measure] final approach to peak "
        f"a={peak.a_m*1e3:+.2f} b={peak.b_m*1e3:+.2f} mm "
        f"(seen P={float(peak.power_w or 0.0):.4e} W, speed={scan.final_speed_scale:.2f})"
    )
    mover.set_cruise_speed(float(scan.final_speed_scale), reason="final approach")
    final_pt = _verify_at(
        mover,
        meter,
        peak.a_m,
        peak.b_m,
        label="opt_final",
        pause_s=max(float(scan.final_hold_s), 0.5),
    )
    # Multi-sample average at peak.
    prev_n = meter.samples_per_read
    prev_settle = meter.settle_s
    meter.samples_per_read = max(1, int(scan.final_meter_samples))
    meter.settle_s = 0.0
    try:
        final = meter.read_watts(label="opt_final")
        final_w = float(final.watts)
    finally:
        meter.samples_per_read = prev_n
        meter.settle_s = prev_settle

    peak_w = float(peak.power_w or 0.0)
    # Final stable power = multi-sample average while held at the peak pose.
    best = PlanePoint(
        a_m=peak.a_m,
        b_m=peak.b_m,
        tcp_m=peak.tcp_m,
        power_w=final_w,
        label="opt_final",
    )
    if final_w < peak_w * 0.7:
        print(
            f"[power_measure] warn: final avg {final_w:.4e} W << climb peak "
            f"{peak_w:.4e} W (same pose; using final avg as stable reading)"
        )
    if final_pt is not None:
        samples.append(final_pt)
    samples.append(best)
    print(
        f"[power_measure] final P={best.power_w:.4e} W at "
        f"a={best.a_m*1e3:+.2f} b={best.b_m*1e3:+.2f} mm "
        f"(climb peak was {peak_w:.4e} W)"
    )
    return samples, best, reason



def record_insert_power(
    meter: PowerMeter,
    robot,
    grasp_seg: np.ndarray,
    scan: PowerScanConfig,
    *,
    label: str = "",
    stream_kw: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], float, float]:
    """One continuous insert stream; snapshot power at sample fracs (no mid-path stops)."""
    from high_level_jobs.grab_sensor.execute import execute_grasp_trajectory

    kw = dict(stream_kw or {})
    traj = np.asarray(grasp_seg, dtype=np.float64)
    n_samp = max(2, int(scan.insert_samples))
    # Sample at equally spaced fractions along the streamed path (including end).
    target_fracs = [float(i) / float(n_samp - 1) for i in range(n_samp)]
    next_k = 0
    samples: list[dict[str, Any]] = []

    # Snapshot only — no settle/multi-average (motion must not pause).
    prev_n = meter.samples_per_read
    prev_settle = meter.settle_s
    meter.samples_per_read = 1
    meter.settle_s = 0.0

    def _snap(frac: float, waypoint_index: int) -> None:
        s = meter.read_watts(label=f"insert_{frac:.2f}")
        samples.append(
            {
                "waypoint_index": int(waypoint_index),
                "frac": float(frac),
                "power_w": float(s.watts),
                "t_s": float(s.t_s),
            }
        )
        print(f"{label}[power_measure] insert frac={frac:.2f}  P={s.watts:.4e} W")

    # Start snapshot before motion (frac=0).
    _snap(target_fracs[0], 0)
    next_k = 1

    def _on_waypoint(i: int, frac: float) -> None:
        nonlocal next_k
        while next_k < len(target_fracs) and frac + 1e-9 >= target_fracs[next_k]:
            _snap(target_fracs[next_k], i)
            next_k += 1

    try:
        ok = execute_grasp_trajectory(
            robot,
            traj,
            segment="insert (continuous + power snapshots)",
            arm=False,
            seed_from_robot=True,
            sync_start_to_robot=True,
            hold_end=True,
            settle_timeout_sec=0.0,
            on_waypoint=_on_waypoint,
            smooth=bool(kw.get("smooth", True)),
            send_interval_ms=float(kw.get("send_interval_ms", 220.0)),
            speed_scale=float(kw.get("speed_scale", 0.12)),
            min_speed_scale=float(kw.get("min_speed_scale", 0.05)),
            label=str(kw.get("label", label)),
        )
    finally:
        meter.samples_per_read = prev_n
        meter.settle_s = prev_settle

    if not ok:
        raise RuntimeError("insert trajectory streaming failed.")

    # Ensure end sample if stream ended before last frac fired.
    if next_k < len(target_fracs):
        s = meter.read_watts(label="insert_1.00")
        samples.append(
            {
                "waypoint_index": -1,
                "frac": 1.0,
                "power_w": float(s.watts),
                "t_s": float(s.t_s),
            }
        )
        print(f"{label}[power_measure] insert frac=1.00  P={s.watts:.4e} W")

    if not samples:
        s = meter.read_watts(label="insert_end")
        return (
            [{"waypoint_index": -1, "frac": 1.0, "power_w": float(s.watts), "t_s": float(s.t_s)}],
            0.0,
            float(s.watts),
        )

    insert_p = float(samples[-1]["power_w"])
    if len(samples) >= 2:
        along_max = max(float(s["power_w"]) for s in samples[:-1])
    else:
        along_max = 0.0
    return samples, along_max, insert_p


def run_power_protocol(
    cfg: PowerMeasureConfig,
    insert: InsertTarget,
    meter: PowerMeter,
    mover: PlaneMover,
    *,
    insert_samples: list[dict[str, Any]],
    along_max_w: float,
    insert_power_w: float,
) -> PowerScanResult:
    scan = cfg.power_scan
    result = PowerScanResult(
        insert_samples=list(insert_samples),
        along_max_w=float(along_max_w),
        insert_power_w=float(insert_power_w),
    )
    result.obvious_at_insert = obvious_power_at_insert(insert_power_w, along_max_w, scan)
    result.log.append(
        f"insert P={insert_power_w:.4e} W  along_max={along_max_w:.4e} W  "
        f"obvious={result.obvious_at_insert}"
    )
    print(f"[power_measure] {result.log[-1]}")

    start_a, start_b = 0.0, 0.0
    start_p = float(insert_power_w)
    center_a, center_b = 0.0, 0.0

    if result.obvious_at_insert and power_detected(insert_power_w, scan):
        result.first_detect = {
            "a_mm": 0.0,
            "b_mm": 0.0,
            "tcp_m": np.asarray(insert.tcp_robot_xyz_m).round(5).tolist(),
            "power_w": float(insert_power_w),
            "label": "insert",
        }
        result.log.append(
            "obvious power at insert — directed climb toward higher intensity"
        )
        print(f"[power_measure] {result.log[-1]}")
    else:
        result.log.append(
            f"no obvious insert contrast — expanding search in "
            f"{scan.side_mm:.0f}×{scan.side_mm:.0f} mm "
            f"(step≈{scan.sample_step_mm():.0f} mm)"
        )
        print(f"[power_measure] {result.log[-1]}")
        grid_pts, first = run_grid_search(mover, meter, scan, center_a=0.0, center_b=0.0)
        result.search_samples = [_point_dict(p) for p in grid_pts]
        if first is None or first.power_w is None:
            result.stop_reason = "no_power_detected"
            result.log.append("grid found no power above threshold — lift/return")
            print(f"[power_measure] {result.log[-1]}")
            return result
        result.first_detect = _point_dict(first)
        start_a, start_b = first.a_m, first.b_m
        start_p = float(first.power_w)
        center_a, center_b = first.a_m, first.b_m
        result.log.append(
            f"directed climb within {scan.side_mm:.0f}×{scan.side_mm:.0f} mm "
            f"centered on first detect (a={center_a*1e3:.2f}, b={center_b*1e3:.2f} mm)"
        )
        print(f"[power_measure] {result.log[-1]}")

    opt_pts, best, reason = run_maximize(
        mover,
        meter,
        scan,
        center_a=center_a,
        center_b=center_b,
        start_a=start_a,
        start_b=start_b,
        start_power=start_p,
    )
    result.optimize_samples = [_point_dict(p) for p in opt_pts]
    result.best = _point_dict(best)
    result.stop_reason = reason
    result.log.append(
        f"done ({reason}) best P={best.power_w:.4e} W at "
        f"a={best.a_m*1e3:+.2f} b={best.b_m*1e3:+.2f} mm"
    )
    print(f"[power_measure] {result.log[-1]}")
    return result


def save_power_scan_debug(cfg: PowerMeasureConfig, result: PowerScanResult) -> Path:
    debug_dir = cfg.save_debug_dir
    if debug_dir is None:
        from .paths import DEBUG_DIR

        debug_dir = DEBUG_DIR
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / "power_scan.json"
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"[power_measure] power scan saved: {path}")
    return path