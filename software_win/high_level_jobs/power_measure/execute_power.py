"""Execute power_measure place / optimize / return on the robot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from high_level_jobs.grab_sensor.execute import (
    _confirm_return_home,
    _log_home_error,
    _motion_segments,
    _populate_pre_grasp_joints_deg,
    _stream_segment_or_raise,
    ensure_robot_ready,
    open_gripper,
)
from high_level_jobs.grab_sensor.models import GraspTargets
from high_level_jobs.grab_sensor.plan_motion import work_joints_deg
from robot_ascii_interface import RobotAsciiInterface

from .meter import open_power_meter_from_config
from .models import PowerMeasureConfig
from .power_scan import (
    PlaneMover,
    record_insert_power,
    run_power_protocol,
    save_power_scan_debug,
)
from .target import InsertTarget


def _stream_kw(params: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    return {
        "smooth": bool(params.get("smooth", True)),
        "send_interval_ms": float(params.get("send_interval_ms", 160.0)),
        "speed_scale": float(params.get("speed_scale", 0.55)),
        "label": label,
        "min_speed_scale": 0.05,
    }


def _ensure_ready(
    robot: RobotAsciiInterface,
    cfg: PowerMeasureConfig,
    params: Mapping[str, Any],
    *,
    label: str,
    home_first: bool,
) -> list[float]:
    grab_cfg = cfg.to_grab_sensor_config()
    work_deg = work_joints_deg(grab_cfg)
    cmd_mode = int(params.get("cmd_mode", 2))
    joint_current_limit_a = float(
        params.get("joint_current_limit_a", cfg.execute.joint_current_limit_a)
    )
    if not ensure_robot_ready(
        robot,
        cmd_mode=cmd_mode,
        home_first=home_first,
        open_hand=False,
        joint_current_limit_a=joint_current_limit_a,
        label=label,
        start_joints_deg=work_deg,
    ):
        raise RuntimeError("Failed to reach WORK before power_measure execution.")
    return work_deg


def execute_power_place_on_robot(
    robot: RobotAsciiInterface,
    _trajectory_rad: np.ndarray,
    targets: GraspTargets,
    cfg: PowerMeasureConfig,
    insert: InsertTarget,
    params: Mapping[str, Any],
    *,
    label: str = "",
) -> dict[str, Any]:
    """WORK → pre → insert, then stay (no search / lift / return)."""
    del targets, insert
    exec_params = dict(params)
    meta = exec_params.get("plan_meta") or {}
    _populate_pre_grasp_joints_deg(exec_params, meta)

    segments = _motion_segments(meta)
    if segments.get("grasp") is None:
        raise RuntimeError(
            "execute_power_place_on_robot requires plan_meta['segments']['grasp']."
        )

    pre_seg = segments.get("pre_grasp")
    grasp_seg = segments.get("grasp")
    home_first = bool(params.get("home_first", True))
    open_gripper_at_pre = bool(params.get("open_gripper", False))
    sample_power = (
        bool(params.get("sample_power_on_place", True))
        and bool(cfg.power_meter.enabled)
        and not bool(params.get("no_power_scan", False))
    )

    meter = None
    insert_samples: list = []
    along_max = 0.0
    insert_p = 0.0
    try:
        if sample_power:
            meter = open_power_meter_from_config(cfg.power_meter)

        _ensure_ready(robot, cfg, params, label=label, home_first=home_first)
        stream_kw = _stream_kw(params, label=label)
        armed = home_first

        if pre_seg is not None:
            print(f"{label}Phase 1: WORK -> pre-insert ({len(pre_seg)} planner samples).")
            approach = float(params.get("approach_speed_scale", 0.40))
            _stream_segment_or_raise(
                robot,
                pre_seg,
                segment="WORK -> pre-insert",
                stream_kw={
                    **stream_kw,
                    "speed_scale": min(float(stream_kw["speed_scale"]), approach),
                },
                arm=False,
                seed_from_robot=False,
                sync_start_to_robot=True,
                ramp_speed_scale=min(approach, float(stream_kw["speed_scale"])),
            )
            armed = True
            if open_gripper_at_pre:
                open_gripper(robot, label=label)

        assert grasp_seg is not None
        phase = "pre-insert -> insert" if pre_seg is not None else "WORK -> insert"
        print(f"{label}Phase 2: {phase} (place; stay at insert).")

        if meter is not None:
            insert_samples, along_max, insert_p = record_insert_power(
                meter,
                robot,
                grasp_seg,
                cfg.power_scan,
                label=label,
                stream_kw={
                    **stream_kw,
                    "speed_scale": min(
                        float(stream_kw["speed_scale"]),
                        float(cfg.power_scan.search_speed_scale),
                    ),
                    "send_interval_ms": max(
                        float(stream_kw.get("send_interval_ms", 120.0)),
                        float(cfg.power_scan.send_interval_ms),
                    ),
                    "min_speed_scale": 0.05,
                },
            )
        else:
            _stream_segment_or_raise(
                robot,
                grasp_seg,
                segment=phase,
                stream_kw=stream_kw,
                arm=not armed,
                seed_from_robot=not armed,
                sync_start_to_robot=True,
            )
        print(f"{label}[power_measure] place done — holding at insert.")
    finally:
        if meter is not None:
            meter.close()

    return {
        "mode": "place",
        "insert_samples": insert_samples,
        "along_max_w": along_max,
        "insert_power_w": insert_p,
        "power_scan": None,
    }


def execute_power_optimize_on_robot(
    robot: RobotAsciiInterface,
    cfg: PowerMeasureConfig,
    insert: InsertTarget,
    params: Mapping[str, Any],
    *,
    label: str = "",
    pull_up_after: bool = False,
) -> dict[str, Any]:
    """From current insert pose: PM100 search/climb; stay (optional in-plane pull-up)."""
    meta = dict(params.get("plan_meta") or {})
    grasp_q = meta.get("grasp_goal_q")
    if grasp_q is None:
        raise RuntimeError(
            "power_optimization requires plan_meta['grasp_goal_q'] "
            "(run power_measure place first)."
        )

    cmd_mode = int(params.get("cmd_mode", 2))
    joint_current_limit_a = float(
        params.get("joint_current_limit_a", cfg.execute.joint_current_limit_a)
    )
    if not ensure_robot_ready(
        robot,
        cmd_mode=cmd_mode,
        home_first=False,
        open_hand=False,
        joint_current_limit_a=joint_current_limit_a,
        label=label,
    ):
        raise RuntimeError("Failed to arm robot for power_optimization.")

    meter = None
    mover = None
    scan_result = None
    try:
        meter = open_power_meter_from_config(cfg.power_meter)
        seed_p = float(params.get("insert_power_w") or 0.0)
        if seed_p <= 0.0:
            seed_p = float(meter.read_watts(label="optimize_seed").watts)
        along_max = float(params.get("along_max_w") or seed_p)
        insert_samples = list(params.get("insert_samples") or [])

        mover = PlaneMover(
            cfg,
            insert,
            robot,
            grasp_q_sim=grasp_q,
            label=label,
            speed_scale=float(
                getattr(cfg.power_scan, "search_speed_scale", cfg.power_scan.move_speed_scale)
            ),
            send_interval_ms=float(cfg.power_scan.send_interval_ms),
            settle_timeout_sec=float(cfg.power_scan.settle_timeout_sec),
        )
        print(
            f"{label}Phase optimize: power protocol "
            f"(λ={cfg.power_meter.wavelength_nm:.0f} nm, "
            f"square {cfg.power_scan.side_mm:.0f}×{cfg.power_scan.side_mm:.0f} mm "
            f"directed climb step={cfg.power_scan.opt_step_mm():.0f} mm)"
        )
        scan_result = run_power_protocol(
            cfg,
            insert,
            meter,
            mover,
            insert_samples=insert_samples,
            along_max_w=along_max,
            insert_power_w=seed_p,
        )
        save_power_scan_debug(cfg, scan_result)
        from high_level_jobs.power_measure.record_power import log_optimization_power

        log_path = params.get("power_log_path")
        if log_path is None:
            from high_level_jobs.session.paths import POWER_LOG_DIR, POWER_LOG_NAME

            log_path = POWER_LOG_DIR / POWER_LOG_NAME
        log_optimization_power(
            scan_result.to_dict(),
            log_path=Path(log_path),
            extras=params.get("power_log_extras") or {},
            settle_s=float(cfg.power_scan.final_hold_s),
            n_reads=int(cfg.power_scan.final_meter_samples),
            label=f"{label}[power_optimization] ",
        )

        if pull_up_after:
            lift_mm = float(cfg.approach.post_grasp_lift_mm)
            if lift_mm > 0.0:
                print(f"{label}Phase pull-up: {lift_mm:.0f} mm along +Z from search pose.")
                mover.set_cruise_speed(
                    float(getattr(cfg.power_scan, "climb_speed_scale", 0.10)),
                    reason="pull up",
                )
                mover.move_to(
                    mover.a_m,
                    mover.b_m + lift_mm * 1e-3,
                    name="pull_up",
                    hold_end=True,
                )
        else:
            print(f"{label}[power_measure] optimize done — holding at peak.")
    finally:
        if mover is not None:
            mover.close()
        if meter is not None:
            meter.close()

    return {
        "mode": "optimize",
        "power_scan": scan_result.to_dict() if scan_result is not None else None,
    }


def execute_power_return_on_robot(
    robot: RobotAsciiInterface,
    cfg: PowerMeasureConfig,
    params: Mapping[str, Any],
    *,
    label: str = "",
    skip_lift: bool = False,
) -> None:
    """Stream planned lift (optional) + home segments back to WORK."""
    meta = dict(params.get("plan_meta") or {})
    segments = _motion_segments(meta)
    lift_seg = None if skip_lift else segments.get("lift")
    home_seg = segments.get("home")
    work_deg = work_joints_deg(cfg.to_grab_sensor_config())
    cmd_mode = int(params.get("cmd_mode", 2))
    return_home = bool(meta.get("plan_return_home", cfg.approach.plan_return_home))
    stream_kw = _stream_kw(params, label=label)

    joint_current_limit_a = float(
        params.get("joint_current_limit_a", cfg.execute.joint_current_limit_a)
    )
    if not ensure_robot_ready(
        robot,
        cmd_mode=cmd_mode,
        home_first=False,
        open_hand=False,
        joint_current_limit_a=joint_current_limit_a,
        label=label,
    ):
        raise RuntimeError("Failed to arm robot for power return.")

    if lift_seg is not None:
        print(f"{label}Phase return: pull up ({len(lift_seg)} planner samples).")
        _stream_segment_or_raise(
            robot,
            lift_seg,
            segment="post-insert lift",
            stream_kw=stream_kw,
            arm=False,
            seed_from_robot=False,
            sync_start_to_robot=True,
        )

    if home_seg is not None:
        print(f"{label}Phase return: WORK ({len(home_seg)} planner samples).")
        home_ramp = params.get("return_home_speed_scale")
        _stream_segment_or_raise(
            robot,
            home_seg,
            segment="return WORK",
            stream_kw=stream_kw,
            arm=False,
            seed_from_robot=False,
            sync_start_to_robot=True,
            ramp_speed_scale=float(home_ramp) if home_ramp is not None else None,
        )
        _log_home_error(
            robot, label=label, step="After return stream", target_joints_deg=work_deg
        )

    if return_home:
        _confirm_return_home(
            robot,
            cmd_mode=cmd_mode,
            params=params,
            label=label,
            target_joints_deg=work_deg,
        )


def execute_power_measure_on_robot(
    robot: RobotAsciiInterface,
    trajectory_rad: np.ndarray,
    targets: GraspTargets,
    cfg: PowerMeasureConfig,
    insert: InsertTarget,
    params: Mapping[str, Any],
    *,
    label: str = "",
) -> dict[str, Any]:
    """One-shot convenience: place → optimize (+ in-plane pull-up) → return WORK."""
    place = execute_power_place_on_robot(
        robot, trajectory_rad, targets, cfg, insert, params, label=label
    )
    opt_params = dict(params)
    opt_params["home_first"] = False
    opt_params["insert_power_w"] = place.get("insert_power_w") or 0.0
    opt_params["along_max_w"] = place.get("along_max_w") or 0.0
    opt_params["insert_samples"] = place.get("insert_samples") or []

    opt = execute_power_optimize_on_robot(
        robot,
        cfg,
        insert,
        opt_params,
        label=label,
        pull_up_after=True,
    )
    # In-plane pull-up already done; skip planned lift segment.
    execute_power_return_on_robot(
        robot, cfg, opt_params, label=label, skip_lift=True
    )
    return {
        "mode": "full",
        "place_result": place,
        "power_scan": opt.get("power_scan"),
    }
