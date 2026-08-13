"""Plan / preview / execute epoxy phases: pre_apply, xy_optimize, z_creep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from high_level_jobs.grab_sensor.execute import execute_on_robot
from high_level_jobs.grab_sensor.grasp import finalize_grasp_targets
from high_level_jobs.grab_sensor.ik_debug import show_ik_goal_debug_gui
from high_level_jobs.grab_sensor.plan_motion import plan_approach_and_grasp

from .approach_config import apply_runtime_calibrations
from .models import ALL_PHASES, EpoxyApproachConfig, EpoxyPhase
from .target import build_rough_approach_targets

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_DEBUG_DIR = _PKG_DIR / "debug"


def normalize_phases(phases: Sequence[str] | None) -> list[EpoxyPhase]:
    """Parse CLI phase list; default = all three in order."""
    if not phases:
        return list(ALL_PHASES)
    out: list[EpoxyPhase] = []
    seen: set[str] = set()
    for raw in phases:
        name = str(raw).strip().lower().replace("-", "_")
        aliases = {
            "pre": "pre_apply",
            "preapply": "pre_apply",
            "pre_apply": "pre_apply",
            "xy": "xy_optimize",
            "xy_opt": "xy_optimize",
            "xy_optimize": "xy_optimize",
            "z": "z_creep",
            "zcreep": "z_creep",
            "z_creep": "z_creep",
            "apply": "z_creep",
        }
        phase = aliases.get(name)
        if phase is None:
            raise ValueError(
                f"Unknown phase {raw!r}. Use: pre_apply, xy_optimize, z_creep"
            )
        if phase not in seen:
            out.append(phase)  # type: ignore[arg-type]
            seen.add(phase)
    # Keep pipeline order even if user passed out of order.
    return [p for p in ALL_PHASES if p in seen]


def _save_debug(cfg: EpoxyApproachConfig, targets, plan_meta: Mapping[str, Any]) -> None:
    debug_dir = cfg.save_debug_dir or _DEFAULT_DEBUG_DIR
    debug_dir.mkdir(parents=True, exist_ok=True)
    epoxy_meta = {
        k: (int(v) if isinstance(v, (np.integer,)) else v)
        for k, v in plan_meta.items()
        if k != "segments" and not isinstance(v, np.ndarray)
    }
    if "n_grasp" in epoxy_meta:
        epoxy_meta["n_rough_tcp"] = epoxy_meta["n_grasp"]
    payload = {
        "pre_apply": {
            k: v
            for k, v in targets.debug.items()
            if not isinstance(v, np.ndarray)
        },
        "plan_meta": epoxy_meta,
    }
    out = debug_dir / "epoxy_approach.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[epoxy_app] debug saved: {out}")


def run_pre_apply_phase(
    cfg: EpoxyApproachConfig,
    params: Mapping[str, Any],
    *,
    robot=None,
    start_joints_deg: list[float] | None = None,
) -> dict[str, Any]:
    """Phase 1: plan/execute WORK → rough TCP (one pose); stay."""
    cfg = apply_runtime_calibrations(cfg, params)
    grab_cfg = cfg.to_grab_sensor_config()
    targets = build_rough_approach_targets(cfg)

    from high_level_jobs.grab_sensor.table_position import (
        format_robot_in_table_display,
        format_table_xyz_display,
    )

    tcp_m = cfg.effective_tcp_table_xyz_m()
    tcp_in = cfg.effective_tcp_table_xyz_in()
    print(f"[epoxy_app] phase=pre_apply (single pose)")
    print(f"[epoxy_app] robot in table frame: {format_robot_in_table_display(cfg.robot_in_table_frame)}")
    print(
        f"[epoxy_app] nominal (table): "
        f"{format_table_xyz_display([v * 0.0254 for v in cfg.nominal_tcp_table_xyz_in])}"
    )
    print(
        f"[epoxy_app] rough TCP (table, safety y-{cfg.safety_offset_y_in:.2f} "
        f"z+{cfg.safety_offset_z_in:.2f}): {format_table_xyz_display(tcp_m)} "
        f"-> {[round(v, 2) for v in tcp_in]} in"
    )

    if start_joints_deg is None:
        from high_level_jobs.grab_sensor.plan_motion import work_joints_deg

        start_joints_deg = list(work_joints_deg(grab_cfg))
        print(f"[epoxy_app] start joints (WORK) deg={start_joints_deg}")

    finalize_grasp_targets(grab_cfg, targets)
    if grab_cfg.ik_ignore_yaw:
        print("[epoxy_app] IK enforces A/B only (yaw free / not taught)")
    print(
        f"[epoxy_app] finalized RPY world deg={targets.debug.get('grasp_rpy_world_deg')}  "
        f"TCP robot m={targets.grasp_pos_m.round(4).tolist()}"
    )

    if bool(params.get("ik_gui", False)):
        print("[epoxy_app] opening PyBullet IK GUI (--ik-gui); close window to exit.")
        show_ik_goal_debug_gui(grab_cfg, targets)
        return {
            "phase": "pre_apply",
            "targets": targets,
            "trajectory_rad": None,
            "plan_meta": {},
            "start_joints_deg": start_joints_deg,
        }

    trajectory, plan_meta = plan_approach_and_grasp(
        grab_cfg,
        start_joints_deg,
        targets,
        use_gui=bool(params.get("plan_gui", False)),
    )
    print(f"[epoxy_app] planned {plan_meta['n_total']} waypoints (WORK -> rough TCP).")
    _save_debug(cfg, targets, plan_meta)

    if bool(params.get("plan_gui", False)):
        from high_level_jobs.grab_sensor.ik_debug import playback_grasp_plan_in_gui

        print("[epoxy_app] opening PyBullet plan playback (--plan-gui); close window to exit.")
        playback_grasp_plan_in_gui(grab_cfg, trajectory, targets)
    else:
        print(
            "[epoxy_app] No PyBullet window. "
            "Add --ik-gui or --plan-gui for robot preview (env loaded)."
        )

    result: dict[str, Any] = {
        "phase": "pre_apply",
        "targets": targets,
        "trajectory_rad": trajectory,
        "plan_meta": plan_meta,
        "start_joints_deg": start_joints_deg,
    }

    if bool(params.get("execute", False)):
        exec_params = dict(params)
        exec_params.setdefault("joint_current_limit_a", cfg.execute.joint_current_limit_a)
        exec_params.setdefault("home_confirm_attempts", cfg.execute.home_confirm_attempts)
        exec_params["open_gripper"] = False
        exec_params["close_gripper"] = False
        exec_params["plan_meta"] = plan_meta

        own_robot = False
        if robot is None:
            from robot_ascii_interface import RobotAsciiInterface
            from robot_config import default_robot_port

            port = str(params.get("port") or default_robot_port())
            robot = RobotAsciiInterface(
                port=port,
                baudrate=int(params.get("baudrate", 115200)),
            )
            if not robot.connect():
                raise RuntimeError(f"Failed to connect robot on {port}")
            own_robot = True
        try:
            execute_on_robot(robot, trajectory, targets, grab_cfg, exec_params, label="[epoxy_app] ")
            print(
                "[epoxy_app] pre_apply done — at rough TCP. "
                "Next: --phase xy_optimize (placeholder) then --phase z_creep.",
                flush=True,
            )
        finally:
            if own_robot:
                from manual_control.robot_session import disconnect_robot

                disconnect_robot(robot)

    return result


def run_xy_optimize_phase(
    cfg: EpoxyApproachConfig,
    params: Mapping[str, Any],
    *,
    robot=None,
) -> dict[str, Any]:
    """Phase 2: camera XY refine at fixed Z — placeholder until camera wired."""
    del robot
    xy = cfg.xy_optimize
    print("[epoxy_app] phase=xy_optimize")
    if not xy.enabled:
        print("[epoxy_app] xy_optimize disabled in config — skip.")
        return {"phase": "xy_optimize", "status": "skipped", "placeholder": True}

    print(
        f"[epoxy_app] xy_optimize PLACEHOLDER "
        f"(camera={xy.camera}, placeholder={xy.placeholder}). "
        "No motion. Future: Zelux tip/pad XY refine at fixed Z, then small XY moves."
    )
    if bool(params.get("execute", False)):
        print(
            "[epoxy_app] xy_optimize --execute: no-op placeholder "
            "(assume arm already at rough TCP from pre_apply)."
        )
    return {
        "phase": "xy_optimize",
        "status": "placeholder",
        "placeholder": True,
        "camera": xy.camera,
    }


def run_z_creep_phase(
    cfg: EpoxyApproachConfig,
    params: Mapping[str, Any],
    *,
    robot=None,
) -> dict[str, Any]:
    """Phase 3: vision Z creeps — placeholder; use epoxy_apply for real FSM."""
    del robot
    zc = cfg.z_creep
    print("[epoxy_app] phase=z_creep")
    if not zc.enabled:
        print("[epoxy_app] z_creep disabled in config — skip.")
        return {"phase": "z_creep", "status": "skipped", "placeholder": True}

    print(
        "[epoxy_app] z_creep PLACEHOLDER. "
        "For live vision Z creeps run:\n"
        "  python -m high_level_jobs.epoxy_app.epoxy_apply"
    )
    if bool(params.get("execute", False)):
        print(
            "[epoxy_app] z_creep --execute: no-op placeholder "
            "(assume XY already optimized)."
        )
    return {
        "phase": "z_creep",
        "status": "placeholder",
        "placeholder": True,
    }


def run_epoxy_phases(
    cfg: EpoxyApproachConfig,
    params: Mapping[str, Any],
    *,
    phases: Sequence[str] | None = None,
    robot=None,
    start_joints_deg: list[float] | None = None,
) -> dict[str, Any]:
    """Run one or more epoxy phases in order (default: all three)."""
    selected = normalize_phases(phases if phases is not None else params.get("phases"))
    print(f"[epoxy_app] phases={list(selected)}")

    # ik-gui / plan-gui only make sense for pre_apply planning.
    if (params.get("ik_gui") or params.get("plan_gui")) and "pre_apply" not in selected:
        raise ValueError("--ik-gui / --plan-gui require --phase pre_apply")

    results: dict[str, Any] = {"phases": list(selected), "results": {}}
    shared_robot = robot

    for phase in selected:
        if phase == "pre_apply":
            results["results"]["pre_apply"] = run_pre_apply_phase(
                cfg,
                params,
                robot=shared_robot,
                start_joints_deg=start_joints_deg,
            )
        elif phase == "xy_optimize":
            results["results"]["xy_optimize"] = run_xy_optimize_phase(
                cfg, params, robot=shared_robot
            )
        elif phase == "z_creep":
            results["results"]["z_creep"] = run_z_creep_phase(
                cfg, params, robot=shared_robot
            )
    return results


# Back-compat alias
def run_rough_approach_pipeline(
    cfg: EpoxyApproachConfig,
    params: Mapping[str, Any],
    *,
    robot=None,
    start_joints_deg: list[float] | None = None,
) -> dict[str, Any]:
    """Deprecated name — runs pre_apply only."""
    return run_pre_apply_phase(
        cfg, params, robot=robot, start_joints_deg=start_joints_deg
    )
