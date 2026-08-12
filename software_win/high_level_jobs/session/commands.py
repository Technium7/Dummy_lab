"""Continuous high-level session: start at WORK, then dispatch skill commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from high_level_jobs.session.motion_store import load_record, save_record
from high_level_jobs.session.paths import (
    GRAB_SENSOR_RECORD,
    POWER_MEASURE_RECORD,
    RECORDS_DIR,
)


@dataclass
class SessionState:
    robot: Any = None
    port: str = ""
    baudrate: int = 115200
    camera_serial: str | None = None
    dry_run: bool = False
    joint_current_limit_a: float | None = None
    records_dir: Path = field(default_factory=lambda: RECORDS_DIR)
    # Last place-mode extras for power_optimization
    power_place_extras: dict[str, Any] = field(default_factory=dict)
    last_command: str = ""
    armed_at_work: bool = False

    def base_params(self) -> dict[str, Any]:
        p: dict[str, Any] = {
            "execute": not self.dry_run,
            "port": self.port,
            "baudrate": self.baudrate,
            "home_first": not self.armed_at_work,
        }
        if self.camera_serial:
            p["camera_serial"] = self.camera_serial
        if self.joint_current_limit_a is not None:
            p["joint_current_limit_a"] = float(self.joint_current_limit_a)
        return p


COMMANDS = (
    "grab_sensor",
    "put_sensor",
    "power_measure",
    "power_optimization",
    "return",
    "standby",
    "manual",
    "stop",
    "epoxy_apply",
    "help",
    "quit",
)

ALIASES = {
    "grab": "grab_sensor",
    "put": "put_sensor",
    "power": "power_measure",
    "power_optimize": "power_optimization",
    "optimize": "power_optimization",
    "power_opt": "power_optimization",
    "home": "standby",
    "work": "standby",
    "man": "manual",
    "teleop": "manual",
    "s": "stop",
    "estop": "stop",
    "e_stop": "stop",
    "exit": "quit",
    "q": "quit",
    "?": "help",
}

_MULTI = {
    "power measure": "power_measure",
    "power optimization": "power_optimization",
    "put sensor": "put_sensor",
    "grab sensor": "grab_sensor",
    "epoxy apply": "epoxy_apply",
    "manual control": "manual",
    "emergency stop": "stop",
}


def normalize_command(raw: str) -> str:
    text = raw.strip().lower().replace("-", "_")
    if not text:
        return ""
    spaced = " ".join(text.replace("_", " ").split())
    if spaced in _MULTI:
        return _MULTI[spaced]
    underscored = spaced.replace(" ", "_")
    if underscored in COMMANDS:
        return underscored
    if underscored in ALIASES:
        return ALIASES[underscored]
    token = text.split()[0]
    if token in COMMANDS:
        return token
    return ALIASES.get(token, "")


def cmd_help(_state: SessionState) -> None:
    print(
        """
Commands (type one per line):
  grab_sensor          Pick sensor, return to WORK, save motion for put_sensor
  put_sensor           Put sensor back using the saved grab motion
  power_measure        Place sensor at click target, stay, record return traj
  power_optimization   Search/climb power from current insert pose, stay
  return               Return to WORK using the recorded power_measure traj
  standby              Fresh plan from current pose back to WORK
  manual (man)         Hand off to manual_control teleop; resume session on exit
  s / stop             Emergency !STOP (hold current pose, soft-disable)
  epoxy_apply          (stub — not wired yet)
  help                 Show this help
  quit                 Disconnect and exit

Natural language:
  llm <text>           Map text to a command via local/hosted LLM
  Or type free text (if it is not a known command) to use the LLM mapper.
""".strip()
    )


def cmd_grab_sensor(state: SessionState) -> None:
    from high_level_jobs.grab_sensor.config_loader import (
        apply_runtime_calibrations,
        load_grab_sensor_config,
    )
    from high_level_jobs.grab_sensor.pipeline import run_grab_sensor_pipeline
    from high_level_jobs.session.motion_store import record_from_plan

    cfg = apply_runtime_calibrations(load_grab_sensor_config(), {})
    params = state.base_params()
    print("[session] grab_sensor: detect → grasp → WORK (recording motion)...")
    if state.dry_run:
        print("[session] dry-run: skip execute")
        return
    result = run_grab_sensor_pipeline(cfg, params, robot=state.robot)
    state.armed_at_work = True
    meta = result.get("plan_meta") or {}
    record = record_from_plan(
        name=GRAB_SENSOR_RECORD,
        kind="grab_sensor",
        plan_meta=meta,
        start_joints_deg=list(result.get("start_joints_deg") or []),
    )
    save_record(record, state.records_dir)
    print("[session] grab_sensor done; put_sensor is available.")


def cmd_put_sensor(state: SessionState) -> None:
    from high_level_jobs.grab_sensor.config_loader import (
        apply_runtime_calibrations,
        load_grab_sensor_config,
    )
    from high_level_jobs.grab_sensor.execute import execute_put_sensor_on_robot

    record = load_record(GRAB_SENSOR_RECORD, state.records_dir)
    if record is None:
        raise RuntimeError(
            f"No grab_sensor record at {state.records_dir / (GRAB_SENSOR_RECORD + '.json')}. "
            "Run grab_sensor first."
        )
    cfg = apply_runtime_calibrations(load_grab_sensor_config(), {})
    params = state.base_params()
    print("[session] put_sensor: reversing recorded grab motion...")
    if state.dry_run:
        print("[session] dry-run: skip execute")
        return
    execute_put_sensor_on_robot(
        state.robot,
        record.segments,
        cfg,
        params,
    )
    state.armed_at_work = True
    print("[session] put_sensor done.")


def cmd_power_measure(state: SessionState) -> None:
    from high_level_jobs.power_measure.config_loader import (
        apply_runtime_calibrations,
        load_power_measure_config,
    )
    from high_level_jobs.power_measure.live_gui import run_live_gui
    from high_level_jobs.power_measure.pipeline import run_insert_pipeline
    from high_level_jobs.session.motion_store import record_from_plan

    cfg = apply_runtime_calibrations(load_power_measure_config(), {})
    params = state.base_params()
    params["power_mode"] = "place"
    print("[session] power_measure: click target → place only (stay + record)...")
    live = run_live_gui(cfg, params)
    if not live.confirmed or live.insert is None:
        print("[session] power_measure cancelled (no confirm).")
        return
    if state.dry_run:
        print("[session] dry-run: skip execute")
        return
    result = run_insert_pipeline(cfg, live.insert, params, robot=state.robot)
    state.armed_at_work = False  # at insert, not WORK
    meta = result.get("plan_meta") or {}
    place = result.get("place_result") or {}
    extras = {
        "click_uv": list(live.insert.click_uv),
        "insert_power_w": place.get("insert_power_w"),
        "along_max_w": place.get("along_max_w"),
        "insert_samples": place.get("insert_samples") or [],
        "grasp_goal_q": meta.get("grasp_goal_q"),
    }
    state.power_place_extras = extras
    record = record_from_plan(
        name=POWER_MEASURE_RECORD,
        kind="power_measure",
        plan_meta=meta,
        start_joints_deg=list(result.get("start_joints_deg") or []),
        extras=extras,
    )
    save_record(record, state.records_dir)
    print("[session] power_measure place done; use power_optimization or return.")


def cmd_power_optimization(state: SessionState) -> None:
    from high_level_jobs.power_measure.config_loader import (
        apply_runtime_calibrations,
        load_power_measure_config,
    )
    from high_level_jobs.power_measure.pipeline import run_power_optimize
    from high_level_jobs.power_measure.target import build_insert_target_from_uv

    record = load_record(POWER_MEASURE_RECORD, state.records_dir)
    extras = dict(state.power_place_extras)
    if record is not None:
        extras = {**dict(record.extras), **extras}
    click = extras.get("click_uv")
    grasp_q = extras.get("grasp_goal_q") or (record.plan_meta.get("grasp_goal_q") if record else None)
    if not click or grasp_q is None:
        raise RuntimeError(
            "power_optimization needs a prior power_measure place "
            f"(missing click_uv / grasp_goal_q in {state.records_dir})."
        )

    cfg = apply_runtime_calibrations(load_power_measure_config(), {})
    insert = build_insert_target_from_uv(cfg, (int(click[0]), int(click[1])))
    params = state.base_params()
    params["home_first"] = False
    params["plan_meta"] = {"grasp_goal_q": grasp_q}
    params["insert_power_w"] = extras.get("insert_power_w") or 0.0
    params["along_max_w"] = extras.get("along_max_w") or 0.0
    params["insert_samples"] = extras.get("insert_samples") or []
    print("[session] power_optimization: search/climb from current pose...")
    if state.dry_run:
        print("[session] dry-run: skip execute")
        return
    run_power_optimize(cfg, insert, params, robot=state.robot)
    state.armed_at_work = False
    print("[session] power_optimization done; use return or standby.")


def cmd_return(state: SessionState) -> None:
    from high_level_jobs.grab_sensor.config_loader import (
        apply_runtime_calibrations,
        load_grab_sensor_config,
    )
    from high_level_jobs.grab_sensor.execute import execute_return_segments_on_robot
    from high_level_jobs.power_measure.config_loader import (
        apply_runtime_calibrations as apply_pm,
        load_power_measure_config,
    )

    record = load_record(POWER_MEASURE_RECORD, state.records_dir)
    if record is None:
        raise RuntimeError(
            "return: no recorded trajectory. Run power_measure first, or use standby."
        )
    segs = record.return_to_work_segments()
    if not segs:
        raise RuntimeError("return: recorded motion has no returnable segments.")

    # Prefer power_measure work joints when returning from a place record.
    if record.kind == "power_measure":
        cfg = apply_pm(load_power_measure_config(), {}).to_grab_sensor_config()
    else:
        cfg = apply_runtime_calibrations(load_grab_sensor_config(), {})
    params = state.base_params()
    params["home_first"] = False
    print("[session] return: streaming recorded trajectory to WORK...")
    if state.dry_run:
        print("[session] dry-run: skip execute")
        return
    execute_return_segments_on_robot(state.robot, segs, cfg, params)
    state.armed_at_work = True
    print("[session] return done.")


def cmd_standby(state: SessionState) -> None:
    from high_level_jobs.grab_sensor.config_loader import (
        apply_runtime_calibrations,
        load_grab_sensor_config,
    )
    from high_level_jobs.grab_sensor.execute import execute_standby_to_work

    cfg = apply_runtime_calibrations(load_grab_sensor_config(), {})
    params = state.base_params()
    params["home_first"] = False
    print("[session] standby: plan current → WORK...")
    if state.dry_run:
        print("[session] dry-run: skip execute")
        return
    execute_standby_to_work(state.robot, cfg, params)
    state.armed_at_work = True
    print("[session] standby done — at WORK.")


def cmd_stop(state: SessionState) -> None:
    """Immediate firmware emergency stop (!STOP)."""
    robot = state.robot
    if robot is None:
        print("[session] stop: no robot handle (already released?).")
        return
    if state.dry_run:
        print("[session] dry-run: would send !STOP")
        return
    try:
        resp = robot.stop()
        print(f"[session] !STOP -> {resp}")
    except Exception as exc:  # noqa: BLE001
        print(f"[session] !STOP failed: {exc}")
        raise
    state.armed_at_work = False
    print("[session] robot soft-disabled. Next skill will !START again; use standby if needed.")


def cmd_manual(state: SessionState) -> None:
    """Release serial, run manual_control teleop, then reconnect session."""
    from high_level_jobs.session.loop import reconnect_robot, release_robot_port
    from high_level_jobs.session.manual_teleop import run_manual_control

    print("[session] manual: releasing port for teleop (close PyBullet window to return)...")
    if state.dry_run:
        print("[session] dry-run: skip manual_control")
        return
    release_robot_port(state)
    try:
        run_manual_control(
            port=state.port,
            baudrate=state.baudrate,
            joint_current_limit_a=state.joint_current_limit_a,
            sim_only=False,
        )
    finally:
        print("[session] manual exited — reconnecting...")
        reconnect_robot(state, go_to_work=False)
    print("[session] back in session loop (pose may have changed; use standby to WORK).")


def cmd_epoxy_apply(_state: SessionState) -> None:
    print(
        "[session] epoxy_apply is not wired into the session yet. "
        "Use: python -m high_level_jobs.epoxy_app.epoxy_apply"
    )


HANDLERS = {
    "help": cmd_help,
    "grab_sensor": cmd_grab_sensor,
    "put_sensor": cmd_put_sensor,
    "power_measure": cmd_power_measure,
    "power_optimization": cmd_power_optimization,
    "return": cmd_return,
    "standby": cmd_standby,
    "manual": cmd_manual,
    "stop": cmd_stop,
    "epoxy_apply": cmd_epoxy_apply,
}
