"""Session lifecycle: connect, go to WORK, REPL until quit."""

from __future__ import annotations

from typing import Any

from high_level_jobs.grab_sensor.config_loader import (
    apply_runtime_calibrations,
    load_grab_sensor_config,
)
from high_level_jobs.grab_sensor.execute import ensure_robot_ready
from high_level_jobs.grab_sensor.plan_motion import work_joints_deg
from high_level_jobs.session.commands import HANDLERS, SessionState, cmd_help, normalize_command
from high_level_jobs.session.llm_map import map_utterance_to_command
from high_level_jobs.session.paths import RECORDS_DIR
from robot_ascii_interface import RobotAsciiInterface
from robot_config import default_robot_port


def start_robot_at_work(
    state: SessionState,
    *,
    joint_current_limit_a: float = 0.0,
) -> None:
    cfg = apply_runtime_calibrations(load_grab_sensor_config(), {})
    work = work_joints_deg(cfg)
    limit = (
        float(state.joint_current_limit_a)
        if state.joint_current_limit_a is not None
        else float(joint_current_limit_a or cfg.execute.joint_current_limit_a)
    )
    print(f"[session] connecting {state.port} @ {state.baudrate}...")
    robot = RobotAsciiInterface(port=state.port, baudrate=state.baudrate)
    if not robot.connect():
        raise RuntimeError(f"Failed to connect robot on {state.port}")
    state.robot = robot
    if state.dry_run:
        print("[session] dry-run: skip WORK move")
        state.armed_at_work = True
        return
    print(f"[session] !START + move to WORK {work}...")
    if not ensure_robot_ready(
        robot,
        cmd_mode=2,
        home_first=True,
        open_hand=False,
        joint_current_limit_a=limit,
        start_joints_deg=work,
        label="[session] ",
    ):
        raise RuntimeError("Failed to reach WORK at session start.")
    state.armed_at_work = True
    print("[session] at WORK — ready for commands.")


def release_robot_port(state: SessionState) -> None:
    """!STOP + close serial so another process (manual_control) can open the port."""
    robot = state.robot
    state.robot = None
    state.armed_at_work = False
    if robot is None:
        return
    from manual_control.robot_session import disconnect_robot

    disconnect_robot(robot, label="[session] ")


def reconnect_robot(state: SessionState, *, go_to_work: bool = False) -> None:
    """Open serial again after manual teleop (or after a released port)."""
    from high_level_jobs.grab_sensor.config_loader import (
        apply_runtime_calibrations,
        load_grab_sensor_config,
    )
    from high_level_jobs.grab_sensor.execute import ensure_robot_ready
    from high_level_jobs.grab_sensor.plan_motion import work_joints_deg

    if state.robot is not None:
        return

    cfg = apply_runtime_calibrations(load_grab_sensor_config(), {})
    work = work_joints_deg(cfg)
    limit = float(state.joint_current_limit_a or cfg.execute.joint_current_limit_a or 0.0)
    print(f"[session] reconnecting {state.port} @ {state.baudrate}...")
    robot = RobotAsciiInterface(port=state.port, baudrate=state.baudrate)
    if not robot.connect():
        raise RuntimeError(f"Failed to reconnect robot on {state.port}")
    state.robot = robot
    if state.dry_run:
        state.armed_at_work = bool(go_to_work)
        return
    if not ensure_robot_ready(
        robot,
        cmd_mode=2,
        home_first=bool(go_to_work),
        open_hand=False,
        joint_current_limit_a=limit,
        start_joints_deg=work if go_to_work else None,
        label="[session] ",
    ):
        raise RuntimeError("Failed to re-arm robot after reconnect.")
    state.armed_at_work = bool(go_to_work)


def shutdown(state: SessionState) -> None:
    release_robot_port(state)


def resolve_command(
    line: str,
    *,
    llm_backend: str,
    llm_model: str | None,
    enable_llm: bool,
) -> str:
    text = line.strip()
    if not text:
        return ""
    direct = normalize_command(text)
    if direct:
        return direct
    # Known multi-word aliases already handled; free text → LLM / keywords.
    if text.lower().startswith("llm ") or enable_llm:
        return map_utterance_to_command(
            text,
            backend=llm_backend,
            model=llm_model,
            use_llm=enable_llm or text.lower().startswith("llm "),
        )
    # Without LLM, still try keywords on free text.
    return map_utterance_to_command(text, use_llm=False)


def run_repl(
    state: SessionState,
    *,
    llm_backend: str = "ollama",
    llm_model: str | None = None,
    enable_llm: bool = True,
) -> int:
    cmd_help(state)
    while True:
        try:
            line = input("session> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            line = "quit"
        if not line:
            continue
        try:
            cmd = resolve_command(
                line,
                llm_backend=llm_backend,
                llm_model=llm_model,
                enable_llm=enable_llm,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[session] could not parse command: {exc}")
            continue
        if not cmd:
            print("[session] empty command.")
            continue
        if cmd == "quit":
            print("[session] quitting...")
            break
        handler = HANDLERS.get(cmd)
        if handler is None:
            print(f"[session] unknown command: {cmd}")
            continue
        state.last_command = cmd
        try:
            handler(state)
        except KeyboardInterrupt:
            print("\n[session] command interrupted — robot stays connected.")
        except Exception as exc:  # noqa: BLE001 — keep REPL alive
            print(f"[session] ERROR in {cmd}: {exc}")
    return 0


def build_state(
    *,
    port: str | None = None,
    baudrate: int = 115200,
    camera_serial: str | None = None,
    dry_run: bool = False,
    joint_current_limit_a: float | None = None,
    records_dir: Any = None,
) -> SessionState:
    return SessionState(
        port=str(port or default_robot_port()),
        baudrate=int(baudrate),
        camera_serial=camera_serial,
        dry_run=bool(dry_run),
        joint_current_limit_a=joint_current_limit_a,
        records_dir=RECORDS_DIR if records_dir is None else records_dir,
    )
