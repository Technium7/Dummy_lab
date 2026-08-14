"""Session lifecycle: connect, go to WORK, REPL until quit."""

from __future__ import annotations

from typing import Any

from high_level_jobs.grab_sensor.config_loader import (
    apply_runtime_calibrations,
    load_grab_sensor_config,
)
from high_level_jobs.grab_sensor.execute import ensure_robot_ready
from high_level_jobs.grab_sensor.plan_motion import work_joints_deg
from high_level_jobs.session.commands import (
    SessionState,
    cmd_help,
    normalize_command,
    run_command,
)
from high_level_jobs.session.llm_map import Plan, map_utterance_to_plan
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


def resolve_plan(
    line: str,
    *,
    llm_backend: str,
    llm_model: str | None,
    enable_llm: bool,
) -> Plan:
    text = line.strip()
    if not text:
        return Plan()
    direct = normalize_command(text)
    # Exact command tokens (with optional args) skip the LLM.
    if direct and not text.lower().startswith("llm "):
        spaced = " ".join(text.lower().replace("-", "_").replace("_", " ").split())
        forms = {direct.replace("_", " "), direct}
        if any(spaced == f or spaced.startswith(f + " ") for f in forms):
            return map_utterance_to_plan(text, use_llm=False)
    if text.lower().startswith("llm ") or enable_llm:
        return map_utterance_to_plan(
            text,
            backend=llm_backend,
            model=llm_model,
            use_llm=enable_llm or text.lower().startswith("llm "),
        )
    return map_utterance_to_plan(text, use_llm=False)


def format_plan_review(plan: Plan) -> str:
    return plan.format_review()


def confirm_llm_plan(plan: Plan) -> bool:
    """Print LLM plan and ask operator to confirm before execute."""
    print(plan.format_review())
    while True:
        try:
            ans = input("[session] run this plan? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if ans in ("", "n", "no"):
            print("[session] plan cancelled.")
            return False
        if ans in ("y", "yes"):
            return True
        print("[session] type y or n")


def run_plan(state: SessionState, plan: Plan) -> str | None:
    """Execute plan steps. Returns 'quit' if a quit step ran, else None."""
    n = len(plan.steps)
    if n == 0:
        print("[session] empty plan.")
        return None
    if n > 1 and not plan.from_llm:
        print(f"[session] plan ({n} steps): {plan.summary()}")
    for i, step in enumerate(plan.steps, start=1):
        if step.action == "quit":
            print("[session] quitting...")
            return "quit"
        if n > 1:
            print(f"[session] step {i}/{n}: {step.action} {step.params or ''}".rstrip())
        try:
            run_command(state, step.action, step.params)
        except KeyboardInterrupt:
            print("\n[session] plan interrupted — robot stays connected.")
            return None
        except Exception as exc:  # noqa: BLE001 — keep REPL alive
            print(f"[session] ERROR in {step.action} (step {i}/{n}): {exc}")
            print("[session] plan aborted.")
            return None
    return None


def run_repl(
    state: SessionState,
    *,
    llm_backend: str = "ollama",
    llm_model: str | None = None,
    enable_llm: bool = True,
    confirm_llm_plans: bool = True,
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
            plan = resolve_plan(
                line,
                llm_backend=llm_backend,
                llm_model=llm_model,
                enable_llm=enable_llm,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[session] could not parse command: {exc}")
            continue
        if plan.is_empty():
            print("[session] empty command.")
            continue
        if plan.from_llm and confirm_llm_plans:
            if not confirm_llm_plan(plan):
                continue
        result = run_plan(state, plan)
        if result == "quit":
            break
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
