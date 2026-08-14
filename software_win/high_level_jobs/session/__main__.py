"""CLI: continuous session — arm to WORK, then command / LLM loop."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SOFTWARE_WIN = Path(__file__).resolve().parents[2]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

# Load repo .env.local before reading OPENAI_* defaults.
import high_level_jobs.llm_client  # noqa: E402,F401

from high_level_jobs.session.loop import (  # noqa: E402
    build_state,
    run_repl,
    shutdown,
    start_robot_at_work,
)
from high_level_jobs.session.paths import RECORDS_DIR  # noqa: E402
from robot_config import default_robot_port  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Continuous Dummy_lab session: start arm at WORK, then run "
            "grab_sensor / put_sensor / power_measure / power_optimization / "
            "return / standby (or NL via LLM)."
        )
    )
    p.add_argument("--port", default=default_robot_port())
    p.add_argument("--baudrate", type=int, default=int(os.environ.get("ROBOT_BAUDRATE", "115200")))
    p.add_argument("--camera-serial", type=str, default=None)
    p.add_argument("--joint-current-limit-a", type=float, default=None)
    p.add_argument(
        "--records-dir",
        type=Path,
        default=RECORDS_DIR,
        help="Where motion records are saved (default: session/records)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Connect logic + command parsing only; no motion",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM; only exact command tokens are accepted",
    )
    p.add_argument(
        "--llm-backend",
        choices=("ollama", "openai"),
        default=("openai" if os.environ.get("OPENAI_API_KEY") else "ollama"),
        help="LLM provider (default: openai if OPENAI_API_KEY set, else ollama)",
    )
    p.add_argument("--llm-model", default="", help="Override OLLAMA_MODEL / OPENAI_MODEL")
    p.add_argument(
        "--yes-llm",
        action="store_true",
        help="Skip interactive review/confirm for LLM plans",
    )
    args = p.parse_args(argv)

    state = build_state(
        port=args.port,
        baudrate=args.baudrate,
        camera_serial=args.camera_serial,
        dry_run=args.dry_run,
        joint_current_limit_a=args.joint_current_limit_a,
        records_dir=args.records_dir,
    )
    try:
        start_robot_at_work(state)
        return run_repl(
            state,
            llm_backend=args.llm_backend,
            llm_model=(args.llm_model or None),
            enable_llm=not args.no_llm,
            confirm_llm_plans=not args.yes_llm,
        )
    except KeyboardInterrupt:
        print("\n[session] interrupted.")
        return 130
    finally:
        shutdown(state)


if __name__ == "__main__":
    raise SystemExit(main())
