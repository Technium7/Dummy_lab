from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

_SOFTWARE_DIR = Path(__file__).resolve().parent.parent
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from high_level_jobs.llm_client import OllamaClient, OpenAICompatClient
from high_level_jobs.nl_compiler import compile_natural_language, load_prompt, save_job_yaml
from high_level_jobs.runner import run_job, run_job_file
from high_level_jobs.subtasks.registry import default_registry
from robot_config import default_robot_port


def _robot_cli_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--robot-port",
        default=default_robot_port(),
        help="Serial port for robot_check_connected (env ROBOT_PORT; auto-detect on Linux).",
    )
    parser.add_argument(
        "--robot-baudrate",
        type=int,
        default=int(os.environ.get("ROBOT_BAUDRATE", "115200")),
        help="Baud rate (env ROBOT_BAUDRATE).",
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="High-level job runner + optional NL compile (LLM).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run a job YAML file (stub subtasks by default).")
    pr.add_argument("job_yaml", type=Path, help="Path to job .yaml (relative paths OK)")
    pr.add_argument(
        "--execute",
        action="store_true",
        help="If set, wait_seconds and robot_* steps hit real hardware; default is dry-run.",
    )
    pr.add_argument("--quiet", action="store_true", help="Less console output from builtins.")
    _robot_cli_flags(pr)

    pd = sub.add_parser(
        "demo",
        help='Compile natural language with an LLM, then run the job (e.g. "go to home position").',
    )
    pd.add_argument("text", help='Human command, e.g. "go to home position"')
    pd.add_argument(
        "--backend",
        choices=("ollama", "openai"),
        default="ollama",
        help="LLM backend for compile step",
    )
    pd.add_argument("--model", default="", help="Override model for the chosen backend")
    pd.add_argument(
        "--system-prompt",
        type=Path,
        default=None,
        help="Override system prompt path (default: prompts/job_compiler_system.txt)",
    )
    pd.add_argument(
        "--save-yaml",
        type=Path,
        default=None,
        help="If set, write the compiled job YAML here before running",
    )
    pd.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip real serial and motion (robot_* no-ops; good for testing the LLM output only).",
    )
    pd.add_argument("--quiet", action="store_true", help="Less console output")
    _robot_cli_flags(pd)

    pc = sub.add_parser("compile", help="Human text -> job YAML via LLM (validate + save).")
    pc.add_argument("text", nargs="?", default="", help="Human command (or use --file)")
    pc.add_argument("--file", type=Path, help="Read human command from this UTF-8 file")
    pc.add_argument(
        "--backend",
        choices=("ollama", "openai"),
        default="ollama",
        help="ollama=local default; openai=OpenAI-compatible HTTP API",
    )
    pc.add_argument("--model", default="", help="Override model env default for the backend")
    pc.add_argument("--out", type=Path, required=True, help="Where to write compiled job YAML")
    pc.add_argument(
        "--system-prompt",
        type=Path,
        default=None,
        help="Override system prompt file (default: packaged prompts/job_compiler_system.txt)",
    )

    args = p.parse_args(argv)

    if args.cmd == "run":
        job_path = args.job_yaml
        if not job_path.is_file():
            print(f"Job file not found: {job_path}", file=sys.stderr)
            return 2
        dry = not args.execute
        _, ctx_out = run_job_file(
            job_path,
            default_registry(),
            dry_run=dry,
            verbose=not args.quiet,
            robot_port=args.robot_port,
            robot_baudrate=args.robot_baudrate,
        )
        if args.execute:
            from manual_control.robot_session import disconnect_robot

            robot = ctx_out.get("robot")
            disconnect_robot(robot)
        return 0

    if args.cmd == "demo":
        pkg = Path(__file__).resolve().parent
        spath = args.system_prompt or (pkg / "prompts" / "job_compiler_system.txt")
        system = load_prompt(spath)

        if args.backend == "ollama":
            model = args.model or None
            client = OllamaClient(model=model) if model else OllamaClient()
        else:
            model = args.model or None
            client = OpenAICompatClient(model=model) if model else OpenAICompatClient()

        spec = compile_natural_language(args.text.strip(), client, system_prompt=system)
        if args.save_yaml:
            save_job_yaml(spec, args.save_yaml)
            print(f"Saved compiled job to {args.save_yaml.resolve()}")

        ctx: dict[str, Any] = {
            "dry_run": args.dry_run,
            "verbose": not args.quiet,
            "robot_port": args.robot_port,
            "robot_baudrate": args.robot_baudrate,
        }
        ctx = run_job(spec, default_registry(), ctx=ctx)
        if not args.dry_run:
            from manual_control.robot_session import disconnect_robot

            robot = ctx.get("robot")
            disconnect_robot(robot)
        return 0

    if args.cmd == "compile":
        human = args.text.strip()
        if args.file:
            human = args.file.read_text(encoding="utf-8").strip()
        if not human:
            print("Provide text on the CLI or --file", file=sys.stderr)
            return 2

        pkg = Path(__file__).resolve().parent
        spath = args.system_prompt or (pkg / "prompts" / "job_compiler_system.txt")
        system = load_prompt(spath)

        if args.backend == "ollama":
            model = args.model or None
            client = OllamaClient(model=model) if model else OllamaClient()
        else:
            model = args.model or None
            client = OpenAICompatClient(model=model) if model else OpenAICompatClient()

        spec = compile_natural_language(human, client, system_prompt=system)
        save_job_yaml(spec, args.out)
        print(f"Wrote {args.out.resolve()}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
