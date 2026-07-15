from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SOFTWARE_DIR = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from robot_config import DEFAULT_ROBOT, default_robot_port, repo_root

from high_level_jobs.subtasks.insert_sensor.config_loader import default_config_path, load_insert_sensor_config
from high_level_jobs.subtasks.insert_sensor.deploy import run_deploy_loop
from high_level_jobs.subtasks.insert_sensor.record import run_record_session
from high_level_jobs.subtasks.insert_sensor.review import check_recordings
from high_level_jobs.subtasks.insert_sensor.train import train_policy


def _resolve_urdf(root: Path) -> tuple[Path, list[Path], str, Path | None]:
    from manual_control.urdf_tools import build_package_map, choose_description, discover_descriptions, materialize_urdf

    descriptions = discover_descriptions(root)
    selected = choose_description(descriptions, DEFAULT_ROBOT, root)
    package_map = build_package_map(root)
    urdf_path, temp_file = materialize_urdf(selected, package_map)
    search_paths = [root, selected.parent, urdf_path.parent, *package_map.values()]
    end_link = "link6_1"
    return urdf_path, search_paths, end_link, temp_file


def main() -> int:
    p = argparse.ArgumentParser(
        description="Insert sensor: record demos, train IL policy, deploy on robot.",
    )
    p.add_argument("--config", type=Path, default=default_config_path())
    p.add_argument("--record", action="store_true", help="Teleop + RGB-D demonstration recording")
    p.add_argument("--train", action="store_true", help="Train policy from recorded episodes")
    p.add_argument("--deploy", action="store_true", help="Run trained policy loop")
    p.add_argument("--check", action="store_true", help="List saved episodes and stats")
    p.add_argument(
        "--list-cameras",
        action="store_true",
        help="List connected RealSense serial numbers",
    )
    p.add_argument(
        "--preview-cameras",
        action="store_true",
        help="Save one RGB preview per RealSense (identify overhead vs side, then pin serials)",
    )
    p.add_argument(
        "--relabel-episode",
        type=str,
        default=None,
        help="Fix swapped overhead/side files for one episode (e.g. episode_000004)",
    )
    p.add_argument("--episode", type=str, default=None, help="With --check: inspect one episode")
    p.add_argument("--frame", type=int, default=None, help="With --check --episode: print frame JSON")
    p.add_argument("--port", default=default_robot_port())
    p.add_argument("--baudrate", type=int, default=int(os.environ.get("ROBOT_BAUDRATE", "115200")))
    p.add_argument("--sim-only", action="store_true", help="Cameras + policy only, no serial / no motors")
    p.add_argument(
        "--countdown",
        type=float,
        default=5.0,
        help="With --deploy: seconds before robot streaming starts",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=None,
        help="With --deploy: motion speed scale (default: config deploy_speed_scale)",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--model", type=Path, default=None, help="Policy checkpoint for deploy")
    p.add_argument("--out", type=Path, default=None, help="Training / preview output path")
    args = p.parse_args()

    if not (
        args.record
        or args.train
        or args.deploy
        or args.check
        or args.list_cameras
        or args.preview_cameras
        or args.relabel_episode
    ):
        p.error(
            "Specify one of --record, --train, --deploy, --check, "
            "--list-cameras, --preview-cameras, or --relabel-episode"
        )

    if args.list_cameras:
        from high_level_jobs.subtasks.insert_sensor.cameras import list_realsense_serials

        serials = list_realsense_serials()
        if not serials:
            print("No RealSense devices found.")
            return 1
        print(f"Found {len(serials)} RealSense device(s):")
        for i, sn in enumerate(serials):
            print(f"  [{i}] {sn}")
        print(
            "\nDo NOT trust list order. Identify with live_view or --preview-cameras, then pin:\n"
            "  cameras.overhead.serial  # top-down robot+table\n"
            "  cameras.side.serial      # horizontal breadboard view\n"
            "  cameras.side2.serial     # additional side (recording only for now)\n"
            "  python software/camera/live_view.py"
        )
        return 0

    if args.preview_cameras:
        from high_level_jobs.subtasks.insert_sensor.cameras import preview_cameras

        out = args.out or (repo_root() / "software/high_level_jobs/subtasks/insert_sensor/camera_previews")
        preview_cameras(out_dir=Path(out))
        return 0

    cfg = load_insert_sensor_config(args.config)
    root = repo_root()

    if args.relabel_episode:
        from high_level_jobs.subtasks.insert_sensor.cameras import relabel_swapped_episode

        ep_name = args.relabel_episode
        ep_dir = cfg.dataset_dir / ep_name if not Path(ep_name).is_dir() else Path(ep_name)
        if not ep_dir.is_dir():
            # also allow bare name under dataset
            ep_dir = cfg.dataset_dir / ep_name
        if not ep_dir.is_dir():
            print(f"Episode not found: {args.relabel_episode}", file=sys.stderr)
            return 1
        oh = cfg.overhead_serial
        side = cfg.side_serial
        if not oh or not side:
            print(
                "Pin cameras.overhead.serial and cameras.side.serial in config before relabel.",
                file=sys.stderr,
            )
            return 1
        n = relabel_swapped_episode(ep_dir, overhead_serial=oh, side_serial=side)
        print(f"Relabeled {n} frames in {ep_dir.name}")
        print(f"  overhead serial -> {oh}")
        print(f"  side serial     -> {side}")
        return 0

    if args.record:
        urdf_path, search_paths, end_link, temp_file = _resolve_urdf(root)
        try:
            run_record_session(
                cfg,
                urdf_path=urdf_path,
                extra_search_paths=search_paths,
                end_link_hint=end_link,
                port=args.port,
                baudrate=args.baudrate,
                sim_only=args.sim_only,
            )
        finally:
            if temp_file is not None and temp_file.exists():
                temp_file.unlink(missing_ok=True)
        return 0

    if args.train:
        train_policy(cfg, device=args.device, out_path=args.out)
        return 0

    if args.check:
        return check_recordings(cfg, episode=args.episode, frame=args.frame)

    if args.deploy:
        run_deploy_loop(
            cfg,
            port=args.port,
            baudrate=args.baudrate,
            sim_only=args.sim_only,
            model_path=args.model,
            countdown_s=args.countdown,
            speed_scale=args.speed,
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
