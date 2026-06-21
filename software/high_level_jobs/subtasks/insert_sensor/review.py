"""Inspect saved insert_sensor demonstration episodes."""

from __future__ import annotations

import json
from pathlib import Path

from .config_loader import InsertSensorConfig
from .dataset import list_episodes


def _episode_stats(ep_dir: Path) -> dict:
    frame_paths = sorted(ep_dir.glob("*.frame.json"))
    n = len(frame_paths)
    record_hz = float(json.loads((ep_dir / "meta.json").read_text()).get("record_hz", 15.0)) if (ep_dir / "meta.json").is_file() else 15.0

    valid_kp = 0
    gripper_events = 0
    home_frames = 0
    first: dict | None = None
    last: dict | None = None

    for fp in frame_paths:
        data = json.loads(fp.read_text(encoding="utf-8"))
        if first is None:
            first = data
        last = data
        if data.get("keypoint_valid"):
            valid_kp += 1
        if data.get("hand_open") or data.get("hand_close"):
            gripper_events += 1
        if data.get("home_latched"):
            home_frames += 1

    duration_s = n / max(record_hz, 1.0)
    return {
        "n_frames": n,
        "duration_s": duration_s,
        "record_hz": record_hz,
        "keypoint_valid_pct": (100.0 * valid_kp / n) if n else 0.0,
        "gripper_labeled_frames": gripper_events,
        "home_latched_frames": home_frames,
        "first_joint_deg": first.get("joint_pos_deg") if first else None,
        "last_joint_deg": last.get("joint_pos_deg") if last else None,
        "first_color": ep_dir / f"{frame_paths[0].name.replace('.frame.json', '')}.color.png" if frame_paths else None,
        "last_color": ep_dir / f"{frame_paths[-1].name.replace('.frame.json', '')}.color.png" if frame_paths else None,
    }


def check_recordings(cfg: InsertSensorConfig, *, episode: str | None = None, frame: int | None = None) -> int:
    dataset_dir = cfg.dataset_dir.resolve()
    episodes = list_episodes(dataset_dir)

    if not episodes:
        print(f"No saved episodes in {dataset_dir}")
        print("Record with: python manual_control/manual_control_app.py --record-demos")
        print("Save with C after stopping with R.")
        staging = dataset_dir / "_staging" / "current"
        if staging.is_dir() and list(staging.glob("*.frame.json")):
            n = len(list(staging.glob("*.frame.json")))
            print(f"\nUnsaved staging recording: {n} frames (press C to save, or discard with R).")
        return 1

    print(f"Dataset: {dataset_dir}")
    print(f"Episodes: {len(episodes)}\n")

    selected = episodes
    if episode:
        selected = [ep for ep in episodes if ep.path.name == episode or episode in ep.path.name]
        if not selected:
            print(f"Episode not found: {episode}")
            return 1

    for ep in selected:
        stats = _episode_stats(ep.path)
        print(f"{ep.path.name}")
        print(f"  frames:      {stats['n_frames']}")
        print(f"  duration:    ~{stats['duration_s']:.1f} s @ {stats['record_hz']:.0f} Hz")
        print(f"  keypoint OK: {stats['keypoint_valid_pct']:.0f}%")
        print(f"  gripper:     {stats['gripper_labeled_frames']} frames with open/close")
        print(f"  home (T):    {stats['home_latched_frames']} frames with home_latched")
        if stats["first_color"] and stats["first_color"].is_file():
            print(f"  first image: {stats['first_color']}")
        if stats["last_color"] and stats["last_color"].is_file():
            print(f"  last image:  {stats['last_color']}")
        print()

    if frame is not None and len(selected) == 1:
        frame_paths = sorted(selected[0].path.glob("*.frame.json"))
        if not frame_paths or frame < 0 or frame >= len(frame_paths):
            print(f"Frame index out of range (0..{len(frame_paths) - 1})")
            return 1
        data = json.loads(frame_paths[frame].read_text(encoding="utf-8"))
        stem = frame_paths[frame].name.replace(".frame.json", "")
        print(f"Frame {frame} ({stem}):")
        print(json.dumps(data, indent=2))
        color = selected[0].path / f"{stem}.color.png"
        depth = selected[0].path / f"{stem}.depth.npy"
        if color.is_file():
            print(f"  color: {color}")
        if depth.is_file():
            print(f"  depth: {depth}")

    history = cfg.policy.history_len
    short = [ep for ep in episodes if ep.n_frames < history]
    if short:
        print(f"Warning: {len(short)} episode(s) have fewer than history_len={history} frames (not usable for training).")

    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .config_loader import default_config_path, load_insert_sensor_config

    p = argparse.ArgumentParser(description="Check insert_sensor demonstration recordings.")
    p.add_argument("--config", type=Path, default=default_config_path())
    p.add_argument("--episode", type=str, default=None, help="e.g. episode_000001")
    p.add_argument("--frame", type=int, default=None, help="Print one frame JSON (with --episode)")
    args = p.parse_args(argv)
    cfg = load_insert_sensor_config(args.config)
    return check_recordings(cfg, episode=args.episode, frame=args.frame)


if __name__ == "__main__":
    raise SystemExit(main())
