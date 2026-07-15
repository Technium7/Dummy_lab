"""Inspect saved insert_sensor demonstration episodes."""

from __future__ import annotations

import json
from pathlib import Path

from .cameras import (
    OVERHEAD_ROLE,
    SIDE2_ROLE,
    SIDE_ROLE,
    color_filename,
    episode_camera_roles,
    episode_has_side,
)
from .config_loader import InsertSensorConfig
from .dataset import list_episodes


def _episode_stats(ep_dir: Path) -> dict:
    frame_paths = sorted(ep_dir.glob("*.frame.json"))
    n = len(frame_paths)
    meta = json.loads((ep_dir / "meta.json").read_text()) if (ep_dir / "meta.json").is_file() else {}
    record_hz = float(meta.get("record_hz", 15.0))

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
        if data.get("hand_open") or data.get("hand_close"):
            gripper_events += 1
        if data.get("home_latched"):
            home_frames += 1

    duration_s = n / max(record_hz, 1.0)
    first_stem = frame_paths[0].name.replace(".frame.json", "") if frame_paths else None
    last_stem = frame_paths[-1].name.replace(".frame.json", "") if frame_paths else None
    has_side = episode_has_side(ep_dir)
    roles = episode_camera_roles(ep_dir)
    return {
        "n_frames": n,
        "duration_s": duration_s,
        "record_hz": record_hz,
        "num_cameras": int(meta.get("num_cameras", len(roles) or (1 if not has_side else 2))),
        "has_side": has_side,
        "has_side2": SIDE2_ROLE in roles,
        "roles": roles,
        "cameras": meta.get("cameras"),
        "gripper_labeled_frames": gripper_events,
        "home_latched_frames": home_frames,
        "first_joint_deg": first.get("joint_pos_deg") if first else None,
        "last_joint_deg": last.get("joint_pos_deg") if last else None,
        "first_color": ep_dir / color_filename(first_stem, OVERHEAD_ROLE) if first_stem else None,
        "last_color": ep_dir / color_filename(last_stem, OVERHEAD_ROLE) if last_stem else None,
        "first_side": ep_dir / color_filename(first_stem, SIDE_ROLE) if first_stem else None,
        "last_side": ep_dir / color_filename(last_stem, SIDE_ROLE) if last_stem else None,
        "first_side2": ep_dir / color_filename(first_stem, SIDE2_ROLE) if first_stem else None,
        "last_side2": ep_dir / color_filename(last_stem, SIDE2_ROLE) if last_stem else None,
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
        print(f"  cameras:     {stats['num_cameras']} ({'+'.join(stats.get('roles') or ['overhead'])})")
        if stats.get("cameras"):
            print(f"  serials:     {stats['cameras']}")
        print(f"  gripper:     {stats['gripper_labeled_frames']} frames with open/close")
        print(f"  home (T):    {stats['home_latched_frames']} frames with home_latched")
        if stats["first_color"] and stats["first_color"].is_file():
            print(f"  first overhead: {stats['first_color']}")
        if stats["last_color"] and stats["last_color"].is_file():
            print(f"  last overhead:  {stats['last_color']}")
        first_side = _resolve_side_image(ep.path, stats["first_side"])
        last_side = _resolve_side_image(ep.path, stats["last_side"])
        if first_side is not None:
            print(f"  first side:     {first_side}")
        if last_side is not None:
            print(f"  last side:      {last_side}")
        first_side2 = stats.get("first_side2")
        last_side2 = stats.get("last_side2")
        if first_side2 is not None and Path(first_side2).is_file():
            print(f"  first side2:    {first_side2}")
        if last_side2 is not None and Path(last_side2).is_file():
            print(f"  last side2:     {last_side2}")
        print()


def _resolve_side_image(ep_dir: Path, preferred: Path | None) -> Path | None:
    """Return existing side (or legacy wrist) RGB path for display."""
    if preferred is None:
        return None
    if preferred.is_file():
        return preferred
    stem = preferred.name.split(".")[0]
    for role in (SIDE_ROLE, "wrist"):
        cand = ep_dir / color_filename(stem, role)
        if cand.is_file():
            return cand
    return None

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
