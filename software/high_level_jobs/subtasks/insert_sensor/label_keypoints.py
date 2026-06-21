"""Offline base keypoint labeling for recorded episodes (run after teleop)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .base_keypoint import detect_base_keypoint
from .config_loader import InsertSensorConfig, load_grasp_config
from .dataset import list_episodes


def label_episode_keypoints(
    ep_dir: Path,
    *,
    grasp_cfg,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Fill keypoint fields in frame JSON from saved RGB-D. Returns (updated, skipped)."""
    ep_dir = ep_dir.resolve()
    updated = 0
    skipped = 0
    for frame_path in sorted(ep_dir.glob("*.frame.json")):
        data = json.loads(frame_path.read_text(encoding="utf-8"))
        if data.get("keypoint_valid") and not overwrite:
            skipped += 1
            continue
        stem = frame_path.name.replace(".frame.json", "")
        color_path = ep_dir / f"{stem}.color.png"
        depth_path = ep_dir / f"{stem}.depth.npy"
        if not color_path.is_file() or not depth_path.is_file():
            skipped += 1
            continue
        color_rgb = np.asarray(Image.open(color_path).convert("RGB"))
        depth_m = np.load(depth_path).astype(np.float32)
        kp = detect_base_keypoint(color_rgb, depth_m, grasp_cfg)
        data["keypoint"] = kp.vector().tolist()
        data["keypoint_valid"] = kp.valid
        frame_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        updated += 1
    return updated, skipped


def label_all_episodes(
    cfg: InsertSensorConfig,
    *,
    episode: str | None = None,
    overwrite: bool = False,
) -> int:
    grasp_cfg = load_grasp_config(cfg)
    episodes = list_episodes(cfg.dataset_dir)
    if episode:
        episodes = [ep for ep in episodes if ep.path.name == episode or episode in ep.path.name]
    if not episodes:
        print(f"No episodes in {cfg.dataset_dir}")
        return 1

    total_updated = 0
    total_skipped = 0
    for ep in episodes:
        updated, skipped = label_episode_keypoints(
            ep.path,
            grasp_cfg=grasp_cfg,
            overwrite=overwrite,
        )
        total_updated += updated
        total_skipped += skipped
        print(f"{ep.path.name}: labeled {updated} frames, skipped {skipped}")

    print(f"Done. labeled={total_updated} skipped={total_skipped}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .config_loader import default_config_path, load_insert_sensor_config

    p = argparse.ArgumentParser(description="Label base keypoints on saved episodes (offline).")
    p.add_argument("--config", type=Path, default=default_config_path())
    p.add_argument("--episode", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    cfg = load_insert_sensor_config(args.config)
    return label_all_episodes(cfg, episode=args.episode, overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
