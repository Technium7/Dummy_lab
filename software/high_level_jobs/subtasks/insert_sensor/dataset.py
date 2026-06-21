"""Episode dataset for insert_sensor imitation learning."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from manual_control.mapping import transformed_deg_to_sim_rad

from .model import ACTION_DIM, KEYPOINT_DIM, STATE_DIM, InsertionPolicyConfig, preprocess_rgbd_batch


@dataclass(frozen=True)
class EpisodeInfo:
    path: Path
    n_frames: int
    meta: dict[str, Any]


def list_episodes(dataset_dir: Path) -> list[EpisodeInfo]:
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        return []
    episodes: list[EpisodeInfo] = []
    for ep_dir in sorted(dataset_dir.glob("episode_*")):
        if not ep_dir.is_dir():
            continue
        meta_path = ep_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        frames = sorted(ep_dir.glob("*.frame.json"))
        if not frames:
            continue
        episodes.append(EpisodeInfo(path=ep_dir, n_frames=len(frames), meta=meta))
    return episodes


def _load_frame_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_stem(path: Path) -> str:
    return path.name.replace(".frame.json", "")


def build_state_vector(frame: dict[str, Any]) -> np.ndarray:
    joint_deg = frame.get("joint_pos_deg") or frame.get("robot_target_deg") or [0.0] * 6
    joint_vel_deg_s = frame.get("joint_vel_deg_s") or [0.0] * 6
    gripper = float(frame.get("gripper", 0.0))
    pos_rad = [transformed_deg_to_sim_rad(i, float(joint_deg[i])) for i in range(6)]
    vel_rad = [float(v) * np.pi / 180.0 for v in joint_vel_deg_s[:6]]
    return np.array([*pos_rad, *vel_rad, gripper], dtype=np.float32)


def build_action_vector(frame: dict[str, Any]) -> np.ndarray:
    if "action_joint_vel_rad_s" in frame:
        vel = frame["action_joint_vel_rad_s"]
    else:
        vel = frame.get("commanded_velocity", [0.0] * 6)
    grip = float(frame.get("action_gripper", frame.get("gripper_action", 0.0)))
    return np.array([*(float(v) for v in vel[:6]), grip], dtype=np.float32)


def compute_norm_stats(episodes: list[EpisodeInfo]) -> dict[str, Any]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    keypoints: list[np.ndarray] = []
    for ep in episodes:
        for frame_path in sorted(ep.path.glob("*.frame.json")):
            data = _load_frame_json(frame_path)
            states.append(build_state_vector(data))
            actions.append(build_action_vector(data))
            keypoints.append(np.asarray(data.get("keypoint", [0.0] * KEYPOINT_DIM), dtype=np.float32))
    if not states:
        return {}
    s = np.stack(states)
    a = np.stack(actions)
    k = np.stack(keypoints)
    return {
        "state_mean": s.mean(axis=0).tolist(),
        "state_std": (s.std(axis=0) + 1e-6).tolist(),
        "action_mean": a.mean(axis=0).tolist(),
        "action_std": (a.std(axis=0) + 1e-6).tolist(),
        "keypoint_mean": k.mean(axis=0).tolist(),
        "keypoint_std": (k.std(axis=0) + 1e-6).tolist(),
    }


def normalize(x: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    m = np.asarray(mean, dtype=np.float32)
    s = np.asarray(std, dtype=np.float32)
    return (x - m) / s


class InsertSensorSequenceDataset(Dataset):
    """Sliding-window sequences over teleop episodes."""

    def __init__(
        self,
        episodes: list[EpisodeInfo],
        *,
        history_len: int,
        policy_cfg: InsertionPolicyConfig,
        norm_stats: dict[str, Any] | None = None,
        augment: bool = False,
    ) -> None:
        self.episodes = episodes
        self.history_len = history_len
        self.policy_cfg = policy_cfg
        self.norm_stats = norm_stats or {}
        self.augment = augment
        self._windows: list[tuple[Path, int]] = []
        for ep in episodes:
            frames = sorted(ep.path.glob("*.frame.json"))
            n = len(frames)
            if n < history_len:
                continue
            for start in range(0, n - history_len + 1):
                self._windows.append((ep.path, start))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_dir, start = self._windows[idx]
        frame_paths = sorted(ep_dir.glob("*.frame.json"))[start : start + self.history_len]
        cfg = self.policy_cfg

        colors: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        keypoints: list[np.ndarray] = []

        for fp in frame_paths:
            data = _load_frame_json(fp)
            stem = _frame_stem(fp)
            color_path = ep_dir / f"{stem}.color.png"
            depth_path = ep_dir / f"{stem}.depth.npy"
            color = np.asarray(Image.open(color_path).convert("RGB"))
            depth = np.load(depth_path).astype(np.float32)
            if self.augment and random.random() < 0.5:
                color = np.ascontiguousarray(color[:, ::-1, :])
                depth = np.ascontiguousarray(depth[:, ::-1])
            colors.append(color)
            depths.append(depth)

            st = build_state_vector(data)
            ac = build_action_vector(data)
            kp = np.asarray(data.get("keypoint", [0.0] * KEYPOINT_DIM), dtype=np.float32)
            if self.norm_stats:
                st = normalize(st, self.norm_stats["state_mean"], self.norm_stats["state_std"])
                ac = normalize(ac, self.norm_stats["action_mean"], self.norm_stats["action_std"])
                kp = normalize(kp, self.norm_stats["keypoint_mean"], self.norm_stats["keypoint_std"])
            states.append(st)
            actions.append(ac)
            keypoints.append(kp)

        rgbd = preprocess_rgbd_batch(
            np.stack(colors),
            np.stack(depths),
            out_w=cfg.img_width,
            out_h=cfg.img_height,
            depth_min_m=cfg.depth_min_m,
            depth_max_m=cfg.depth_max_m,
        )
        return {
            "rgbd": torch.from_numpy(rgbd),
            "state": torch.from_numpy(np.stack(states)),
            "action": torch.from_numpy(np.stack(actions)),
            "keypoint": torch.from_numpy(np.stack(keypoints)),
        }


def make_dataloaders(
    dataset_dir: Path,
    *,
    policy_cfg: InsertionPolicyConfig,
    batch_size: int,
    val_frac: float,
    seed: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader | None, dict[str, Any]]:
    episodes = list_episodes(dataset_dir)
    if len(episodes) < 1:
        raise ValueError(f"No episodes found in {dataset_dir}")

    rng = random.Random(seed)
    shuffled = episodes[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_frac))) if len(shuffled) > 1 else 0
    val_eps = shuffled[:val_count] if val_count else []
    train_eps = shuffled[val_count:] if val_count else shuffled

    norm_stats = compute_norm_stats(train_eps if train_eps else episodes)
    train_ds = InsertSensorSequenceDataset(
        train_eps,
        history_len=policy_cfg.history_len,
        policy_cfg=policy_cfg,
        norm_stats=norm_stats,
        augment=True,
    )
    val_ds = (
        InsertSensorSequenceDataset(
            val_eps,
            history_len=policy_cfg.history_len,
            policy_cfg=policy_cfg,
            norm_stats=norm_stats,
            augment=False,
        )
        if val_eps
        else None
    )
    if len(train_ds) == 0:
        raise ValueError(
            f"No training windows (need episodes with >= {policy_cfg.history_len} frames)"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=len(train_ds) > batch_size,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if val_ds and len(val_ds) > 0
        else None
    )
    return train_loader, val_loader, norm_stats
