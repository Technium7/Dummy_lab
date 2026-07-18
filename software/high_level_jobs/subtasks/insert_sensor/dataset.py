"""Episode dataset for insert_sensor ACT / diffusion (obs history + action chunk)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from manual_control.mapping import transformed_deg_to_sim_rad

from .cameras import OVERHEAD_ROLE, POLICY_CAMERA_ROLES, load_rgbd_pair


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
    for ep in episodes:
        for frame_path in sorted(ep.path.glob("*.frame.json")):
            data = _load_frame_json(frame_path)
            states.append(build_state_vector(data))
            actions.append(build_action_vector(data))
    if not states:
        return {}
    s = np.stack(states)
    a = np.stack(actions)
    return {
        "state_mean": s.mean(axis=0).tolist(),
        "state_std": (s.std(axis=0) + 1e-6).tolist(),
        "action_mean": a.mean(axis=0).tolist(),
        "action_std": (a.std(axis=0) + 1e-6).tolist(),
    }


def normalize(x: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    m = np.asarray(mean, dtype=np.float32)
    s = np.asarray(std, dtype=np.float32)
    return (x - m) / s


def _preprocess_fn(policy_cfg: Any):
    """ACT uses model.preprocess; diffusion may append table XYZ channels."""
    if bool(getattr(policy_cfg, "use_table_xyz", False)) or hasattr(
        policy_cfg, "diffusion_steps"
    ):
        from .policies.diffusion import preprocess_rgbd_batch

        return preprocess_rgbd_batch
    from .model import preprocess_rgbd_batch

    return preprocess_rgbd_batch


class InsertSensorSequenceDataset(Dataset):
    """Obs window [i, i+H) + action chunk starting at i+H-1 of length Ta."""

    def __init__(
        self,
        episodes: list[EpisodeInfo],
        *,
        history_len: int,
        policy_cfg: Any,
        norm_stats: dict[str, Any] | None = None,
        augment: bool = False,
    ) -> None:
        self.episodes = episodes
        self.history_len = history_len
        self.action_horizon = int(policy_cfg.action_horizon)
        self.policy_cfg = policy_cfg
        self.norm_stats = norm_stats or {}
        self.augment = augment
        self._preprocess = _preprocess_fn(policy_cfg)
        self._table_xyz_ctx = None
        if bool(getattr(policy_cfg, "use_table_xyz", False)):
            from .table_xyz import TableXyzContext

            self._table_xyz_ctx = TableXyzContext.from_config(
                table_calib_path=getattr(policy_cfg, "table_calib_path", "") or None,
                xy_min_m=float(getattr(policy_cfg, "table_xyz_xy_min_m", -0.40)),
                xy_max_m=float(getattr(policy_cfg, "table_xyz_xy_max_m", 0.40)),
                z_min_m=float(getattr(policy_cfg, "table_xyz_z_min_m", -0.05)),
                z_max_m=float(getattr(policy_cfg, "table_xyz_z_max_m", 0.35)),
            )
        self._windows: list[tuple[Path, int]] = []
        need = history_len + self.action_horizon - 1
        for ep in episodes:
            frames = sorted(ep.path.glob("*.frame.json"))
            n = len(frames)
            if n < need:
                continue
            for start in range(0, n - need + 1):
                self._windows.append((ep.path, start))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_dir, start = self._windows[idx]
        all_paths = sorted(ep_dir.glob("*.frame.json"))
        obs_paths = all_paths[start : start + self.history_len]
        act_start = start + self.history_len - 1
        act_paths = all_paths[act_start : act_start + self.action_horizon]

        cfg = self.policy_cfg
        n_cam = int(cfg.num_cameras)
        roles = POLICY_CAMERA_ROLES[: max(1, n_cam)]
        flip = self.augment and random.random() < 0.5
        use_table_xyz = bool(getattr(cfg, "use_table_xyz", False))

        per_cam_colors: list[list[np.ndarray]] = [[] for _ in roles]
        per_cam_depths: list[list[np.ndarray]] = [[] for _ in roles]
        overhead_xyz: list[np.ndarray] = []
        states: list[np.ndarray] = []

        for fp in obs_paths:
            data = _load_frame_json(fp)
            stem = _frame_stem(fp)
            for cam_i, role in enumerate(roles):
                color, depth = load_rgbd_pair(ep_dir, stem, role)
                if color is None or depth is None:
                    if cam_i == 0:
                        raise FileNotFoundError(f"Missing overhead RGB-D for {ep_dir}/{stem}")
                    h, w = per_cam_colors[0][-1].shape[:2]
                    color = np.zeros((h, w, 3), dtype=np.uint8)
                    depth = np.zeros((h, w), dtype=np.float32)

                xyz_map = None
                if use_table_xyz and role == OVERHEAD_ROLE and self._table_xyz_ctx is not None:
                    from .table_xyz import depth_to_table_xyz

                    xyz_map = depth_to_table_xyz(depth, self._table_xyz_ctx)

                if flip:
                    color = np.ascontiguousarray(color[:, ::-1, :])
                    depth = np.ascontiguousarray(depth[:, ::-1])
                    if xyz_map is not None:
                        xyz_map = np.ascontiguousarray(xyz_map[:, :, ::-1])

                per_cam_colors[cam_i].append(color)
                per_cam_depths[cam_i].append(depth)
                if role == OVERHEAD_ROLE and xyz_map is not None:
                    overhead_xyz.append(xyz_map)

            st = build_state_vector(data)
            if self.norm_stats:
                st = normalize(st, self.norm_stats["state_mean"], self.norm_stats["state_std"])
            states.append(st)

        cam_tensors = []
        for cam_i, role in enumerate(roles):
            kwargs: dict[str, Any] = {
                "out_w": cfg.img_width,
                "out_h": cfg.img_height,
                "depth_min_m": cfg.depth_min_m,
                "depth_max_m": cfg.depth_max_m,
                "use_depth": bool(getattr(cfg, "use_depth", True)),
            }
            if use_table_xyz:
                kwargs["use_table_xyz"] = True
                kwargs["table_xyz_ctx"] = self._table_xyz_ctx
                kwargs["is_overhead"] = role == OVERHEAD_ROLE
                if role == OVERHEAD_ROLE and overhead_xyz:
                    kwargs["table_xyz_maps"] = np.stack(overhead_xyz)
            cam_tensors.append(
                self._preprocess(
                    np.stack(per_cam_colors[cam_i]),
                    np.stack(per_cam_depths[cam_i]),
                    **kwargs,
                )
            )
        rgbd = np.stack(cam_tensors, axis=1)  # (T, N, C, H, W)

        actions: list[np.ndarray] = []
        for fp in act_paths:
            data = _load_frame_json(fp)
            ac = build_action_vector(data)
            if self.norm_stats:
                ac = normalize(ac, self.norm_stats["action_mean"], self.norm_stats["action_std"])
            actions.append(ac)

        return {
            "rgbd": torch.from_numpy(rgbd),
            "state": torch.from_numpy(np.stack(states)),
            "action": torch.from_numpy(np.stack(actions)),
        }


def make_dataloaders(
    dataset_dir: Path,
    *,
    policy_cfg: Any,
    batch_size: int,
    val_frac: float,
    seed: int,
    num_workers: int = 0,
    max_episodes: int | None = None,
) -> tuple[DataLoader, DataLoader | None, dict[str, Any]]:
    episodes = list_episodes(dataset_dir)
    if len(episodes) < 1:
        raise ValueError(f"No episodes found in {dataset_dir}")
    if max_episodes is not None and max_episodes > 0:
        episodes = episodes[: int(max_episodes)]
        print(
            f"[insert_sensor] Using first {len(episodes)} episode(s) "
            f"(max_episodes={max_episodes})",
            flush=True,
        )

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
    need = policy_cfg.history_len + policy_cfg.action_horizon - 1
    if len(train_ds) == 0:
        raise ValueError(f"No training windows (need episodes with >= {need} frames)")

    loader_kw: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_ds) > batch_size,
        **loader_kw,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kw)
        if val_ds and len(val_ds) > 0
        else None
    )
    return train_loader, val_loader, norm_stats
