"""Episode-level dataset: labels live in episode_######/labels.json (Zelux full frames)."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import torch
from torch.utils.data import Dataset

from high_level_jobs.epoxy_app.vision.preprocess import (
    EpoxyAugment,
    bgr_to_gray,
    build_temporal_mono,
    crop_around_tip,
    gaussian_heatmap,
    resize_hwc,
    to_model_tensor,
)

# Labels: contact binary + phase among contact frames
# REMOVING = lift-off; that frame and everything after are dropped from training.
STATE_TO_LABELS = {
    "NOT_TOUCHING": (0, 0),
    "TOUCHING": (1, 0),
    "FLOWING": (1, 1),
    "SETTLED": (1, 0),
}

ALLOWED_LABEL_STATES = (
    "NOT_TOUCHING",
    "TOUCHING",
    "FLOWING",
    "SETTLED",
    "REMOVING",
)

# Phase index for stage-2. SETTLED is contact but not a phase class.
STATE_TO_PHASE = {
    "TOUCHING": 0,
    "FLOWING": 1,
}

LABELS_SCHEMA = "epoxy_cascade_v3"


@dataclass
class FrameSample:
    episode_id: str
    stem: str
    frame_path: Path
    prev_frame_path: Path | None
    contact: int
    flow: int
    state: str | None = None
    phase: int | None = None  # 0 TOUCHING, 1 FLOWING; None if not phase-train
    tip_xy: tuple[float, float] | None = None  # full-frame tip (x, y) if labeled/interpolated


def is_removing_row(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    state = str(row.get("state") or "").strip().upper()
    if state == "REMOVING":
        return True
    try:
        return int(row.get("removing") or 0) == 1
    except (TypeError, ValueError):
        return False


def first_removing_index(stems: Sequence[str], labels: dict[str, dict[str, Any]]) -> int | None:
    """Index of first REMOVING label in stem order, or None."""
    for i, stem in enumerate(stems):
        if is_removing_row(labels.get(stem)):
            return i
    return None


def stems_before_removing(stems: Sequence[str], labels: dict[str, dict[str, Any]]) -> list[str]:
    """Keep frames strictly before the first REMOVING mark."""
    cut = first_removing_index(stems, labels)
    if cut is None:
        return list(stems)
    return list(stems[:cut])


def parse_tip(row: dict[str, Any] | None) -> tuple[float, float] | None:
    if not row:
        return None
    tip = row.get("tip")
    if not isinstance(tip, dict):
        return None
    if "x" not in tip or "y" not in tip:
        return None
    return float(tip["x"]), float(tip["y"])


def interpolate_tips_along_stems(
    stems: Sequence[str],
    tip_by_stem: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Linear interpolate tip (x,y) between sparsely labeled stems (by stem order)."""
    if not tip_by_stem:
        return {}
    indexed = [(i, stems[i], tip_by_stem[stems[i]]) for i in range(len(stems)) if stems[i] in tip_by_stem]
    if not indexed:
        return {}
    out = dict(tip_by_stem)
    for (i0, _s0, (x0, y0)), (i1, _s1, (x1, y1)) in zip(indexed, indexed[1:]):
        span = i1 - i0
        if span <= 1:
            continue
        for k in range(i0 + 1, i1):
            t = (k - i0) / float(span)
            out[stems[k]] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    return out


def normalize_record_result_names(record_result_dir: Path) -> list[tuple[str, str]]:
    """Rename leftover episode_* folders under record_result to result_* (no name clash)."""
    root = Path(record_result_dir)
    if not root.is_dir():
        return []
    renamed: list[tuple[str, str]] = []
    for p in sorted(root.glob("episode_*")):
        if not p.is_dir():
            continue
        suffix = p.name.split("_")[-1]
        dest = root / f"result_{suffix}"
        if dest.exists():
            k = 1
            while True:
                dest = root / f"result_{suffix}_{k:02d}"
                if not dest.exists():
                    break
                k += 1
        p.rename(dest)
        renamed.append((p.name, dest.name))
        print(f"[epoxy_app] Renamed record_result/{p.name} -> {dest.name}", flush=True)
    return renamed


def discover_tip_labeled_episodes(*dataset_dirs: Path) -> list[Path]:
    found: list[Path] = []
    for dataset_dir in dataset_dirs:
        if not Path(dataset_dir).is_dir():
            continue
        for pattern in ("episode_*", "result_*"):
            for p in sorted(Path(dataset_dir).glob(pattern)):
                if not p.is_dir():
                    continue
                labels = load_labels(p)
                if any(parse_tip(row) is not None for row in labels.values()):
                    found.append(p)
    return sorted(found, key=lambda p: p.name)


@dataclass
class EpisodeSplit:
    train: list[str]
    val: list[str]
    test: list[str]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EpisodeSplit":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(train=data["train"], val=data["val"], test=data["test"])


def labels_path(episode_dir: Path) -> Path:
    return episode_dir / "labels.json"


def load_labels(episode_dir: Path) -> dict[str, dict[str, Any]]:
    path = labels_path(episode_dir)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames", data)
    return {str(k): v for k, v in frames.items()}


def save_labels(episode_dir: Path, frames: dict[str, dict[str, Any]]) -> None:
    payload = {
        "schema": LABELS_SCHEMA,
        "camera": "zelux",
        "frames": frames,
    }
    labels_path(episode_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def list_zelux_stems(episode_dir: Path) -> list[str]:
    paths = sorted(episode_dir.glob("*_zelux.jpg"))
    return [p.name[: -len("_zelux.jpg")] for p in paths]


def discover_labeled_episodes(*dataset_dirs: Path) -> list[Path]:
    found: list[Path] = []
    for dataset_dir in dataset_dirs:
        if not Path(dataset_dir).is_dir():
            continue
        for pattern in ("episode_*", "result_*"):
            for p in sorted(Path(dataset_dir).glob(pattern)):
                if not p.is_dir():
                    continue
                labels = load_labels(p)
                if labels:
                    found.append(p)
    return sorted(found, key=lambda p: p.name)


def find_episode_dir(dataset_dirs: Sequence[Path] | Path, name: str) -> Path | None:
    roots = [dataset_dirs] if isinstance(dataset_dirs, Path) else list(dataset_dirs)
    for root in roots:
        cand = Path(root) / name
        if cand.is_dir():
            return cand
    return None


def load_episode_samples(
    episode_dir: Path,
    *,
    interpolate_tips: bool = True,
) -> list[FrameSample]:
    labels = load_labels(episode_dir)
    if not labels:
        return []
    stems_all = list_zelux_stems(episode_dir)
    stems = stems_before_removing(stems_all, labels)
    cut = first_removing_index(stems_all, labels)
    if cut is not None:
        n_drop = len(stems_all) - cut
        print(
            f"  {episode_dir.name}: dataset cut at first REMOVING "
            f"(dropped {n_drop} frames from index {cut}; train+val)"
        )

    tip_marked = {s: t for s in stems if (t := parse_tip(labels.get(s))) is not None}
    tip_map = (
        interpolate_tips_along_stems(stems, tip_marked) if interpolate_tips else dict(tip_marked)
    )
    samples: list[FrameSample] = []
    prev: Path | None = None
    for stem in stems:
        row = labels.get(stem)
        if row is None:
            prev = episode_dir / f"{stem}_zelux.jpg"
            continue

        state = (row.get("state") or "").strip().upper() or None
        if is_removing_row(row):
            break

        if state in STATE_TO_LABELS:
            contact, flow = STATE_TO_LABELS[state]
        elif "contact" in row and "flow" in row and row["contact"] != "" and row["flow"] != "":
            contact, flow = int(row["contact"]), int(row["flow"])
        else:
            continue

        phase = None
        if contact == 1 and state in STATE_TO_PHASE:
            phase = STATE_TO_PHASE[state]
        elif contact == 1 and state is None:
            # Legacy contact/flow without state name
            phase = 1 if flow == 1 else 0

        path = episode_dir / f"{stem}_zelux.jpg"
        if not path.exists():
            continue
        samples.append(
            FrameSample(
                episode_id=episode_dir.name,
                stem=stem,
                frame_path=path,
                prev_frame_path=prev,
                contact=contact,
                flow=flow,
                state=state,
                phase=phase,
                tip_xy=tip_map.get(stem),
            )
        )
        prev = path
    return samples


def load_tip_samples(
    episode_dir: Path,
    *,
    interpolate_tips: bool = True,
) -> list[FrameSample]:
    """Frames with tip coordinates (marked or interpolated) for TipNet training."""
    labels = load_labels(episode_dir)
    stems_all = list_zelux_stems(episode_dir)
    stems = stems_before_removing(stems_all, labels)
    tip_marked = {s: t for s in stems if (t := parse_tip(labels.get(s))) is not None}
    if not tip_marked:
        return []
    tip_map = (
        interpolate_tips_along_stems(stems, tip_marked) if interpolate_tips else dict(tip_marked)
    )
    samples: list[FrameSample] = []
    prev: Path | None = None
    for stem in stems:
        path = episode_dir / f"{stem}_zelux.jpg"
        tip = tip_map.get(stem)
        if tip is None or not path.exists():
            prev = path if path.exists() else prev
            continue
        samples.append(
            FrameSample(
                episode_id=episode_dir.name,
                stem=stem,
                frame_path=path,
                prev_frame_path=prev,
                contact=0,
                flow=0,
                tip_xy=tip,
            )
        )
        prev = path
    return samples


def split_episodes(
    episode_names: Sequence[str],
    *,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    train_episodes: Sequence[str] | None = None,
    val_episodes: Sequence[str] | None = None,
    test_episodes: Sequence[str] | None = None,
) -> EpisodeSplit:
    names = sorted(set(episode_names))
    if train_episodes or val_episodes or test_episodes:
        train = list(train_episodes or [])
        val = list(val_episodes or [])
        test = list(test_episodes or [])
        assigned = set(train) | set(val) | set(test)
        train.extend(n for n in names if n not in assigned)
        return EpisodeSplit(train=train, val=val, test=test)

    if len(names) <= 1:
        return EpisodeSplit(train=list(names), val=[], test=[])

    rng = random.Random(seed)
    shuffled = names[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 and n >= 3 else 0
    n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 and n >= 2 else 0
    while n_test + n_val >= n and n_test > 0:
        n_test -= 1
    while n_test + n_val >= n and n_val > 0:
        n_val -= 1
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return EpisodeSplit(train=train, val=val, test=test)


def samples_for_episodes(
    dataset_dir: Path | Sequence[Path],
    names: Sequence[str],
    *,
    interpolate_tips: bool = True,
) -> list[FrameSample]:
    out: list[FrameSample] = []
    for name in names:
        ep = find_episode_dir(dataset_dir, name)
        if ep is None:
            print(f"  skip missing episode: {name}", flush=True)
            continue
        out.extend(load_episode_samples(ep, interpolate_tips=interpolate_tips))
    return out


def tip_samples_for_episodes(
    dataset_dir: Path | Sequence[Path],
    names: Sequence[str],
    *,
    interpolate_tips: bool = True,
) -> list[FrameSample]:
    out: list[FrameSample] = []
    for name in names:
        ep = find_episode_dir(dataset_dir, name)
        if ep is None:
            print(f"  skip missing tip episode: {name}", flush=True)
            continue
        out.extend(load_tip_samples(ep, interpolate_tips=interpolate_tips))
    return out


def compute_pos_weights(samples: Sequence[FrameSample]) -> tuple[float, float]:
    n = max(len(samples), 1)
    c_pos = sum(s.contact for s in samples) or 1
    f_pos = sum(s.flow for s in samples) or 1
    return float((n - c_pos) / c_pos), float((n - f_pos) / f_pos)


def phase_samples(samples: Sequence[FrameSample]) -> list[FrameSample]:
    """TOUCHING/FLOWING only — SETTLED is contact but excluded from phase train."""
    return [
        s
        for s in samples
        if s.contact == 1 and s.phase is not None and (s.state or "").upper() != "SETTLED"
    ]


def _maybe_tip_crop(
    gray: Any,
    tip_xy: tuple[float, float] | None,
    *,
    tip_enabled: bool,
    crop_size: int,
) -> Any:
    if tip_enabled and tip_xy is not None:
        return crop_around_tip(gray, tip_xy[0], tip_xy[1], crop_size)
    return gray


class ContactDataset(Dataset):
    """Stage 1: mono gray (optional tip crop) → contact label."""

    def __init__(
        self,
        samples: Sequence[FrameSample],
        image_size: int = 224,
        augment: EpoxyAugment | None = None,
        train_cfg: dict | None = None,
        tip_cfg: dict | None = None,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.augment = augment
        self.train_cfg = train_cfg or {}
        tip = tip_cfg or {}
        self.tip_enabled = bool(tip.get("enabled", False))
        self.crop_size = int(tip.get("crop_size", 256))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        bgr = cv2.imread(str(s.frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(s.frame_path)
        gray = bgr_to_gray(bgr)
        gray = _maybe_tip_crop(
            gray, s.tip_xy, tip_enabled=self.tip_enabled, crop_size=self.crop_size
        )
        gray = resize_hwc(gray, self.image_size)
        if self.augment is not None:
            gray = self.augment(gray)
        w = sample_weights_from_config(s, self.train_cfg) if self.train_cfg else _sample_weight(s)
        return {
            "image": to_model_tensor(gray, self.image_size),
            "contact": torch.tensor(s.contact, dtype=torch.float32),
            "sample_weight": torch.tensor(w, dtype=torch.float32),
            "stem": s.stem,
        }


class PhaseDataset(Dataset):
    """Stage 2: mono+|diff| (optional tip crop) → phase class."""

    def __init__(
        self,
        samples: Sequence[FrameSample],
        image_size: int = 224,
        augment: EpoxyAugment | None = None,
        tip_cfg: dict | None = None,
    ) -> None:
        self.samples = phase_samples(samples)
        self.image_size = image_size
        self.augment = augment
        tip = tip_cfg or {}
        self.tip_enabled = bool(tip.get("enabled", False))
        self.crop_size = int(tip.get("crop_size", 256))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        bgr = cv2.imread(str(s.frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(s.frame_path)
        gray = bgr_to_gray(bgr)
        prev = None
        if s.prev_frame_path is not None and s.prev_frame_path.exists():
            pbgr = cv2.imread(str(s.prev_frame_path), cv2.IMREAD_COLOR)
            if pbgr is not None:
                prev = bgr_to_gray(pbgr)
        if self.tip_enabled and s.tip_xy is not None:
            gray = crop_around_tip(gray, s.tip_xy[0], s.tip_xy[1], self.crop_size)
            if prev is not None:
                prev = crop_around_tip(prev, s.tip_xy[0], s.tip_xy[1], self.crop_size)
        two = build_temporal_mono(gray, prev)
        two = resize_hwc(two, self.image_size)
        if self.augment is not None:
            two = self.augment(two)
        assert s.phase is not None
        return {
            "image": to_model_tensor(two, self.image_size),
            "phase": torch.tensor(s.phase, dtype=torch.long),
            "stem": s.stem,
        }


class TipDataset(Dataset):
    """Full-frame mono → tip heatmap target."""

    def __init__(
        self,
        samples: Sequence[FrameSample],
        image_size: int = 224,
        heatmap_size: int = 64,
        sigma: float = 2.0,
        augment: EpoxyAugment | None = None,
    ) -> None:
        self.samples = [s for s in samples if s.tip_xy is not None]
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        assert s.tip_xy is not None
        bgr = cv2.imread(str(s.frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(s.frame_path)
        gray = bgr_to_gray(bgr)
        h0, w0 = gray.shape[:2]
        tx, ty = s.tip_xy
        gray_r = resize_hwc(gray, self.image_size)
        if self.augment is not None:
            gray_r = self.augment(gray_r)
        scale = self.heatmap_size / float(self.image_size)
        hx = (tx / max(w0, 1)) * self.image_size * scale
        hy = (ty / max(h0, 1)) * self.image_size * scale
        heat = gaussian_heatmap(self.heatmap_size, self.heatmap_size, hx, hy, sigma=self.sigma)
        return {
            "image": to_model_tensor(gray_r, self.image_size),
            "heatmap": torch.from_numpy(heat).unsqueeze(0),
            "tip_xy": torch.tensor([tx, ty], dtype=torch.float32),
            "frame_w": w0,
            "frame_h": h0,
            "stem": s.stem,
        }


# Back-compat name used by evaluate scripts
EpoxyEpisodeDataset = ContactDataset


def _sample_weight(s: FrameSample) -> float:
    if s.contact:
        return 4.0
    return 0.2


def sample_weights_from_config(s: FrameSample, tcfg: dict) -> float:
    if s.contact:
        return float(tcfg.get("contact_sample_weight", 4.0))
    return float(tcfg.get("not_touching_weight", 0.2))
