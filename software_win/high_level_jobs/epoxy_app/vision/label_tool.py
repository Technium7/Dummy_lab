"""Seed / correct labels for apply recordings and demos.

Writes labels.json in the same format as demo episodes so results can train.

::

    # Label an apply recording (dataset/record_result/result_######)
    python -m high_level_jobs.epoxy_app.vision.label_tool --results --episode 1

    # Prefill from cascade checkpoint, then correct with keys 1-5
    python -m high_level_jobs.epoxy_app.vision.label_tool --results --episode 1 --seed-checkpoint

Keys: 1 NOT · 2 TOUCH · 3 FLOW · 4 SETTLED · 5 REMOVING
      a/d prev/next · [/] ±10 · r range-fill · w save · q quit
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import torch

_SOFTWARE_WIN = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.epoxy_app.config_loader import load_epoxy_config
from high_level_jobs.epoxy_app.view_episode import resolve_episode
from high_level_jobs.epoxy_app.vision.config_util import load_vision_config, resolve_repo_path
from high_level_jobs.epoxy_app.vision.dataset import (
    ALLOWED_LABEL_STATES,
    STATE_TO_LABELS,
    camera_frame_path,
    list_camera_stems,
    load_labels,
    normalize_camera_role,
    normalize_record_result_names,
    save_labels,
)
from high_level_jobs.epoxy_app.vision.infer_crop import infer_frame_cascade, maybe_load_tip_model
from high_level_jobs.epoxy_app.vision.model import load_checkpoint
from high_level_jobs.epoxy_app.vision.preprocess import bgr_to_gray
from high_level_jobs.epoxy_app.vision.temporal_filter import (
    TemporalFilterConfig,
    TemporalVisionFilter,
    VisionState,
)
from high_level_jobs.epoxy_app.vision.tip_track import TipTrackConfig, TipTracker

LABEL_KEYS = {
    ord("1"): "NOT_TOUCHING",
    ord("n"): "NOT_TOUCHING",
    ord("2"): "TOUCHING",
    ord("t"): "TOUCHING",
    ord("3"): "FLOWING",
    ord("f"): "FLOWING",
    ord("4"): "SETTLED",
    ord("s"): "SETTLED",
    ord("5"): "REMOVING",
    ord("m"): "REMOVING",
}

_NAV_MIN_INTERVAL_S = 0.10
_NAV_KEYS = {ord("a"), ord("d"), 81, 83, ord("["), ord("]")}

_VISION_TO_LABEL = {
    VisionState.NOT_TOUCHING: "NOT_TOUCHING",
    VisionState.TOUCHING: "TOUCHING",
    VisionState.FLOWING: "FLOWING",
    VisionState.SETTLED: "SETTLED",
    VisionState.UNCERTAIN: "TOUCHING",
}


def set_state(labels: dict, stem: str, state: str) -> None:
    state = state.strip().upper()
    if state not in ALLOWED_LABEL_STATES:
        raise ValueError(f"Unsupported label state: {state}")
    if state == "REMOVING":
        labels[stem] = {"contact": 0, "flow": 0, "removing": 1, "state": "REMOVING"}
        return
    c, fl = STATE_TO_LABELS[state]
    labels[stem] = {"contact": c, "flow": fl, "state": state}


def seed_labels_from_checkpoint(
    episode_dir: Path,
    stems: list[str],
    labels: dict,
    *,
    vision_config: str | None,
    checkpoint: str | None,
    device_s: str,
    overwrite: bool,
    role: str = "zelux",
) -> int:
    """Run cascade + temporal filter; fill unlabeled (or all if overwrite) stems."""
    vcfg = load_vision_config(vision_config)
    ckpt_path = resolve_repo_path(checkpoint or vcfg["paths"]["best_checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if device_s == "cuda" and not torch.cuda.is_available():
        device_s = "cpu"
    device = torch.device(device_s)
    model, ckpt = load_checkpoint(ckpt_path, device=device, cfg=vcfg)
    tip_cfg = dict(vcfg.get("tip", {}))
    tip_model = maybe_load_tip_model(vcfg, device, cascade_ckpt=ckpt)
    if tip_cfg.get("enabled") and tip_model is None:
        tip_cfg["enabled"] = False
    filt = TemporalVisionFilter(TemporalFilterConfig.from_dict(vcfg))
    tip_tracker = TipTracker(TipTrackConfig.from_tip_cfg(tip_cfg))
    image_size = int(vcfg["model"]["image_size"])
    contact_thr = float(vcfg["thresholds"]["contact"])

    prev_gray = None
    n_set = 0
    for stem in stems:
        if not overwrite and stem in labels and labels[stem].get("state"):
            # Still advance temporal state from image so later frames stay consistent
            path = camera_frame_path(episode_dir, stem, role)
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            gray = bgr_to_gray(bgr)
            result = infer_frame_cascade(
                model=model,
                tip_model=tip_model,
                gray=gray,
                prev_gray=prev_gray,
                device=device,
                image_size=image_size,
                contact_thr=contact_thr,
                tip_cfg=tip_cfg,
                tip_tracker=tip_tracker,
            )
            filt.update(result["contact_probability"], result["phase_probs"])
            prev_gray = gray
            continue

        path = camera_frame_path(episode_dir, stem, role)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gray = bgr_to_gray(bgr)
        result = infer_frame_cascade(
            model=model,
            tip_model=tip_model,
            gray=gray,
            prev_gray=prev_gray,
            device=device,
            image_size=image_size,
            contact_thr=contact_thr,
            tip_cfg=tip_cfg,
            tip_tracker=tip_tracker,
        )
        prev_gray = gray
        state = filt.update(result["contact_probability"], result["phase_probs"])
        label_name = _VISION_TO_LABEL.get(state, "TOUCHING")
        set_state(labels, stem, label_name)
        n_set += 1
    return n_set


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label epoxy episode Zelux frames")
    parser.add_argument("--episode", required=True, help="Episode/result number or folder name")
    parser.add_argument("--config", default=None, help="epoxy_app default_epoxy.yaml")
    parser.add_argument(
        "--results",
        action="store_true",
        help="Use dataset/record_result (apply runs) instead of training episodes/",
    )
    parser.add_argument(
        "--seed-checkpoint",
        action="store_true",
        help="Prefill unlabeled frames from cascade vision (then correct with 1-5)",
    )
    parser.add_argument(
        "--reseed",
        action="store_true",
        help="With --seed-checkpoint, overwrite existing labels too",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--vision-config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--camera",
        default="zelux",
        help="zelux/side (*_zelux.jpg) or zelux_front/front (*_zelux_front.jpg)",
    )
    args = parser.parse_args(argv)

    epoxy_cfg = load_epoxy_config(args.config)
    dataset_dir = epoxy_cfg.record_result_dir if args.results else epoxy_cfg.dataset_dir
    if args.results:
        normalize_record_result_names(dataset_dir)
    episode_dir = resolve_episode(
        dataset_dir, args.episode, prefer_prefix="result" if args.results else "episode"
    )
    if episode_dir is None:
        print(f"Episode not found under {dataset_dir}: {args.episode}", file=sys.stderr)
        return 1

    role = normalize_camera_role(args.camera)
    stems = list_camera_stems(episode_dir, role)
    if not stems:
        print(f"No *_{role}.jpg in {episode_dir}", file=sys.stderr)
        return 1

    labels = load_labels(episode_dir)
    if args.seed_checkpoint:
        try:
            n = seed_labels_from_checkpoint(
                episode_dir,
                stems,
                labels,
                vision_config=args.vision_config,
                checkpoint=args.checkpoint,
                device_s=args.device,
                overwrite=bool(args.reseed),
                role=role,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Seed failed: {exc}", file=sys.stderr)
            return 1
        save_labels(episode_dir, labels)
        print(f"Seeded {n} frames from checkpoint → {episode_dir / 'labels.json'}")

    idx = 0
    brush = "NOT_TOUCHING"
    last_marked = 0
    dirty = False
    last_nav_t = 0.0
    last_nav_key: int | None = None
    pending: deque[int] = deque()
    cached_idx = -1
    cached_img = None
    win = "epoxy_label"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    src = "record_result" if args.results else "episodes"
    print(f"Labeling [{src}] {episode_dir.name} {role} ({len(stems)} frames)")
    print("1/n NOT  2/t TOUCH  3/f FLOW  4/s SETTLED  5/m REMOVING")
    print("a/d nav  [/] ±10  r range-fill  w save  q quit")
    print("Train cuts at first REMOVING (that frame and after are ignored).")
    if args.results:
        print("After labeling, retrain: python -m high_level_jobs.epoxy_app.vision.train --device cuda")

    def next_key() -> int:
        if pending:
            return pending.popleft()
        return cv2.waitKey(30) & 0xFF

    while True:
        stem = stems[idx]
        if idx != cached_idx:
            path = camera_frame_path(episode_dir, stem, role)
            cached_img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            cached_idx = idx
        img = cached_img
        if img is None:
            idx = min(idx + 1, len(stems) - 1)
            cached_idx = -1
            continue

        row = labels.get(stem)
        saved = str(row["state"]).upper() if row and row.get("state") else None
        shown = saved if saved else "UNLABELED"

        display = img.copy()
        lines = [
            f"{idx + 1}/{len(stems)}  {stem}_{role}.jpg  [{src}]",
            f"label: {shown}",
            f"brush: {brush}  (space applies brush)",
            "1 NOT  2 TOUCH  3 FLOW  4 SETTLED  5 REMOVING | a/d [/] r w q",
        ]
        for i, text in enumerate(lines):
            y = 24 + i * 22
            cv2.putText(display, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(display, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow(win, display)

        key = next_key()
        if key == 255:
            continue

        now = time.perf_counter()
        if key in _NAV_KEYS:
            same_hold = last_nav_key == key and (now - last_nav_t) < _NAV_MIN_INTERVAL_S
            if same_hold:
                while True:
                    k2 = cv2.waitKey(1) & 0xFF
                    if k2 == 255:
                        break
                    if k2 != key:
                        pending.append(k2)
                        break
                continue
            last_nav_t = now
            last_nav_key = key
        else:
            last_nav_key = None

        if key in LABEL_KEYS:
            brush = LABEL_KEYS[key]
            set_state(labels, stem, brush)
            last_marked = idx
            dirty = True
            idx = min(idx + 1, len(stems) - 1)
        elif key == ord(" "):
            set_state(labels, stem, brush)
            dirty = True
            idx = min(idx + 1, len(stems) - 1)
        elif key in (ord("d"), 83):
            idx = min(idx + 1, len(stems) - 1)
        elif key in (ord("a"), 81):
            idx = max(idx - 1, 0)
        elif key == ord("]"):
            idx = min(idx + 10, len(stems) - 1)
        elif key == ord("["):
            idx = max(idx - 10, 0)
        elif key == ord("r"):
            lo, hi = sorted((last_marked, idx))
            for j in range(lo, hi + 1):
                set_state(labels, stems[j], brush)
            dirty = True
            print(f"Range-filled {stems[lo]}..{stems[hi]} -> {brush}")
        elif key == ord("w"):
            save_labels(episode_dir, labels)
            dirty = False
            print(f"Wrote {episode_dir / 'labels.json'}")
        elif key in (ord("q"), 27):
            if dirty:
                save_labels(episode_dir, labels)
                print(f"Auto-saved {episode_dir / 'labels.json'}")
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
