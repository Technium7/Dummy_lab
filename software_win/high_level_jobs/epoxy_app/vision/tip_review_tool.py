"""Review / auto-label tips with TipNet; click to correct.

Loads tip checkpoint, predicts tip per frame. You accept or click to fix.
Corrections are saved into labels.json for more training data.

::

    # After tip_train:
    python -m high_level_jobs.epoxy_app.vision.tip_review_tool --episode 6 --device cuda
    # Re-run tracked fill over old model tips (keeps human corrections):
    python -m high_level_jobs.epoxy_app.vision.tip_review_tool --episode 6 --device cuda --refill-model

Keys:
  click     set tip (human correction)
  space     accept model prediction for this frame
  f         fill all frames that lack a tip with model predictions
  a/d       prev/next frame
  [/]       ±10 frames
  x         clear tip on this frame
  w         save
  q         save if dirty and quit
"""

from __future__ import annotations

import argparse
import sys
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
    camera_frame_path,
    list_camera_stems,
    load_labels,
    normalize_camera_role,
    parse_tip,
    save_labels,
    tip_key_for_role,
)
from high_level_jobs.epoxy_app.vision.preprocess import bgr_to_gray, to_model_tensor
from high_level_jobs.epoxy_app.vision.tip_model import build_tip_model, load_tip_checkpoint
from high_level_jobs.epoxy_app.vision.tip_track import TipTrackConfig, TipTracker


def _ensure_row(labels: dict, stem: str) -> dict:
    row = labels.get(stem)
    if row is None:
        row = {}
        labels[stem] = row
    return row


def set_tip(labels: dict, stem: str, x: float, y: float, *, source: str = "human", key: str = "tip") -> None:
    row = _ensure_row(labels, stem)
    row[key] = {"x": float(x), "y": float(y), "source": source}


def clear_tip(labels: dict, stem: str, *, key: str = "tip") -> None:
    row = labels.get(stem)
    if row and key in row:
        del row[key]


def _load_tip_net(vcfg: dict, device: torch.device, checkpoint: str | None):
    tip_cfg = vcfg.get("tip", {})
    path = None
    if checkpoint:
        path = resolve_repo_path(checkpoint)
    elif tip_cfg.get("checkpoint"):
        path = resolve_repo_path(tip_cfg["checkpoint"])
    else:
        ckpt_dir = resolve_repo_path(vcfg["paths"]["checkpoints"])
        for name in ("tip_best_full.pt", "tip_best.pt"):
            cand = ckpt_dir / name
            if cand.exists():
                path = cand
                break
    if path is None or not path.exists():
        return None, None
    if path.name.endswith("_full.pt") or "full" in path.name:
        model, ckpt = load_tip_checkpoint(path, device=device, cfg=vcfg)
        return model, path
    model = build_tip_model(vcfg)
    model.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, path


@torch.inference_mode()
def predict_tip_tracked(
    model,
    gray,
    image_size: int,
    device: torch.device,
    tracker: TipTracker,
) -> tuple[float, float, float]:
    h, w = gray.shape[:2]
    t = to_model_tensor(gray, image_size).unsqueeze(0).to(device)
    return tracker.step(model, t, frame_w=w, frame_h=h)


def _run_tracked_preds(
    *,
    stems: list[str],
    episode_dir: Path,
    tip_model,
    image_size: int,
    device: torch.device,
    tip_cfg: dict,
    role: str = "zelux",
) -> dict[str, tuple[float, float, float]]:
    """Predict tips in order with temporal tracking (avoids apply-blob lock)."""
    tracker = TipTracker(TipTrackConfig.from_tip_cfg(tip_cfg))
    preds: dict[str, tuple[float, float, float]] = {}
    for stem in stems:
        path = camera_frame_path(episode_dir, stem, role)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            preds[stem] = (0.0, 0.0, 0.0)
            continue
        gray = bgr_to_gray(bgr)
        preds[stem] = predict_tip_tracked(tip_model, gray, image_size, device, tracker)
    return preds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-label tips with TipNet + click to fix")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--vision-config", default=None)
    parser.add_argument("--epoxy-config", default=None)
    parser.add_argument("--checkpoint", default=None, help="tip_best_full.pt or tip_best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--results",
        action="store_true",
        help="Use dataset/record_result (apply runs)",
    )
    parser.add_argument(
        "--fill-on-start",
        action="store_true",
        help="On open, write model tips onto all frames missing tip",
    )
    parser.add_argument(
        "--refill-model",
        action="store_true",
        help="Overwrite tips with source=model using tracked predictions (keeps human tips)",
    )
    parser.add_argument(
        "--camera",
        default="zelux",
        help="zelux/side (*_zelux.jpg) or zelux_front/front (*_zelux_front.jpg)",
    )
    args = parser.parse_args(argv)

    vcfg = load_vision_config(args.vision_config)
    epoxy = load_epoxy_config(args.epoxy_config)
    dataset_dir = epoxy.record_result_dir if args.results else epoxy.dataset_dir
    episode_dir = resolve_episode(
        dataset_dir, args.episode, prefer_prefix="result" if args.results else "episode"
    )
    if episode_dir is None:
        print(f"Episode not found under {dataset_dir}: {args.episode}", file=sys.stderr)
        return 1

    role = normalize_camera_role(args.camera)
    tip_key = tip_key_for_role(role)
    stems = list_camera_stems(episode_dir, role)
    if not stems:
        print(f"No *_{role}.jpg in {episode_dir}", file=sys.stderr)
        return 1

    device_s = args.device
    if device_s == "cuda" and not torch.cuda.is_available():
        device_s = "cpu"
    device = torch.device(device_s)
    image_size = int(vcfg["model"]["image_size"])

    tip_model, tip_path = _load_tip_net(vcfg, device, args.checkpoint)
    if tip_model is None:
        print(
            "No tip checkpoint found. Train first:\n"
            "  python -m high_level_jobs.epoxy_app.vision.tip_train --device cuda",
            file=sys.stderr,
        )
        return 1
    print(f"Loaded tip model: {tip_path} ({device})")
    tip_cfg = dict(vcfg.get("tip", {}))
    track_on = bool(TipTrackConfig.from_tip_cfg(tip_cfg).enabled)
    print(f"Tip tracking={'on' if track_on else 'off'} (tip.track in config)")

    labels = load_labels(episode_dir)
    # Cache predictions: stem -> (x, y, conf) — filled sequentially with tracker
    preds: dict[str, tuple[float, float, float]] = {}

    def ensure_all_preds() -> None:
        nonlocal preds
        if len(preds) == len(stems):
            return
        preds = _run_tracked_preds(
            stems=stems,
            episode_dir=episode_dir,
            tip_model=tip_model,
            image_size=image_size,
            device=device,
            tip_cfg=tip_cfg,
            role=role,
        )

    def ensure_pred(stem: str) -> tuple[float, float, float]:
        ensure_all_preds()
        return preds[stem]

    dirty = False
    if args.fill_on_start or args.refill_model:
        ensure_all_preds()
        n_fill = 0
        for stem in stems:
            row = labels.get(stem) or {}
            tip = row.get(tip_key) if isinstance(row, dict) else None
            src = str(tip.get("source") or "") if isinstance(tip, dict) else ""
            if args.refill_model:
                if tip is not None and src == "human":
                    continue
            elif parse_tip(labels.get(stem), key=tip_key) is not None:
                continue
            x, y, _c = preds[stem]
            set_tip(labels, stem, x, y, source="model", key=tip_key)
            n_fill += 1
        dirty = n_fill > 0
        mode = "refill-model" if args.refill_model else "fill missing"
        print(f"Wrote {n_fill} tracked model tips ({mode})")

    idx = 0
    win = "epoxy_tip_review"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    click = {"xy": None}

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            click["xy"] = (float(x), float(y))

    cv2.setMouseCallback(win, on_mouse)

    print(f"Review {episode_dir.name} {role} ({len(stems)} frames)")
    print("click=fix  space=accept pred  f=fill missing  a/d [/]  x clear  w save  q quit")

    while True:
        stem = stems[idx]
        path = camera_frame_path(episode_dir, stem, role)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            idx = min(idx + 1, len(stems) - 1)
            continue

        if click["xy"] is not None:
            set_tip(labels, stem, click["xy"][0], click["xy"][1], source="human", key=tip_key)
            click["xy"] = None
            dirty = True

        px, py, pconf = ensure_pred(stem)
        saved = parse_tip(labels.get(stem), key=tip_key)
        src = ""
        row = labels.get(stem) or {}
        if isinstance(row.get(tip_key), dict):
            src = str(row[tip_key].get("source") or "")

        display = img.copy()
        # Model prediction (green)
        cv2.drawMarker(
            display,
            (int(round(px)), int(round(py))),
            (0, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=2,
        )
        # Saved tip (cyan/yellow)
        if saved is not None:
            color = (0, 255, 255) if src == "human" else (255, 200, 0)
            cv2.circle(display, (int(round(saved[0])), int(round(saved[1]))), 10, color, 2)

        tip_line = (
            f"saved=({saved[0]:.0f},{saved[1]:.0f}) [{src or '?'}]"
            if saved
            else "saved=NONE"
        )
        lines = [
            f"{idx + 1}/{len(stems)}  {path.name}",
            f"pred=({px:.0f},{py:.0f}) conf={pconf:.2f}",
            tip_line,
            "click fix | space accept | f fill | a/d [/] | x | w | q",
        ]
        for i, text in enumerate(lines):
            y = 24 + i * 22
            cv2.putText(display, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(display, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(" "):
            set_tip(labels, stem, px, py, source="model", key=tip_key)
            dirty = True
            idx = min(idx + 1, len(stems) - 1)
        elif key == ord("f"):
            ensure_all_preds()
            n_fill = 0
            for s in stems:
                if parse_tip(labels.get(s), key=tip_key) is not None:
                    continue
                x, y, _c = preds[s]
                set_tip(labels, s, x, y, source="model", key=tip_key)
                n_fill += 1
            dirty = dirty or n_fill > 0
            print(f"Filled {n_fill} missing tips with tracked model")
        elif key in (ord("d"), 83):
            idx = min(idx + 1, len(stems) - 1)
        elif key in (ord("a"), 81):
            idx = max(idx - 1, 0)
        elif key == ord("]"):
            idx = min(idx + 10, len(stems) - 1)
        elif key == ord("["):
            idx = max(idx - 10, 0)
        elif key == ord("x"):
            clear_tip(labels, stem, key=tip_key)
            dirty = True
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
