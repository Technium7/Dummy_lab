"""Run cascade vision over an episode; save overlay MP4.

::

    python -m high_level_jobs.epoxy_app.vision.demo_episode --episode 10
"""

from __future__ import annotations

import argparse
import json
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
    list_zelux_stems,
    load_labels,
    stems_before_removing,
)
from high_level_jobs.epoxy_app.vision.infer_crop import infer_frame_cascade, maybe_load_tip_model
from high_level_jobs.epoxy_app.vision.model import load_checkpoint
from high_level_jobs.epoxy_app.vision.preprocess import bgr_to_gray
from high_level_jobs.epoxy_app.vision.temporal_filter import TemporalFilterConfig, TemporalVisionFilter
from high_level_jobs.epoxy_app.vision.tip_track import TipTrackConfig, TipTracker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--vision-config", default=None)
    parser.add_argument("--epoxy-config", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-tip", action="store_true", help="Disable tip crop even if configured")
    parser.add_argument(
        "--results",
        action="store_true",
        help="Load from dataset/record_result (apply runs)",
    )
    parser.add_argument(
        "--full-episode",
        action="store_true",
        help="Include frames at/after first REMOVING (default: cut like train/val)",
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

    ckpt_path = resolve_repo_path(args.checkpoint or vcfg["paths"]["best_checkpoint"])
    if not ckpt_path.exists():
        print(f"Missing checkpoint: {ckpt_path}", file=sys.stderr)
        return 1

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model, ckpt = load_checkpoint(ckpt_path, device=device, cfg=vcfg)
    tip_cfg = dict(vcfg.get("tip", {}))
    if args.no_tip:
        tip_cfg["enabled"] = False
    tip_model = maybe_load_tip_model(vcfg, device, cascade_ckpt=ckpt)
    if tip_cfg.get("enabled") and tip_model is None:
        print("Tip enabled but no tip checkpoint — falling back to full frame.", file=sys.stderr)
        tip_cfg["enabled"] = False

    filt = TemporalVisionFilter(TemporalFilterConfig.from_dict(vcfg))
    tip_tracker = TipTracker(TipTrackConfig.from_tip_cfg(tip_cfg))
    image_size = int(vcfg["model"]["image_size"])
    contact_thr = float(vcfg["thresholds"]["contact"])
    gt = load_labels(episode_dir)
    stems_all = list_zelux_stems(episode_dir)
    if args.full_episode:
        stems = stems_all
    else:
        stems = stems_before_removing(stems_all, gt)
        if len(stems) < len(stems_all):
            print(
                f"Demo cut at first REMOVING: {len(stems)}/{len(stems_all)} frames "
                "(same as train/val; use --full-episode to include lift-off)"
            )

    meta_hz = 15.0
    meta_path = episode_dir / "meta.json"
    if meta_path.is_file():
        meta_hz = float(json.loads(meta_path.read_text(encoding="utf-8")).get("record_hz", 15.0) or 15.0)
    fps = float(args.fps or meta_hz)
    out_path = Path(args.out) if args.out else episode_dir / f"{episode_dir.name}_vision_demo.mp4"

    writer = None
    prev_gray = None
    print(
        f"Demo {episode_dir.name}: {len(stems)} frames -> {out_path} "
        f"({device}, tip_crop={bool(tip_cfg.get('enabled'))}, tip_track={tip_tracker.cfg.enabled})"
    )

    for i, stem in enumerate(stems):
        path = episode_dir / f"{stem}_zelux.jpg"
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
        contact_p = float(result["contact_probability"])
        phase_probs = result["phase_probs"]
        flow_p = float(phase_probs[1]) if len(phase_probs) > 1 else 0.0
        state = filt.update(contact_p, phase_probs)
        gt_state = ""
        row = gt.get(stem)
        if row and row.get("state"):
            gt_state = str(row["state"])

        vis = bgr.copy()
        if result["tip_xy"] is not None:
            tx, ty = int(round(result["tip_xy"][0])), int(round(result["tip_xy"][1]))
            cv2.drawMarker(
                vis, (tx, ty), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2
            )
            if result["used_crop"]:
                half = int(result["crop_size"]) // 2
                x0, y0 = max(0, tx - half), max(0, ty - half)
                x1, y1 = tx + half, ty + half
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 255), 2)

        # Match epoxy_apply live overlay: probs + raw phase + filtered vision state
        lines = [
            f"{stem}  {i + 1}/{len(stems)}",
            f"vision {state.value}",
            f"contact {contact_p:.3f}  flow {flow_p:.3f}  phase {result.get('phase_name', '-')}",
            f"tip_conf {result.get('tip_conf', 0):.2f} crop={int(bool(result.get('used_crop')))}",
        ]
        if gt_state:
            lines.append(f"gt {gt_state}")
        for li, text in enumerate(lines):
            y = 28 + li * 26
            cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
        writer.write(vis)
        if args.show:
            cv2.imshow("epoxy_demo", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(stems)}")

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
