"""Evaluate cascade checkpoint (contact + phase)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_SOFTWARE_WIN = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.epoxy_app.config_loader import load_epoxy_config
from high_level_jobs.epoxy_app.vision.config_util import load_vision_config, resolve_repo_path
from high_level_jobs.epoxy_app.vision.dataset import (
    ContactDataset,
    EpisodeSplit,
    PhaseDataset,
    samples_for_episodes,
)
from high_level_jobs.epoxy_app.vision.model import load_checkpoint
from high_level_jobs.epoxy_app.vision.train import eval_contact, eval_phase


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-config", default=None)
    parser.add_argument("--epoxy-config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    vcfg = load_vision_config(args.vision_config)
    epoxy = load_epoxy_config(args.epoxy_config)
    ckpt_path = resolve_repo_path(args.checkpoint or vcfg["paths"]["best_checkpoint"])
    if not ckpt_path.exists():
        print(f"Missing {ckpt_path}", file=sys.stderr)
        return 1

    device_s = args.device or vcfg.get("inference", {}).get("device", "cuda")
    if device_s == "cuda" and not torch.cuda.is_available():
        device_s = "cpu"
    device = torch.device(device_s)

    model, ckpt = load_checkpoint(ckpt_path, device=device, cfg=vcfg)
    split = EpisodeSplit(**ckpt["split"]) if "split" in ckpt else EpisodeSplit.load(
        resolve_repo_path(vcfg["paths"]["split_json"])
    )
    names = getattr(split, args.split) or split.train
    samples = samples_for_episodes([epoxy.dataset_dir, epoxy.record_result_dir], names)
    if not samples:
        print("No samples", file=sys.stderr)
        return 1

    bs = int(vcfg["training"]["batch_size"])
    image_size = int(vcfg["model"]["image_size"])
    tip_cfg = vcfg.get("tip", {})
    tcfg = vcfg.get("training", {})
    c_ds = ContactDataset(samples, image_size, train_cfg=tcfg, tip_cfg=tip_cfg)
    p_ds = PhaseDataset(samples, image_size, tip_cfg=tip_cfg)
    c_loader = DataLoader(c_ds, batch_size=bs, shuffle=False)
    p_loader = DataLoader(p_ds, batch_size=bs, shuffle=False)

    contact_stats = eval_contact(model.contact, c_loader, device)
    phase_stats = eval_phase(model.phase, p_loader, device) if len(p_ds) else {}

    report = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "episodes": names,
        "contact": contact_stats,
        "phase": phase_stats,
    }
    out_dir = resolve_repo_path(vcfg["paths"]["checkpoints"]) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
