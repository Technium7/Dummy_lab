"""Train tip heatmap detector.

::

    python -m high_level_jobs.epoxy_app.vision.tip_train --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_SOFTWARE_WIN = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.epoxy_app.config_loader import load_epoxy_config
from high_level_jobs.epoxy_app.vision.config_util import load_vision_config, resolve_repo_path
from high_level_jobs.epoxy_app.vision.dataset import (
    TipDataset,
    discover_tip_labeled_episodes,
    normalize_record_result_names,
    split_episodes,
    tip_samples_for_episodes,
)
from high_level_jobs.epoxy_app.vision.preprocess import EpoxyAugment
from high_level_jobs.epoxy_app.vision.tip_model import build_tip_model


@torch.no_grad()
def eval_tip(model, loader, device) -> dict[str, float]:
    model.eval()
    errs: list[float] = []
    confs: list[float] = []
    for batch in loader:
        images = batch["image"].to(device)
        tip_xy = batch["tip_xy"].numpy()
        fw = batch["frame_w"]
        fh = batch["frame_h"]
        for i in range(images.size(0)):
            xi, yi, ci = model.predict_tip(
                images[i : i + 1],
                frame_w=int(fw[i]),
                frame_h=int(fh[i]),
            )
            gt = tip_xy[i]
            errs.append(float(np.hypot(float(xi[0]) - gt[0], float(yi[0]) - gt[1])))
            confs.append(float(ci[0]))
    if not errs:
        return {"mean_px_err": 1e9, "median_px_err": 1e9, "mean_conf": 0.0, "score": -1e9}
    mean_err = float(np.mean(errs))
    return {
        "mean_px_err": mean_err,
        "median_px_err": float(np.median(errs)),
        "mean_conf": float(np.mean(confs)),
        "score": -mean_err,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train epoxy tip heatmap detector")
    parser.add_argument("--vision-config", default=None)
    parser.add_argument("--epoxy-config", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=None,
        help="Restrict to these episode numbers or names (e.g. 1 2 3 or episode_000001)",
    )
    args = parser.parse_args(argv)

    vcfg = load_vision_config(args.vision_config)
    epoxy = load_epoxy_config(args.epoxy_config)
    tcfg = vcfg["training"]
    tip_cfg = vcfg.get("tip", {})
    ckpt_dir = resolve_repo_path(vcfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device_s = args.device or vcfg.get("inference", {}).get("device", "cuda")
    if device_s == "cuda" and not torch.cuda.is_available():
        device_s = "cpu"
    device = torch.device(device_s)

    normalize_record_result_names(epoxy.record_result_dir)
    labeled = discover_tip_labeled_episodes(epoxy.dataset_dir, epoxy.record_result_dir)
    names = [p.name for p in labeled]
    if args.episodes:
        want: set[str] = set()
        for e in args.episodes:
            s = str(e).strip()
            if s.isdigit():
                want.add(f"episode_{int(s):06d}")
                want.add(f"result_{int(s):06d}")
            elif s.startswith(("episode_", "result_")):
                want.add(s)
            else:
                want.add(f"episode_{s}")
                want.add(f"result_{s}")
        names = [n for n in names if n in want]
        if not names:
            print(f"No tip-labeled episodes match --episodes {args.episodes}", file=sys.stderr)
            return 1
    if not names:
        print("No tip labels found. Run tip_label_tool on episodes/results first.", file=sys.stderr)
        return 1

    interp = bool(tip_cfg.get("interpolate_labels", True))
    name_set = set(names)
    data_roots = [epoxy.dataset_dir, epoxy.record_result_dir]
    # When --episodes is set, train on exactly those (ignore contact train_episodes spill)
    if args.episodes:
        split = split_episodes(
            names,
            val_ratio=0.0,
            test_ratio=0.0,
            seed=int(tcfg.get("seed", 42)),
            train_episodes=list(names),
            val_episodes=[],
            test_episodes=[],
        )
    else:
        train_eps = [n for n in (tcfg.get("train_episodes") or []) if n in name_set] or None
        val_eps = [n for n in (tcfg.get("val_episodes") or []) if n in name_set] or None
        test_eps = [n for n in (tcfg.get("test_episodes") or []) if n in name_set] or None
        if train_eps is not None:
            holdout = set(val_eps or []) | set(test_eps or [])
            result_names = [n for n in names if n.startswith("result_") and n not in holdout]
            train_eps = list(dict.fromkeys([*train_eps, *result_names]))
        split = split_episodes(
            names,
            val_ratio=float(tcfg.get("val_ratio", 0.2)),
            test_ratio=float(tcfg.get("test_ratio", 0.0)),
            seed=int(tcfg.get("seed", 42)),
            train_episodes=train_eps,
            val_episodes=val_eps,
            test_episodes=test_eps,
        )
    train_s = tip_samples_for_episodes(data_roots, split.train, interpolate_tips=interp)
    val_s = (
        tip_samples_for_episodes(data_roots, split.val, interpolate_tips=interp)
        if split.val
        else []
    )
    if not train_s:
        print("No tip train samples.", file=sys.stderr)
        return 1

    aug_cfg = vcfg.get("augmentation", {})
    augment = EpoxyAugment(
        brightness=float(aug_cfg.get("brightness", 0.15)),
        contrast=float(aug_cfg.get("contrast", 0.15)),
        exposure=float(aug_cfg.get("exposure", 0.10)),
        noise_std=float(aug_cfg.get("noise_std", 0.02)),
        translate_frac=float(aug_cfg.get("translate_frac", 0.02)),
    )
    image_size = int(vcfg["model"]["image_size"])
    heat_size = int(tip_cfg.get("heatmap_size", 64))
    bs = int(tcfg["batch_size"])
    nw = int(tcfg.get("num_workers", 0))
    epochs = int(args.epochs or tcfg["epochs"])
    patience = int(tcfg.get("early_stopping_patience", 6))

    train_ds = TipDataset(train_s, image_size, heat_size, augment=augment)
    val_ds = TipDataset(val_s or train_s, image_size, heat_size)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)

    model = build_tip_model(vcfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg.get("tip_lr", max(float(tcfg["lr"]), 1e-4))),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    print(
        f"tip train n={len(train_ds)} val n={len(val_ds)} "
        f"episodes train={split.train} val={split.val}"
    )

    best = -1e18
    stale = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        for batch in tqdm(train_loader, desc="tip", leave=False):
            images = batch["image"].to(device)
            target = batch["heatmap"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.mse_loss(torch.sigmoid(logits), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * images.size(0)
            n += images.size(0)
        loss_avg = total / max(n, 1)
        stats = eval_tip(model, val_loader, device)
        print(
            f"tip {epoch:03d} loss={loss_avg:.4f} mean_px={stats['mean_px_err']:.1f} "
            f"med_px={stats['median_px_err']:.1f} conf={stats['mean_conf']:.3f} "
            f"({time.perf_counter()-t0:.1f}s)"
        )
        history.append({"epoch": epoch, "loss": loss_avg, **stats})
        if stats["score"] > best:
            best = stats["score"]
            stale = 0
            torch.save(model.state_dict(), ckpt_dir / "tip_best.pt")
            torch.save(
                {
                    "tip_state": model.state_dict(),
                    "config": vcfg,
                    "split": {"train": split.train, "val": split.val, "test": split.test},
                    "best_tip_score": best,
                    "history": history,
                },
                ckpt_dir / "tip_best_full.pt",
            )
            print(f"  saved tip best mean_px={stats['mean_px_err']:.1f}")
        else:
            stale += 1
            if stale >= patience:
                print(f"Tip early stop @ {epoch}")
                break

    print(f"Done. tip -> {ckpt_dir / 'tip_best.pt'} score={best:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
