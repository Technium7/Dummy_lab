"""Train cascade: (1) contact on all frames, (2) phase on contact frames only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

_SOFTWARE_WIN = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.epoxy_app.config_loader import load_epoxy_config
from high_level_jobs.epoxy_app.vision.config_util import load_vision_config, resolve_repo_path
from high_level_jobs.epoxy_app.vision.dataset import (
    ContactDataset,
    PhaseDataset,
    discover_labeled_episodes,
    normalize_record_result_names,
    phase_samples,
    sample_weights_from_config,
    samples_for_episodes,
    split_episodes,
)
from high_level_jobs.epoxy_app.vision.model import (
    PHASE_NAMES,
    build_contact_model,
    build_phase_model,
)
from high_level_jobs.epoxy_app.vision.preprocess import EpoxyAugment


class FocalBCE(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        return ((1.0 - p_t) ** self.gamma) * bce


def _bin_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


@torch.no_grad()
def eval_contact(model, loader, device) -> dict[str, Any]:
    model.eval()
    yt, yp = [], []
    for batch in loader:
        logits = model(batch["image"].to(device))
        yt.append(batch["contact"].numpy())
        yp.append(torch.sigmoid(logits).cpu().numpy())
    if not yt:
        return {"f1": 0.0, "recall": 0.0, "mean_p": 0.0}
    y_true = np.concatenate(yt)
    y_prob = np.concatenate(yp)
    m = _bin_metrics(y_true, y_prob)
    m["mean_p"] = float(y_prob.mean())
    m["score"] = 0.6 * m["f1"] + 0.4 * m["recall"]
    return m


@torch.no_grad()
def eval_phase(model, loader, device) -> dict[str, Any]:
    model.eval()
    yt, yp = [], []
    for batch in loader:
        logits = model(batch["image"].to(device))
        pred = logits.argmax(dim=-1).cpu().numpy()
        yt.append(batch["phase"].numpy())
        yp.append(pred)
    if not yt:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "harm": 0.0,
            "f1_touching": 0.0,
            "f1_flowing": 0.0,
            "score": 0.0,
        }
    y_true = np.concatenate(yt)
    y_pred = np.concatenate(yp)
    acc = float(accuracy_score(y_true, y_pred))
    labels = list(range(len(PHASE_NAMES)))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels))
    per = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    f_touch = float(per[0])
    f_flow = float(per[1]) if len(per) > 1 else 0.0
    # Prefer balanced models: harmonic of both F1s (0 if either class is dead)
    if f_touch > 0 and f_flow > 0:
        harm = 2.0 * f_touch * f_flow / (f_touch + f_flow)
    else:
        harm = 0.0
    # Blend with macro so mild imbalance still improves score
    score = 0.5 * f1 + 0.5 * harm
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "harm": harm,
        "f1_touching": f_touch,
        "f1_flowing": f_flow,
        "score": score,
    }


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def train_contact_epoch(model, loader, device, optimizer, crit) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="contact", leave=False):
        images = batch["image"].to(device)
        contact = batch["contact"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        per = crit(logits, contact)
        loss = per.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item()) * images.size(0)
        n += images.size(0)
    return total / max(n, 1)


def train_phase_epoch(model, loader, device, optimizer, crit) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="phase", leave=False):
        images = batch["image"].to(device)
        phase = batch["phase"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = crit(logits, phase)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * images.size(0)
        n += images.size(0)
    return total / max(n, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train epoxy cascade (contact then phase)")
    parser.add_argument("--vision-config", default=None)
    parser.add_argument("--epoxy-config", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--stage", choices=["both", "contact", "phase"], default="both")
    args = parser.parse_args(argv)

    vcfg = load_vision_config(args.vision_config)
    epoxy = load_epoxy_config(args.epoxy_config)
    tcfg = vcfg["training"]
    ckpt_dir = resolve_repo_path(vcfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device_s = args.device or vcfg.get("inference", {}).get("device", "cuda")
    if device_s == "cuda" and not torch.cuda.is_available():
        device_s = "cpu"
    device = torch.device(device_s)

    normalize_record_result_names(epoxy.record_result_dir)
    labeled = discover_labeled_episodes(epoxy.dataset_dir, epoxy.record_result_dir)
    names = [p.name for p in labeled]
    if not names:
        print("No labeled episodes (checked episodes/ and record_result/).", file=sys.stderr)
        return 1

    train_eps = list(tcfg.get("train_episodes") or []) or None
    val_eps = list(tcfg.get("val_episodes") or []) or None
    test_eps = list(tcfg.get("test_episodes") or []) or None
    # Auto-include labeled apply results (result_*) into train unless listed in val/test.
    if train_eps is not None:
        holdout = set(val_eps or []) | set(test_eps or [])
        result_names = [n for n in names if n.startswith("result_") and n not in holdout]
        train_eps = list(dict.fromkeys([*train_eps, *result_names]))
        if result_names:
            print(f"Including {len(result_names)} labeled record_result episodes in train: {result_names}")

    data_roots = [epoxy.dataset_dir, epoxy.record_result_dir]
    split = split_episodes(
        names,
        val_ratio=float(tcfg.get("val_ratio", 0.2)),
        test_ratio=float(tcfg.get("test_ratio", 0.1)),
        seed=int(tcfg.get("seed", 42)),
        train_episodes=train_eps,
        val_episodes=val_eps,
        test_episodes=test_eps,
    )
    split.save(resolve_repo_path(vcfg["paths"]["split_json"]))

    tip_cfg = vcfg.get("tip", {})
    interp = bool(tip_cfg.get("interpolate_labels", True))
    train_s = samples_for_episodes(data_roots, split.train, interpolate_tips=interp)
    val_s = samples_for_episodes(data_roots, split.val, interpolate_tips=interp) if split.val else []
    n_tip = sum(1 for s in train_s if s.tip_xy is not None)
    print(f"tip crops enabled={bool(tip_cfg.get('enabled'))} train_frames_with_tip={n_tip}/{len(train_s)}")
    if not train_s:
        print("No train samples.", file=sys.stderr)
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
    bs = int(tcfg["batch_size"])
    nw = int(tcfg.get("num_workers", 0))
    epochs = int(args.epochs or tcfg["epochs"])
    patience = int(tcfg.get("early_stopping_patience", 10))

    contact_model = build_contact_model(vcfg).to(device)
    phase_model = build_phase_model(vcfg).to(device)
    best_contact, best_phase = -1.0, -1.0
    best_contact_mean_p = float("inf")
    history: dict[str, list] = {"contact": [], "phase": []}

    # ---------- Stage 1: contact ----------
    if args.stage in ("both", "contact"):
        print("=== Stage 1: CONTACT (mono gray, all frames) ===")
        c_train = ContactDataset(
            train_s, image_size, augment=augment, train_cfg=tcfg, tip_cfg=tip_cfg
        )
        c_val = ContactDataset(val_s or train_s, image_size, train_cfg=tcfg, tip_cfg=tip_cfg)
        if tcfg.get("use_weighted_sampler", True):
            weights = [sample_weights_from_config(s, tcfg) for s in train_s]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            c_loader = DataLoader(c_train, batch_size=bs, sampler=sampler, num_workers=nw)
        else:
            c_loader = DataLoader(c_train, batch_size=bs, shuffle=True, num_workers=nw)
        c_vloader = DataLoader(c_val, batch_size=bs, shuffle=False, num_workers=nw)

        n_pos = sum(s.contact for s in train_s) or 1
        n_neg = len(train_s) - n_pos
        # Mild class balance only (no sample-weight stacking; that caused collapse)
        pw = float(tcfg.get("contact_pos_weight", 1.0)) * (n_neg / max(n_pos, 1))
        gamma = float(tcfg.get("contact_focal_gamma", 0.0))
        crit = FocalBCE(torch.tensor([pw], device=device), gamma)
        # Head-only warmup, then full fine-tune at lower LR
        freeze_epochs = int(tcfg.get("contact_freeze_epochs", 3))
        finetune_backbone = bool(tcfg.get("contact_finetune_backbone", False))
        head_lr = float(tcfg.get("contact_head_lr", tcfg["lr"]))
        backbone_lr = float(tcfg.get("contact_backbone_lr", float(tcfg["lr"]) * 0.1))
        _set_requires_grad(contact_model.encoder, False)
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, contact_model.parameters()),
            lr=head_lr,
            weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        )
        print(f"split train={len(split.train)} eps val={split.val} test={split.test}")
        print(
            f"contact frames n={len(train_s)} pos={n_pos} neg={n_neg} "
            f"pos_weight={pw:.2f} gamma={gamma} "
            f"freeze_epochs={freeze_epochs} finetune_backbone={finetune_backbone}"
        )
        stale = 0
        for epoch in range(1, epochs + 1):
            if finetune_backbone and epoch == freeze_epochs + 1:
                _set_requires_grad(contact_model.encoder, True)
                opt = torch.optim.AdamW(
                    [
                        {"params": contact_model.encoder.parameters(), "lr": backbone_lr},
                        {"params": contact_model.head.parameters(), "lr": head_lr},
                    ],
                    weight_decay=float(tcfg.get("weight_decay", 1e-4)),
                )
                print(f"  unfreeze encoder lr_backbone={backbone_lr:g} lr_head={head_lr:g}")
            t0 = time.perf_counter()
            loss = train_contact_epoch(contact_model, c_loader, device, opt, crit)
            stats = eval_contact(contact_model, c_vloader, device)
            print(
                f"contact {epoch:03d} loss={loss:.4f} f1={stats['f1']:.3f} "
                f"rec={stats['recall']:.3f} mean_p={stats['mean_p']:.3f} "
                f"score={stats['score']:.3f} ({time.perf_counter()-t0:.1f}s)"
            )
            history["contact"].append({"epoch": epoch, "loss": loss, **stats})
            # Save on score gain; if score ties, prefer lower mean_p (better calibrated)
            improved = stats["score"] > best_contact + 1e-12
            tied_better_cal = (
                abs(stats["score"] - best_contact) < 1e-9
                and stats["mean_p"] < best_contact_mean_p - 1e-6
            )
            if improved or tied_better_cal:
                best_contact = stats["score"]
                best_contact_mean_p = float(stats["mean_p"])
                stale = 0
                torch.save(contact_model.state_dict(), ckpt_dir / "contact_best.pt")
                reason = "score" if improved else "mean_p"
                print(
                    f"  saved contact best={best_contact:.3f} "
                    f"mean_p={best_contact_mean_p:.3f} ({reason})"
                )
            else:
                stale += 1
                if stale >= patience:
                    print(f"Contact early stop @ {epoch}")
                    break
        contact_model.load_state_dict(torch.load(ckpt_dir / "contact_best.pt", map_location=device, weights_only=True))
        _set_requires_grad(contact_model.encoder, True)

    # ---------- Stage 2: phase (contact only) ----------
    if args.stage in ("both", "phase"):
        print("=== Stage 2: PHASE (mono+|diff|, contact frames only) ===")
        p_train_s = phase_samples(train_s)
        p_val_s = phase_samples(val_s) if val_s else p_train_s
        if not p_train_s:
            print("No contact frames for phase training.", file=sys.stderr)
            return 1
        counts = [sum(1 for s in p_train_s if s.phase == i) for i in range(len(PHASE_NAMES))]
        print(f"phase counts touching/flowing = {counts}  (classes {PHASE_NAMES})")
        print(
            f"phase val: {len(p_val_s)} frames "
            f"(REMOVING-truncated like train; SETTLED excluded from phase)"
        )

        p_train = PhaseDataset(train_s, image_size, augment=augment, tip_cfg=tip_cfg)
        p_val = PhaseDataset(val_s or train_s, image_size, tip_cfg=tip_cfg)
        val_counts = [sum(1 for s in p_val.samples if s.phase == i) for i in range(len(PHASE_NAMES))]
        print(f"phase val counts touching/flowing = {val_counts}")

        # Oversample TOUCHING so FLOWING does not dominate every batch
        if tcfg.get("phase_use_weighted_sampler", tcfg.get("use_weighted_sampler", False)):
            inv = [1.0 / max(c, 1) for c in counts]
            sample_w = [inv[int(s.phase)] for s in p_train.samples]
            sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
            p_loader = DataLoader(p_train, batch_size=bs, sampler=sampler, num_workers=nw)
            print("phase WeightedRandomSampler on (oversample minority phase)")
        else:
            p_loader = DataLoader(p_train, batch_size=bs, shuffle=True, num_workers=nw)
        p_vloader = DataLoader(p_val, batch_size=bs, shuffle=False, num_workers=nw)

        # class weights inverse frequency × config multipliers
        total = sum(counts) or 1
        cw = [
            float(tcfg.get("phase_class_weight_touching", tcfg.get("phase_class_weight_still", 1.2)))
            * total
            / max(counts[0], 1),
            float(tcfg.get("phase_class_weight_flowing", 1.0)) * total / max(counts[1], 1),
        ]
        # normalize mean to 1
        mean_w = sum(cw) / len(cw)
        cw = [w / mean_w for w in cw]
        crit_p = nn.CrossEntropyLoss(weight=torch.tensor(cw, device=device, dtype=torch.float32))
        print(f"phase CE class_weight={ [round(w,2) for w in cw] }")

        if (ckpt_dir / "contact_best.pt").exists() and args.stage == "phase":
            contact_model.load_state_dict(
                torch.load(ckpt_dir / "contact_best.pt", map_location=device, weights_only=True)
            )

        opt_p = torch.optim.AdamW(
            phase_model.parameters(),
            lr=float(tcfg["lr"]),
            weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        )
        stale = 0
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            loss = train_phase_epoch(phase_model, p_loader, device, opt_p, crit_p)
            stats = eval_phase(phase_model, p_vloader, device)
            print(
                f"phase {epoch:03d} loss={loss:.4f} acc={stats['accuracy']:.3f} "
                f"macro_f1={stats['macro_f1']:.3f} harm={stats['harm']:.3f} "
                f"score={stats['score']:.3f} "
                f"(touch={stats['f1_touching']:.2f} flow={stats['f1_flowing']:.2f}) "
                f"({time.perf_counter()-t0:.1f}s)"
            )
            history["phase"].append({"epoch": epoch, "loss": loss, **stats})
            if stats["score"] > best_phase:
                best_phase = stats["score"]
                stale = 0
                torch.save(phase_model.state_dict(), ckpt_dir / "phase_best.pt")
                print(f"  saved phase best={best_phase:.3f}")
            elif abs(stats["score"] - best_phase) < 1e-9:
                # Tie with best — keep going (val discrete on small set)
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    print(f"Phase early stop @ {epoch}")
                    break
        phase_model.load_state_dict(torch.load(ckpt_dir / "phase_best.pt", map_location=device, weights_only=True))

    # Combined checkpoint
    if (ckpt_dir / "contact_best.pt").exists():
        contact_model.load_state_dict(
            torch.load(ckpt_dir / "contact_best.pt", map_location=device, weights_only=True)
        )
    if (ckpt_dir / "phase_best.pt").exists():
        phase_model.load_state_dict(
            torch.load(ckpt_dir / "phase_best.pt", map_location=device, weights_only=True)
        )
    # Preserve prior contact score when phase-only
    if args.stage == "phase" and best_contact < 0:
        prev = ckpt_dir / "best.pt"
        if prev.exists():
            try:
                old = torch.load(prev, map_location="cpu", weights_only=False)
                if "best_contact" in old and float(old["best_contact"]) >= 0:
                    best_contact = float(old["best_contact"])
            except Exception:  # noqa: BLE001
                pass

    ckpt = {
        "contact_state": contact_model.state_dict(),
        "phase_state": phase_model.state_dict(),
        "config": vcfg,
        "split": {"train": split.train, "val": split.val, "test": split.test},
        "best_contact": best_contact,
        "best_phase": best_phase,
        "history": history,
    }
    tip_full = ckpt_dir / "tip_best_full.pt"
    tip_weights = ckpt_dir / "tip_best.pt"
    if tip_full.exists():
        tip_ckpt = torch.load(tip_full, map_location="cpu", weights_only=False)
        if "tip_state" in tip_ckpt:
            ckpt["tip_state"] = tip_ckpt["tip_state"]
    elif tip_weights.exists():
        ckpt["tip_state"] = torch.load(tip_weights, map_location="cpu", weights_only=True)
    torch.save(ckpt, ckpt_dir / "best.pt")
    torch.save(ckpt, resolve_repo_path(vcfg["paths"]["best_checkpoint"]))
    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
    c_txt = f"{best_contact:.3f}" if best_contact >= 0 else "skipped"
    p_txt = f"{best_phase:.3f}" if best_phase >= 0 else "skipped"
    print(f"Done. cascade -> {ckpt_dir / 'best.pt'}  contact={c_txt} phase={p_txt}")
    return 0


# evaluate_loader stub for evaluate.py imports during transition
def evaluate_loader(model, loader, device):
    return eval_contact(model.contact if hasattr(model, "contact") else model, loader, device)


if __name__ == "__main__":
    raise SystemExit(main())
