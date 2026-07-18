"""Train insert_sensor ACT policy from teleop episodes."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config_loader import InsertSensorConfig, default_config_path, load_insert_sensor_config
from .dataset import list_episodes, make_dataloaders
from .model import InsertionPolicy, compute_losses, save_policy

CHECKPOINT_NAME = "train_checkpoint.pt"
DONE_MARKER_NAME = "TRAINING_DONE"


def _atomic_torch_save(obj: object, path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _save_train_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: InsertionPolicy,
    optimizer: torch.optim.Optimizer,
    best_val: float,
    best_state: dict | None,
    history: list[dict[str, float]],
    norm_stats: dict,
    max_episodes: int | None,
    total_epochs: int,
) -> None:
    _atomic_torch_save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val": best_val,
            "best_state": best_state,
            "history": history,
            "norm_stats": norm_stats,
            "max_episodes": max_episodes,
            "total_epochs": total_epochs,
        },
        path,
    )


def train_policy(
    cfg: InsertSensorConfig,
    *,
    device: str | None = None,
    out_path: Path | None = None,
    max_episodes: int | None = None,
    resume: bool = True,
    checkpoint_every: int = 1,
) -> Path:
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy_cfg = cfg.policy
    train_loader, val_loader, norm_stats = make_dataloaders(
        cfg.dataset_dir,
        policy_cfg=policy_cfg,
        batch_size=cfg.batch_size,
        val_frac=cfg.val_frac,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        max_episodes=max_episodes,
    )

    model = InsertionPolicy(policy_cfg).to(torch_device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr)
    out_path = (out_path or cfg.model_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = out_path.parent
    ckpt_path = run_dir / CHECKPOINT_NAME
    done_marker = run_dir / DONE_MARKER_NAME
    history: list[dict[str, float]] = []

    best_val = float("inf")
    best_state: dict | None = None
    start_epoch = 0

    if resume and ckpt_path.is_file():
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        best_val = float(payload.get("best_val", float("inf")))
        best_state = payload.get("best_state")
        history = list(payload.get("history") or [])
        # Prefer checkpoint norm stats so resumed runs stay consistent.
        if payload.get("norm_stats"):
            norm_stats = payload["norm_stats"]
        start_epoch = int(payload.get("epoch", 0))
        print(
            f"[insert_sensor] Resuming from {ckpt_path}  "
            f"completed_epochs={start_epoch}/{cfg.epochs}  best_val={best_val:.6f}",
            flush=True,
        )
    elif done_marker.is_file() and out_path.is_file():
        print(f"[insert_sensor] Training already complete ({done_marker}); skipping.", flush=True)
        return out_path

    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[insert_sensor] Training ACT  "
        f"backbone={policy_cfg.vision_backbone} freeze={policy_cfg.freeze_backbone}  "
        f"obs_H={policy_cfg.history_len} act_H={policy_cfg.action_horizon}  "
        f"temporal={policy_cfg.temporal}  kl_weight={policy_cfg.kl_weight}  "
        f"use_depth={policy_cfg.use_depth}  params={n_train}/{n_total}  "
        f"device={torch_device}  checkpoint_every={checkpoint_every}",
        flush=True,
    )

    if start_epoch >= cfg.epochs:
        print(f"[insert_sensor] Checkpoint already at epoch {start_epoch}; finalizing.", flush=True)
    else:
        for epoch in range(start_epoch, cfg.epochs):
            model.train()
            train_losses: list[float] = []
            train_action: list[float] = []
            train_kl: list[float] = []
            for batch in train_loader:
                batch_t = {k: v.to(torch_device) for k, v in batch.items()}
                optimizer.zero_grad()
                losses = compute_losses(model, batch_t, cfg=policy_cfg)
                losses["total"].backward()
                optimizer.step()
                train_losses.append(float(losses["total"].item()))
                train_action.append(float(losses["action"].item()))
                train_kl.append(float(losses["kl"].item()))

            avg_train = sum(train_losses) / max(len(train_losses), 1)
            val_total = float("nan")
            improved = False
            if val_loader is not None:
                model.eval()
                val_losses: list[float] = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch_t = {k: v.to(torch_device) for k, v in batch.items()}
                        losses = compute_losses(model, batch_t, cfg=policy_cfg)
                        val_losses.append(float(losses["total"].item()))
                val_total = sum(val_losses) / max(len(val_losses), 1)
                if val_total < best_val:
                    best_val = val_total
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    improved = True

            completed = epoch + 1
            history.append({"epoch": completed, "train": avg_train, "val": val_total})
            print(
                f"  epoch {completed}/{cfg.epochs}  train={avg_train:.6f}  val={val_total:.6f}  "
                f"action={sum(train_action)/max(len(train_action),1):.4f}  "
                f"kl={sum(train_kl)/max(len(train_kl),1):.4f}",
                flush=True,
            )

            should_ckpt = (
                checkpoint_every > 0
                and (completed % checkpoint_every == 0 or completed == cfg.epochs or improved)
            )
            if should_ckpt:
                _save_train_checkpoint(
                    ckpt_path,
                    epoch=completed,
                    model=model,
                    optimizer=optimizer,
                    best_val=best_val,
                    best_state=best_state,
                    history=history,
                    norm_stats=norm_stats,
                    max_episodes=max_episodes,
                    total_epochs=cfg.epochs,
                )
                # Keep a deployable best snapshot as soon as we have one.
                if best_state is not None:
                    snap = InsertionPolicy(policy_cfg)
                    snap.load_state_dict(best_state)
                    save_policy(
                        out_path,
                        snap,
                        policy_cfg,
                        norm_stats=norm_stats,
                        extra={
                            "epochs": completed,
                            "n_episodes": len(list_episodes(cfg.dataset_dir)[: max_episodes or None]),
                            "max_episodes": max_episodes,
                            "best_val": best_val,
                            "arch": "act_v1",
                            "partial": completed < cfg.epochs,
                        },
                    )
                (run_dir / "training_history.json").write_text(
                    json.dumps(history, indent=2), encoding="utf-8"
                )
                print(
                    f"[insert_sensor] Checkpoint -> {ckpt_path}  "
                    f"(best_policy={'yes' if best_state is not None else 'no'})",
                    flush=True,
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    save_policy(
        out_path,
        model,
        policy_cfg,
        norm_stats=norm_stats,
        extra={
            "epochs": cfg.epochs,
            "n_episodes": len(list_episodes(cfg.dataset_dir)[: max_episodes or None]),
            "max_episodes": max_episodes,
            "best_val": None if best_val == float("inf") else best_val,
            "arch": "act_v1",
            "partial": False,
        },
    )
    (run_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    # Final resume checkpoint at full epochs so a late preempt after last epoch is safe.
    _save_train_checkpoint(
        ckpt_path,
        epoch=cfg.epochs,
        model=model,
        optimizer=optimizer,
        best_val=best_val,
        best_state=best_state,
        history=history,
        norm_stats=norm_stats,
        max_episodes=max_episodes,
        total_epochs=cfg.epochs,
    )
    done_marker.write_text(
        json.dumps({"out": str(out_path), "best_val": best_val, "epochs": cfg.epochs}, indent=2),
        encoding="utf-8",
    )
    print(f"[insert_sensor] Saved policy -> {out_path}", flush=True)
    print(f"[insert_sensor] Wrote done marker -> {done_marker}", flush=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Train insert_sensor ACT policy.")
    p.add_argument("--config", type=Path, default=default_config_path())
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Use only the first N episodes (sorted by name) for a fair subset comparison",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from run_dir/train_checkpoint.pt if present (default: true)",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save train_checkpoint.pt every N epochs (also on best-val improvements)",
    )
    args = p.parse_args(argv)

    cfg = load_insert_sensor_config(args.config)
    episodes = list_episodes(cfg.dataset_dir)
    if not episodes:
        raise SystemExit(
            f"No episodes in {cfg.dataset_dir}. Run:\n"
            "  python -m high_level_jobs.subtasks.insert_sensor --record"
        )
    train_policy(
        cfg,
        device=args.device,
        out_path=args.out,
        max_episodes=args.max_episodes,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
