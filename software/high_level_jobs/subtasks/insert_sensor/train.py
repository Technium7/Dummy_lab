"""Train insert_sensor visuomotor policy from teleop episodes."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config_loader import InsertSensorConfig, default_config_path, load_insert_sensor_config
from .dataset import list_episodes, make_dataloaders
from .model import InsertionPolicy, compute_losses, save_policy


def train_policy(
    cfg: InsertSensorConfig,
    *,
    device: str | None = None,
    out_path: Path | None = None,
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
    )

    model = InsertionPolicy(policy_cfg).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    out_path = (out_path or cfg.model_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = out_path.parent
    history: list[dict[str, float]] = []

    best_val = float("inf")
    best_state: dict | None = None

    for epoch in range(cfg.epochs):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch_t = {k: v.to(torch_device) for k, v in batch.items()}
            optimizer.zero_grad()
            losses = compute_losses(model, batch_t, cfg=policy_cfg)
            losses["total"].backward()
            optimizer.step()
            train_losses.append(float(losses["total"].item()))

        avg_train = sum(train_losses) / max(len(train_losses), 1)
        val_total = float("nan")
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

        history.append({"epoch": epoch + 1, "train": avg_train, "val": val_total})
        print(
            f"  epoch {epoch + 1}/{cfg.epochs}  train={avg_train:.6f}  val={val_total:.6f}",
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
            "n_episodes": len(list_episodes(cfg.dataset_dir)),
            "best_val": None if best_val == float("inf") else best_val,
        },
    )
    (run_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[insert_sensor] Saved policy -> {out_path}", flush=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Train insert_sensor IL policy.")
    p.add_argument("--config", type=Path, default=default_config_path())
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    cfg = load_insert_sensor_config(args.config)
    episodes = list_episodes(cfg.dataset_dir)
    if not episodes:
        raise SystemExit(
            f"No episodes in {cfg.dataset_dir}. Run:\n"
            "  python -m high_level_jobs.subtasks.insert_sensor --record"
        )
    train_policy(cfg, device=args.device, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
