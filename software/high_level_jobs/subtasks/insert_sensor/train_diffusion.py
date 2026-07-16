"""Train insert_sensor diffusion policy from teleop episodes (no aux heads)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config_loader import InsertSensorConfig, default_config_path, load_insert_sensor_config
from .dataset import list_episodes, make_dataloaders
from .policies.diffusion import DiffusionPolicy, compute_losses, save_policy


def train_diffusion_policy(
    cfg: InsertSensorConfig,
    *,
    device: str | None = None,
    out_path: Path | None = None,
    max_episodes: int | None = None,
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

    model = DiffusionPolicy(policy_cfg).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    out_path = (out_path or cfg.model_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = out_path.parent
    history: list[dict[str, float]] = []

    best_val = float("inf")
    best_state: dict | None = None

    print(
        f"[insert_sensor] Training Diffusion Policy (no aux)  "
        f"obs_H={policy_cfg.history_len} act_H={policy_cfg.action_horizon}  "
        f"temporal={policy_cfg.temporal}  T_diff={policy_cfg.diffusion_steps}  "
        f"infer_steps={policy_cfg.inference_steps}",
        flush=True,
    )
    for epoch in range(cfg.epochs):
        model.train()
        train_losses: list[float] = []
        train_diff: list[float] = []
        for batch in train_loader:
            batch_t = {k: v.to(torch_device) for k, v in batch.items()}
            optimizer.zero_grad()
            losses = compute_losses(model, batch_t, cfg=policy_cfg)
            losses["total"].backward()
            optimizer.step()
            train_losses.append(float(losses["total"].item()))
            train_diff.append(float(losses["diffusion"].item()))

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
            f"  epoch {epoch + 1}/{cfg.epochs}  train={avg_train:.6f}  val={val_total:.6f}  "
            f"diffusion={sum(train_diff)/max(len(train_diff),1):.4f}",
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
            "arch": "diffusion_policy_v1",
        },
    )
    (run_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[insert_sensor] Saved diffusion policy -> {out_path}", flush=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Train insert_sensor diffusion policy.")
    p.add_argument("--config", type=Path, default=default_config_path(arch="diffusion"))
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Use only the first N episodes (sorted by name) for a fair subset comparison",
    )
    args = p.parse_args(argv)

    cfg = load_insert_sensor_config(args.config, arch="diffusion")
    episodes = list_episodes(cfg.dataset_dir)
    if not episodes:
        raise SystemExit(
            f"No episodes in {cfg.dataset_dir}. Record demos first, then retrain."
        )
    train_diffusion_policy(
        cfg, device=args.device, out_path=args.out, max_episodes=args.max_episodes
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
