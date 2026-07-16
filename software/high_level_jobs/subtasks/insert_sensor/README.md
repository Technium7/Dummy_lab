# Insert sensor — place sensor in holder (ACT or Diffusion)

Two policies share the same demo episodes. Train and deploy them separately.

| Arch | Vision | Action head | Config | Checkpoint |
|------|--------|-------------|--------|------------|
| **act** (default) | Same RGB-D CNN as diffusion (`160×120`) | ACT CVAE chunk | `config/default_insert_sensor.yaml` | `runs/act_cnn/latest/policy.pt` |
| **act** (DINOv2) | Frozen DINOv2 + depth CNN | ACT CVAE chunk | `config/default_act_dinov2.yaml` | `runs/latest/policy.pt` |
| **diffusion** | RGB-D CNN | DDPM action-chunk denoiser (no aux) | `config/default_diffusion.yaml` | `runs/diffusion/latest/policy.pt` |

For apple-to-apple ACT vs diffusion, use the default ACT config (`vision_backbone: cnn`) and the same `--max-episodes` / seed / episode set.

Recording can also capture a third **`side2`** RealSense; train/deploy still use overhead + side until models support 3 cams.

## Train / deploy

```bash
cd software

# ACT (existing)
python -m high_level_jobs.subtasks.insert_sensor --arch act --train
python -m high_level_jobs.subtasks.insert_sensor --arch act --deploy

# Diffusion (reproduced; no auxiliary heads)
python -m high_level_jobs.subtasks.insert_sensor --arch diffusion --train
python -m high_level_jobs.subtasks.insert_sensor --arch diffusion --deploy
```

`--arch` selects the default config and train/deploy path. You can still override with `--config` / `--model` / `--out`.

## ACT architecture

| Component | Role |
|-----------|------|
| Vision (default) | Per-cam RGB-D CNN (`160×120`) — same as diffusion |
| Vision (optional) | Frozen DINOv2 + depth CNN (`default_act_dinov2.yaml`) |
| Temporal | Transformer over observation history `H` |
| ACT | CVAE decoder predicts action horizon `Ta` |

## Diffusion architecture

| Component | Role |
|-----------|------|
| Vision | Per-cam RGB-D CNN + state MLP |
| Temporal | Transformer / GRU → condition vector |
| Diffusion | Noise prediction on action chunk; DDPM sample at deploy |

## Multi-camera setup

USB enumeration order is **not stable**. Identify serials, then pin in config:

```bash
python software/camera/live_view.py
python -m high_level_jobs.subtasks.insert_sensor --preview-cameras
```

| Config key | Role |
|------------|------|
| `cameras.overhead` | Top-down robot + table |
| `cameras.side` | Horizontal breadboard (used by policies) |
| `cameras.side2` | Extra side / oblique — **recorded**, not used by policies yet |

## Test on robot

1. `python manual_control/manual_control_app.py` — **T** home, grasp, **P** close, ESC  
2. Deploy with the matching `--arch` / `--model`

| Flag | Meaning |
|------|---------|
| `--sim-only` | No motors |
| `--speed 2.5` | Faster motion |
| `--arch act\|diffusion` | Which policy family |

## Config highlights (ACT)

| Key | Purpose |
|-----|---------|
| `vision_backbone` | `dinov2_vits14` (default) or `cnn` |
| `freeze_backbone` | Freeze DINOv2 weights (`true`) |
| `img_width` / `img_height` | `224` (DINOv2 needs multiples of 14) |
| `use_depth` | Include depth in vision (`true` / `false`) |
| `action_horizon` | ACT chunk length |
| `style_dim` / `kl_weight` | ACT CVAE |
| `cameras.overhead` / `side` / `side2` | RealSense serials (`side2` = record only) |

## Config highlights (Diffusion)

| Key | Purpose |
|-----|---------|
| `img_width` / `img_height` | `160` / `120` (CNN) |
| `diffusion_steps` | Train noise schedule length |
| `inference_steps` | Denoising steps at deploy |
| `beta_start` / `beta_end` | Linear β schedule |
