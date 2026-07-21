# Insert sensor — place sensor in holder (ACT or Diffusion)

Two policies share the same demo episodes. Train and deploy them separately.

| Arch | Vision | Action head | Config | Checkpoint |
|------|--------|-------------|--------|------------|
| **act** (default) | Same RGB-D CNN as diffusion (`160×120`) | ACT CVAE chunk | `config/default_insert_sensor.yaml` | `runs/act_cnn/latest/policy.pt` |
| **act** (DINOv2) | Frozen DINOv2 + depth CNN | ACT CVAE chunk | `config/default_act_dinov2.yaml` | `runs/latest/policy.pt` |
| **diffusion** (CNN) | RGB-D CNN + dense table XYZ; 3 cams | DDPM action-chunk denoiser (no aux) | `config/default_diffusion.yaml` | `runs/diffusion/ep20_tablexyz/policy.pt` |
| **diffusion** (DINOv2) | Frozen DINOv2 RGB + depth/XYZ CNN; same 3 cams + table XYZ | DDPM (same head) | `config/default_diffusion_dinov2.yaml` | `runs/diffusion_dinov2/latest/policy.pt` |

For apple-to-apple ACT vs diffusion (2-cam RGB-D only), use ACT `vision_backbone: cnn`, diffusion with `num_cameras: 2` / `use_table_xyz: false`, and the same `--max-episodes` / seed.

For CNN vs DINOv2 diffusion (same demos / 3 cams / table XYZ), train both configs with the same `--max-episodes` / `seed`.

## Record more demos

Episodes live under `dataset/episodes/episode_XXXXXX/` (gitignored). After moving **side2**, delete (or exclude) takes that used the old view; new recordings pick up the new aim. Serials usually stay the same — only re-pin YAML if you swapped USB cameras.

```bash
cd software

# 1) Confirm views (especially after moving side2)
python -m high_level_jobs.subtasks.insert_sensor --preview-cameras
# Open camera_previews/preview_<serial>.png — overhead / side / side2 should match config

# 2) Record (same teleop as manual_control; cameras stay off until R)
python manual_control/manual_control_app.py --record-demos
# equivalent: python -m high_level_jobs.subtasks.insert_sensor --record
```

| Key | Action |
|-----|--------|
| **T** | Home |
| **O** / **P** | Gripper open / close |
| **R** | Start or stop RGB-D capture (all pinned cams, incl. side2) |
| **C** | Save episode after stopping with **R** (writes next `episode_*`) |
| **R** again (while unsaved) | Discard staging and record again |
| Esc / close | Exit (unsaved staging is discarded) |

Typical take: **T** home → grasp sensor → **P** close → **R** start → insert into holder → **R** stop → **C** save. Repeat.

```bash
# 3) Sanity-check dataset
python -m high_level_jobs.subtasks.insert_sensor --check
```

New episodes append alongside whatever remains in `dataset/episodes/`. Retrain when you have enough takes (`--arch act` / `diffusion`).

## Train / deploy

```bash
cd software

# ACT (existing)
python -m high_level_jobs.subtasks.insert_sensor --arch act --train
python -m high_level_jobs.subtasks.insert_sensor --arch act --deploy

# Diffusion CNN (3 cams + dense table XYZ; side2 zero-filled when missing)
python -m high_level_jobs.subtasks.insert_sensor --arch diffusion --train
python -m high_level_jobs.subtasks.insert_sensor --arch diffusion --deploy

# Diffusion DINOv2 (same cams / table XYZ; separate checkpoint)
python -m high_level_jobs.subtasks.insert_sensor --arch diffusion --train \
  --config high_level_jobs/subtasks/insert_sensor/config/default_diffusion_dinov2.yaml
python -m high_level_jobs.subtasks.insert_sensor --arch diffusion --deploy \
  --config high_level_jobs/subtasks/insert_sensor/config/default_diffusion_dinov2.yaml
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
| Vision (CNN) | Per-cam RGB-D CNN (7ch when `use_table_xyz`: RGB-D + table XYZ on overhead; side/side2 get zero XYZ) + state MLP |
| Vision (DINOv2) | Frozen DINOv2 RGB + aux CNN on depth/(table XYZ); same preprocess / cams (`default_diffusion_dinov2.yaml`) |
| Cameras | `num_cameras: 3` — overhead, side, optional side2 (missing → zero-fill) |
| Table XYZ | Overhead depth → cam XYZ → table frame via `calibration_table` (no YOLO) |
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
| `cameras.overhead` | Top-down robot + table (also dense table XYZ for diffusion) |
| `cameras.side` | Horizontal breadboard |
| `cameras.side2` | Extra side / oblique — used by diffusion when present; optional at deploy |

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
| `cameras.overhead` / `side` / `side2` | RealSense serials |

## Config highlights (Diffusion)

| Key | Purpose |
|-----|---------|
| `vision_backbone` | `cnn` (default) or `dinov2_vits14` |
| `freeze_backbone` | Freeze DINOv2 weights (`true` for DINOv2) |
| `img_width` / `img_height` | `160` / `120` (CNN) or `224` / `224` (DINOv2) |
| `num_cameras` | `3` (overhead + side + optional side2) |
| `use_table_xyz` | Append dense table XYZ on overhead |
| `table_calib_path` | Empty → newest valid `calibration_table/runs/*/table_calib.json` |
| `diffusion_steps` | Train noise schedule length |
| `inference_steps` | Denoising steps at deploy |
| `beta_start` / `beta_end` | Linear β schedule |
