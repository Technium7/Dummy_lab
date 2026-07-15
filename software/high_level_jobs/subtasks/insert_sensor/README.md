# Insert sensor — ACT for placing sensor in holder

**RGB (2 cameras for ACT)** + **Depth** (optional) + **joints** + **gripper**
→ Frozen **DINOv2** (+ depth CNN) → Temporal Transformer → **ACT** → Action chunk.

Recording can also capture a third **`side2`** RealSense; train/deploy still use overhead + side until the model is updated.

## Architecture

| Component | Role |
|-----------|------|
| Vision | Shared frozen DINOv2 (`dinov2_vits14`) on RGB; small CNN on depth; MLP(state) |
| Temporal | Transformer over observation history `H` |
| ACT | CVAE decoder predicts action horizon `Ta` |

## Workflow

```bash
cd software
python -m high_level_jobs.subtasks.insert_sensor --train
python -m high_level_jobs.subtasks.insert_sensor --deploy \
  --model high_level_jobs/subtasks/insert_sensor/runs/latest/policy.pt
```

Retrain after model changes — older CNN / diffusion checkpoints are incompatible.

## Multi-camera setup

USB enumeration order is **not stable**. Identify serials with live mosaic or snapshots, then pin in config:

```bash
python software/camera/live_view.py
python -m high_level_jobs.subtasks.insert_sensor --preview-cameras
```

| Config key | Role |
|------------|------|
| `cameras.overhead` | Top-down robot + table |
| `cameras.side` | Horizontal breadboard (used by ACT) |
| `cameras.side2` | Extra side / oblique — **recorded**, not used by ACT yet |

`num_cameras: 2` remains the policy size. New episodes save `*.side2.color.png` / `*.side2.depth.npy` alongside the existing files.

## Test on robot

1. `python manual_control/manual_control_app.py` — **T** home, grasp, **P** close, ESC  
2. `python -m high_level_jobs.subtasks.insert_sensor --deploy --model .../policy.pt`

| Flag | Meaning |
|------|---------|
| `--sim-only` | No motors |
| `--speed 2.5` | Faster motion |

## Config highlights

| Key | Purpose |
|-----|---------|
| `vision_backbone` | `dinov2_vits14` (default) or `cnn` |
| `freeze_backbone` | Freeze DINOv2 weights (`true`) |
| `img_width` / `img_height` | `224` (DINOv2 needs multiples of 14) |
| `use_depth` | Include depth in vision (`true` / `false`) |
| `action_horizon` | ACT chunk length |
| `style_dim` / `kl_weight` | ACT CVAE |
| `cameras.overhead` / `side` / `side2` | RealSense serials (`side2` = record only) |
