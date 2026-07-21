---
name: insert-sensor
description: >-
  Record, train, and deploy insert_sensor IL policies (ACT and diffusion).
  Use when the user mentions insert_sensor, ACT, diffusion policy, demo
  recording, --arch, --max-episodes, side2 camera, table XYZ, or deploying
  after manual_control grasp.
---

# insert_sensor

Work from `software/`. Policies share the same episodes; train/deploy separately.

## Cameras

| Role | Used by |
|------|---------|
| `overhead`, `side` | ACT (`num_cameras: 2`); diffusion always |
| `side2` | Diffusion when `num_cameras: 3` — **optional** (zero-filled if missing). Still recorded when pinned. |

Identify serials: `python software/camera/live_view.py` or `--preview-cameras`. Pin in YAML; USB order is unstable.

Configs:

- ACT (fair vs diffusion, CNN vision): `config/default_insert_sensor.yaml` → `runs/act_cnn/latest/policy.pt`
- ACT (DINOv2): `config/default_act_dinov2.yaml` → `runs/latest/policy.pt`
- Diffusion CNN (3-cam + dense table XYZ): `config/default_diffusion.yaml` → `runs/diffusion/ep20_tablexyz/policy.pt`
- Diffusion DINOv2 (same cams / table XYZ): `config/default_diffusion_dinov2.yaml` → `runs/diffusion_dinov2/latest/policy.pt`

Diffusion overhead input is **7 channels** (RGB-D + table XYZ from depth + `calibration_table`, no YOLO). Side/side2 keep RGB-D with zero XYZ pads. DINOv2 uses the same preprocess at `224×224` (frozen RGB backbone + aux depth/XYZ CNN).

## Record demos

```bash
cd software
python manual_control/manual_control_app.py --record-demos
# or: python -m high_level_jobs.subtasks.insert_sensor --record
```

**R** start/stop capture · **C** save · **O/P** gripper. All configured cameras (incl. side2) record.

## Train

```bash
# ACT
python -m high_level_jobs.subtasks.insert_sensor --arch act --train

# Diffusion CNN (all episodes; optional side2 + table XYZ)
python -m high_level_jobs.subtasks.insert_sensor --arch diffusion --train

# Diffusion DINOv2 (same demos / cams / table XYZ; compare vs CNN)
python -m high_level_jobs.subtasks.insert_sensor --arch diffusion --train \
  --config high_level_jobs/subtasks/insert_sensor/config/default_diffusion_dinov2.yaml

# Fair subset (first N episode_* by name) — use same N/seed for CNN vs DINOv2
python -m high_level_jobs.subtasks.insert_sensor --arch diffusion --train --max-episodes 10 \
  --out high_level_jobs/subtasks/insert_sensor/runs/diffusion/ep10/policy.pt
```

## Deploy (after manual prep)

1. Teleop: `python manual_control/manual_control_app.py` → **T** home → grasp → **P** close → **ESC** (free serial).
2. Deploy (only one process may hold the port):

```bash
python -m high_level_jobs.subtasks.insert_sensor --arch diffusion --deploy
# optional: --sim-only | --speed 3 | --model path/to/policy.pt
```

## Agent rules

1. Do not change ACT checkpoints when adding diffusion (separate dirs).
2. Keep CNN vs DINOv2 diffusion checkpoints separate (`runs/diffusion/` vs `runs/diffusion_dinov2/`); do not overwrite the CNN baseline when adding DINOv2.
3. Diffusion may use `num_cameras: 3` with optional side2; keep ACT at `num_cameras: 2`.
4. If deploy fails with missing `cameras` on config, ensure `InsertSensorConfig` includes `cameras=` from YAML.
5. Large demos: sync with [lab-data-sync](../lab-data-sync/SKILL.md); never git-push episodes.
