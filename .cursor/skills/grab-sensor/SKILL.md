---
name: grab-sensor
description: >-
  grab_sensor grasp pipeline, live labeling, and local dataset paths. Use when
  working on grab_sensor, live_label, live_review, YOLO grasp, flat Windows
  cascade detector, or table calibration tied to grasping.
---

# grab_sensor

## Linux (`software/`)

Work from `software/`. Detection data and weights are **gitignored** — keep on disk / sync separately.

| Path | Purpose |
|------|---------|
| `high_level_jobs/subtasks/grab_sensor/` | Pipeline, teleop hooks, config |
| `detection/top_detection/` | Overhead base-point labeling (YOLO + base CNN) |
| `detection/side_detection/` | Side grasp + base ellipse |
| `detection/*/dataset/` | Images, depth, labels, `.pt` weights (local only) |
| `debug/` | Live capture overlays (local only) |

```bash
cd software
python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.live_label
python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.live_review
python -m high_level_jobs.subtasks.grab_sensor --detect-only --visualize
```

## Windows flat grab (`software_win/`)

Perpendicular end-link grasp; yaw from sensor orientation axis; custom cascade (no YOLO v1); posts → 1″ SCP cylinders.

| Path | Purpose |
|------|---------|
| `high_level_jobs/grab_sensor/` | Pipeline, config, IK/plan/debug |
| `vision/` | live_label / live_review / train / infer |
| `vision/dataset/` | Labels + weights (gitignored) |
| `camera/realsense.py` | D415 RGB+depth backend |

```powershell
$env:PYTHONPATH = "$PWD\software_win"
python -m high_level_jobs.grab_sensor.vision.live_label --serial <D415>
python -m high_level_jobs.grab_sensor.vision.live_detect --serial <D415>
python -m high_level_jobs.grab_sensor.vision.train_yolo
python -m high_level_jobs.grab_sensor.vision.train_heads
python -m high_level_jobs.grab_sensor --detect-only --visualize
python -m high_level_jobs.grab_sensor --ik-gui
python -m high_level_jobs.grab_sensor --execute --port COM3
```

Live detect: Space freezes NN proposals for edit/save; A accepts NN as a new label.

Label schema: **sensor** = 2 keypoints (**contact** + direction **along holder**; yaw from axis; table XY = contact + `grasp_offset_from_holder_mm`); **post/other** = 1 center. Sensor head = RGB+depth+edge → contact UV + cos/sin(θ). Relabel default: one click updates contact only (keeps holder dir).

Table calib still consumed from `software_win/camera/calibration_table/runs/*/table_calib.json` (falls back to `software/camera/...`).

```powershell
$env:PYTHONPATH = "$PWD\software_win"
python -m camera.calibration_table
```

## Agent rules

1. Do not commit Linux `detection/**/dataset/...` / `debug/`, or Windows `grab_sensor/vision/dataset/{images,depth,labels,previews,model}` / `debug/`.
2. Prefer table-grid calibration under `software_win/camera/calibration_table/` (Windows) or `software/camera/calibration_table/` (Linux) for overhead extrinsics.
3. For multi-machine copies of large detection datasets, use [lab-data-sync](../lab-data-sync/SKILL.md) patterns (rsync to `/share/mehta`, not home quota).
4. Platform overview: [dummy-lab-platforms](../dummy-lab-platforms/SKILL.md).
