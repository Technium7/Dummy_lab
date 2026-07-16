---
name: grab-sensor
description: >-
  grab_sensor grasp pipeline, live labeling, and local dataset paths. Use when
  working on grab_sensor, live_label, live_review, YOLO grasp, or table
  calibration tied to grasping.
---

# grab_sensor

Work from `software/`. Detection data and weights are **gitignored** — keep on disk / sync separately.

## Layout

| Path | Purpose |
|------|---------|
| `high_level_jobs/subtasks/grab_sensor/` | Pipeline, teleop hooks, config |
| `detection/top_detection/` | Overhead base-point labeling |
| `detection/side_detection/` | Side grasp + base ellipse |
| `detection/*/dataset/` | Images, depth, labels, `.pt` weights (local only) |
| `debug/` | Live capture overlays (local only) |

## Common commands

```bash
cd software
# Side: label / review
python -m high_level_jobs.subtasks.grab_sensor.detection.side_detection.live_label
python -m high_level_jobs.subtasks.grab_sensor.detection.side_detection.live_review

# Top / overhead
python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.live_label
python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.live_review
```

Pass `--serial SERIAL` when multiple RealSense devices are connected.

## Agent rules

1. Do not commit `detection/**/dataset/{images,depth,labels,previews,model}` or `debug/`.
2. Prefer table-grid calibration under `software/camera/calibration_table/` for overhead extrinsics.
3. For multi-machine copies of large detection datasets, use [lab-data-sync](../lab-data-sync/SKILL.md) patterns (rsync to `/share/mehta`, not home quota).
4. Platform overview: [dummy-lab-platforms](../dummy-lab-platforms/SKILL.md).
