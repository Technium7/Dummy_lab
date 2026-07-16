---
name: camera-lab
description: >-
  RealSense live view, calibration, and serial pinning for Dummy_lab cameras.
  Use when positioning cameras, live_view, preview-cameras, table calib, or
  multi-RealSense USB issues.
---

# Camera lab utilities

## Live mosaic (all cameras)

```bash
python software/camera/live_view.py
python software/camera/live_view.py --list
python software/camera/live_view.py --serial SERIAL1 --serial SERIAL2
```

**Q** / **Esc** quit. Prefer this for aiming a third camera.

## One-shot ID for insert_sensor

```bash
cd software
python -m high_level_jobs.subtasks.insert_sensor --list-cameras
python -m high_level_jobs.subtasks.insert_sensor --preview-cameras
```

Pin serials in job YAML (`cameras.overhead` / `side` / `side2`). Never trust USB enumeration order.

## Calibration

| Tool | When |
|------|------|
| `software/camera/calibration_table/table_calib_app.py` | Default: optical table hole grid |
| `software/camera/calibration/calibration_app.py` | Hand-eye / checkerboard fallback |
| `software/camera/single_shot.py` | Single RGB-D capture, no GUI |

## Agent rules

1. Separate USB ports for each RealSense and the robot serial.
2. If a camera is stuck: `jobs -l`, kill stopped viewers, unplug/replug; avoid leaving Ctrl+Z’d OpenCV processes.
3. insert_sensor models still use **2** cams; side2 is record-only until models change ([insert-sensor](../insert-sensor/SKILL.md)).
