# Local training data (not in git)

This directory holds machine-specific captures, labels, and trained weights. Copy or recreate on each workstation.

## Layout

```
dataset/
  grasp_keypoints/          # Grasp-point labeling output
    images/                 # RGB frames (*.png)
    depth/                  # Aligned depth (*.npy, meters)
    labels/                 # Grasp pixel JSON per sample
    previews/               # Overlay previews
    manifest.jsonl
    model/
      sensor_yolo.pt        # YOLO sensor detector
      grasp_point.pt        # Grasp CNN (2 outputs)
      base_point.pt         # Base CNN (2 outputs, separate net)
      runs/                 # Ultralytics training logs

  sensor_yolo/              # Exported YOLO dataset (from export_yolo_dataset)
    images/train|val/
    labels/train|val/
    data.yaml
```

## Minimum workflow

1. Collect labels (grasp + base): `python -m high_level_jobs.subtasks.grab_sensor.detection.live_label`
2. Fix missing bases on old samples: `python -m high_level_jobs.subtasks.grab_sensor.detection.label_base`
3. Train YOLO: `python -m high_level_jobs.subtasks.grab_sensor.detection.train_yolo_model --export-first`
4. Train grasp + base nets: `python -m high_level_jobs.subtasks.grab_sensor.detection.train_grasp_model --crop --depth`

Models must exist before running `vision_mode: yolo_grasp` in production.
