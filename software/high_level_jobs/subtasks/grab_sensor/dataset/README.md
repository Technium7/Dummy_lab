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
      grasp_point.pt        # RGBD grasp CNN
      runs/                 # Ultralytics training logs

  sensor_yolo/              # Exported YOLO dataset (from export_yolo_dataset)
    images/train|val/
    labels/train|val/
    data.yaml
```

## Minimum workflow

1. Collect labels: `python -m high_level_jobs.subtasks.grab_sensor.label_grasp`
2. Train YOLO: `python -m high_level_jobs.subtasks.grab_sensor.train_yolo_model --export-first`
3. Train grasp net: `python -m high_level_jobs.subtasks.grab_sensor.train_grasp_model --crop --depth`

Models must exist before running `vision_mode: yolo_grasp` in production.
