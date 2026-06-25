# Local training data (not in git)

Top/overhead camera captures and trained weights for `grab_sensor` (base point only).

## Layout

```
dataset/
  base_keypoints/
    images/                 # RGB frames (*.png)
    depth/                  # Aligned depth (*.npy, meters)
    labels/                 # Base point JSON per sample
    previews/
    manifest.jsonl
    model/
      sensor_yolo.pt
      base_point.pt
      runs/

  sensor_yolo/              # Exported YOLO dataset
    images/train|val/
    labels/train|val/
    data.yaml
```

## Workflow

1. `python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.live_label`
2. `python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.train_yolo_model --export-first`
3. `python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.train_model --crop --depth`

Models must exist before running `vision_mode: yolo_grasp` with `grasp_net.base_only: true`.
