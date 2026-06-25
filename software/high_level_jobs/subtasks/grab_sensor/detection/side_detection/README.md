# Side-view detection (oblique D415)

Legacy side/oblique camera workflow: **YOLO** → crop → **`grasp_point.pt`** + **`base_point.pt`** (ellipse).

Saved dataset from the original grab_sensor labeling lives here under `dataset/grasp_keypoints/`.

## Commands (from `software/`)

```bash
# Label grasp + base ellipse (live camera)
python -m high_level_jobs.subtasks.grab_sensor.detection.side_detection.live_label

# Offline base ellipse relabel
python -m high_level_jobs.subtasks.grab_sensor.detection.side_detection.relabel

# Train YOLO
python -m high_level_jobs.subtasks.grab_sensor.detection.side_detection.train_yolo_model --export-first

# Train grasp + base ellipse CNNs
python -m high_level_jobs.subtasks.grab_sensor.detection.side_detection.train_model --crop --depth --epochs 120

# Review NN on live camera
python -m high_level_jobs.subtasks.grab_sensor.detection.side_detection.live_review
```

## Layout

```
side_detection/
  dataset/
    grasp_keypoints/     # images, depth, labels, model/
    sensor_yolo/         # exported YOLO train/val split
```

Models: `dataset/grasp_keypoints/model/sensor_yolo.pt`, `grasp_point.pt`, `base_point.pt`
