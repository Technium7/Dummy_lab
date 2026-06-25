# Grab sensor — vision & training

Camera capture, YOLO sensor detection, grasp/base CNN, labeling GUIs, and model training.

Runtime inference: `yolo_grasp.infer_yolo_grasp_points` → `grasp.build_grasp_targets` in `pipeline.py`.

## Detection layouts

| Folder | Camera | Models |
|--------|--------|--------|
| **`top_detection/`** | Overhead (production default) | YOLO + **base point** only |
| **`side_detection/`** | Oblique / side view | YOLO + grasp + **base ellipse** |

Shared modules live in this directory (`capture.py`, `grasp_point_net.py`, `yolo_detect.py`, …).

## Top view (overhead) — default

From `software/`:

```bash
python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.live_label
python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.train_yolo_model --export-first
python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.train_model --crop --depth --epochs 120
python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.live_review
```

See [`top_detection/README.md`](top_detection/README.md).

## Side view (legacy / extra cameras)

```bash
python -m high_level_jobs.subtasks.grab_sensor.detection.side_detection.live_label
python -m high_level_jobs.subtasks.grab_sensor.detection.side_detection.relabel
python -m high_level_jobs.subtasks.grab_sensor.detection.side_detection.train_yolo_model --export-first
python -m high_level_jobs.subtasks.grab_sensor.detection.side_detection.train_model --crop --depth --epochs 120
python -m high_level_jobs.subtasks.grab_sensor.detection.side_detection.live_review
```

See [`side_detection/README.md`](side_detection/README.md). Saved side-view data is under `side_detection/dataset/grasp_keypoints/`.

## Legacy entry points

These still work and default to **side_detection** paths:

- `detection.live_label`, `detection.live_review`, `detection.relabel`
- `detection.train_model`, `detection.train_yolo_model`
