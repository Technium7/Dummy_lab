# Camera

Utilities for RealSense capture and extrinsics for `grab_sensor` / robot vision jobs.

## Live view (all cameras)

Tile every connected RealSense into one window — useful when mounting or identifying cameras:

```bash
python software/camera/live_view.py
python software/camera/live_view.py --list
python software/camera/live_view.py --serial 104122062828 --serial 024122062290
```

**Q** / **Esc** quits. Each tile is labeled with the device serial.

One-shot capture (no GUI): [`single_shot.py`](single_shot.py).

## Calibration

Two ways to estimate fixed RealSense extrinsics:

| Package | When to use | Output frame |
|---------|-------------|--------------|
| [`calibration_table/`](calibration_table/) | **Default.** Camera views an optical table hole grid; robot need not move. | Table grid (`camera_in_table_frame`) |
| [`calibration/`](calibration/) | **Reference / fallback.** Moving checkerboard on the gripper; needs varied arm poses. | Robot base (`camera_to_robot`) |

Both solve full **6-DOF** pose from vision (OpenCV `solvePnP` or joint SE(3) fit). Prefer the **4×4 transforms** in JSON for robot chaining; `rpy_deg` fields are human-readable XYZ Euler labels, not individual bracket angles.

### Quick start (table grid)

```bash
python software/camera/calibration_table/table_calib_app.py
```

See [calibration_table/README.md](calibration_table/README.md).

### Reference (hand-eye)

```bash
python software/camera/calibration/calibration_app.py
```

See [calibration/README.md](calibration/README.md).

## Chaining table → robot

When table calib gives `T_table_cam` and you know robot base pose in the table frame:

```text
T_robot_cam = inv(T_table_robot) @ T_table_cam
```

Table +X/+Y match robot base +X/+Y (grid column/row index direction). Hand-eye gives `T_robot_cam` directly as `camera_to_robot` in `default_grasp.yaml` format.
