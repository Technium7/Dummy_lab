# Camera hand-eye calibration (reference)

> **Prefer table-grid calibration** when the optical table is visible: [`../calibration_table/`](../calibration_table/)  
> See [`../README.md`](../README.md) for when to use each method.

Calibrate the fixed RealSense camera pose relative to the robot base (`camera_to_robot`) while teleoperating the arm with the same stack as `manual_control/manual_control_app.py`.

## Setup

1. Print or mount a **10×10** checkerboard with **10.128 mm** square size.
2. Grasp the board at a **fixed** point — default **`corner_tip`** is the **outer corner on the wrist-facing side** (not the pattern-origin corner). Use `--grasp-point origin_corner_tip` for the old origin-adjacent corner.
3. Install vision dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-vision.txt
```

4. Connect the robot over USB (Type-C) and the RealSense camera.

## Run

From the repo root:

```bash
python software/camera/calibration/calibration_app.py
```

Or from `software/`:

```bash
python -m camera.calibration.calibration_app
```

### Controls

Keys work in the **camera window** or the **PyBullet** window (click either first).

| Key | Action |
|-----|--------|
| WASD / QE | Linear motion |
| HK / UJ / IY | Angular / wrist |
| **C** | Capture calibration sample |
| O / P | Open / close gripper |
| T | Home |

Move the arm to varied poses so the board is visible from different angles, then press **C** after the robot settles.

Two threads:

- **Control** (`control_thread.py`) — identical teleop to `manual_control_app.py` (PyBullet + robot + IK). Publishes joint pose; never runs OpenCV.
- **CV** (`cv_thread.py`) — RealSense preview, checkerboard overlay, capture, and hand-eye solve. Reads joint snapshots from the control thread.

**Joint mapping:** calibration does not copy `manual_control/mapping.py`. Teleop goes through `manual_control/teleop_tick.py`, which imports `JOINT_SIGNS`, `JOINT_OFFSETS_DEG`, `HOME_JOINTS_DEG`, and speed helpers from `mapping.py`. Edit mapping once — both apps pick it up.

**Serial / Linux:** control uses `manual_control/teleop_runtime.py` + `robot_control_thread.py` — same Linux tty low-latency setup, status/end-link GUI windows, and per-waypoint async `&` streaming as `manual_control_app.py`. Defaults match manual control: `--control-dt-ms 70`, `--speed-factor 1`, `--timeout 1`. Try `--control-dt-ms 20` for snappier response if serial keeps up.

**Options shared with manual control:**

| Flag | Purpose |
|------|---------|
| `--home-firmware` | `T` also runs firmware `!HOME` (~3 s block; default is stream-only home) |
| `--debug-serial` | Log all serial TX/RX |
| `--joint-current-limit-a` | Set `#I_LIMIT_J` at connect (A per joint; 0 = skip) |
| `--sim-only` | PyBullet only, no robot serial |
| `--teleop-only` | Teleop without RealSense (no capture/solve; useful when camera is unplugged) |

## Outputs

Each run writes under `software/camera/calibration/runs/run_YYYYMMDD_HHMMSS/`:

- `samples/shot_*/` — color image, annotated overlay, joint pose metadata
- `camera_to_robot.json` — solved extrinsics in the same format as `grab_sensor` config (requires ≥ 3 valid captures). Includes `camera_origin_in_base_m` and `board_pose_error_m`.
- `convergence_history.json` — distance, RPY, and spread after each valid solve (from sample 3 onward)
- `convergence_trend.png` — static plot of the same convergence curves shown live during the session

## Convergence plot

A separate matplotlib window tracks estimates as samples are added:

- **distance** — camera origin distance from robot base (m)
- **roll / pitch / yaw** — camera orientation (deg)
- **spread** — leave-one-out stability (lower = converged)

OpenCV hand-eye solve needs **at least 3** valid captures; spread metrics appear from the 4th. Take more poses until distance and angles plateau and spread drops.

### Solver (all data, joint fit)

The solve uses **every valid capture together**:

```text
T_base_board ≈ T_base_cam @ T_cam_board     (same for each pose)
```

**Default:** full **6-DOF** rotation + translation jointly fit over all captures.

- **Fitted:** camera origin `(tx, ty, tz)` and full `rpy_deg` in the **robot base frame**
- **Legacy (reference only):** `--camera-constrained-mount` fixes roll/pitch about base X/Y and fits translation + base-Z yaw. Roll/pitch values are in the **robot frame**, not the table frame — do not reuse those angles for table-grid calibration.

Copy `translation_m` and `rpy_deg` from `camera_to_robot.json` into `default_grasp.yaml`, or use `transform_robot_cam` when chaining transforms.

## Grasp point / gripper→board offset

Hand-eye only needs a **constant** transform from gripper to checkerboard origin (OpenCV’s first inner corner). Pinching the corner tip is fine.

**Board mount** (`--board-mount`, default `gripper_yz`): checkerboard lies in the end-link **Y/Z plane** (parallel to PyBullet **green** and **blue**). Pattern **+X** extends along link **+Y** (green); pattern **+Y** extends along link **−Z** (opposite blue). RPY defaults to `(-90, 0, 90)` deg. Default grasp is the outer corner on the wrist-facing side.

The default **TCP** is **6.5 cm along link6 −Y** (`end_link_to_tcp` = `(0, −0.065, 0)` m). The board extends from the pinch along link **+Y** and **−Z**.

Transform chain:

```text
T_base_board = T_base_link @ T_link_tcp @ T_tcp_board
```

- `T_link_tcp` — wrist to pinch along link **−Y** (default: `(0, −0.065, 0)` m)
- `T_tcp_board` — board origin relative to pinch on the **link-side** outer corner; pattern extends along link **+Y** and **−Z**

| Mount | Board plane in gripper | RPY (deg) |
|-------|------------------------|-----------|
| `gripper_yz` (default) | Y + (−Z) (green + opposite blue) | `(-90, 0, 90)` |
| `gripper_xz` | X + Z (red + blue) | `(-90, 0, 0)` |
| `gripper_xy` | X + Y (red + green) | `(0, 0, 0)` |

| `--grasp-point` | Meaning | TCP→board xyz with `gripper_yz` (10×10 @ 10.128 mm) |
|-----------------|---------|-----------------------------------------------------|
| `corner_tip` (default) | Outer corner on wrist-facing side (+board X) | `(0, −0.091152, 0.091152)` m |
| `origin_corner_tip` | Outer corner near pattern origin (wrong end) | `(0, −0.010128, 0.010128)` m |
| `link_side_edge` | Midpoint of wrist-facing outer edge | `(0, −0.091152, 0.040512)` m |
| `edge_center` | Midpoint of bottom outer edge | `(0, −0.05064, 0)` m |
| `far_corner_tip` | Same as `origin_corner_tip` | `(0, −0.010128, 0.010128)` m |

```bash
# default: link-side corner tip, board parallel to gripper Y/Z (green + blue)
python software/camera/calibration/calibration_app.py

# other mount presets if your physical grasp differs
python software/camera/calibration/calibration_app.py --board-mount gripper_xz
python software/camera/calibration/calibration_app.py --board-mount gripper_xy
```

Override rotation or translation only if your physical grasp differs:

```bash
python software/camera/calibration/calibration_app.py \
  --board-mount gripper_yz \
  --grasp-point corner_tip
```

## Apply result

Copy `translation_m` and `rpy_deg` from `camera_to_robot.json` into:

`software/high_level_jobs/subtasks/grab_sensor/config/default_grasp.yaml`

## Preview in PyBullet

```bash
python software/camera/calibration/view_calibration.py --run-dir run_20260612_185557
```

| Color | Meaning |
|-------|---------|
| **Blue** rectangle | Solved camera |
| **Green** rectangle | Physical checkerboard edge |
| **Cyan** rectangle | OpenCV 9×9 pattern |
| **Yellow** crosses | TCP (pinch) + pattern origin |

Uses the same `CalibrationCardConfig` / `T_base_board = T_base_link @ T_link_tcp @ T_tcp_board` chain as capture. Board dimensions may load from saved `card_config`; TCP and board offsets always use current code defaults.
