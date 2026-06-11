# Camera hand-eye calibration

Calibrate the fixed RealSense camera pose relative to the robot base (`camera_to_robot`) while teleoperating the arm with the same stack as `manual_control/manual_control_app.py`.

## Setup

1. Print or mount a **10×10** checkerboard with **10.128 mm** square size.
2. Grasp the board at a **fixed** point — default is the **outer corner tip nearest the pattern origin** (`--grasp-point corner_tip`). Edge-center and other presets are available; any grasp is fine as long as the offset matches.
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

**Serial / Linux:** control uses `manual_control/teleop_runtime.py` + `robot_session.py` — same Linux tty low-latency setup and per-waypoint `drain_rx()` before async `&` streaming as `manual_control_app.py`. Defaults: `--control-dt-ms 20`, `--speed-factor 1`. Try `--control-dt-ms 15` for snappier response.

## Outputs

Each run writes under `software/camera/calibration/runs/run_YYYYMMDD_HHMMSS/`:

- `samples/shot_*/` — color image, annotated overlay, joint pose metadata
- `camera_to_robot.json` — solved extrinsics in the same format as `grab_sensor` config (requires ≥ 3 valid captures)
- `convergence_history.json` — distance, RPY, and spread after each valid solve (from sample 3 onward)
- `convergence_trend.png` — static plot of the same convergence curves shown live during the session

## Convergence plot

A separate matplotlib window tracks estimates as samples are added:

- **distance** — camera origin distance from robot base (m)
- **roll / pitch / yaw** — camera orientation (deg)
- **spread** — leave-one-out stability (lower = converged)

OpenCV hand-eye solve needs **at least 3** valid captures; spread metrics appear from the 4th. Take more poses until distance and angles plateau and spread drops.

## Grasp point / gripper→board offset

Hand-eye only needs a **constant** transform from gripper to checkerboard origin (OpenCV’s first inner corner). Pinching the corner tip is fine.

**Board mount** (`--board-mount`, default `gripper_xz`): checkerboard axes lie on the end-link **X** and **Z** axes (pattern Y → gripper Z). RPY defaults to `(-90, 0, 0)` deg. Use `gripper_xy` if the board lies in the gripper XY plane instead.

| `--grasp-point` | Meaning | Default xyz with `gripper_xz` (10×10 @ 10.128 mm) |
|-----------------|---------|-----------------------------------------------------|
| `corner_tip` | Outer corner nearest pattern origin | `(0.010128, 0, 0.010128)` m |
| `edge_center` | Midpoint of bottom outer edge | `(0.05064, 0, 0)` m |
| `far_corner_tip` | Outer corner opposite pattern origin | `(0.10128, 0, 0.10128)` m |

```bash
# default: corner tip, board on gripper X/Z
python software/camera/calibration/calibration_app.py

# board in gripper XY plane (older assumption)
python software/camera/calibration/calibration_app.py --board-mount gripper_xy
```

Override rotation or translation only if your physical grasp differs:

```bash
python software/camera/calibration/calibration_app.py \
  --board-mount gripper_xz \
  --grasp-point corner_tip
```

## Apply result

Copy `translation_m` and `rpy_deg` from `camera_to_robot.json` into:

`software/high_level_jobs/subtasks/grab_sensor/config/default_grasp.yaml`
