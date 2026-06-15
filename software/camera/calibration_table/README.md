# Optical table calibration

Calibrate the RealSense camera pose relative to an optical table hole grid. The **chosen origin hole** is table `(0, 0)`; grid columns and rows are spaced at **1 inch** (25.4 mm) by default.

Use this when the table fills the camera view and the robot cannot reach enough poses for hand-eye calibration. See [`../README.md`](../README.md) for how this relates to [`../calibration/`](../calibration/) (checkerboard hand-eye, kept for reference).

## Setup

1. Ensure the table fills a reasonable portion of the image and several holes are visible.
2. Lighting should make holes appear as dark circles on a lighter surface.
3. Install vision dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-vision.txt
```

## Workflow

1. **Live preview** — Canny/Otsu edge detections (cyan circles)
2. **Click grow-origin** — freezes the frame
3. **Click 4 neighbor holes** — E/W/N/S assigned automatically by angle from origin
4. **Grid grows** — recursive walk; press **D** to draw a polygon region (or **A** = all holes)
5. **Enter** — apply region filter
6. **Click final origin hole** — re-indexes grid so that hole is table `(0,0)` and runs `solvePnP`
7. **S** — save `table_calib.json`

Press **E** to toggle the Canny edge debug view. Press **R** to unfreeze and reset.

### Tuning

| Flag | Default | Purpose |
|------|---------|---------|
| `--min-circularity` | `0.65` | How round a contour must be to count as a hole |
| `--edge-margin-px` | `15` | Ignore circles near the image border |
| `--max-radius-px` | off | Optional cap for huge blobs only |
| `--min-holes` | `6` | Minimum on-grid holes for pose solve |

## Run (live camera)

Exposure is **locked by default** after a short warmup so edge detection stays stable. Auto exposure is restored when you quit.

From the repo root:

```bash
python software/camera/calibration_table/table_calib_app.py
```

Or from `software/`:

```bash
python -m camera.calibration_table
```

### Controls

| Input | Action |
|-------|--------|
| **Click** | Grow-origin → 4 seeds → region corners → final origin (step depends on phase) |
| **Enter** | Apply polygon region |
| **A** | Skip region — use all grown holes |
| **U** | Undo last region polygon point |
| **D** | Redraw region after grid grow |
| **S** | Save calibration (after final origin) |
| **R** | Reset — resume live preview |
| **E** | Toggle Canny edge debug view |
| **Q** / Esc | Quit |

The overlay shows:

- **Green circles** — holes on the grown grid
- **Gray lines** — connect grown holes along rows/columns
- **Cyan circles** — live preview before origin click
- **Red cross** — origin hole at table `(0, 0)`
- **Magenta dots** — reprojected centers from solved pose

## Run (saved image)

```bash
python software/camera/calibration_table/table_calib_app.py \
  --image path/to/shot.color.png \
  --meta path/to/shot.meta.json
```

## Options

| Flag | Default | Purpose |
|------|---------|---------|
| `--pitch-in` | `1.0` | Hole spacing (inches) |
| `--hole-diameter-in` | `0.25` | Expected hole diameter |
| `--min-holes` | `6` | Minimum holes for `solvePnP` |
| `--camera-serial` | first device | RealSense serial |
| `--warmup` | `15` | Frames before exposure lock |
| `--no-lock-exposure` | off | Keep auto exposure (not recommended) |
| `--exposure-us` | auto | Fixed exposure after warmup (microseconds) |
| `--out-dir` | `calibration_table/runs` | Output base directory |

## Outputs

Each save writes under `software/camera/calibration_table/runs/run_*/`:

- `table_calib.json` — solved extrinsics and detected hole map
- `table_calib.color.png` — raw color frame
- `table_calib.overlay.png` — detection overlay

### JSON fields

**`camera_in_table_frame`** — camera pose in the table frame (origin = chosen hole):

- `translation_m` — camera position; **Z should be positive** when the camera is above the table (+Z = up from table surface)
- `rpy_deg` — XYZ Euler decomposition of the rotation (for display; prefer the 4×4 matrix for chaining)
- `transform_table_cam` — 4×4 homogeneous transform (**authoritative**)

**`table_to_camera`** — inverse: table origin in the camera frame (`transform_cam_table`, OpenCV `rvec`/`tvec`).

**`holes`** — each detected hole with grid index, image pixel center, and table XY (meters).

## Coordinate frames

**Table frame** (aligned with robot base XY):

```text
  +X → increasing grid column (East on grow overlay)
  +Y → decreasing internal grow row (North on grow overlay; opposite image row down)
  +Z → up from table surface
```

Green overlay labels show table-frame indices `(col, row)` where `row = -grow_row`.
Hole one inch north of origin is labeled `(0, 1)`; one inch east is `(1, 0)`.
Internal grow uses `grow_row` (+ = image down / South) only for detection; saved JSON uses table-frame indices.

**OpenCV / RealSense optical frame** (fixed by the sensor):

```text
  +X → right in the image
  +Y → down in the image
  +Z → optical axis (into the scene)
```

There is **no** assumption that these frames align at “zero rotation.” Full 6-DOF `solvePnP` finds whatever rotation maps grid 3D points to pixels. Identity rotation would point the optical axis along table +Z (sideways), not down at the table.

To get `camera_to_robot`, combine with a known robot pose in the table frame:

```text
T_robot_cam = inv(T_table_robot) @ T_table_cam
```

(`T_table_robot` = robot base origin expressed in the table frame.)

## PyBullet preview

Show the calibrated camera and robot in the **table frame** (origin = chosen hole, +Z up):

```bash
python software/camera/calibration_table/view_table_calibration.py \
  --robot-x-in -6 --robot-y-in -5 --robot-z-in 1.25
```

Uses the latest `table_calib.json` under `calibration_table/runs/` unless `--calib` is set.

- **Red axes** — table origin `(0,0,0)`
- **Yellow cross** — robot base at `--robot-*-in` (inches in table frame)
- **Blue/yellow** — camera frustum from `camera_in_table_frame`
- **Gray ticks** — detected hole centers on the table plane

## Tips

- Use **6+ holes** spread across the view for a stable solve.
- Too many magenta X marks? Usually duplicate detections — raise `--min-circularity 0.72`.
- Too few green circles? Lower `--min-circularity 0.55`.
- Press **E** to inspect whether Canny edges ring the holes.

Target: **under 5 px** reprojection for a usable calibration.
