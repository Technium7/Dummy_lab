# Optical table calibration (Windows)

Calibrate the RealSense camera pose relative to an optical table hole grid. The **chosen origin hole** is table `(0, 0)`; grid columns and rows are spaced at **1 inch** (25.4 mm) by default.

Same workflow as `software/camera/calibration_table/`; this copy uses the Windows RealSense backend in `software_win/camera/realsense.py`.

## Setup

1. Table fills a reasonable portion of the image; several holes visible.
2. Lighting should make holes appear as dark circles on a lighter surface.
3. From repo root:

```powershell
$env:PYTHONPATH = "$PWD\software_win"
pip install -r software_win\requirements.txt
# needs opencv + pyrealsense2 (and torch if using the hole NN filter)
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

## Run (live camera)

Exposure is **locked by default** at **166 µs** after a short warmup (same as prior good runs). Override with `--exposure-us`, or `--exposure-us 0` to snapshot auto. Auto exposure is restored when you quit.

```powershell
$env:PYTHONPATH = "$PWD\software_win"
python -m camera.calibration_table
# or:
python software_win\camera\calibration_table\table_calib_app.py
```

Pin a camera:

```powershell
python -m camera.calibration_table --camera-serial <SERIAL>
```

### Controls

| Input | Action |
|-------|--------|
| **Click** | Grow-origin → 4 seeds → region corners → final origin |
| **Enter** | Apply polygon region |
| **A** | Skip region — use all grown holes |
| **U** | Undo last region polygon point |
| **D** | Redraw region after grid grow |
| **S** | Save calibration (after final origin) |
| **R** | Reset — resume live preview |
| **E** | Toggle Canny edge debug view |
| **Q** / Esc | Quit |

## Outputs

Each save writes under `software_win/camera/calibration_table/runs/run_*/`:

- `table_calib.json` — solved extrinsics and detected hole map
- `table_calib.color.png` — raw color frame
- `table_calib.overlay.png` — detection overlay

`grab_sensor` / `power_measure` auto-pick the newest valid run (Windows path preferred over the Linux `software/` copy).

## PyBullet preview

```powershell
$env:PYTHONPATH = "$PWD\software_win"
python software_win\camera\calibration_table\view_table_calibration.py `
  --robot-x-in 10.5 --robot-y-in 5.4 --robot-z-in -0.5 `
  --robot-rpy-deg 0 0 -90
```

Default view is **top-down, image-aligned** (same as the calib overlay):

- **Red / green / blue** — table `+X` / `+Y` / `+Z`
- **Yellow cross** — robot base
- **Blue frustum** — camera
- **Dark ticks** — holes

## Tips

- Use **6+ holes** spread across the view for a stable solve.
- Too many magenta X marks? Raise `--min-circularity 0.72`.
- Too few green circles? Lower `--min-circularity 0.55`.
- Target: **under 5 px** reprojection for a usable calibration.
