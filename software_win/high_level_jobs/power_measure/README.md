# Windows power_measure

Live top-view **D415** click → table XY (calib) + fixed measured height → TCP offset
(radial-left + above) → top-down insert. Gripper stays closed (sensor already held).

## Modes

| Mode | Entry | Behavior |
|------|-------|----------|
| **Place** | Session `power_measure` or `--place-only` | Insert and **stay**; record traj for `return` |
| **Optimize** | Session `power_optimization` or `--optimize-only` | PM100 search/climb from insert; **stay** |
| **Full** | `--execute` (default CLI) | Composed: place → optimize → return WORK |

During power protocols, records optical power with **ThorlabsPM100** (PyVISA) at **400 nm**.
If needed, expands gently inside a **20×20 mm** square until power is found, then
**climbs toward higher intensity** (first-improvement + gradient step). Search starts
slow (`search_speed_scale`); once power is seen, motion slows further
(`climb_speed_scale`) until the local max, then a slower final approach. Every
speed change ramps up/down via smooth streaming; no settle waits — not a full-area
raster. Limited to the square centered on insert / first detection.

Interactive place → optimize → return:

```powershell
$env:PYTHONPATH = "$PWD\software_win"
python -m high_level_jobs.session --port COM3
```

See [`../session/README.md`](../session/README.md).

## Geometry

- **Measured point:** click pixel → table XY via `table_calib.json`; Z = `measured_height_in` (default 4 in)
- **TCP XY:** from measured, offset `tcp_left_of_radial_in` (1.25 in) on the **left** of the robot→click radial in table XY (left = +90° CCW about +Z)
- **TCP Z:** measured Z + `tcp_above_measured_in` (4.25 in)
- **Orientation:** `fixed_grasp_ab_deg: [-90, 90]` — A=-90 plus B=+90 so the longer red axis in `--ik-gui` (link local X) aligns with table **+Z**; yaw from radial then `yaw_cw_from_radial_deg` (90° CW)
- **Power plane:** ⊥ to robot→target radial; offsets along left (a) and world +Z (b); trial region is a `side_mm`×`side_mm` square (default 20×20)

## Setup

```powershell
$env:PYTHONPATH = "$PWD\software_win"
pip install ThorlabsPM100   # + NI-VISA / Thorlabs PM100 drivers
```

Close Thorlabs Optical Power Monitor before running (exclusive USB). Attach a
supported sensor head. Pin a meter with `power_meter.serial` or `--pm-serial`.

If open fails with `VI_ERROR_IO`: unplug the PM100 USB cable for a few seconds,
plug it back in, then re-run (USB session can stick after a failed/aborted open).

Table calib: `software_win/camera/calibration_table/runs/*/table_calib.json` (auto-picked newest valid; falls back to `software/camera/...`).

## Run

```powershell
# Live click, then plan (no PyBullet)
python -m high_level_jobs.power_measure --camera-serial <D415_SN>

# Same as grab_sensor: PyBullet IK goal at insert TCP (click + Enter, then close camera)
python -m high_level_jobs.power_measure --ik-gui --camera-serial <D415_SN>

# Same as grab_sensor: plan + PyBullet trajectory playback
python -m high_level_jobs.power_measure --plan-gui --camera-serial <D415_SN>

# Full one-shot: place → optimize → return WORK
python -m high_level_jobs.power_measure --execute --port COM3 --camera-serial <D415_SN>

# Place only (stay at insert; session-style)
python -m high_level_jobs.power_measure --execute --place-only --port COM3

# Optimize only (after a place that saved session/records/power_measure.json)
python -m high_level_jobs.power_measure --execute --optimize-only --port COM3

# Insert only (no power meter protocol; still lift+WORK unless --place-only)
python -m high_level_jobs.power_measure --execute --no-power-scan --port COM3

# Headless from a pixel (no live window)
python -m high_level_jobs.power_measure --click-uv 320,240 --no-gui --ik-gui
python -m high_level_jobs.power_measure --click-uv 320,240 --no-gui --plan-gui
```

Flow with `--ik-gui` / `--plan-gui`: live click → Enter/E → **camera/OpenCV closes** → PyBullet opens (avoids OpenCV fighting PyBullet).

### Live keys

| Key | Action |
|-----|--------|
| Mouse move | Hover depth (m) + table XY (in) |
| Left click | Set measured / TCP target |
| Enter / E | Confirm: close live view, then IK/plan/execute per flags |
| C | Clear target |
| Q / Esc | Quit without planning |

Config: [`config/default_power_measure.yaml`](config/default_power_measure.yaml).
Debug power log: `debug/power_scan.json`.
Session records: `../session/records/power_measure.json`.
Final optimization power also appends to `../session/records/power_logs/record_power.jsonl` (`source: power_optimization`).
