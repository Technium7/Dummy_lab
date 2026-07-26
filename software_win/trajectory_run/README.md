# Trajectory run (Windows)

```powershell
$env:PYTHONPATH = "$PWD\software_win"
python software_win\trajectory_run\run_planned_trajectory.py --dry-run
python software_win\trajectory_run\run_planned_trajectory.py --port COM3 --repeats 5
```

Serial defaults: `ROBOT_PORT` → USB auto-detect → `COM3`.

## Smoothing pipeline

1. Load full `last_plan.npz` (all planner samples, not deduped)
2. Convert sim rad → robot degrees
3. **Quintic spline** through knots (C2, zero vel/acc at start/end)
4. **Retime** — scale duration until max joint speed/acceleration are within limits
5. Resample at `--sample-dt` (default 50 ms) for streaming
6. Reverse resampled path for return to HOME

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--port` | platform default | Serial port (`ROBOT_PORT` or auto-detect) |
| `--repeats` | 5 | HOME->target->HOME cycles |
| `--interval-ms` | 70 | `&` waypoint period (matches `manual_control --control-dt-ms`) |
| `--sample-dt` | 0.05 | Output sample period (s) after smoothing |
| `--max-vel` | 25 | Retiming speed limit (deg/s) |
| `--max-accel` | 25 | Retiming acceleration limit (deg/s²) |
| `--no-smooth` | off | Stream raw planner waypoints (legacy) |
| `--debug-serial` | off | Log TX/RX like manual_control |
| `--joint-current-limit-a` | 0 | Optional `#I_LIMIT_J` at connect (A) |

Raw planner path (jerky): `--no-smooth`

Slower / smoother: `--max-vel 15 --max-accel 8 --sample-dt 0.08`

Snappier streaming on Linux (if serial keeps up): `--interval-ms 50`
