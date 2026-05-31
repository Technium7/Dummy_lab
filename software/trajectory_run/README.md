# Trajectory run

Execute a cached planner trajectory on the **physical robot** with quintic spline smoothing and retiming.

## Run

```bash
python software/trajectory_run/run_planned_trajectory.py --port COM3 --repeats 5
```

Dry-run:

```bash
python software/trajectory_run/run_planned_trajectory.py --dry-run
```

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
| `--port` | COM3 | Serial port |
| `--repeats` | 5 | HOME->target->HOME cycles |
| `--sample-dt` | 0.05 | Output sample period (s) after smoothing |
| `--max-vel` | ~25.5 | Retiming speed limit (deg/s) |
| `--max-accel` | 12 | Retiming acceleration limit (deg/s²) |
| `--no-smooth` | off | Stream raw planner waypoints (legacy) |
| `--interval-ms` | 50 | Send spacing when `--no-smooth` |

Raw planner path (jerky): `--no-smooth`

Slower / smoother: `--max-vel 15 --max-accel 8 --sample-dt 0.08`
