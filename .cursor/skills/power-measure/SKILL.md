---
name: power-measure
description: >-
  Windows power_measure click-to-insert, place-only vs power_optimization, PM100
  scan. Use when editing power_measure place/optimize/return, Thorlabs meter, or
  session power commands.
---

# power_measure (Windows)

Package: `software_win/high_level_jobs/power_measure/`.

## Modes

| Mode | How | Behavior |
|------|-----|----------|
| **Place** | Session `power_measure` or CLI `--place-only` | WORK→insert, **stay**, record traj for `return` |
| **Optimize** | Session `power_optimization` or CLI `--optimize-only` | Search/climb from current insert; **stay** |
| **Full** | CLI `--execute` (no place/optimize flags) | Composed place → optimize → return |

Prefer the **session** for interactive place → optimize → return:

```powershell
$env:PYTHONPATH = "$PWD\software_win"
python -m high_level_jobs.session --port COM3
# session> power_measure
# session> power_optimization
# session> return
```

## Standalone

```powershell
python -m high_level_jobs.power_measure --execute --place-only --port COM3
python -m high_level_jobs.power_measure --execute --optimize-only --port COM3
python -m high_level_jobs.power_measure --execute --port COM3   # full composed
```

Records: `software_win/high_level_jobs/session/records/power_measure.json`.

## Agent rules

1. Session `power_measure` must remain place-only; do not auto-run optimization.
2. Optimization requires prior place extras (`click_uv`, `grasp_goal_q`).
3. Do not commit `debug/power_scan.json` or session record JSON.
4. Session overview: [lab-session](../lab-session/SKILL.md).
