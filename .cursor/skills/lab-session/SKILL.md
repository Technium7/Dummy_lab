---
name: lab-session
description: >-
  Continuous Dummy_lab Windows session loop: WORK start, grab_sensor, put_sensor,
  power_measure place, power_optimization, return, standby, LLM command mapping.
  Use when running the robot interactively, wiring subtasks together, or editing
  software_win/high_level_jobs/session.
---

# Lab session (Windows)

## Entry

```powershell
$env:PYTHONPATH = "$PWD\software_win"
python -m high_level_jobs.session --port COM3
```

Docs: `software_win/high_level_jobs/session/README.md`.

## Command contract

| Input | Effect |
|-------|--------|
| `grab_sensor` | Pick → WORK; save `session/records/grab_sensor.json` |
| `put_sensor` | Reverse grab record → release → WORK |
| `power_measure` | **Place only** at click; stay; save return traj |
| `power_optimization` | PM100 search/climb from insert; stay |
| `return` | Recorded traj → WORK |
| `standby` | Replan current → WORK |
| `manual` / `man` | Hand off to `manual_control` teleop; reconnect when GUI closes |
| `s` / `stop` | Immediate `!STOP` |
| `epoxy_apply` | Stub (use `epoxy_app` module) |
| `llm …` / free text | Map to a command via Ollama/OpenAI or keywords |

## Agent rules

1. Keep the long-lived robot handle in the session — except `manual`, which must release the serial port for teleop and reconnect afterward.
2. Do not merge place + optimize back into a single session `power_measure` command.
3. Motion JSON belongs under `software_win/high_level_jobs/session/records/` (the general place).
4. `epoxy_apply` stays stubbed until the user specifies session wiring.
5. Related skills: [grab-sensor](../grab-sensor/SKILL.md), [camera-lab](../camera-lab/SKILL.md).
