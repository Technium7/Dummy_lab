# Continuous session (`high_level_jobs.session`)

Long-lived Windows runner: **start the arm → WORK**, then accept skill commands
(or natural language) without reconnecting each time.

Motion recordings live in the shared folder
[`records/`](records/) (gitignored payloads; keep the directory).

## Quick start

```powershell
cd C:\Users\PQE_339_ThinkPad\Documents\Dummy_lab
$env:PYTHONPATH = "$PWD\software_win"
python -m high_level_jobs.session --port COM3
# optional: --camera-serial <D415>  --no-llm  --dry-run
```

At the prompt:

```text
session> grab_sensor
session> power_measure
session> power_optimization
session> return
session> put_sensor
session> standby
session> llm put the sensor back
session> quit
```

## Command semantics

| Command | Behavior | Recording |
|---------|----------|-----------|
| `grab_sensor` | Detect → grasp → lift → **WORK** (sensor held) | Saves `records/grab_sensor.json` for `put_sensor` |
| `put_sensor` | Reverse grab segments → open at grasp → **WORK** | Reads `grab_sensor.json` |
| `power_measure` | Live click → **place only** (WORK→insert) → **stay** | Saves `power_measure.json` (includes planned lift/home for later `return`) |
| `power_optimization` | From current insert: PM100 search/climb → **stay** | Needs prior `power_measure` extras (`click_uv`, `grasp_goal_q`) |
| `return` | Stream recorded return path to **WORK** | Uses `power_measure.json` lift+home (else reverse place) |
| `standby` | **Fresh plan** current joints → WORK | — |
| `manual` / `man` | Release port → `manual_control` teleop → reconnect | Pose may change; use `standby` after |
| `s` / `stop` | Immediate firmware `!STOP` (hold + soft-disable) | Next skill re-`!START`s |
| `epoxy_apply` | Stub for now | Use `python -m high_level_jobs.epoxy_app.epoxy_apply` |

Typical power flow: `grab_sensor` → `power_measure` → (`power_optimization`) → `return` → `put_sensor`.

## Layout

```text
software_win/high_level_jobs/session/
  __main__.py      CLI entry
  loop.py          connect + WORK + REPL
  commands.py      command handlers
  motion_store.py  save / load / reverse segments
  llm_map.py       NL → command (Ollama / OpenAI / keywords)
  records/         shared motion JSON (general place)
  README.md        this file

software_win/high_level_jobs/
  grab_sensor/     pick / put helpers (`execute_put_sensor_on_robot`, `execute_standby_to_work`)
  power_measure/   place vs optimize (`execute_power_place_on_robot`, `execute_power_optimize_on_robot`)
  epoxy_app/       epoxy (not session-wired yet)
```

## LLM input

- Exact tokens always win (`grab_sensor`, …).
- `llm <utterance>` forces the mapper.
- Free text uses the mapper when LLM is enabled (default); `--no-llm` uses keyword fallback only.
- Reuses `software/high_level_jobs/llm_client.py` (`OLLAMA_*` or `OPENAI_*` env).

## Standalone CLIs (still available)

```powershell
python -m high_level_jobs.grab_sensor --execute --port COM3
python -m high_level_jobs.power_measure --execute --place-only --port COM3
python -m high_level_jobs.power_measure --execute --optimize-only --port COM3
```

One-shot `power_measure --execute` (no `--place-only`) runs place → optimize → return as a composed sequence.
