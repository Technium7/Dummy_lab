# Epoxy cascade vision (`epoxy_app/vision`)

Builds on recorded episodes and manual_control DLS IK — not a parallel stack.

```text
Zelux (full frame, mono)
  -> TipNet (heatmap keypoint)           # optional; crop around tip
  -> ContactNet (1-ch gray crop/full)
  -> PhaseNet (2-ch gray+|diff|)         # only when contact
  -> temporal filter                     # SETTLED = touch after flow
  -> FSM -> IkCartesianMover
```

Contact/phase train on **GT tip crops** when tips are labeled; runtime uses TipNet (full-frame fallback if tip conf is low).

## Install

```powershell
cd C:\Users\PQE_339_ThinkPad\Documents\Dummy_lab
conda activate .\.conda
pip install -r software_win\high_level_jobs\epoxy_app\vision\requirements.txt
$env:PYTHONPATH = "$PWD\software_win"
```

## 1. Record demos (existing)

```powershell
python software_win\manual_control\manual_control_app.py --record-demos
```

## 2. Label states

```powershell
python -m high_level_jobs.epoxy_app.vision.label_tool --episode 1
```

Keys: `1` NOT · `2` TOUCH · `3` FLOW · `4` SETTLED · `5` REMOVING · `r` range-fill · `w` save.

- **SETTLED** = post-flow contact (contact train yes; **not** in phase TOUCH/FLOW)
- **REMOVING** = lift-off — training **drops that frame and everything after**
- Runtime FSM can still infer SETTLED temporally after flow

### 2a. Label / correct apply recordings (`record_result`)

Apply runs save as `dataset/record_result/result_######/`. Prefill from the cascade, then fix phase:

```powershell
python -m high_level_jobs.epoxy_app.vision.label_tool --results --episode 1 --seed-checkpoint --device cuda
# same contact/phase keys while viewing the front camera:
python -m high_level_jobs.epoxy_app.vision.label_tool --results --episode 1 --camera front
# optional tip labels on the same take:
python -m high_level_jobs.epoxy_app.vision.tip_label_tool --results --episode 1
python -m high_level_jobs.epoxy_app.vision.tip_label_tool --results --episode 1 --camera front
# preview predictions:
python -m high_level_jobs.epoxy_app.vision.demo_episode --results --episode 1 --device cuda
```

Labeled `result_*` episodes are **auto-added to train** the next time you run `vision.train` / `tip_train` (alongside demo `episode_*`).
## 2b. Label needle tip (click)

```powershell
python -m high_level_jobs.epoxy_app.vision.tip_label_tool --episode 1
# front Zelux (writes tip_front, does not overwrite side tip):
python -m high_level_jobs.epoxy_app.vision.tip_label_tool --results --episode 14 --camera front
```

Click = tip · `a/d` navigate · `x` clear · `w` save. Sparse tips OK — training interpolates between marks (`tip.interpolate_labels`).

Writes `"tip"` (side / `*_zelux.jpg`) or `"tip_front"` (front / `*_zelux_front.jpg`) into `labels.json`.
`tip_train` / `vision.train` still use **side** frames only (`tip` + `*_zelux.jpg`).

## 2c. Review / auto-label more episodes (after tip_train)

```powershell
python -m high_level_jobs.epoxy_app.vision.tip_review_tool --episode 6 --device cuda
# or pre-fill missing tips on open:
python -m high_level_jobs.epoxy_app.vision.tip_review_tool --episode 6 --device cuda --fill-on-start
# re-predict model tips with temporal tracking (keeps human clicks):
python -m high_level_jobs.epoxy_app.vision.tip_review_tool --episode 6 --device cuda --refill-model
# same on the front camera:
python -m high_level_jobs.epoxy_app.vision.tip_review_tool --results --episode 14 --device cuda --camera front --fill-on-start
```

Green cross = model pred · cyan circle = human tip · orange = model-saved tip.  
`space` accept pred · click to fix · `f` fill all missing · `w` save. Then retrain tip on the expanded set.

## 3. Train tip → contact → phase

```powershell
python -m high_level_jobs.epoxy_app.vision.tip_train --device cuda
python -m high_level_jobs.epoxy_app.vision.train --device cuda
python -m high_level_jobs.epoxy_app.vision.demo_episode --episode 17 --device cuda
```

Checkpoints under `epoxy_app/dataset/checkpoints/`:
- `tip_best.pt` / `tip_best_full.pt`
- `contact_best.pt`, `phase_best.pt`, combined `best.pt` (embeds tip weights when present)

Config (`tip:` in `config.yaml`): `enabled`, `crop_size: 256`, `conf_threshold`, `heatmap_size`.
Near apply, TipNet’s global peak often locks onto the epoxy blob — `tip.track: true` keeps a local search around the last tip (demo / run_apply / tip_review).

Demo overlay shows tip crosshair + crop box. Use `--no-tip` to force full-frame.

## 4. Test controller (no hardware)

```powershell
python -m high_level_jobs.epoxy_app.vision.run_apply --smoke
python -m pytest software_win\high_level_jobs\epoxy_app\vision\tests -q
```

## 5. Vision + FSM with Zelux (mock robot)

```powershell
python -m high_level_jobs.epoxy_app.vision.run_apply --mock
```

## 6. Real apply (teleop + vision + motion)

Starts the same PyBullet / firmware stack as `manual_control`, opens Zelux with live predictions, then runs the FSM from the current tip pose.

```powershell
python -m high_level_jobs.epoxy_app.epoxy_apply
# or dry-run without serial:
python -m high_level_jobs.epoxy_app.epoxy_apply --sim-only
```

1. Focus **PyBullet** and jog the tip above the pad (`WASD` / `QE` / …).
2. Press **`G`** to start approach → contact → flow → settle → retract.
3. Press **`X`** anytime to emergency-stop and **hold the current pose** (also works with focus on the OpenCV window).
4. After an estop: **`G`** clears the hold (teleop jog back on); **`G`** again starts a new apply.
5. **`R`** start/stop recording the run; **`C`** saves to `dataset/record_result/result_######/` (not training `episodes/`).
6. Label/correct that take, then retrain — see §2a.

Live overlays: `epoxy_apply Zelux side` (tip / contact / phase + FSM, `*_zelux.jpg`) and `epoxy_apply Zelux front` (same weights, `*_zelux_front.jpg`). Apply FSM uses **side**. Infinity is live view only. **R** records all live cameras into the same `result_######` frames.

## Safety

- Approach: creep with **vision paused**, then **~0.40 s** tip settle, then vision runs during the post-creep window (~0.85 s total before next move)
- CHECK_FLOW: same creep → settle → vision sequence
- After FLOWING: retract on SETTLED, or after **3 s** even if SETTLED never fires
- IK `settle_tol` must stay much smaller than `min_z_step` (tiny steps were being skipped)
- `robot.stop()` only holds pose; estop uses `abort()` so retract is not blocked
- Max downward travel / Z steps / approach time
- Camera or inference failure → ERROR
- Vision UNCERTAIN streak → ERROR
- Retract: **+Z 10 mm** first (direct lift), then **+Y** slide (`retract_distance`; set negative to flip)
- `X` estop aborts in-flight IK and holds pose
