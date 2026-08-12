# Windows cameras (epoxy / high-level jobs)

Live viewers and grab backends for:

| Camera | Backend |
|--------|---------|
| Thorlabs **Zelux** | `pylablib` → Thorlabs TLCamera SDK (ThorCam) |
| Lumenera / Teledyne **Infinity 2-1RM** | `lucam` → `lucamapi.dll`, with OpenCV DirectShow fallback |
| Intel **RealSense D415** | `pyrealsense2` (RGB + aligned depth; used by `grab_sensor`) |

### RealSense (D415)

1. Install [Intel RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense/releases) for Windows.
2. `pip install pyrealsense2`
3. Use from Python:

```python
from camera import RealSenseCamera, list_realsense_devices

print(list_realsense_devices())
with RealSenseCamera(serial="") as cam:
    bgr = cam.grab_bgr()
    rgb, depth_m = cam.grab_rgb_depth()
```

Pin a serial in `high_level_jobs/grab_sensor/config/default_grasp.yaml` (`camera_serial`).

## Install

1. Install vendor software / drivers (ThorCam, Infinity Capture).
2. From repo root:

```powershell
pip install -r software_win\requirements.txt
```

Optional: set `THORLABS_TSI_DLL_PATH` if ThorCam DLLs are not under the default Program Files path.

## Scripts

```powershell
python software_win\camera\list_devices.py
python software_win\camera\live_view.py          # Zelux front + side + Infinity
python software_win\camera\live_view.py --demo  # no hardware
python software_win\camera\live_view.py --front-serial <SN1> --side-serial <SN2>
python software_win\camera\live_view.py --zelux-only   # front + side only
python software_win\camera\live_zelux.py --serial <SN>
python software_win\camera\live_infinity.py --opencv-index 0
```

Keys: **Q** / **Esc** quit. Pin serials so front/side do not swap after USB reordering.

### Table calibration (optical hole grid)

```powershell
$env:PYTHONPATH = "$PWD\software_win"
python -m camera.calibration_table
python -m camera.calibration_table --camera-serial <SERIAL>
```

Saves under `software_win/camera/calibration_table/runs/`. See `calibration_table/README.md`.

Infinity uses **Lucam** (Infinity Capture drivers). It will **not** silently fall
back to OpenCV camera #0 (that is usually a different device). Close the
Infinity Capture GUI before running so Lucam can open the USB camera.

If colors look wrong, try `--infinity-bayer GB` (or `RG` / `GR`).


Each live window has manual **Exposure_0.1ms** and **Gain_0.1** trackbars
(auto exposure/gain disabled when the driver supports it). Overlay text shows
current `exp=…ms` and `gain=…`.

- Exposure trackbar value `10` → **1.0 ms**
- Gain trackbar value `100` → **10.0**
- Zelux gain range is typically **0–48**; Infinity lucam often **1–20**

Initial values:

```powershell
python software_win\camera\live_view.py `
  --front-exposure-ms 5 --front-gain 6 `
  --side-exposure-ms 5 --side-gain 6 `
  --infinity-exposure-ms 8 --infinity-gain 2
```


## Python API (for epoxy_app)

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("software_win").resolve()))

from camera import ZeluxCamera, InfinityCamera

with ZeluxCamera(exposure_s=0.01) as zelux, InfinityCamera(camera_number=1) as infinity:
    z_frame = zelux.grab_bgr()   # HxWx3 BGR uint8
    i_frame = infinity.grab_bgr()
```

Both backends implement `CameraBackend` (`open` / `grab_bgr` / `close`).
