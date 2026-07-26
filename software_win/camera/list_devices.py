#!/usr/bin/env python3
"""List Windows cameras used by epoxy_app (Zelux + Infinity).

  python software_win/camera/list_devices.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SOFTWARE_WIN = Path(__file__).resolve().parents[1]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from camera.infinity import list_infinity_devices
from camera.zelux import list_zelux_devices


def main() -> int:
    print("=== Thorlabs Zelux / TLCamera ===")
    zelux = list_zelux_devices()
    if not zelux:
        print("(none detected - need ThorCam + pylablib)")
    for info in zelux:
        print(f"  [{info.index}] {info.label()}")

    print("=== Infinity 2-1RM / OpenCV ===")
    infinity = list_infinity_devices()
    if not infinity:
        print("(none detected - need Infinity Capture + lucam, or DirectShow devices)")
    for info in infinity:
        print(f"  [{info.index}] {info.label()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
