"""OpenCV highgui helpers (Qt on Linux needs imshow before mouse callbacks)."""

from __future__ import annotations

import re
import sys

import numpy as np

PANEL_H = 120


def require_cv2_gui():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "opencv-python is required for the labeling GUI.\n"
            "Install with: pip uninstall opencv-python-headless -y && pip install opencv-python"
        ) from exc

    build = cv2.getBuildInformation()
    gui_match = re.search(r"^\s*GUI:\s*(\S+)", build, re.MULTILINE)
    if gui_match and gui_match.group(1).upper() == "NONE":
        raise SystemExit(
            "OpenCV was built without GUI support (opencv-python-headless).\n"
            "Install the GUI build:\n"
            "  pip uninstall opencv-python-headless -y\n"
            "  pip install opencv-python"
        )

    if sys.platform.startswith("linux") and hasattr(cv2, "startWindowThread"):
        try:
            cv2.startWindowThread()
        except cv2.error:
            pass

    probe = "__cv2_gui_probe__"
    try:
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.namedWindow(probe, cv2.WINDOW_NORMAL)
        cv2.imshow(probe, blank)
        cv2.waitKey(1)
        cv2.destroyWindow(probe)
        cv2.waitKey(1)
    except cv2.error as exc:
        raise SystemExit(
            "OpenCV cannot open windows on this machine.\n"
            "Check DISPLAY is set and you are not running headless.\n"
            "Try: pip uninstall opencv-python-headless -y && pip install opencv-python"
        ) from exc
    return cv2


def init_cv2_window(
    cv2,
    window: str,
    width: int,
    height: int,
    on_mouse,
) -> None:
    """Create a resizable window and attach mouse handler (Qt needs a shown frame first)."""
    blank = np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, blank)
    cv2.waitKey(1)
    try:
        cv2.resizeWindow(window, width, height)
    except cv2.error:
        pass
    cv2.imshow(window, blank)
    cv2.waitKey(1)
    try:
        cv2.setMouseCallback(window, on_mouse)
    except cv2.error as exc:
        raise SystemExit(
            f"OpenCV could not attach mouse handler to window {window!r}.\n"
            "Qt highgui often fails without a display or when fonts are missing.\n"
            "Ensure DISPLAY is set, or try: sudo apt install fontconfig fonts-dejavu-core"
        ) from exc
