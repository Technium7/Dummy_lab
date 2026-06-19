"""Backward-compatible entry point — use live_review instead."""

from .live_review import main, run_live_review_gui as run_review_gui

__all__ = ["main", "run_review_gui"]

if __name__ == "__main__":
    raise SystemExit(main())
