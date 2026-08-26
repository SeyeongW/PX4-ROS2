#!/usr/bin/env python3
"""Qt binding shim.

PySide6 is the target binding (`pip install --user PySide6`).  PyQt5 is what
Ubuntu 22.04 ships, so it stays available as a fallback and the GUI code is held
to the subset both bindings share.  Import Qt only through this module:

    from .qt import QtCore, QtGui, QtWidgets, Signal, Slot, BINDING
"""

from __future__ import annotations

BINDING = ""

try:
    from PySide6 import QtCore, QtGui, QtWidgets  # noqa: F401
    from PySide6.QtCore import Signal, Slot  # noqa: F401

    BINDING = "PySide6"
except ImportError:  # pragma: no cover - exercised only on the fallback binding
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: F401
        from PyQt5.QtCore import pyqtSignal as Signal  # noqa: F401
        from PyQt5.QtCore import pyqtSlot as Slot  # noqa: F401

        BINDING = "PyQt5"
    except ImportError as exc:
        raise ImportError(
            "no Qt binding found. Install the target one with "
            "`pip install --user PySide6` (see gcs/PLAN.md)."
        ) from exc


def exec_app(app) -> int:
    """Run an application event loop under either binding."""
    runner = getattr(app, "exec", None) or app.exec_
    return int(runner())
