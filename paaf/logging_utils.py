"""Uniform logging setup for CLI, GUI and library use."""
from __future__ import annotations

import logging
import sys


_FMT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"


def _configure_root(level: int) -> logging.Logger:
    """One stdout handler on the ``paaf`` logger; every child propagates to it,
    so a handler attached to ``paaf`` (e.g. the GUI console) sees all records."""
    root = logging.getLogger("paaf")
    if not any(getattr(h, "_paaf_stream", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        handler._paaf_stream = True
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
    return root


def get_logger(name: str = "paaf", level: int = logging.INFO) -> logging.Logger:
    root = _configure_root(level)
    if name == "paaf":
        return root
    if not name.startswith("paaf."):
        name = f"paaf.{name}"
    logger = logging.getLogger(name)
    logger.propagate = True
    return logger


class QtLogHandler(logging.Handler):
    """Emits log records to a Qt signal so the GUI console can render them.

    Records may come from worker threads; the signal is delivered to
    ``callback`` on the GUI thread through a queued connection. The Qt import
    is deferred so importing this module never requires PyQt.
    """

    def __init__(self, callback):
        super().__init__()
        self.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        from PyQt5.QtCore import QObject, Qt, pyqtSignal

        class _Bridge(QObject):
            message = pyqtSignal(str)

        self._bridge = _Bridge()
        self._bridge.message.connect(callback, Qt.QueuedConnection)

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            self._bridge.message.emit(self.format(record))
        except Exception:
            self.handleError(record)
