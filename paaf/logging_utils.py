"""Uniform logging setup for CLI, GUI and library use."""
from __future__ import annotations

import logging
import sys


_FMT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str = "paaf", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class QtLogHandler(logging.Handler):
    """Emits log records to a Qt signal so the GUI console can render them.

    The Qt import is deferred so importing this module never requires PyQt.
    """

    def __init__(self, callback):
        super().__init__()
        self._callback = callback
        self.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            self._callback(self.format(record))
        except Exception:
            pass
