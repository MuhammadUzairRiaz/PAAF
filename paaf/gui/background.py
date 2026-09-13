"""Run a slow call off the GUI thread.

Packing, LAMMPS minimisation and gmx calls can take minutes. Called straight
from a click slot they freeze the window (no repaint, no log lines) until they
finish. :func:`run_in_background` moves the call to a QThread and hands the
result back on the GUI thread, where message boxes are safe to show.
"""
from __future__ import annotations

import traceback
from typing import Callable, Optional

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot


class _Worker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, fn: Callable):
        super().__init__()
        self._fn = fn

    @pyqtSlot()
    def run(self) -> None:
        try:
            # progress.emit is thread-safe: the slot runs on the GUI thread.
            result = self._fn(self.progress.emit)
        except Exception as exc:                       # reported, not swallowed
            self.failed.emit(f"{exc}\n\n{traceback.format_exc(limit=6)}")
            return
        self.finished.emit(result)


def run_in_background(owner: QObject, fn: Callable, *,
                      on_done: Callable[[object], None],
                      on_fail: Callable[[str], None],
                      on_progress: Optional[Callable[[str], None]] = None,
                      busy_widget=None) -> QThread:
    """Call ``fn(progress)`` on a worker thread.

    ``on_done(result)`` / ``on_fail(message)`` run on the GUI thread.
    ``busy_widget`` (e.g. the Pack button) is disabled until the run ends.
    References are kept on ``owner`` so the thread is not garbage-collected.
    """
    thread = QThread(owner)
    worker = _Worker(fn)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if on_progress is not None:
        worker.progress.connect(on_progress)
    worker.finished.connect(on_done)
    worker.failed.connect(on_fail)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    if busy_widget is not None:
        busy_widget.setEnabled(False)
        thread.finished.connect(lambda: busy_widget.setEnabled(True))

    runs = getattr(owner, "_background_runs", None)
    if runs is None:
        runs = []
        setattr(owner, "_background_runs", runs)
    runs.append((thread, worker))

    def _forget():
        try:
            runs.remove((thread, worker))
        except ValueError:
            pass
    thread.finished.connect(_forget)
    thread.start()
    return thread
