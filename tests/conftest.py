"""Shared test setup.

A modal QMessageBox blocks forever in an offscreen test run (nobody can click
OK), which froze the whole suite once. Make the static dialogs return at once.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _no_modal_message_boxes(monkeypatch):
    try:
        from PyQt5.QtWidgets import QMessageBox
    except Exception:
        yield
        return
    for name, answer in (("information", QMessageBox.Ok),
                         ("warning", QMessageBox.Ok),
                         ("critical", QMessageBox.Ok),
                         ("question", QMessageBox.Yes)):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, _r=answer, **k: _r))
    yield
