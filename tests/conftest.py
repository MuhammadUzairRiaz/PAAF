"""Shared test setup.

A modal QMessageBox blocks forever in an offscreen test run (nobody can click
OK), which froze the whole suite once. Make the static dialogs return at once.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _benchmark_history_in_tmp(tmp_path, monkeypatch):
    """Every export appends to a benchmark history; keep tests out of ~/.paaf."""
    monkeypatch.setenv("PAAF_BENCH_HISTORY",
                       str(tmp_path / "benchmark_history.csv"))


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
