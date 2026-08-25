"""Tests for the redesigned sidebar (pipeline stepper + tools).

Skips cleanly when PyQt5 is unavailable.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 not installed")

from PyQt5.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from paaf.gui.main_window import NAV_ITEMS, PIPELINE_INDICES  # noqa: E402
from paaf.gui.sidebar import DONE, LOCKED, NOW, READY, Sidebar  # noqa: E402


def _states(sb):
    return {it.text_lb.text(): it._state for it in sb._items}


def test_sidebar_lists_every_nav_item():
    sb = Sidebar(NAV_ITEMS, PIPELINE_INDICES)
    assert sb.count() == len(NAV_ITEMS)


def test_step_states_follow_the_spec():
    """Spec screenshot: step 1 is NOW, step 2 ready, everything else locked."""
    sb = Sidebar(NAV_ITEMS, PIPELINE_INDICES)
    sb.setCurrentRow(0)
    st = _states(sb)
    assert st["Builder"] == NOW
    assert st["Chain"] == READY
    assert st["Optimize"] == LOCKED
    assert st["Export"] == LOCKED


def test_completing_a_step_unlocks_the_next_and_updates_progress():
    sb = Sidebar(NAV_ITEMS, PIPELINE_INDICES)
    sb.setCurrentRow(0)
    assert sb.progress_lb.text().startswith("0 of 6")

    sb.mark_done(0, True)
    sb.setCurrentRow(1)
    st = _states(sb)
    assert st["Builder"] == DONE
    assert st["Chain"] == NOW
    assert st["Optimize"] == READY          # unlocked by being next
    assert sb.progress_lb.text().startswith("1 of 6")

    # Un-completing rolls the progress back.
    sb.mark_done(0, False)
    assert sb.progress_lb.text().startswith("0 of 6")


def test_step_label_tracks_position_and_tools():
    sb = Sidebar(NAV_ITEMS, PIPELINE_INDICES)
    sb.setCurrentRow(0)
    assert sb.step_lb.text() == "STEP 1 OF 6 · BUILDER"
    sb.setCurrentRow(5)
    assert sb.step_lb.text() == "STEP 6 OF 6 · EXPORT"
    # Independent tools are not pipeline steps.
    tool_row = len(PIPELINE_INDICES)
    sb.setCurrentRow(tool_row)
    assert sb.step_lb.text().startswith("TOOL ·")


def test_current_row_signal_emitted_once_per_change():
    sb = Sidebar(NAV_ITEMS, PIPELINE_INDICES)
    seen = []
    sb.currentRowChanged.connect(seen.append)
    sb.setCurrentRow(2)
    sb.setCurrentRow(2)          # same row -> no repeat signal
    sb.setCurrentRow(4)
    assert seen == [2, 4]
    assert sb.currentRow() == 4


def test_project_name_is_shown_in_the_sidebar():
    sb = Sidebar(NAV_ITEMS, PIPELINE_INDICES)
    sb.set_project_name("PLA-crosslink-study")
    assert sb.project_lb.text() == "PLA-crosslink-study"
    sb.set_project_name("")
    assert sb.project_lb.text() == "untitled project"
