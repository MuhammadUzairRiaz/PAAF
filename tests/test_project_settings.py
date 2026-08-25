"""Project name / output folder: the settings that decide where files land.

Regression cover for a real bug: the sidebar was seeded with the literal string
"untitled project" while the project field already said "polymer", and
``textChanged`` never fires for a QLineEdit's *initial* value — so the two were
out of sync from the moment the window opened.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 not installed")

from PyQt5.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from paaf.gui.project_dialog import ProjectDialog  # noqa: E402


# --------------------------------------------------------------- dialog
def test_dialog_round_trips_values():
    d = ProjectDialog("PBS-20", "/tmp/runs")
    assert d.values() == ("PBS-20", "/tmp/runs")


def test_dialog_previews_the_real_destination():
    d = ProjectDialog("PBS-20", "/tmp/runs")
    assert str(Path("/tmp/runs") / "PBS-20") in d.preview.text()
    # The preview follows edits.
    d.name_edit.setText("PLA-study")
    assert str(Path("/tmp/runs") / "PLA-study") in d.preview.text()


def test_dialog_falls_back_to_sane_defaults():
    d = ProjectDialog("", "")
    name, out = d.values()
    assert name == "polymer"
    assert out.endswith("output")


def test_preview_slot_accepts_the_textchanged_argument():
    """QLineEdit.textChanged emits a str; the slot must tolerate it."""
    d = ProjectDialog("x", "/tmp")
    d._update_preview("some text")     # must not raise


# ------------------------------------------------------------ main window
def test_sidebar_shows_the_real_project_name_at_startup():
    from paaf.gui.main_window import MainWindow
    w = MainWindow()
    assert w.project_name.text() == w.sidebar.project_lb.text()
    assert w.sidebar.project_lb.text() != "untitled project"


def test_renaming_the_project_updates_the_sidebar():
    from paaf.gui.main_window import MainWindow
    w = MainWindow()
    w.project_name.setText("PLA-crosslink-study")
    w._sync_project_label()
    assert w.sidebar.project_lb.text() == "PLA-crosslink-study"


def test_project_is_reachable_from_the_sidebar_and_menu():
    from paaf.gui.main_window import MainWindow
    w = MainWindow()
    assert hasattr(w, "_edit_project")
    assert hasattr(w.sidebar, "projectClicked")


def test_output_path_hint_is_shown_on_the_export_page():
    from paaf.gui.main_window import MainWindow
    w = MainWindow()
    hint = w.project_path_hint.text()
    assert "written to" in hint
    assert w.project_name.text() in hint
