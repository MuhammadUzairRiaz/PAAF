"""Where the Blend tab writes its files.

The tab used to write ``packed_blend.data`` beside whichever component the
user happened to add first, with no field to change it and nothing on screen
saying where it went. These tests pin the replacement: an explicit folder,
defaulting to the project's, with the first component's directory kept only as
a last-resort fallback.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 not installed")

from PyQt5.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from paaf.gui.blend_tab import BlendTab  # noqa: E402


class _Component:
    """Just enough of BlendComponent for the path logic."""

    def __init__(self, path: str):
        self.data_file = Path(path)


def _components(tmp_path):
    return [_Component(tmp_path / "runA" / "lammps1.data"),
            _Component(tmp_path / "runB" / "lammps1.data")]


def test_a_typed_folder_wins(tmp_path):
    t = BlendTab()
    t.out_dir.setText(str(tmp_path / "chosen"))
    got = t.resolved_out_dir(_components(tmp_path))
    print(f"\n  typed -> {got}")
    assert got == tmp_path / "chosen"


def test_the_project_folder_is_used_when_nothing_is_typed(tmp_path):
    """The default that was missing entirely before."""
    t = BlendTab()
    t.set_settings_provider(lambda: {"out_dir": str(tmp_path / "out"),
                                     "project": "myblend"})
    got = t.resolved_out_dir(_components(tmp_path))
    print(f"\n  project default -> {got}")
    assert got == tmp_path / "out" / "myblend" / "blends"
    # And it is filled into the field, so the user can see and change it.
    assert t.out_dir.text() == str(tmp_path / "out" / "myblend" / "blends")


def test_the_first_component_is_only_a_last_resort(tmp_path):
    """It used to be the ONLY behaviour, which is the bug being fixed."""
    t = BlendTab()
    comps = _components(tmp_path)
    got = t.resolved_out_dir(comps)
    print(f"\n  no folder, no project -> {got}")
    assert got == comps[0].data_file.parent


def test_the_preview_states_where_files_will_go(tmp_path):
    t = BlendTab()
    t.set_settings_provider(lambda: {"out_dir": str(tmp_path / "out"),
                                     "project": "p"})
    text = t.out_preview.text()
    print(f"\n  with project : {text}")
    assert "packed_blend.data" in text
    assert str(tmp_path / "out" / "p" / "blends") in text

    t2 = BlendTab()
    print(f"  without      : {t2.out_preview.text()}")
    assert "beside the first component" in t2.out_preview.text()


def test_the_preview_follows_the_write_input_checkbox(tmp_path):
    t = BlendTab()
    t.out_dir.setText(str(tmp_path))
    t.write_in.setChecked(True)
    assert "packed_blend.in" in t.out_preview.text()
    t.write_in.setChecked(False)
    print(f"\n  .in off -> {t.out_preview.text()}")
    assert "packed_blend.in" not in t.out_preview.text()
    assert "packed_blend.data" in t.out_preview.text()


def test_blank_file_names_fall_back_to_defaults(tmp_path):
    """An empty name must not produce a path ending in a bare directory."""
    t = BlendTab()
    t.out_dir.setText(str(tmp_path))
    t.out_data.setText("")
    text = t.out_preview.text()
    print(f"\n  blank data name -> {text}")
    assert "packed_blend.data" in text


def test_a_missing_settings_provider_is_not_an_error(tmp_path):
    """The tab must work standalone, outside the main window."""
    t = BlendTab()
    assert t.default_out_dir() == ""
    t.set_settings_provider(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert t.default_out_dir() == ""
    assert t.resolved_out_dir(_components(tmp_path)) is not None
