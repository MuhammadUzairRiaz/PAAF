"""A polymer the user adds shows up in every library list, not just one.

Each Builder panel (Homopolymer, Monomer A, B, C…) keeps its own copy of the
library, so a polymer added under Homopolymer used to be missing from Monomer
A, and one added under Monomer A missing from Monomer B. The pickers on the
other pages (Amorphous cell, CG, Reaction scheme, Copolymer dialog) must list
it too.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 not installed")
pytest.importorskip("rdkit", reason="the Builder needs RDKit")

from PyQt5.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

PIB = ("MY_PIB", "[*]CC(C)(C)[*]", "Polyisobutylene")


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A store of our own, so a test never touches the user's real one."""
    from paaf import builder
    monkeypatch.setenv("PAAF_CUSTOM_LIBRARY", str(tmp_path / "custom.csv"))
    builder.reload_library()
    yield
    monkeypatch.delenv("PAAF_CUSTOM_LIBRARY")
    builder.reload_library()


def _pids(panel):
    t = panel.lib_table
    return {t.item(r, 0).text() for r in range(t.rowCount())}


def _add_from(panel):
    from paaf.custom_library import add_custom
    entry = add_custom(*PIB)
    panel._reload_library_rows(select_pid=entry.pid)


def _panels(tab):
    return [tab.panel_solo, tab.panel_a, tab.panel_b, *tab.extra_panels]


@pytest.mark.parametrize("source", ["panel_solo", "panel_a", "panel_b"])
def test_added_in_one_builder_panel_is_listed_in_all(store, source):
    from paaf.gui.builder_tab import BuilderTab
    tab = BuilderTab()
    tab._add_extra_panel()                       # Monomer C
    assert all(PIB[0] not in _pids(p) for p in _panels(tab))

    _add_from(getattr(tab, source))

    for p in _panels(tab):
        assert PIB[0] in _pids(p), p._title


def test_removed_in_one_builder_panel_is_gone_from_all(store):
    from paaf.custom_library import delete_custom
    from paaf.gui.builder_tab import BuilderTab
    tab = BuilderTab()
    _add_from(tab.panel_solo)
    delete_custom(PIB[0])
    tab.panel_a._reload_library_rows()
    for p in _panels(tab):
        assert PIB[0] not in _pids(p), p._title


def test_a_monomer_panel_added_later_lists_it_too(store):
    from paaf.gui.builder_tab import BuilderTab
    tab = BuilderTab()
    _add_from(tab.panel_solo)
    assert PIB[0] in _pids(tab._add_extra_panel())


def test_refreshing_another_panel_keeps_its_hand_edited_smiles(store):
    from paaf.gui.builder_tab import BuilderTab
    tab = BuilderTab()
    tab.panel_a.lib_table.selectRow(0)
    kept = tab.panel_a._selected_pid()
    tab.panel_a.smiles_edit.setText("CCOCC")     # typed by hand

    _add_from(tab.panel_solo)

    assert tab.panel_a.smiles_edit.text() == "CCOCC"
    assert tab.panel_a._selected_pid() == kept


def test_the_shared_picker_lists_it(store):
    """The picker behind Amorphous cell, CG, Reaction scheme and Copolymer."""
    from paaf.custom_library import add_custom
    from paaf.gui.library_picker import LibraryPicker
    add_custom(*PIB)                             # no reload_library() call
    assert PIB[0] in {r.pid for r in LibraryPicker()._records}
