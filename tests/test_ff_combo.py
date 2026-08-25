"""The force-field chooser, grouped by typing engine.

Forty-six force fields in one flat list is more than anyone can scan, and the
split that matters is which tool produces the parameters: Moltemplate reads
bundled .lt libraries, DL_FIELD types from its own .sf/.par. That also decides
which relaxation engine is available.

Two things here are easy to get wrong and would be invisible until a user hit
them: a heading that can be selected (giving ``currentData() is None`` to code
expecting a key), and a combo left on index 0 — which is now a heading rather
than a force field.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 not installed")

from PyQt5.QtCore import Qt                                    # noqa: E402
from PyQt5.QtWidgets import QApplication, QComboBox            # noqa: E402

_app = QApplication.instance() or QApplication([])

from paaf.ff_registry import REGISTRY, list_ffs                # noqa: E402
from paaf.gui.ff_combo import (                                # noqa: E402
    GROUPS, populate_force_field_combo, select_force_field,
)


def _combo(default=None) -> QComboBox:
    c = QComboBox()
    populate_force_field_combo(c, default)
    return c


def _headings(c: QComboBox):
    return [c.itemText(i) for i in range(c.count()) if c.itemData(i) is None]


def _keys(c: QComboBox):
    return [c.itemData(i) for i in range(c.count()) if c.itemData(i) is not None]


# ============================================================== grouping
def test_the_list_is_grouped_by_typing_engine():
    c = _combo()
    print("\n  " + "\n  ".join(_headings(c)))
    assert "— MOLTEMPLATE —" in _headings(c)
    assert "— DL_FIELD —" in _headings(c)


def test_every_force_field_still_appears():
    """Grouping must not drop anything — 46 in, 46 out."""
    c = _combo()
    keys = _keys(c)
    print(f"\n  {len(keys)} force fields under {len(_headings(c))} headings")
    assert sorted(keys) == sorted(REGISTRY)


def test_each_force_field_sits_under_its_own_engine():
    c = _combo()
    current = None
    seen = {}
    for i in range(c.count()):
        key = c.itemData(i)
        if key is None:
            current = c.itemText(i)
        else:
            seen[key] = current
    titles = {kind: f"— {title} —" for kind, title, _ in GROUPS}
    wrong = [(k, seen[k], titles.get(REGISTRY[k].kind))
             for k in seen if seen[k] != titles.get(REGISTRY[k].kind)]
    print(f"\n  pcff -> {seen['pcff']}; oplsaa -> {seen['oplsaa']}")
    assert not wrong, f"filed under the wrong heading: {wrong}"


def test_dlfield_force_fields_are_all_together():
    c = _combo()
    positions = [i for i in range(c.count())
                 if c.itemData(i) and REGISTRY[c.itemData(i)].kind == "dlfield"]
    assert positions == list(range(min(positions), max(positions) + 1)), \
        "the DL_FIELD block is interrupted"


# ============================================================== headings
def test_a_heading_cannot_be_selected():
    """Otherwise currentData() returns None to code expecting a key."""
    c = _combo()
    model = c.model()
    for i in range(c.count()):
        if c.itemData(i) is None:
            item = model.item(i)
            print(f"\n  {c.itemText(i)!r} flags={int(item.flags())}")
            assert not (item.flags() & Qt.ItemIsSelectable)
            assert not (item.flags() & Qt.ItemIsEnabled)


def test_the_combo_never_starts_on_a_heading():
    """Index 0 is a heading now; a default must still be a real force field."""
    c = _combo()
    print(f"\n  starts on {c.currentText()!r} -> {c.currentData()!r}")
    assert c.currentData() is not None


def test_an_unknown_default_still_lands_on_something_real():
    c = _combo("no_such_force_field")
    assert c.currentData() is not None


def test_the_requested_default_is_selected():
    c = _combo("pcff")
    assert c.currentData() == "pcff"


# ========================================================== availability
def test_a_force_field_dl_field_cannot_select_is_greyed_out():
    """OPLS2020.par ships in lib/ but 4.13 has no control-file key for it.

    Hidden would be worse: a user can see the file on disk and would hunt for
    the missing entry. Disabled with a reason answers the question.
    """
    c = _combo()
    model = c.model()
    index = next(i for i in range(c.count())
                 if c.itemData(i) == "opls2020_dl")
    item = model.item(index)
    print(f"\n  {c.itemText(index)}\n  tooltip: {item.toolTip()}")
    assert not item.isEnabled()
    assert "unavailable" in c.itemText(index)
    assert "no control-file key" in item.toolTip()


def test_selectable_dlfield_force_fields_stay_enabled():
    c = _combo()
    model = c.model()
    for key in ("pcff", "compass", "cvff", "opls2005_dl", "charmm36_carb"):
        index = next(i for i in range(c.count()) if c.itemData(i) == key)
        assert model.item(index).isEnabled(), f"{key} should be selectable"


def test_selecting_by_key_reports_failure_rather_than_moving():
    c = _combo("pcff")
    assert select_force_field(c, "cvff") is True
    assert c.currentData() == "cvff"
    assert select_force_field(c, "nonsense") is False
    assert c.currentData() == "cvff", "a failed lookup must not move the combo"


# ================================================= every tab uses the same one
@pytest.mark.parametrize("factory,attr", [
    ("paaf.gui.amorphous_tab:AmorphousTab", "ff_combo"),
    ("paaf.gui.reaction_scheme_tab:ReactionSchemeTab", "rx_ff_combo"),
])
def test_the_tabs_show_the_grouped_list(factory, attr):
    """Three tabs used to build this list independently."""
    import importlib

    module_name, class_name = factory.split(":")
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
        tab = cls()
    except Exception as exc:                            # pragma: no cover
        pytest.skip(f"{class_name} needs more than a bare QApplication: {exc}")
    combo = getattr(tab, attr)
    texts = [combo.itemText(i) for i in range(combo.count())]
    print(f"\n  {class_name}.{attr}: {combo.count()} rows")
    assert "— DL_FIELD —" in texts
    assert combo.currentData() is not None
