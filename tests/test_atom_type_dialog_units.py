"""The typing dialog must offer all three units, not just the repeat unit.

The user's complaint was that the chain ends came out wrong no matter how
carefully the monomer was typed — because the ends were never theirs to type.
The dialog showed one unit's worth of atoms and everything else was inferred.

These drive the real dialog (offscreen) and check what it actually puts in
front of the user: a row per atom per unit, each row labelled with which unit
it belongs to, each carrying a distinct key, and the head/tail rows suggesting
terminal types where the repeat unit suggests backbone ones.
"""
from __future__ import annotations

from collections import Counter

import pytest

pytest.importorskip("rdkit", reason="needs RDKit")
PyQt5 = pytest.importorskip("PyQt5", reason="needs PyQt5")

from PyQt5.QtCore import Qt                                      # noqa: E402
from PyQt5.QtWidgets import QApplication                         # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def isoprene_file(tmp_path):
    """1,4-isoprene as an .xyz, links at C0 and C4."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles("CC=C(C)C"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    AllChem.UFFOptimizeMolecule(m)
    conf = m.GetConformer()
    lines = [str(m.GetNumAtoms()), "isoprene"]
    for i, a in enumerate(m.GetAtoms()):
        p = conf.GetAtomPosition(i)
        lines.append(f"{a.GetSymbol()} {p.x:.4f} {p.y:.4f} {p.z:.4f}")
    path = tmp_path / "isoprene.xyz"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _spec(path):
    from paaf.config import MonomerSpec

    def cap(k):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        m = Chem.AddHs(Chem.MolFromSmiles("CC=C(C)C"))
        return [n.GetIdx() for n in m.GetAtomWithIdx(k).GetNeighbors()
                if n.GetSymbol() == "H"][0]

    return MonomerSpec(file=path, name="isoprene", head=0, tail=4,
                       head_h=cap(0), tail_h=cap(4))


def _dialog(app, isoprene_file):
    from paaf.ff_registry import get_ff
    from paaf.gui.atom_type_dialog import AtomTypingDialog

    ff = get_ff("oplsaa")
    lt = ff.bundled_path() if ff.bundled_lt else None
    if lt is None or not lt.exists():
        pytest.skip("the OPLS-AA .lt library is not bundled here")
    return AtomTypingDialog(monomer_files=[isoprene_file], ff_lt_path=lt,
                            ff_key=ff.key, ff_inherit=ff.inherit,
                            monomer_specs=[_spec(isoprene_file)])


def _rows(dlg):
    """``[(displayed index, unit label, element, key)]`` from the table."""
    out = []
    for r in range(dlg.atom_table.rowCount()):
        item = dlg.atom_table.item(r, 0)
        out.append((item.text(),
                    dlg.atom_table.item(r, 1).text(),
                    dlg.atom_table.item(r, 2).text(),
                    item.data(Qt.UserRole)))
    return out


def test_the_table_lists_every_unit_of_the_chain(app, isoprene_file):
    dlg = _dialog(app, isoprene_file)
    units = Counter(unit for _i, unit, _e, _k in _rows(dlg))
    print(f"\n  {dict(units)}")
    assert len(units) == 3, "head, repeat and tail must all be listed"
    assert any("Head" in u for u in units)
    assert any("Tail" in u for u in units)
    assert any("Repeat" in u for u in units)


def test_the_end_units_have_one_more_atom_than_the_repeat_unit(app, isoprene_file):
    """Each end keeps a cap the interior gave up at its junction."""
    dlg = _dialog(app, isoprene_file)
    units = Counter(unit for _i, unit, _e, _k in _rows(dlg))
    repeat = next(v for k, v in units.items() if "Repeat" in k)
    for label, count in units.items():
        if "Repeat" not in label:
            print(f"\n  {label}: {count} atoms vs repeat {repeat}")
            assert count == repeat + 1


def test_every_row_has_a_distinct_key(app, isoprene_file):
    """Head atom 4 and tail atom 4 must not overwrite one another."""
    dlg = _dialog(app, isoprene_file)
    keys = [k for _i, _u, _e, k in _rows(dlg)]
    assert len(set(keys)) == len(keys), "two rows share a key"
    assert all(k is not None for k in keys)


def test_the_displayed_index_is_still_the_monomer_atom_number(app, isoprene_file):
    """The user sees the numbering in their own file, not an internal key."""
    dlg = _dialog(app, isoprene_file)
    shown = {int(i) for i, _u, _e, _k in _rows(dlg)}
    print(f"\n  displayed indices: {sorted(shown)}")
    assert max(shown) < 15, "an internal key leaked into the visible column"


def test_the_repeat_unit_keys_are_plain_monomer_indices(app, isoprene_file):
    """So anything reading overrides() from before still reads what it expects."""
    dlg = _dialog(app, isoprene_file)
    for shown, unit, _e, key in _rows(dlg):
        if "Repeat" in unit:
            assert key == int(shown)


def test_the_chain_ends_are_suggested_terminal_types(app, isoprene_file):
    """The whole point: 135 at the ends where the middle says 136."""
    dlg = _dialog(app, isoprene_file)
    from paaf.typing_context import split_role_key

    by_role = dlg.types_by_role()
    print(f"\n  head link C: {by_role['head'].get(0)}  "
          f"middle link C: {by_role['middle'].get(0)}  "
          f"tail link C: {by_role['tail'].get(4)}")
    assert by_role["middle"].get(0) == "136", "interior link carbon is a CH2"
    assert by_role["head"].get(0) == "135", "the chain start is a terminal CH3"
    assert by_role["tail"].get(4) == "135", "the chain end is a terminal CH3"


def test_every_atom_of_every_unit_is_seeded_with_a_type(app, isoprene_file):
    dlg = _dialog(app, isoprene_file)
    assert len(dlg.overrides()) == dlg.atom_table.rowCount()


def test_assigning_to_a_head_row_leaves_the_repeat_unit_alone(app, isoprene_file):
    dlg = _dialog(app, isoprene_file)
    before = dict(dlg.types_by_role()["middle"])

    head_row = next(r for r in range(dlg.atom_table.rowCount())
                    if "Head" in dlg.atom_table.item(r, 1).text()
                    and dlg.atom_table.item(r, 2).text() == "C")
    dlg.atom_table.selectRow(head_row)
    tid_row = next(r for r in range(dlg.type_table.rowCount())
                   if dlg.type_table.item(r, 0).text() == "139")
    dlg.type_table.selectRow(tid_row)
    dlg._assign()

    after = dlg.types_by_role()
    print(f"\n  head now has 139 on "
          f"{sum(1 for v in after['head'].values() if v == '139')} atoms")
    assert "139" in after["head"].values()
    assert after["middle"] == before, "typing an end changed the repeat unit"


def test_a_click_in_3d_selects_the_row_for_that_unit(app, isoprene_file):
    """Clicking the tail's link carbon must not select the head's."""
    dlg = _dialog(app, isoprene_file)
    tail_row = next(r for r in range(dlg.atom_table.rowCount())
                    if "Tail" in dlg.atom_table.item(r, 1).text())
    key = dlg.atom_table.item(tail_row, 0).data(Qt.UserRole)
    dlg._on_viewer_atom_clicked(key)
    selected = {i.row() for i in dlg.atom_table.selectedIndexes()}
    print(f"\n  clicked key {key} -> row {selected}")
    assert selected == {tail_row}


def test_an_element_mismatch_is_still_refused_on_an_end_row(app, isoprene_file,
                                                           monkeypatch):
    dlg = _dialog(app, isoprene_file)
    warned = []
    monkeypatch.setattr(dlg, "_alert", lambda m: warned.append(m))

    h_row = next(r for r in range(dlg.atom_table.rowCount())
                 if "Tail" in dlg.atom_table.item(r, 1).text()
                 and dlg.atom_table.item(r, 2).text() == "H")
    key = dlg.atom_table.item(h_row, 0).data(Qt.UserRole)
    before = dlg.overrides().get(key)
    dlg.atom_table.selectRow(h_row)
    tid_row = next(r for r in range(dlg.type_table.rowCount())
                   if dlg.type_table.item(r, 0).text() == "136")
    dlg.type_table.selectRow(tid_row)
    dlg._assign()
    print(f"\n  refused: {warned[:1]}")
    assert warned, "a carbon type on a hydrogen was accepted"
    assert dlg.overrides().get(key) == before


def test_equivalent_assignment_stays_inside_one_unit(app, isoprene_file,
                                                     monkeypatch):
    """A head CH3 hydrogen and a middle CH2 hydrogen are not interchangeable."""
    dlg = _dialog(app, isoprene_file)
    monkeypatch.setattr(dlg, "_alert", lambda m: None)
    before_middle = dict(dlg.types_by_role()["middle"])

    h_row = next(r for r in range(dlg.atom_table.rowCount())
                 if "Head" in dlg.atom_table.item(r, 1).text()
                 and dlg.atom_table.item(r, 2).text() == "H")
    dlg.atom_table.selectRow(h_row)
    tid_row = next(r for r in range(dlg.type_table.rowCount())
                   if dlg.type_table.item(r, 0).text() == "140")
    dlg.type_table.selectRow(tid_row)
    dlg._assign_equivalent()

    assert dlg.types_by_role()["middle"] == before_middle


def test_auto_type_all_fills_the_ends_too(app, isoprene_file):
    dlg = _dialog(app, isoprene_file)
    dlg._types_by_atom.clear()
    dlg._auto_type_all()
    by_role = dlg.types_by_role()
    print(f"\n  head {len(by_role['head'])}, middle {len(by_role['middle'])}, "
          f"tail {len(by_role['tail'])}")
    assert by_role["head"] and by_role["middle"] and by_role["tail"]
    assert by_role["head"].get(0) == "135"
    assert by_role["middle"].get(0) == "136"


# ================================================================ the 3D view
#
# The point of the 3D view is that you pick atoms by looking at the molecule
# instead of matching numbers in a table. That only works if every atom you
# are allowed to type is actually pickable there — which, while only the
# repeat unit could be typed, the chain ends were not.
def test_every_atom_in_the_3d_view_is_pickable(app, isoprene_file):
    dlg = _dialog(app, isoprene_file)
    if dlg.viewer3d is None:
        pytest.skip("QtWebEngine is not available here")
    mapping = dlg.viewer3d._serial_to_index
    print(f"\n  {len(mapping)} atoms in the scene, "
          f"{sum(1 for i in mapping.values() if i >= 0)} pickable")
    assert mapping, "nothing was handed to the viewer"
    assert all(i >= 0 for i in mapping.values()), \
        "some atoms are still context-only and cannot be clicked"


def test_the_3d_view_shows_all_three_units(app, isoprene_file):
    dlg = _dialog(app, isoprene_file)
    if dlg.viewer3d is None:
        pytest.skip("QtWebEngine is not available here")
    from paaf.typing_context import split_role_key

    roles = Counter(split_role_key(i)[0]
                    for i in dlg.viewer3d._serial_to_index.values())
    print(f"\n  {dict(roles)}")
    assert set(roles) == {"head", "middle", "tail"}


def test_a_head_atom_and_its_tail_twin_are_separate_in_the_view(app, isoprene_file):
    """Clicking the tail's link carbon must not land on the head's."""
    dlg = _dialog(app, isoprene_file)
    if dlg.viewer3d is None:
        pytest.skip("QtWebEngine is not available here")
    keys = list(dlg.viewer3d._serial_to_index.values())
    assert len(set(keys)) == len(keys), "two atoms in the scene share a key"


def test_the_labels_show_the_atom_number_not_the_internal_key(app, isoprene_file):
    """A head atom must read h0, not 1000000."""
    dlg = _dialog(app, isoprene_file)
    if dlg.viewer3d is None:
        pytest.skip("QtWebEngine is not available here")
    labels = dlg._viewer_labels()
    sample = {labels[k] for k in list(labels)[:60]}
    print(f"\n  e.g. {sorted(sample)[:8]}")
    assert all(len(v) <= 4 for v in labels.values()), \
        f"an internal key leaked into a 3D label: {sorted(labels.values())[:3]}"
    assert any(v.startswith("h") for v in labels.values())
    assert any(v.startswith("t") for v in labels.values())
    assert any(v.isdigit() for v in labels.values())


def test_clicking_an_atom_in_3d_reaches_the_right_row(app, isoprene_file):
    """The signal path, end to end, for an end-unit atom."""
    dlg = _dialog(app, isoprene_file)
    if dlg.viewer3d is None:
        pytest.skip("QtWebEngine is not available here")
    serial, key = next((s, i) for s, i in dlg.viewer3d._serial_to_index.items()
                       if i >= 1_000_000)                  # a head-unit atom
    dlg.viewer3d._on_serial_clicked(serial)
    app.processEvents()
    rows = {i.row() for i in dlg.atom_table.selectedIndexes()}
    assert len(rows) == 1
    row = rows.pop()
    print(f"\n  serial {serial} -> key {key} -> row {row} "
          f"({dlg.atom_table.item(row, 1).text()})")
    assert dlg.atom_table.item(row, 0).data(Qt.UserRole) == key
    assert "Head" in dlg.atom_table.item(row, 1).text()


# ====================================================== the viewer's controls
#
# The view was unusable at 41 atoms: every label drawn at once, no way to move
# the structure sideways, no way to look at one unit alone. These check the
# controls that fixed that are wired to something real.
def test_the_viewer_offers_one_group_per_chain_unit(app, isoprene_file):
    dlg = _dialog(app, isoprene_file)
    if dlg.viewer3d is None:
        pytest.skip("QtWebEngine is not available here")
    names = [dlg.viewer3d.unit_combo.itemText(i)
             for i in range(dlg.viewer3d.unit_combo.count())]
    print(f"\n  {names}")
    assert names[0] == "All", "the default must show the whole trimer"
    assert names[1:] == ["Head unit", "Repeat unit", "Tail unit"], \
        "units must be offered in chain order, not dictionary order"


def test_isolating_a_unit_hides_exactly_the_other_two(app, isoprene_file):
    dlg = _dialog(app, isoprene_file)
    if dlg.viewer3d is None:
        pytest.skip("QtWebEngine is not available here")
    viewer = dlg.viewer3d
    total = len(viewer._serial_to_index)

    viewer.show_group("Repeat unit")
    shown = total - len(viewer._hidden)
    repeat_rows = sum(1 for r in range(dlg.atom_table.rowCount())
                      if "Repeat" in dlg.atom_table.item(r, 1).text())
    print(f"\n  {shown} of {total} atoms shown, {repeat_rows} repeat-unit rows")
    assert shown == repeat_rows

    viewer.show_group(None)
    assert viewer._hidden == [], "'All' must bring everything back"


def test_isolating_the_head_shows_the_head_and_not_the_tail(app, isoprene_file):
    dlg = _dialog(app, isoprene_file)
    if dlg.viewer3d is None:
        pytest.skip("QtWebEngine is not available here")
    from paaf.typing_context import split_role_key

    viewer = dlg.viewer3d
    viewer.show_group("Head unit")
    visible = {split_role_key(i)[0]
               for s, i in viewer._serial_to_index.items()
               if s not in set(viewer._hidden)}
    print(f"\n  roles still visible: {visible}")
    assert visible == {"head"}


def test_the_label_mode_defaults_to_selected_not_all(app, isoprene_file):
    """All-at-once is what made the picture unreadable."""
    dlg = _dialog(app, isoprene_file)
    if dlg.viewer3d is None:
        pytest.skip("QtWebEngine is not available here")
    assert dlg.viewer3d._label_mode == "selected"
    assert dlg.viewer3d.label_combo.currentData() == "selected"


def test_choosing_a_label_mode_reaches_the_viewer(app, isoprene_file):
    dlg = _dialog(app, isoprene_file)
    if dlg.viewer3d is None:
        pytest.skip("QtWebEngine is not available here")
    combo = dlg.viewer3d.label_combo
    combo.setCurrentIndex([combo.itemData(i) for i in
                           range(combo.count())].index("all"))
    assert dlg.viewer3d._label_mode == "all"


def test_the_pan_button_toggles_pan_mode(app, isoprene_file):
    dlg = _dialog(app, isoprene_file)
    if dlg.viewer3d is None:
        pytest.skip("QtWebEngine is not available here")
    assert dlg.viewer3d._pan_mode is False
    dlg.viewer3d.pan_button.setChecked(True)
    assert dlg.viewer3d._pan_mode is True
    dlg.viewer3d.pan_button.setChecked(False)
    assert dlg.viewer3d._pan_mode is False


def test_selecting_a_row_highlights_that_atom_in_3d(app, isoprene_file):
    """With labels on 'selected', the table and the picture drive each other."""
    dlg = _dialog(app, isoprene_file)
    if dlg.viewer3d is None:
        pytest.skip("QtWebEngine is not available here")
    picked = []
    dlg.viewer3d.select_index = lambda i: picked.append(i)
    row = next(r for r in range(dlg.atom_table.rowCount())
               if "Tail" in dlg.atom_table.item(r, 1).text())
    dlg.atom_table.selectRow(row)
    app.processEvents()
    print(f"\n  row {row} -> select_index{picked}")
    assert picked == [dlg.atom_table.item(row, 0).data(Qt.UserRole)]


def test_popping_out_moves_the_view_rather_than_cloning_it(app, isoprene_file):
    """Two viewers would mean two states, and one of them silently stale."""
    dlg = _dialog(app, isoprene_file)
    if dlg.viewer3d is None:
        pytest.skip("QtWebEngine is not available here")
    viewer = dlg.viewer3d
    home = viewer.parentWidget()

    dlg._popout_viewer()
    window = dlg._popout
    print(f"\n  popped out into {window.windowTitle()!r}")
    assert viewer.parentWidget() is window, "the view was not moved"
    assert dlg.viewer3d is viewer, "the dialog lost its reference to the view"
    assert not viewer.popout_button.isEnabled()

    window.close()
    app.processEvents()
    assert viewer.parentWidget() is home, "the view was not handed back"
    assert viewer.popout_button.isEnabled()


def test_popping_out_twice_raises_the_same_window(app, isoprene_file):
    dlg = _dialog(app, isoprene_file)
    if dlg.viewer3d is None:
        pytest.skip("QtWebEngine is not available here")
    dlg._popout_viewer()
    first = dlg._popout
    dlg._popout_viewer()
    assert dlg._popout is first, "a second window was opened"
    first.close()
