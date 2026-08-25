"""Logic tests for the Reaction scheme tab.

The tab holds one or more independent :class:`ReactionBlock` objects — a scheme
may run several reactions, each with its own molecules, its own atom-map
numbers, its own validation and its own export folder.

PyQt5 is not importable in every environment, so these skip cleanly when it is
missing. They exercise the tab's *logic*, not its pixels.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt5", reason="PyQt5 not installed")

from PyQt5.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from paaf.gui.reaction_scheme_tab import ReactionBlock, ReactionSchemeTab  # noqa: E402
from paaf.reaction import complete_mapping, extract_template  # noqa: E402
from paaf.reaction_smiles import Issue, SchemeResult  # noqa: E402
from paaf.structure import Atom, Molecule  # noqa: E402


def _mol(elements, bonds, name="m"):
    return Molecule(
        atoms=[Atom(index=i, element=e, xyz=np.zeros(3))
               for i, e in enumerate(elements)],
        bonds=[(i, j, float(o)) for i, j, o in bonds], name=name)


# ------------------------------------------------------------ scheme shape
def test_scheme_starts_with_one_reaction():
    tab = ReactionSchemeTab()
    assert len(tab._blocks) == 1
    assert tab.library() == []


def test_reactions_can_be_added_and_removed():
    tab = ReactionSchemeTab()
    tab._add_reaction()
    tab._add_reaction()
    assert len(tab._blocks) == 3
    assert [b.title_lb.text() for b in tab._blocks] == [
        "Reaction 1", "Reaction 2", "Reaction 3"]

    tab._remove_reaction(tab._blocks[1])
    assert len(tab._blocks) == 2
    # Numbering closes the gap.
    assert [b.title_lb.text() for b in tab._blocks] == ["Reaction 1", "Reaction 2"]

    # A scheme always keeps at least one reaction.
    tab._remove_reaction(tab._blocks[0])
    tab._remove_reaction(tab._blocks[0])
    assert len(tab._blocks) == 1


def test_a_reaction_can_hold_many_reactants_and_one_product():
    blk = ReactionBlock(1)
    blk.add_reactant()                     # 3 total
    assert len(blk._rows) == 3
    assert len(blk._prod_rows) == 1
    data = [("epoxide", "[C:1]1CO1"), ("amine", "[N:2]CC"), ("cat", "CO")]
    for row, (nm, smi) in zip(blk._rows, data):
        row.name.setText(nm); row.smiles.setText(smi)
    blk._prod_rows[0].smiles.setText("[C:1]C[N:2]CC")
    name, reactants, products = blk.values()
    assert len(reactants) == 3 and len(products) == 1


def test_each_side_keeps_at_least_one_molecule():
    blk = ReactionBlock(1)
    while len(blk._rows) > 1:
        blk._remove(blk._rows, blk._rows[-1], "reactant")
    blk._remove(blk._rows, blk._rows[0], "reactant")
    assert len(blk._rows) == 1
    blk._remove(blk._prod_rows, blk._prod_rows[0], "product")
    assert len(blk._prod_rows) == 1


# ------------------------------------------------------- independent maps
def test_two_reactions_may_reuse_the_same_map_numbers():
    """Map numbers are scoped to a reaction, so Reaction 2 may reuse :1."""
    tab = ReactionSchemeTab()
    b1 = tab._blocks[0]
    b1.load("first", [("a", "[C:1]CC")], [("p", "[C:1]CO")])
    b2 = tab._add_reaction()
    b2.load("second", [("a", "[C:1]NN")], [("p", "[C:1]NO")])

    r1, p1 = b1.maps()
    r2, p2 = b2.maps()
    assert set(r1) == {1} and set(r2) == {1}     # no collision between blocks
    assert b1.reaction_name() != b2.reaction_name()


def test_map_panel_follows_the_reaction_being_edited():
    tab = ReactionSchemeTab()
    b1 = tab._blocks[0]
    b1.load("first", [("a", "[C:1]CC")], [("p", "[C:1]CO")])
    b2 = tab._add_reaction()
    b2.load("second", [("acid", "CC(=O)[OH:1]"), ("alcohol", "[C:2]CO")],
            [("ester", "CC(=O)O[C:2]"), ("water", "O")])

    tab._on_block_changed(b2)
    assert tab.map_scope_lb.text() == "Reaction 2"
    rows = {tab.map_table.item(r, 0).text():
            (tab.map_table.item(r, 1).text(), tab.map_table.item(r, 2).text())
            for r in range(tab.map_table.rowCount())}
    assert rows[":2"] == ("reactant 2", "product 1")
    assert rows[":1"][1] == "absent"          # the OH leaves as water

    tab._on_block_changed(b1)
    assert tab.map_scope_lb.text() == "Reaction 1"


# ---------------------------------------------------------------- example
def test_load_example_fills_the_first_reaction():
    tab = ReactionSchemeTab()
    tab._load_example()
    b = tab._blocks[0]
    names = [r.values()[0] for r in b._rows]
    pnames = [r.values()[0] for r in b._prod_rows]
    assert "acid" in names and "alcohol" in names
    assert "ester" in pnames and "water" in pnames


# ------------------------------------------------------------- validation
def test_validate_all_marks_each_reaction_separately():
    tab = ReactionSchemeTab()
    tab._load_example()
    tab._add_reaction()                  # left blank on purpose
    tab._validate_all()                  # must not raise
    # The blank reaction is skipped rather than reported as broken.
    assert tab._blocks[1].result is None


def test_blocks_render_a_successful_result():
    r = _mol(["C", "O", "H", "C", "O", "H"],
             [(0, 1, 1), (1, 2, 1), (3, 4, 1), (4, 5, 1)], "R")
    p = _mol(["C", "O", "C"], [(0, 1, 1), (1, 2, 1)], "P")
    mapping, ev = complete_mapping(r, p, seed={0: 0, 1: 1, 3: 2})
    tmpl = extract_template(r, p, mapping, ev, name="condensation",
                            reactant="R", product="P")
    res = SchemeResult(
        ok=True, template=tmpl,
        bonds_formed=[tuple(b) for b in tmpl.created_bonds],
        bonds_broken=[tuple(b) for b in tmpl.deleted_bonds],
        n_bonds_formed=len(tmpl.created_bonds),
        n_bonds_broken=len(tmpl.deleted_bonds),
        n_atoms_deleted=len(tmpl.deleted_atoms))

    blk = ReactionBlock(1)
    blk._render_result(res)              # success path must not raise
    blk._set_status(res)
    assert "Validated" in blk.status.text()

    err = SchemeResult(ok=False)
    err.issues.append(Issue("E-MAP-01", "error", "map :3 has no match"))
    blk._render_result(err)              # error path must not raise
    blk._set_status(err)
    assert "error" in blk.status.text()


def test_library_collects_only_valid_reactions():
    tab = ReactionSchemeTab()
    tab._load_example()
    tab._add_to_library()                # nothing validated yet
    assert tab.library() == []


# ---------------------------------------------------------------- export
def test_the_branch_keeps_its_own_force_field():
    """Step 2 is independent: changing it must not touch the pipeline."""
    from paaf.gui.main_window import MainWindow
    w = MainWindow()
    rs = w.builder_tab.reaction_scheme
    pipeline_ff = w.ff_combo.currentData()

    rs.rx_ff_combo.setCurrentIndex(0)
    assert rs.rx_ff_combo.currentData() != pipeline_ff or rs.rx_ff_combo.count() == 1
    assert w.ff_combo.currentData() == pipeline_ff      # pipeline untouched

    # ...and the export reads the BRANCH's settings, not the pipeline's.
    cfg = rs._export_settings()
    assert cfg["ff_key"] == rs.rx_ff_combo.currentData()


def test_copy_from_pipeline_pulls_the_main_selection_in():
    from paaf.gui.main_window import MainWindow
    w = MainWindow()
    rs = w.builder_tab.reaction_scheme
    rs.rx_ff_combo.setCurrentIndex(0)
    rs._copy_ff_from_pipeline()
    assert rs.rx_ff_combo.currentData() == w.ff_combo.currentData()


def test_standalone_tab_still_exports_with_its_own_settings():
    """No main window: the branch has its own FF, so export is still defined."""
    tab = ReactionSchemeTab()
    cfg = tab._export_settings()
    assert cfg["resolved"] is True
    assert cfg["ff_key"]


def test_three_steps_with_their_own_actions():
    tab = ReactionSchemeTab()
    assert [tab.steps.tabText(i) for i in range(tab.steps.count())] == [
        "1 · Scheme", "2 · Force field", "3 · Export"]

    tab.steps.setCurrentIndex(0)
    assert tab.b_validate.isVisible() and not tab.b_export.isVisible()
    tab.steps.setCurrentIndex(2)
    assert tab.b_export.isVisible() and not tab.b_validate.isVisible()


def test_export_preview_lists_one_folder_per_reaction():
    from paaf.reaction_smiles import SchemeResult
    tab = ReactionSchemeTab()
    tab._blocks[0].load("crosslink", [("e", "[C:1]1CO1")], [("p", "[C:1]CO")])
    b2 = tab._add_reaction()
    b2.load("ester", [("a", "CC(=O)[OH:1]")], [("p", "CC(=O)[O:1]C")])
    for b in tab._blocks:
        b.result = SchemeResult(ok=True)

    tab.rx_out_dir.setText("/tmp/runs/reactions")
    tab._refresh_export_preview()
    assert tab.rx_preview.rowCount() == 2
    folders = {tab.rx_preview.item(r, 1).text()
               for r in range(tab.rx_preview.rowCount())}
    assert "/tmp/runs/reactions/crosslink" in folders
    assert "/tmp/runs/reactions/ester" in folders


def test_untyped_export_previews_structures_only():
    from paaf.reaction_smiles import SchemeResult
    tab = ReactionSchemeTab()
    tab._blocks[0].load("r1", [("a", "[C:1]CC")], [("p", "[C:1]CO")])
    tab._blocks[0].result = SchemeResult(ok=True)
    tab.rx_run_typing.setChecked(False)
    tab._refresh_export_preview()
    assert "xyz" in tab.rx_preview.item(0, 3).text()


def test_export_and_library_buttons_start_disabled():
    tab = ReactionSchemeTab()
    assert not tab.b_export.isEnabled()
    assert not tab.b_add_lib.isEnabled()


# ------------------------------------------- library picker & chain length
def test_library_picker_lists_and_filters_polymers():
    from paaf.gui.library_picker import LibraryPicker
    lp = LibraryPicker()
    assert lp.table.rowCount() > 0
    total = lp.table.rowCount()

    lp.filter.setText("PBS"); lp._apply_filter()
    assert 0 < lp.table.rowCount() < total
    rec = lp.selected()
    assert rec is not None
    assert "[*]" in rec.smiles          # polymerization form, not closed-shell

    lp.filter.setText("no-such-polymer"); lp._apply_filter()
    assert lp.table.rowCount() == 0
    assert lp.selected() is None        # must not raise on an empty table


def test_every_molecule_row_can_reach_the_library():
    blk = ReactionBlock(1)
    for row in blk._rows + blk._prod_rows:
        assert hasattr(row, "b_lib")


def test_polymer_rows_are_flagged_with_the_repeat_marker():
    blk = ReactionBlock(1)
    blk._rows[0].smiles.setText("[*]CC[*]")
    blk._rows[0]._on_changed()
    assert "×n" in blk._rows[0].maps_lb.text()
    assert blk._rows[0].is_polymer()

    blk._rows[1].smiles.setText("O")
    blk._rows[1]._on_changed()
    assert "×n" not in blk._rows[1].maps_lb.text()
    assert not blk._rows[1].is_polymer()


def test_one_chain_length_keeps_both_sides_matched():
    """The whole point: the product never drifts from the reactant length."""
    blk = ReactionBlock(1)
    unit = "[*]OCCCCOC(=O)CCC(=O)[*]"
    blk._rows[0].smiles.setText(unit)
    blk._rows[1].smiles.setText("[O:1]=C1OC(=O)C=C1")   # small co-reactant
    blk._prod_rows[0].smiles.setText(unit)

    for n in (1, 3, 10):
        blk.chain_n.setValue(n)
        _, reactants, products = blk.values()
        repeat = "OCCCCOC(=O)CCC(=O)"
        assert reactants[0].count(repeat) == n
        assert products[0].count(repeat) == n
        assert reactants[0] == products[0]
        # The small molecule is never repeated.
        assert reactants[1] == "[O:1]=C1OC(=O)C=C1"


def test_values_can_return_the_unexpanded_smiles():
    blk = ReactionBlock(1)
    blk._rows[0].smiles.setText("[*]CC[*]")
    blk.chain_n.setValue(4)
    assert blk.values(expand=True)[1][0] == "CCCCCCCC"
    assert blk.values(expand=False)[1][0] == "[*]CC[*]"


def test_chain_length_is_per_reaction():
    tab = ReactionSchemeTab()
    b1 = tab._blocks[0]
    b2 = tab._add_reaction()
    b1._rows[0].smiles.setText("[*]CC[*]")
    b2._rows[0].smiles.setText("[*]CC[*]")
    b1.chain_n.setValue(2)
    b2.chain_n.setValue(5)
    assert b1.values()[1][0] == "CC" * 2
    assert b2.values()[1][0] == "CC" * 5
