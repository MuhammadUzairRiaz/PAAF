"""Logic tests for the Amorphous builder tab.

PyQt5 is not importable in every environment, so these skip cleanly when it is
missing. They exercise the tab's *logic* — the composition it hands to the
grower, the mode switch, the file manifest — not its pixels.

The property worth protecting here is that what the user sees on screen is
what actually gets built: the preview table, the box edge and the file list
must all come from the same solved composition the worker is handed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 not installed")
pytest.importorskip("rdkit", reason="composition needs RDKit for masses")

from PyQt5.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from paaf.gui.amorphous_tab import AmorphousTab, ComponentRow  # noqa: E402
from paaf.gui.main_window import NAV_ITEMS  # noqa: E402

PE = "[*]CC[*]"
PS = "[*]CC([*])c1ccccc1"


def _tab():
    t = AmorphousTab()
    t.target_beads.setValue(2000)
    return t


# ------------------------------------------------------------------ layout
def test_it_has_its_own_four_steps():
    t = _tab()
    titles = [t.steps.tabText(i) for i in range(t.steps.count())]
    print(f"\n  steps: {titles}")
    assert titles == ["1 · Composition", "2 · Force field", "3 · Relax",
                      "4 · Export"]


def test_it_is_reachable_from_the_sidebar():
    names = [n for n, _ in NAV_ITEMS]
    print(f"\n  nav items: {names}")
    assert "Amorphous cell" in names


def test_it_starts_with_a_usable_two_component_blend():
    t = _tab()
    smiles = [r.smiles.text() for r in t._rows]
    print(f"\n  default components: {smiles}")
    assert len(t._rows) == 2
    assert all(s.strip() for s in smiles)


# ------------------------------------------------------------------- solve
def test_solving_produces_a_composition_the_grower_can_consume():
    t = _tab()
    t._solve()
    comp = t._composition
    assert comp is not None
    specs = comp.grow_specs()
    print(f"\n  {comp.total_chains} chains, {comp.total_beads} beads, "
          f"box {comp.box_edge_a:.2f} Å")
    for s in specs:
        print(f"    {s.name:4s} n={s.n_chains:3d} DP={s.degree_of_polymerisation}"
              f" mass={s.mass_amu:7.2f} backbone_atoms={s.backbone_atoms}")
    assert specs and all(s.repeat_unit for s in specs)
    assert all(s.mass_amu and s.mass_amu > 0 for s in specs)
    assert t.b_build.isEnabled()


def test_the_preview_table_shows_requested_against_realised():
    """The realised composition must be visible, not implied."""
    t = _tab()
    t._rows[0].wt.setValue(70.0)
    t._rows[1].wt.setValue(30.0)
    t._solve()
    rows = t.preview.rowCount()
    got = [[t.preview.item(r, c).text() for c in range(6)] for r in range(rows)]
    print("\n  " + "\n  ".join(" | ".join(r) for r in got))
    assert rows == 2
    header = [t.preview.horizontalHeaderItem(c).text() for c in range(6)]
    assert "wt% asked" in header and "wt% realised" in header


def test_editing_an_input_re_solves_rather_than_going_stale():
    """What is on screen must always describe what would be built.

    The earlier design blanked the preview and disabled Build until the user
    pressed Solve again. That is safe but unhelpful: the solver is pure
    arithmetic and takes microseconds, so an edit re-solves immediately and
    the numbers on screen are never stale.
    """
    t = _tab()
    t._solve()
    before = t._composition.total_beads
    t._rows[0].dp.setValue(80)
    after = t._composition.total_beads
    print(f"\n  DP 50 -> {before} beads;  DP 80 -> {after} beads")
    assert t._composition is not None, "the solve went stale instead of updating"
    assert t.b_build.isEnabled()
    assert t.preview.rowCount() == len(t._rows)
    assert after != before, "the preview did not respond to the edit"


def test_density_change_resizes_the_box_immediately():
    t = _tab()
    t._solve()
    edge_before = t._composition.box_edge_a
    t.density.setValue(1.10)
    edge_after = t._composition.box_edge_a
    print(f"\n  box {edge_before:.2f} A -> {edge_after:.2f} A on density change")
    assert t._composition is not None
    assert edge_after < edge_before


# -------------------------------------------------------------- both modes
def test_weight_mode_and_count_mode_give_different_counts():
    t = _tab()
    t._solve()
    solved = list(t._composition.chain_counts)

    t.rb_counts.setChecked(True)
    for r, n in zip(t._rows, (5, 3)):
        r.n_chains.setValue(n)
    t._solve()
    explicit = list(t._composition.chain_counts)
    print(f"\n  solved from wt%: {solved}")
    print(f"  entered by hand : {explicit}")
    assert explicit == [5, 3]
    assert solved != explicit


def test_mode_switch_shows_the_right_field():
    t = _tab()
    t.rb_weight.setChecked(True)
    assert t._rows[0].wt.isVisibleTo(t._rows[0])
    assert not t._rows[0].n_chains.isVisibleTo(t._rows[0])
    t.rb_counts.setChecked(True)
    assert not t._rows[0].wt.isVisibleTo(t._rows[0])
    assert t._rows[0].n_chains.isVisibleTo(t._rows[0])


def test_target_size_field_is_only_shown_in_weight_mode():
    """Cell size drives the solver; in count mode the counts already do."""
    t = _tab()
    t.rb_counts.setChecked(True)
    assert not t.beads_row.isVisibleTo(t)
    t.rb_weight.setChecked(True)
    assert t.beads_row.isVisibleTo(t)


# ---------------------------------------------------------------- components
def test_components_can_be_added_and_removed():
    t = _tab()
    t._add_component("PEO", "[*]OCC[*]", 10.0)
    assert len(t._rows) == 3
    t._remove_component(t._rows[-1])
    assert len(t._rows) == 2


def test_the_last_component_cannot_be_removed():
    t = _tab()
    while len(t._rows) > 1:
        t._remove_component(t._rows[-1])
    t._remove_component(t._rows[0])
    print(f"\n  components remaining after trying to remove the last: "
          f"{len(t._rows)}")
    assert len(t._rows) == 1


def test_a_row_reports_the_repeat_unit_mass_as_you_type():
    """The weight arithmetic should be legible before anything is solved."""
    row = ComponentRow(1)
    row.smiles.setText(PE)
    print(f"\n  {PE} -> {row.info.text()}")
    assert "28.05" in row.info.text()
    assert "2 skel" in row.info.text()


def test_a_bad_smiles_is_marked_not_swallowed():
    row = ComponentRow(1)
    row.smiles.setText("this is not a smiles")
    print(f"\n  garbage SMILES -> {row.info.text()!r}")
    assert row.info.text() == "unreadable"


def test_known_polymers_get_their_literature_ris_model():
    """PE/PP/PS/PMMA have real parameterisations; anything else is generic
    and the grower says so in its notes."""
    assert ComponentRow._guess_ris(PE) == "PE"
    assert ComponentRow._guess_ris(PS) == "PS"
    assert ComponentRow._guess_ris("[*]OCC[*]") == ""


# ------------------------------------------------------------------ export
def test_the_file_manifest_tracks_the_atomistic_switch():
    t = _tab()
    t.atomistic.setChecked(False)
    beads_only = [t.files.item(r, 0).text() for r in range(t.files.rowCount())]
    t.atomistic.setChecked(True)
    with_atoms = [t.files.item(r, 0).text() for r in range(t.files.rowCount())]
    print(f"\n  coarse-grained only : {beads_only}")
    print(f"  with back-mapping   : {with_atoms}")
    assert not any("atomistic" in f for f in beads_only)
    assert any("atomistic" in f for f in with_atoms)
    assert any("cell.data" in f for f in with_atoms)


def test_scan_depth_warning_appears_only_when_scanning_is_on():
    t = _tab()
    # The "off" guidance must be there on first view, without touching the
    # control — setValue(0) on a fresh spin box emits nothing.
    assert t.scan_warn.text(), "the depth-0 guidance never rendered"
    off = t.scan_warn.text()
    t.scan_depth.setValue(2)
    on = t.scan_warn.text()
    print(f"\n  depth 0: {off[:60]}…")
    print(f"  depth 2: {on[:60]}…")
    assert "Rosenbluth" not in off
    assert "Rosenbluth" in on


def test_force_field_page_names_the_route_it_will_take():
    """DL_FIELD and Moltemplate are different paths and the user should know
    which one their choice implies before pressing Build."""
    t = _tab()
    seen = set()
    for i in range(t.ff_combo.count()):
        t.ff_combo.setCurrentIndex(i)
        text = t.ff_route.text()
        assert text.startswith("Route:")
        seen.add("DL_FIELD" in text)
    print(f"\n  both routes reachable from the combo: {seen == {True, False}}")
    assert seen == {True, False}


def test_settings_provider_is_explicit_not_guessed():
    """Reaching up through self.window() fails silently on reparent, and the
    failure mode is exporting with the wrong force field."""
    t = _tab()
    assert t.default_out_dir() == ""
    t.set_settings_provider(lambda: {"out_dir": "/tmp/paaf_out",
                                     "project": "myblend",
                                     "ff_key": "oplsaa"})
    print(f"\n  out dir from the pipeline: {t.default_out_dir()}")
    # Nested under the project so a cell called "cell" cannot collide with
    # the pipeline's own artefacts.
    assert t.default_out_dir() == "/tmp/paaf_out/myblend/cells"
    assert t.out_dir.text() == "/tmp/paaf_out/myblend/cells"


def test_solve_with_no_smiles_does_not_crash_or_produce_a_composition():
    t = _tab()
    for r in t._rows:
        r.smiles.setText("")
    t._solve()
    assert t._composition is None
    assert not t.b_build.isEnabled()


# ====================================================== live composition
def test_weight_percentages_stay_linked_at_100():
    """Raising one component's share has to come out of the others."""
    t = _tab()
    t._rows[0].wt.setValue(70.0)
    got = [r.wt.value() for r in t._rows]
    print(f"\n  set row 1 to 70 -> {[f'{w:.2f}' for w in got]}")
    assert abs(sum(got) - 100.0) < 1e-6
    assert abs(got[1] - 30.0) < 1e-6


def test_linking_preserves_the_ratio_between_the_others():
    """With three components the untouched two keep their relative split.

    The starting split has to be set atomically. Typing 50, then 30, then 20
    does NOT land on 50/30/20, because each entry rescales the ones already
    entered — which is the whole point of the linking and also the reason
    set_weights exists.
    """
    t = _tab()
    t._add_component("PEO", "[*]OCC[*]", 0.0)
    t.set_weights([50.0, 30.0, 20.0])
    assert [round(r.wt.value(), 2) for r in t._rows] == [50.0, 30.0, 20.0]

    t._rows[0].wt.setValue(60.0)
    got = [r.wt.value() for r in t._rows]
    print(f"\n  50/30/20, then row 1 -> 60 gives "
          f"{[f'{w:.2f}' for w in got]}")
    assert abs(sum(got) - 100.0) < 1e-6
    # 30:20 was 3:2 and must still be 3:2 -> 24 and 16.
    assert abs(got[1] / got[2] - 1.5) < 1e-6
    assert abs(got[1] - 24.0) < 1e-6 and abs(got[2] - 16.0) < 1e-6


def test_set_weights_normalises_whatever_it_is_given():
    """The escape hatch for an exact split, including un-normalised input."""
    t = _tab()
    t._add_component("PEO", "[*]OCC[*]", 0.0)
    t.set_weights([3.0, 1.0, 1.0])           # a 3:1:1 ratio, not percentages
    got = [round(r.wt.value(), 2) for r in t._rows]
    print(f"\n  3:1:1 -> {got}")
    assert got == [60.0, 20.0, 20.0]


def test_linking_does_not_recurse_or_drift():
    t = _tab()
    for _ in range(20):
        t._rows[0].wt.setValue(t._rows[0].wt.value() + 1.0)
    got = [r.wt.value() for r in t._rows]
    print(f"\n  after 20 successive edits -> {[f'{w:.2f}' for w in got]}")
    assert abs(sum(got) - 100.0) < 1e-6


def test_each_row_shows_the_chains_and_realised_weight_it_gets():
    """The consequence of a setting must be visible without pressing Solve."""
    t = _tab()
    t._rows[0].wt.setValue(70.0)
    texts = [r.outcome.text() for r in t._rows]
    print(f"\n  row readouts: {texts}")
    assert all("ch" in x and "wt%" in x for x in texts), texts


def test_changing_dp_moves_the_realised_weight_without_touching_the_request():
    """A DP change alters chain mass, so the same request lands differently.

    This is the case that is easy to get wrong: the requested 50/50 is
    untouched, but PBS chains become far heavier than PE chains, so the
    integer chain counts that best approximate 50/50 change.
    """
    t = _tab()
    t._rows[0].smiles.setText(PE)
    t._rows[1].smiles.setText("[*]OCCCCOC(=O)CCC(=O)[*]")   # PBS
    t._rows[0].wt.setValue(50.0)

    t._rows[0].dp.setValue(30)
    t._rows[1].dp.setValue(50)
    before = [r.outcome.text() for r in t._rows]
    requested_before = [r.wt.value() for r in t._rows]

    t._rows[0].dp.setValue(2)
    t._rows[1].dp.setValue(5)
    after = [r.outcome.text() for r in t._rows]
    requested_after = [r.wt.value() for r in t._rows]

    print(f"\n  DP 30/50 -> {before}")
    print(f"  DP  2/5  -> {after}")
    assert requested_before == requested_after, \
        "changing DP must not silently rewrite the requested weights"
    assert before != after, "the realised outcome did not respond to DP"


def test_chain_count_mode_shows_the_weight_percent_those_counts_produce():
    t = _tab()
    t.rb_counts.setChecked(True)
    t._rows[0].smiles.setText(PE)
    t._rows[1].smiles.setText(PS)
    t._rows[0].n_chains.setValue(5)
    t._rows[1].n_chains.setValue(3)
    texts = [r.outcome.text() for r in t._rows]
    print(f"\n  5 PE : 3 PS -> {texts}")
    assert "5 ch" in texts[0] and "3 ch" in texts[1]
    # PS chains are much heavier, so 3 of them outweigh 5 PE chains.
    pe_wt = float(texts[0].split("·")[1].strip().replace(" wt%", ""))
    ps_wt = float(texts[1].split("·")[1].strip().replace(" wt%", ""))
    assert ps_wt > pe_wt


def test_editing_a_target_updates_the_box_without_pressing_solve():
    t = _tab()
    t.density.setValue(0.90)
    edge_a = t._composition.box_edge_a
    t.density.setValue(1.20)
    edge_b = t._composition.box_edge_a
    print(f"\n  0.90 g/cm3 -> {edge_a:.2f} A;  1.20 g/cm3 -> {edge_b:.2f} A")
    assert edge_b < edge_a, "a denser cell must be a smaller box"


def test_cell_size_is_editable_and_changes_the_chain_count():
    t = _tab()
    t.target_beads.setValue(1000)
    small = t._composition.total_chains
    t.target_beads.setValue(8000)
    big = t._composition.total_chains
    print(f"\n  1000 beads -> {small} chains;  8000 beads -> {big} chains")
    assert big > small


def test_temperature_is_editable():
    t = _tab()
    t.temperature.setValue(350.0)
    assert abs(t.temperature.value() - 350.0) < 1e-9
    assert t.temperature.isEnabled()


def test_the_targets_are_all_spin_boxes_the_user_can_type_into():
    """They looked like labels; they are not."""
    from PyQt5.QtWidgets import QAbstractSpinBox
    t = _tab()
    for name in ("density", "target_beads", "temperature", "tolerance",
                 "scan_depth", "seed"):
        w = getattr(t, name)
        assert isinstance(w, QAbstractSpinBox), name
        assert not w.isReadOnly(), name
        assert w.isEnabled(), name


# ====================================================== copolymer components
ISOPRENE = "[*]C/C=C(C)\\C[*]"
EPOXIDE = "[*]CC1(C)OC1C[*]"
PBS = "[*]OCCCCOC(=O)CCC(=O)[*]"


def _monomers(fr_epoxide=0.25):
    from paaf.cell.sequence import Monomer
    return [Monomer(ISOPRENE, 1.0 - fr_epoxide, "isoprene"),
            Monomer(EPOXIDE, fr_epoxide, "epoxide")]


def test_a_row_can_become_a_copolymer():
    t = _tab()
    row = t._rows[0]
    assert not row.is_copolymer
    row.set_copolymer(_monomers(), "random")
    print(f"\n  readout: {row.info.text()!r}")
    assert row.is_copolymer
    assert "monomers" in row.info.text()
    # The SMILES box is locked, because the chemistry now lives in the dialog.
    assert not row.smiles.isEnabled()


def test_a_copolymer_row_can_go_back_to_a_homopolymer():
    t = _tab()
    row = t._rows[0]
    row.set_copolymer(_monomers(), "random")
    row.set_copolymer(None)
    assert not row.is_copolymer
    assert row.smiles.isEnabled()


def test_the_copolymer_reaches_the_grow_spec():
    """The whole point: what the row holds must survive into the grower."""
    t = _tab()
    t._rows[0].name.setText("ENR")
    t._rows[0].set_copolymer(_monomers(0.25), "random")
    t._rows[1].name.setText("PBS")
    t._rows[1].smiles.setText(PBS)
    t._solve()

    specs = t._composition.grow_specs()
    by_name = {s.name: s for s in specs}
    print(f"\n  specs: "
          f"{[(s.name, s.is_copolymer, s.arrangement) for s in specs]}")
    assert by_name["ENR"].is_copolymer
    assert len(by_name["ENR"].monomers) == 2
    assert by_name["ENR"].arrangement == "random"
    assert not by_name["PBS"].is_copolymer


def test_the_solver_uses_the_composition_weighted_mass():
    """A copolymer has no single repeat unit; the mean is used for sizing."""
    t = _tab()
    t._rows[0].set_copolymer(_monomers(0.25), "random")
    t._rows[1].smiles.setText(PBS)
    t._solve()
    enr = t._composition.components[0]
    expected = 0.75 * 68.12 + 0.25 * 84.12
    print(f"\n  mean repeat unit {enr.unit_mass:.2f} g/mol "
          f"(expected {expected:.2f})")
    assert abs(enr.unit_mass - expected) < 0.05


def test_an_epoxide_row_is_not_flagged_as_beads_only():
    """Its ring spans one backbone bond, so it CAN be rebuilt as all-atom."""
    row = ComponentRow(1)
    row.smiles.setText(EPOXIDE)
    print(f"\n  epoxide readout: {row.info.text()!r}")
    assert "beads only" not in row.info.text()
    assert "84.1" in row.info.text()


def test_a_pet_row_is_still_flagged_as_beads_only():
    """Its benzene spans three backbone bonds — refused."""
    row = ComponentRow(1)
    row.smiles.setText("[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]")
    print(f"\n  PET readout: {row.info.text()!r}")
    assert "beads only" in row.info.text()


def test_the_copolymer_dialog_offers_an_enr_preset():
    from paaf.gui.copolymer_dialog import CopolymerDialog
    dlg = CopolymerDialog()
    mons = dlg.monomers()
    print(f"\n  preset: "
          f"{[(m.name, round(m.fraction, 2)) for m in mons]}")
    print(f"  arrangement: {dlg.arrangement()}")
    assert len(mons) == 2
    assert abs(mons[0].fraction - 0.75) < 1e-6
    assert dlg.arrangement() == "random"


def test_the_dialog_can_clear_back_to_a_homopolymer():
    from paaf.gui.copolymer_dialog import CopolymerDialog
    dlg = CopolymerDialog()
    dlg._make_homopolymer()
    assert dlg.monomers() is None


def test_is_copolymer_is_a_property_everywhere():
    """A method here reads as truthy and silently breaks every row.

    `if self.is_copolymer:` on a bound method is always True, so every
    component was treated as a copolymer and every mass came out
    "unreadable". The three classes must agree that it is a property.
    """
    from paaf.cell.composition import Component
    from paaf.cell.grow import GrowSpec

    row = ComponentRow(1)
    comp = Component(name="x", repeat_unit=PE)
    spec = GrowSpec(repeat_unit=PE)
    for obj in (row, comp, spec):
        value = obj.is_copolymer
        print(f"  {type(obj).__name__}.is_copolymer -> {value!r}")
        assert isinstance(value, bool), (
            f"{type(obj).__name__}.is_copolymer is {type(value).__name__}, "
            f"not a bool — `if x.is_copolymer:` would always be true")
        assert value is False


def test_the_dialog_round_trips_fractions_without_inflating_them():
    """The table shows mole PERCENT; the API takes fractions summing to 1.

    Without normalising, editing an existing copolymer would multiply every
    fraction by 100 each time it was opened and saved.
    """
    from paaf.gui.copolymer_dialog import CopolymerDialog

    first = CopolymerDialog().monomers()
    again = CopolymerDialog(first, "random").monomers()
    print(f"\n  pass 1: {[round(m.fraction, 4) for m in first]}")
    print(f"  pass 2: {[round(m.fraction, 4) for m in again]}")
    assert abs(sum(m.fraction for m in first) - 1.0) < 1e-9
    assert abs(sum(m.fraction for m in again) - 1.0) < 1e-9
    for a, b in zip(first, again):
        assert abs(a.fraction - b.fraction) < 1e-9


def test_the_dialog_accepts_any_consistent_ratio():
    """3:1 typed as 3 and 1, or as 75 and 25, must mean the same thing."""
    from paaf.gui.copolymer_dialog import CopolymerDialog
    from paaf.cell.sequence import Monomer

    a = CopolymerDialog([Monomer(ISOPRENE, 3.0, "iso"),
                         Monomer(EPOXIDE, 1.0, "ep")]).monomers()
    b = CopolymerDialog([Monomer(ISOPRENE, 0.75, "iso"),
                         Monomer(EPOXIDE, 0.25, "ep")]).monomers()
    print(f"\n  3:1   -> {[round(m.fraction, 4) for m in a]}")
    print(f"  75:25 -> {[round(m.fraction, 4) for m in b]}")
    assert abs(a[0].fraction - 0.75) < 1e-9
    assert abs(b[0].fraction - 0.75) < 1e-9
