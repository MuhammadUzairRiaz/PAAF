"""Amorphous cell steps: Next gated on Solve, Build on Export, output formats."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 not installed")
pytest.importorskip("rdkit", reason="composition needs RDKit for masses")

from PyQt5.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from paaf.cell.cell_export import CellExport, _write_nonhybrid  # noqa: E402
from paaf.gui.amorphous_tab import AmorphousTab  # noqa: E402


def _tab():
    t = AmorphousTab()
    t.target_beads.setValue(2000)
    return t


def _select_ff(t, key):
    i = t.ff_combo.findData(key)
    assert i >= 0, key
    t.ff_combo.setCurrentIndex(i)


def _files(t):
    return [t.files.item(r, 0).text() for r in range(t.files.rowCount())]


# ------------------------------------------------------------ navigation
def test_composition_step_offers_solve_and_next_not_build():
    t = _tab()
    assert t.steps.currentIndex() == 0
    assert not t.b_solve.isHidden()
    assert not t.b_next.isHidden()
    assert t.b_build.isHidden()
    assert t.b_back.isHidden()


def test_next_is_disabled_until_solve_is_pressed():
    t = _tab()
    # The tab re-solves quietly on edits, which must not unlock Next.
    t._invalidate()
    assert t._composition is not None
    assert not t.b_next.isEnabled()
    assert not any(t.steps.isTabEnabled(i) for i in (1, 2, 3))

    t.b_solve.click()

    assert t.b_next.isEnabled()
    assert all(t.steps.isTabEnabled(i) for i in (1, 2, 3))


def test_next_walks_the_steps_and_build_appears_only_on_export():
    t = _tab()
    t.b_solve.click()
    seen = []
    for _ in range(3):
        t.b_next.click()
        seen.append((t.steps.currentIndex(), t.b_build.isHidden(),
                     t.b_next.isHidden(), t.b_solve.isHidden()))
    assert seen == [(1, True, False, True),
                    (2, True, False, True),
                    (3, False, True, True)]
    assert not t.b_back.isHidden()
    t.b_back.click()
    assert t.steps.currentIndex() == 2


def test_an_unsolvable_composition_locks_next_again():
    t = _tab()
    t.b_solve.click()
    for r in t._rows:
        r.smiles.setText("")
    t._solve(quiet=True)
    assert not t.b_next.isEnabled()
    assert not t.steps.isTabEnabled(1)


def test_changing_build_density_requires_solving_again():
    t = _tab()
    t.b_solve.click()
    t.build_fraction.setValue(t.build_fraction.value() + 5)
    assert not t.b_next.isEnabled()
    t.b_solve.click()
    assert t.b_next.isEnabled()


def test_busy_build_disables_navigation():
    t = _tab()
    t.b_solve.click()
    t._set_busy(True)
    assert not t.b_next.isEnabled()
    t._set_busy(False)
    assert t.b_next.isEnabled()


# ---------------------------------------------------------- output formats
def test_format_choices_live_on_the_export_step():
    t = _tab()
    export_page = t.steps.widget(3)
    assert export_page.isAncestorOf(t.output_format)
    assert export_page.isAncestorOf(t.lammps_styles)
    assert not t.steps.widget(1).isAncestorOf(t.output_format)


def test_gromacs_lists_gro_top_and_itp_for_a_dlfield_force_field():
    t = _tab()
    _select_ff(t, "opls2005_dl")
    t.output_format.setCurrentIndex(t.output_format.findData("gromacs"))
    files = _files(t)
    assert "cell/cell.gro" in files
    assert "cell/cell.top" in files
    assert "cell/*.itp" in files
    assert "cell/cell.data" not in files


def test_non_hybrid_choice_adds_the_non_hybrid_files():
    t = _tab()
    _select_ff(t, "opls2005_dl")
    t.output_format.setCurrentIndex(t.output_format.findData("both"))
    assert "cell/non_hybrid/cell.in" not in _files(t)
    t.lammps_styles.setCurrentIndex(t.lammps_styles.findData("non_hybrid"))
    files = _files(t)
    assert "cell/non_hybrid/cell.in" in files
    assert "cell/non_hybrid/cell.data" in files
    assert "cell/cell.data" in files and "cell/cell.gro" in files


def test_moltemplate_force_field_says_gromacs_is_unavailable():
    t = _tab()
    _select_ff(t, "oplsaa")
    t.output_format.setCurrentIndex(t.output_format.findData("gromacs"))
    assert "only LAMMPS" in t.format_note.text()
    assert not t.lammps_styles.isEnabled()
    files = _files(t)
    assert "cell/cell.gro" not in files and "cell/cell.data" in files


def test_typing_off_disables_the_format_choices():
    t = _tab()
    t.run_typing.setChecked(False)
    assert not t.output_format.isEnabled()
    assert not t.lammps_styles.isEnabled()
    assert "typing is off" in t.format_note.text()


# --------------------------------------------------------- export backend
_IN = """units real
atom_style full
bond_style      hybrid harmonic
pair_style      hybrid lj/cut/coul/long 12.0
read_data cell.data
pair_coeff 1 1 lj/cut/coul/long 0.066 3.5
"""

_DATA = """LAMMPS data

2 atoms
1 bonds
1 atom types
1 bond types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

   1  12.011

Bond Coeffs

   1 harmonic   317.0    1.51

Atoms  # full

1 1 1 0.0 1.0 1.0 1.0
2 1 1 0.0 2.0 1.0 1.0

Bonds

1 1 1 2
"""


def test_export_writes_non_hybrid_copies_and_keeps_originals(tmp_path):
    (tmp_path / "cell.in").write_text(_IN)
    (tmp_path / "cell.data").write_text(_DATA)
    exp = CellExport(folder=tmp_path, typed_input=tmp_path / "cell.in",
                     typed_data=tmp_path / "cell.data")

    _write_nonhybrid(exp, tmp_path, lambda _m: None)

    nh_in = tmp_path / "non_hybrid" / "cell.in"
    nh_data = tmp_path / "non_hybrid" / "cell.data"
    assert exp.nonhybrid_files == [nh_in, nh_data]
    assert "hybrid" not in nh_in.read_text()
    assert "read_data cell.data" in nh_in.read_text()
    assert "1 317.0" in " ".join(nh_data.read_text().split())
    assert "hybrid harmonic" in (tmp_path / "cell.in").read_text()


def test_export_skips_non_hybrid_when_already_plain(tmp_path):
    (tmp_path / "cell.in").write_text(_IN.replace("hybrid ", ""))
    (tmp_path / "cell.data").write_text(_DATA)
    exp = CellExport(folder=tmp_path, typed_input=tmp_path / "cell.in",
                     typed_data=tmp_path / "cell.data")
    _write_nonhybrid(exp, tmp_path, lambda _m: None)
    assert exp.nonhybrid_files == []
    assert not (tmp_path / "non_hybrid").exists()
    assert any("plain" in m for m in exp.messages)
