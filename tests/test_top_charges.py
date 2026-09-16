"""Net-charge summary for GROMACS topologies.

The LAMMPS ``.data`` files have been checked since ``data_file_charges``;
``top_file_charges`` does the same for the ``.top`` that grompp reads. The
arithmetic is not the same: a ``.top`` states charges once per moleculetype
and multiplies them by the ``[ molecules ]`` counts, and the moleculetype
itself often lives in an ``#include``d ``.itp``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from paaf.cell.cell_export import top_file_charges


_NEUTRAL_TOP = """\
; a hand-written two-moleculetype system
[ defaults ]
; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ
  1       3          yes        0.5      0.5

[ moleculetype ]
; Name    nrexcl
ETH   3

[ atoms ]
;   nr  type  resnr  residu  atom  cgnr  charge  mass
  1  opls_135  1  ETH  C1  1  -0.1200  12.011
  2  opls_140  1  ETH  H1  2   0.0600   1.008
  3  opls_140  1  ETH  H2  3   0.0600   1.008

[ bonds ]
  1  2  1
  1  3  1

[ moleculetype ]
ION   1

[ atoms ]
  1  opls_401  1  ION  NA  1   1.0000  22.990

[ moleculetype ]
ANI   1

[ atoms ]
  1  opls_402  1  ANI  CL  1  -1.0000  35.453

[ system ]
two moleculetypes

[ molecules ]
ETH   4
ION   2
ANI   2
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_neutral_system_sums_to_zero_over_the_molecules_list(tmp_path):
    top = _write(tmp_path, "system.top", _NEUTRAL_TOP)
    n_atoms, total, biggest = top_file_charges(top)
    # 4 x ETH (3 atoms) + 2 x ION + 2 x ANI
    assert n_atoms == 4 * 3 + 2 + 2
    assert total == pytest.approx(0.0, abs=1e-9)
    assert biggest == pytest.approx(1.0)


def test_a_changed_charge_shows_up_as_the_system_total(tmp_path):
    # +0.03 e on each of the four ETH copies = +0.12 e for the system.
    text = _NEUTRAL_TOP.replace("  2  opls_140  1  ETH  H1  2   0.0600   1.008",
                                "  2  opls_140  1  ETH  H1  2   0.0900   1.008")
    top = _write(tmp_path, "system.top", text)
    n_atoms, total, _ = top_file_charges(top)
    assert n_atoms == 16
    assert total == pytest.approx(0.12, abs=1e-9)


def test_all_zero_charges_are_reported_as_zero_not_as_missing(tmp_path):
    text = """\
[ moleculetype ]
UA   3

[ atoms ]
  1  CH2  1  UA  C1  1  0.0000  14.027
  2  CH2  1  UA  C2  2  0.0000  14.027

[ molecules ]
UA   5
"""
    top = _write(tmp_path, "flat.top", text)
    n_atoms, total, biggest = top_file_charges(top)
    assert (n_atoms, total, biggest) == (10, 0.0, 0.0)


def test_moleculetype_in_an_included_itp_is_counted(tmp_path):
    (tmp_path / "gromacs1.itp").write_text("""\
[ moleculetype ]
; the packed box keeps its moleculetype in its own .itp
PE   3

[ atoms ]
  1  opls_135  1  PE  C1  1  -0.0600  12.011
  2  opls_140  1  PE  H1  2   0.0600   1.008
""")
    top = _write(tmp_path, "packed_box.top", """\
[ defaults ]
  1  3  yes  0.5  0.5

#include "gromacs1.itp"

[ system ]
packed box

[ molecules ]
PE   7
""")
    n_atoms, total, biggest = top_file_charges(top)
    assert n_atoms == 14
    assert total == pytest.approx(0.0, abs=1e-9)
    assert biggest == pytest.approx(0.06)


def test_a_missing_force_field_include_does_not_stop_the_parse(tmp_path):
    top = _write(tmp_path, "gromacs.top", """\
#include "oplsaa.ff/forcefield.itp"

[ moleculetype ]
MOL   3

[ atoms ]
  1  opls_135  1  MOL  C1  1   0.2500  12.011
  2  opls_140  1  MOL  H1  2  -0.2500   1.008

[ molecules ]
MOL   1
""")
    assert top_file_charges(top) == (2, pytest.approx(0.0, abs=1e-9),
                                     pytest.approx(0.25))


def test_no_atoms_section_anywhere_reads_as_nothing_to_report(tmp_path):
    top = _write(tmp_path, "empty.top", "[ defaults ]\n  1  3  yes  0.5  0.5\n")
    assert top_file_charges(top) is None
    assert top_file_charges(tmp_path / "does_not_exist.top") is None


def test_an_include_cycle_terminates(tmp_path):
    (tmp_path / "a.itp").write_text('#include "b.itp"\n')
    (tmp_path / "b.itp").write_text("""\
#include "a.itp"

[ moleculetype ]
X   3

[ atoms ]
  1  cX  1  X  C1  1  0.1000  12.011

[ molecules ]
X   1
""")
    top = _write(tmp_path, "cyc.top", '#include "a.itp"\n')
    assert top_file_charges(top) == (1, pytest.approx(0.1), pytest.approx(0.1))


# ------------------------------------------------------------------ pipeline
def _ethylene_mol2(tmp_path: Path) -> Path:
    """A minimal ethylene monomer file for a two-unit polyethylene build."""
    import numpy as np

    from paaf import structure
    from paaf.structure import Atom, Molecule

    atoms = [
        Atom(0, "C", np.array([0.00, 0.00, 0.00])),
        Atom(1, "C", np.array([1.34, 0.00, 0.00])),
        Atom(2, "H", np.array([-0.50, 0.90, 0.00])),
        Atom(3, "H", np.array([-0.50, -0.90, 0.00])),
        Atom(4, "H", np.array([1.84, 0.90, 0.00])),
        Atom(5, "H", np.array([1.84, -0.90, 0.00])),
    ]
    bonds = [(0, 1, 2), (0, 2, 1), (0, 3, 1), (1, 4, 1), (1, 5, 1)]
    path = tmp_path / "ethylene.mol2"
    structure.write(Molecule(atoms=atoms, bonds=bonds, name="ETH"), path)
    return path


def test_a_gromacs_run_reports_the_topology_charges(tmp_path):
    """End to end: engine='gromacs' prints Charges: for a .top and the
    result carries net_charge_gromacs."""
    from paaf.config import Config, MonomerSpec
    from paaf.pipeline import run_pipeline

    cfg = Config()
    cfg.output_dir = str(tmp_path)
    cfg.project_name = "system"          # -> system.top, one of the candidates
    cfg.engine = "gromacs"
    cfg.monomers = [MonomerSpec(file=str(_ethylene_mol2(tmp_path)), name="ETH",
                                head=1, tail=2, head_h=3, tail_h=5)]
    cfg.chain.n_monomers = 2
    cfg.chain.backend = "simple"
    cfg.optimizer.enabled = False
    cfg.run_moltemplate = False
    cfg.force_field.key = "oplsaa"

    seen: list = []
    result = run_pipeline(cfg, progress=seen.append)

    log = "\n".join(seen)
    charge_lines = [ln for ln in seen if ln.startswith("Charges:")]
    assert charge_lines, f"no charge summary in:\n{log}"
    assert any(".top" in ln for ln in charge_lines), charge_lines
    assert "net_charge_gromacs" in result
    assert result["net_charge_gromacs"] is not None
