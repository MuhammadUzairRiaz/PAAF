"""The soft push-off inputs must be valid, minimal LAMMPS.

Why it exists: the geometric push-off stalls — measured, 1,167 non-bonded
pairs still under 1.65 A after 150 iterations, hydrogens trapped between
pinned backbone atoms. DL_FIELD bonds anything that close and refuses the
cell. The standard preparation-stage push-off (Kremer & Grest: pair_style
soft ramped under nve/limit) resolves contacts by concerted motion and needs
NO atom types, which is what breaks the circularity: typing needs the
contacts gone, a force-field run needs typing.

The run itself needs a LAMMPS binary, so what is pinned here is everything
short of it: the data file, the script, and the graceful refusal without.
"""
from __future__ import annotations

import numpy as np
import pytest

from paaf.cell.soft_pushoff import soft_pushoff, write_pushoff_inputs
from paaf.structure import Atom, Molecule


def _cell(n=60):
    atoms, bonds = [], []
    for i in range(n):
        e = "C" if i % 3 == 0 else "H"
        atoms.append(Atom(index=i, element=e,
                          xyz=np.array([i * 1.0, 0.0, 0.0])))
    for i in range(0, n - 2, 3):
        bonds += [(i, i + 1, 1.0), (i, i + 2, 1.0)]
    return Molecule(atoms=atoms, bonds=bonds, name="t")


def test_the_data_file_is_complete_and_untyped(tmp_path):
    mol = _cell()
    data, inp = write_pushoff_inputs(mol, [70.0, 70.0, 70.0], tmp_path)
    text = data.read_text()
    print(f"\n{text[:400]}")
    assert f"{len(mol.atoms)} atoms" in text
    assert f"{len(mol.bonds)} bonds" in text
    assert "2 atom types" in text          # C and H, nothing force-field
    assert "Masses" in text and "Atoms # molecular" in text


def test_bond_classes_get_their_own_rest_length(tmp_path):
    """C-C and C-H are different classes with different medians."""
    mol = _cell()
    data, _ = write_pushoff_inputs(mol, [70.0] * 3, tmp_path)
    lines = data.read_text().splitlines()
    n_bond_types = next(int(l.split()[0]) for l in lines
                        if l.endswith("bond types"))
    assert n_bond_types == 1               # only C-H bonds in this toy
    mol2 = _cell()
    mol2.bonds.append((0, 3, 1.0))         # one C-C bond
    data2, inp2 = write_pushoff_inputs(mol2, [70.0] * 3, tmp_path / "b")
    assert "2 bond types" in data2.read_text()
    assert "bond_coeff 2" in inp2.read_text()


def test_the_script_is_the_kremer_grest_recipe(tmp_path):
    _, inp = write_pushoff_inputs(_cell(), [70.0] * 3, tmp_path)
    text = inp.read_text()
    print(f"\n{text}")
    assert "pair_style soft" in text
    assert "nve/limit" in text
    assert "ramp(" in text, "the repulsion must be ramped, not slammed on"
    assert "special_bonds lj 0.0 0.0 1.0" in text, \
        "1-3 pairs are held by ANGLE terms, not repulsion — repelling them " \
        "splayed every H-C-H to 127 degrees and DL_FIELD read the carbons " \
        "as sp2"
    assert "angle_style harmonic" in text, \
        "without angle terms nothing holds H-C-H at 107 degrees"
    assert "write_dump" in text


def test_atoms_are_wrapped_into_the_box_first(tmp_path):
    mol = _cell()
    for a in mol.atoms:
        a.xyz = a.xyz + 200.0              # far outside
    data, _ = write_pushoff_inputs(mol, [70.0] * 3, tmp_path)
    xs = [float(l.split()[3]) for l in data.read_text().splitlines()
          if l and l.split()[0].isdigit() and len(l.split()) == 6]
    assert xs and max(xs) < 70.0 and min(xs) >= 0.0


def test_no_lammps_means_a_clean_refusal_and_untouched_coordinates(tmp_path,
                                                                   monkeypatch):
    import paaf.cell.relax as relax
    monkeypatch.setattr(relax, "find_lammps", lambda *a, **k: None)
    mol = _cell()
    before = [a.xyz.copy() for a in mol.atoms]
    said = []
    ok = soft_pushoff(mol, [70.0] * 3, tmp_path, emit=said.append)
    assert ok is False
    assert any("skipped" in m for m in said)
    assert all(np.allclose(b, a.xyz) for b, a in zip(before, mol.atoms))
