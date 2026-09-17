"""Typing an amorphous cell chain by chain (paaf.cell.chain_typing).

DL_FIELD stops on large cells ("Error: too many C identified." for 200 PE
chains). These tests use a DL_FIELD stand-in with the same file shapes and,
where it matters, the same kind of size limit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paaf.cell import chain_typing as ct
from paaf.cell.chain_typing import ChainTypingError, group_chains, type_cell_by_chain
from paaf.structure import Atom, Molecule

from _fake_dlfield import FakeDLField


def _ethane_chain(offset, n_c=4):
    """A zig-zag alkane: C and H in DL_FIELD-plausible geometry."""
    atoms, bonds = [], []
    for k in range(n_c):
        c = np.array([1.26 * k, 0.8 * (k % 2), 0.0]) + offset
        ci = len(atoms)
        atoms.append(("C", c))
        if k:
            bonds.append((ci - 3, ci))
        atoms.append(("H", c + [0.0, 0.0, 1.09]))
        atoms.append(("H", c + [0.0, 0.0, -1.09]))
        bonds += [(ci, ci + 1), (ci, ci + 2)]
    return atoms, bonds


def _cell(n_chains=3, n_c=4, extra=None, box=30.0):
    mol = Molecule()
    per = []
    for c in range(n_chains):
        atoms, bonds = _ethane_chain(np.array([2.0, 3.0 + 5.0 * c, 4.0]), n_c)
        base = len(mol.atoms)
        for el, xyz in atoms:
            mol.atoms.append(Atom(len(mol.atoms), el, np.asarray(xyz, float)))
        mol.bonds += [(base + i, base + j, 1.0) for i, j in bonds]
        per.append(len(atoms))
    return mol, per, np.array([box, box, box])


# ------------------------------------------------------------------ graph
def test_identical_chains_share_one_template():
    mol, per, _ = _cell(n_chains=5)
    templates, which = group_chains(mol, per, ["PE"] * 5)
    assert len(templates) == 1
    assert which == [0] * 5
    assert templates[0].name == "PE"


def test_different_chains_get_numbered_templates():
    mol, per, _ = _cell(n_chains=2, n_c=4)
    atoms, bonds = _ethane_chain(np.array([2.0, 20.0, 4.0]), 6)
    base = len(mol.atoms)
    for el, xyz in atoms:
        mol.atoms.append(Atom(len(mol.atoms), el, np.asarray(xyz, float)))
    mol.bonds += [(base + i, base + j, 1.0) for i, j in bonds]
    per.append(len(atoms))
    templates, which = group_chains(mol, per, ["PE", "PE", "PE"])
    assert which == [0, 0, 1]
    assert [t.name for t in templates] == ["PE_1", "PE_2"]


def test_a_bond_between_chains_is_refused():
    mol, per, _ = _cell(n_chains=2)
    mol.bonds.append((0, per[0], 1.0))
    with pytest.raises(ChainTypingError, match="different chains"):
        group_chains(mol, per)


def test_unwrap_joins_a_chain_split_by_the_boundary():
    xyz = np.array([[29.5, 1.0, 1.0], [0.8, 1.0, 1.0], [2.3, 1.0, 1.0]])
    out = ct._unwrap(xyz, [(0, 1), (1, 2)], np.array([30.0, 30.0, 30.0]))
    assert np.allclose(np.linalg.norm(np.diff(out, axis=0), axis=1), [1.3, 1.5])


# ----------------------------------------------------------------- typing
def _type(tmp_path, mol, per, dims, fake, engines=("lammps",), names=None):
    return type_cell_by_chain(mol, per, dims, "opls2005_dl", None,
                              tmp_path / "_typing", tmp_path, run_dlfield=fake,
                              engines=engines, chain_names=names)


def _data_atoms(path):
    sec = ct._parse_data(path)
    return sec


def test_every_chain_is_typed_from_one_dlfield_run(tmp_path):
    mol, per, dims = _cell(n_chains=6)
    fake = FakeDLField()
    res = _type(tmp_path, mol, per, dims, fake)
    assert len(fake.calls) == 1
    assert fake.calls[0]["n_atoms"] == per[0]          # one representative
    sec = _data_atoms(res.data_file)
    assert len(sec["Atoms"]) == len(mol.atoms)
    assert len(sec["Bonds"]) == len(mol.bonds)
    mols = {int(l.split()[1]) for l in sec["Atoms"]}
    assert mols == set(range(1, 7))
    # the cell's own coordinates, in the cell's own order
    first = sec["Atoms"][per[0]].split()
    assert int(first[0]) == per[0] + 1
    assert np.allclose([float(v) for v in first[4:7]], mol.atoms[per[0]].xyz)
    text = res.input_file.read_text()
    assert "read_data cell.data" in text
    assert "pair_coeff 1 2 lj/cut/coul/long" in text
    assert "group XYZ" not in text


def test_a_cell_bigger_than_dlfield_limits_is_still_typed(tmp_path):
    """The reported failure: DL_FIELD refuses the whole cell, not a chain."""
    mol, per, dims = _cell(n_chains=40)
    fake = FakeDLField(max_atoms=100)          # whole cell: 480 atoms
    res = _type(tmp_path, mol, per, dims, fake, engines=("lammps", "gromacs"))
    assert res.data_file.exists() and res.gro_file.exists()
    assert all(c["n_atoms"] <= 100 for c in fake.calls)
    top = res.top_file.read_text()
    assert "POLY    40" in top
    assert res.gro_file.read_text().splitlines()[1] == str(len(mol.atoms))


def test_many_batches_give_the_same_cell_as_one(tmp_path, monkeypatch):
    mol, per, dims = _cell(n_chains=2, n_c=3)
    atoms, bonds = _ethane_chain(np.array([2.0, 20.0, 4.0]), 5)
    base = len(mol.atoms)
    for el, xyz in atoms:
        mol.atoms.append(Atom(len(mol.atoms), el, np.asarray(xyz, float)))
    mol.bonds += [(base + i, base + j, 1.0) for i, j in bonds]
    per.append(len(atoms))

    one = _type(tmp_path / "one", mol, per, dims, FakeDLField())
    monkeypatch.setattr(ct, "BATCH_ATOMS", 5)
    fake = FakeDLField()
    many = _type(tmp_path / "many", mol, per, dims, fake)
    assert len(fake.calls) == 2
    assert one.data_file.read_text() == many.data_file.read_text()
    assert one.input_file.read_text() == many.input_file.read_text()


def test_dlfield_error_is_reported_by_name(tmp_path):
    mol, per, dims = _cell(n_chains=2)
    with pytest.raises(ChainTypingError, match="too many C identified"):
        _type(tmp_path, mol, per, dims, FakeDLField(max_atoms=3))


def test_a_bond_dlfield_did_not_see_stops_the_typing(tmp_path):
    mol, per, dims = _cell(n_chains=2)
    with pytest.raises(ChainTypingError, match="perceived"):
        _type(tmp_path, mol, per, dims, FakeDLField(drop_bond=True))


def test_representatives_are_far_apart_in_the_batch(tmp_path):
    mol, per, dims = _cell(n_chains=1)
    atoms, bonds = _ethane_chain(np.array([2.0, 3.2, 4.0]), 6)   # overlaps chain 1
    base = len(mol.atoms)
    for el, xyz in atoms:
        mol.atoms.append(Atom(len(mol.atoms), el, np.asarray(xyz, float)))
    mol.bonds += [(base + i, base + j, 1.0) for i, j in bonds]
    per.append(len(atoms))
    res = _type(tmp_path, mol, per, dims, FakeDLField())   # would mis-bond if close
    assert len(ct._parse_data(res.data_file)["Bonds"]) == len(mol.bonds)


def test_gromacs_output_lists_molecules_and_splits_the_itp(tmp_path):
    mol, per, dims = _cell(n_chains=4)
    res = _type(tmp_path, mol, per, dims, FakeDLField(), engines=("gromacs",),
                names=["PE"] * 4)
    top = res.top_file.read_text()
    assert '#include "cell.itp"' in top and "PE    4" in top
    itp = res.itp_files[0].read_text()
    assert itp.count("[ moleculetype ]") == 1
    atoms_block = itp.split("[ atoms ]")[1].split("[ bonds ]")[0]
    rows = [l for l in atoms_block.splitlines() if l.strip() and not l.startswith(";")]
    assert len(rows) == per[0]
    gro = res.gro_file.read_text().splitlines()
    assert int(gro[1]) == len(mol.atoms)
    assert gro[-1].split() == ["3.00000", "3.00000", "3.00000"]
