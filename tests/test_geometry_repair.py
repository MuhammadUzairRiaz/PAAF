"""Clashed chain geometries are detected and rebuilt before typing."""
import numpy as np
import pytest

from paaf.geometry_repair import (ensure_clean_geometry, find_clashes,
                                  rebuild_coordinates)
from paaf.structure import Atom, Molecule

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402


def _from_smiles(smi: str, name="m", collapse=False) -> Molecule:
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=1)
    xyz = m.GetConformer().GetPositions()
    mol = Molecule(name=name)
    for a in m.GetAtoms():
        p = xyz[a.GetIdx()] * (0.3 if collapse else 1.0)   # collapse => clashes
        mol.atoms.append(Atom(index=a.GetIdx(), element=a.GetSymbol(), xyz=np.array(p)))
    for b in m.GetBonds():
        mol.bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                          b.GetBondTypeAsDouble()))
    return mol


def test_clean_molecule_has_no_clashes():
    mol = _from_smiles("CCCCC(CCCCC)CCCC")
    c, b = find_clashes(mol)
    assert c == [] and b == []


def test_collapsed_geometry_is_detected():
    mol = _from_smiles("CCCC", collapse=True)
    c, b = find_clashes(mol)
    assert c and b          # everything overlaps and bonds are 0.3x too short


def test_rebuild_gives_clean_geometry_and_keeps_topology():
    mol = _from_smiles("CC(CCCCC)CC(CCCCC)CC(CCCCC)CC(CCCCC)C", "p1h", collapse=True)
    before = list(mol.bonds)
    assert not ensure_clean_geometry(mol) is None
    c, b = find_clashes(mol)
    assert c == [] and b == []
    assert mol.bonds == before
    # bonded C–C ~1.53, C–H ~1.09
    xyz = mol.coords(); el = mol.elements()
    for i, j, _ in mol.bonds:
        d = np.linalg.norm(xyz[i] - xyz[j])
        if el[i] == el[j] == "C":
            assert 1.45 < d < 1.62
        else:
            assert 1.0 < d < 1.15


def test_side_chain_overlap_like_mbuild_is_repaired():
    """Two heptene units laid on one axis so the pentyl branches overlap."""
    mol = _from_smiles("CC(CCCCC)CC(CCCCC)C", "two_units")
    xyz = mol.coords()
    # push the second half's side chain onto the first
    el = mol.elements()
    for k in range(len(el) // 2, len(el)):
        mol.atoms[k].xyz = xyz[k - len(el) // 2] + np.array([0.4, 0.0, 0.0])
    c, _ = find_clashes(mol)
    assert len(c) > 5
    assert rebuild_coordinates(mol, seed=3)
    c, b = find_clashes(mol)
    assert c == [] and b == []


def test_aromatic_and_heteroatoms_embed():
    mol = _from_smiles("CC(c1ccccc1)CC(c1ccccc1)CC(=O)OC", "ps_like", collapse=True)
    assert ensure_clean_geometry(mol)


def test_pipeline_calls_geometry_check():
    from pathlib import Path
    import paaf.pipeline as pl
    src = Path(pl.__file__).read_text()
    assert "ensure_clean_geometry(chain" in src
    assert src.index("ensure_clean_geometry(chain") < src.index("Force field: ")
