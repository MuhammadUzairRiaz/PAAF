"""Every library polymer, built as a chain, must reach a geometry that a
distance-based typer (dl_field) reads as the right molecule.

Uses the numpy chain backend (mbuild is not needed) so it runs anywhere.
"""
from __future__ import annotations

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from paaf.builder import list_library  # noqa: E402
from paaf.chain_builder import build_chain  # noqa: E402
from paaf.geometry_repair import ensure_clean_geometry, find_clashes  # noqa: E402
from paaf.monomer import Monomer, _capped_from_wildcards  # noqa: E402
from paaf.structure import Atom, Molecule  # noqa: E402


def _monomer(poly_smiles: str, name: str):
    got = _capped_from_wildcards(poly_smiles)
    if got is None:
        return None
    m, caps, links = got
    if AllChem.EmbedMolecule(m, randomSeed=3) < 0:
        return None
    xyz = m.GetConformer().GetPositions()
    mol = Molecule(name=name)
    for a in m.GetAtoms():
        mol.atoms.append(Atom(index=a.GetIdx(), element=a.GetSymbol(),
                              xyz=np.array(xyz[a.GetIdx()])))
    for b in m.GetBonds():
        mol.bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                          b.GetBondTypeAsDouble()))
    return Monomer(mol, links[0], links[1], [caps[0]], [caps[1]], name=name)


def _records():
    out = []
    for r in list_library("all"):
        if r.smiles.count("[*]") == 2:
            out.append(r)
    return out


@pytest.mark.parametrize("rec", _records(), ids=lambda r: r.pid or r.name)
def test_library_chain_is_clean_after_repair(rec):
    mon = _monomer(rec.smiles, rec.name)
    if mon is None:
        pytest.skip("RDKit cannot embed this repeat unit")
    chain = build_chain([mon], n=6, backend="simple")
    ok = ensure_clean_geometry(chain, seed=1)
    c, b = find_clashes(chain)
    assert ok and not c and not b, (rec.name, c[:3], b[:3])


def test_records_found():
    assert len(_records()) > 50
