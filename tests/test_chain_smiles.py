"""Tests for the SMILES-based chain builder (port of create_chain.py).

The heavy graph-isomorphism build needs rdkit+mbuild+networkx+openbabel,
so those are only smoke-tested when the deps are importable. The SMILES
string preparation is dependency-free and always tested.
"""
from __future__ import annotations

import pytest


def test_smiles_prep_matches_create_chain():
    from paaf.chain_smiles import _rdkit_and_mbuild_smiles
    cases = {
        "[*]OCCCCOC(=O)CCC(=O)[*]": ("*OCCCCOC(=O)CCC(=O)*", "OCCCCOC(=O)CCC(=O)"),
        "[*]CC[*]":                 ("*CC*",                 "CC"),
        "[*]CC([*])c1ccccc1":       ("*CC(*)c1ccccc1",       "CCc1ccccc1"),
        "[*]CC([*])Cl":             ("*CC(*)Cl",             "CCCl"),
    }
    for poly, (rd, mb) in cases.items():
        got_rd, got_mb = _rdkit_and_mbuild_smiles(poly)
        assert got_rd == rd, poly
        assert got_mb == mb, poly


def test_copolymer_sequence_to_letters():
    """The explicit monomer-index sequence must map to mBuild letters with
    one letter per DISTINCT monomer that actually appears, preserving order."""
    from paaf.chain_smiles import _seq_letters
    # alternating AB
    unique, mapping, s = _seq_letters([0, 1, 0, 1])
    assert unique == [0, 1] and mapping == {0: "A", 1: "B"} and s == "ABAB"
    # block AABB
    _, _, s = _seq_letters([0, 0, 1, 1])
    assert s == "AABB"
    # only one monomer actually appears -> single letter (no phantom B)
    unique, mapping, s = _seq_letters([1, 1, 1])
    assert unique == [1] and mapping == {1: "A"} and s == "AAA"
    # three-monomer alternating
    _, _, s = _seq_letters([0, 1, 2, 0, 1, 2])
    assert s == "ABCABC"
    import pytest as _pt
    with _pt.raises(ValueError):
        _seq_letters([])


def _deps_ok() -> bool:
    from paaf.chain_smiles import available
    return available()


@pytest.mark.skipif(not _deps_ok(), reason="rdkit/mbuild/networkx/openbabel not installed")
def test_right_dummy_is_carbonyl():
    from paaf.chain_smiles import _right_dummy_is_carbonyl
    assert _right_dummy_is_carbonyl("[*]OCCCCOC(=O)CCC(=O)[*]") is True   # PBS
    assert _right_dummy_is_carbonyl("[*]CC[*]") is False                  # PE
    assert _right_dummy_is_carbonyl("[*]CC([*])c1ccccc1") is False        # PS


@pytest.mark.skipif(not _deps_ok(), reason="rdkit/mbuild/networkx/openbabel not installed")
def test_build_pbs_chain_has_tight_backbone():
    """End-to-end: a 3-unit PBS chain must have backbone C-C bonds near
    1.5 Å (not stretched to 2 Å like the old PDB-roundtrip path)."""
    import numpy as np
    from paaf.chain_smiles import build_chain_from_smiles
    mol = build_chain_from_smiles("[*]OCCCCOC(=O)CCC(=O)[*]", n=3, optimize=True)
    # Find all C-C bonds and check their lengths.
    coords = mol.coords()
    max_cc = 0.0
    for i, j, _ in mol.bonds:
        if mol.atoms[i].element == "C" and mol.atoms[j].element == "C":
            d = float(np.linalg.norm(coords[i] - coords[j]))
            max_cc = max(max_cc, d)
    assert max_cc < 1.7, f"backbone C-C too long: {max_cc:.3f} Å"
