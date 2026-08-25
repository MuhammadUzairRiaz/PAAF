"""Tests for the wildcard-based head/tail identification used to fix PBS
(and other polyester) polymer builds.

Covers:
* the SMILES parser that reads the two ``[*]``-neighbour elements
* the terminal-atom picker that maps those elements onto a 3D molecule
"""
from __future__ import annotations

import numpy as np
import pytest


def test_wildcard_neighbour_symbols_covers_common_polymers():
    from paaf.monomer import _wildcard_neighbour_symbols
    cases = {
        # trailing-wildcard patterns
        "[*]OCCCCOC(=O)CCC(=O)[*]":            ("O", "C"),   # PBS
        "[*]NCCCCCCC(=O)[*]":                  ("N", "C"),   # nylon-7-ish
        # simple diradical
        "[*]CC[*]":                            ("C", "C"),   # PE
        # in-branch wildcard
        "[*]CC([*])c1ccccc1":                  ("C", "C"),   # PS
        "[*]CC([*])Cl":                        ("C", "C"),   # PVC
        "[*]CC([*])O":                         ("C", "C"),   # PVA
        "[*]CC([*])C":                         ("C", "C"),   # PP
        "[*]CC([*])(C)C(=O)OC":                ("C", "C"),   # PMMA
        # branch on the trailing side
        "[*]OCCCCOC(=O)c1ccc(cc1)C(=O)[*]":    ("O", "C"),   # PBAT
    }
    for smi, expected in cases.items():
        assert _wildcard_neighbour_symbols(smi) == expected, smi


def test_pick_head_tail_symmetric_HO_polyester_end():
    """HO-…-COOH monomer should have head/tail routed through the terminal
    Os (not through some CH2 hydrogen). That's what PAAF was getting wrong
    for PBS before the fix."""
    from paaf.monomer import _pick_head_tail_by_element
    from paaf.structure import Atom, Molecule

    atoms = [
        Atom(0,  "O", np.array([-2.0, 0.0, 0.0])),
        Atom(1,  "H", np.array([-2.9, 0.0, 0.0])),  # terminal -OH proton (left)
        Atom(2,  "C", np.array([-1.0, 0.0, 0.0])),
        Atom(3,  "H", np.array([-1.0, 1.0, 0.0])),
        Atom(4,  "H", np.array([-1.0,-1.0, 0.0])),
        Atom(5,  "C", np.array([ 0.0, 0.0, 0.0])),
        Atom(6,  "H", np.array([ 0.0, 1.0, 0.0])),
        Atom(7,  "H", np.array([ 0.0,-1.0, 0.0])),
        Atom(8,  "C", np.array([ 1.0, 0.0, 0.0])),
        Atom(9,  "O", np.array([ 1.0, 1.0, 0.0])),  # C=O (not terminal)
        Atom(10, "O", np.array([ 2.0, 0.0, 0.0])),
        Atom(11, "H", np.array([ 2.9, 0.0, 0.0])),  # terminal -OH proton (right)
    ]
    bonds = [
        (0, 1, 1), (0, 2, 1), (2, 3, 1), (2, 4, 1),
        (2, 5, 1), (5, 6, 1), (5, 7, 1), (5, 8, 1),
        (8, 9, 2), (8, 10, 1), (10, 11, 1),
    ]
    mol = Molecule(atoms=atoms, bonds=bonds, name="minipes")
    picked = _pick_head_tail_by_element(mol, "O", "O")
    assert picked is not None
    head, tail, head_h, tail_h = picked
    assert {head, tail} == {0, 10},   f"head/tail should be the terminal Os, got {(head, tail)}"
    assert {head_h[0], tail_h[0]} == {1, 11}, \
        f"head_h/tail_h should be the -OH protons, got {(head_h[0], tail_h[0])}"


def test_pick_head_tail_asymmetric_ends():
    """When head_elem != tail_elem (e.g. PBS's O + C after wildcard analysis),
    each end has exactly one terminal candidate and picking is deterministic."""
    from paaf.monomer import _pick_head_tail_by_element
    from paaf.structure import Atom, Molecule

    # HO-CH2-C(=O)-H  (minimal glycolic-aldehyde stub)
    atoms = [
        Atom(0, "O", np.array([-2.0, 0.0, 0.0])),
        Atom(1, "H", np.array([-2.9, 0.0, 0.0])),  # terminal -OH
        Atom(2, "C", np.array([-1.0, 0.0, 0.0])),
        Atom(3, "H", np.array([-1.0, 1.0, 0.0])),
        Atom(4, "H", np.array([-1.0,-1.0, 0.0])),
        Atom(5, "C", np.array([ 0.0, 0.0, 0.0])),  # terminal carbonyl C
        Atom(6, "O", np.array([ 0.0, 1.0, 0.0])),  # =O
        Atom(7, "H", np.array([ 0.9, 0.0, 0.0])),  # aldehydic H
    ]
    bonds = [(0,1,1),(0,2,1),(2,3,1),(2,4,1),(2,5,1),(5,6,2),(5,7,1)]
    mol = Molecule(atoms=atoms, bonds=bonds, name="mini")
    picked = _pick_head_tail_by_element(mol, "O", "C")
    assert picked is not None
    head, tail, head_h, tail_h = picked
    assert (head, tail) == (0, 5)
    assert head_h == [1] and tail_h == [7]
