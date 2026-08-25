"""A bond threaded through a ring must be freed by rotating the ring.

The clash-score rotation cannot see this defect: a bond through the ring's
CENTRE touches nothing, scores zero, and is left interlocked. Reseeding lost
to arithmetic — with ~19,000 bonds and ~300 rings, every seed at every
workable density threaded something (measured: 9, 6, then 29-before-rotation
across eight seeds). The unthreader tests the intersection itself and turns
the ring about its own attachment bond, an exact rigid move, until nothing
passes through.
"""
from __future__ import annotations

import numpy as np
import pytest

from paaf.cell.backmap import count_speared_rings, unthread_rings
from paaf.structure import Atom, Molecule


def _threaded_cell():
    """A hexagonal 'phenyl' on a stub, with a foreign bond through it."""
    atoms, bonds = [], []
    # backbone C (pinned) at origin; ipso ring atom above it
    atoms.append(Atom(index=0, element="C", xyz=np.array([0.0, 0.0, 0.0])))
    ring0 = len(atoms)
    for k in range(6):                        # ring in the y-z plane, r=1.4
        ang = 2 * np.pi * k / 6
        atoms.append(Atom(index=ring0 + k, element="C",
                          xyz=np.array([0.0, 1.5 + 1.4 * np.cos(ang),
                                        1.4 * np.sin(ang)])))
    bonds.append((0, ring0, 1.0))             # attachment
    for k in range(6):
        bonds.append((ring0 + k, ring0 + (k + 1) % 6, 1.0))
    # the intruder: a 2-atom bond passing through the ring centre along x
    a = len(atoms)
    atoms.append(Atom(index=a, element="C", xyz=np.array([-1.0, 1.5, 0.0])))
    atoms.append(Atom(index=a + 1, element="C", xyz=np.array([1.0, 1.5, 0.0])))
    bonds.append((a, a + 1, 1.0))
    mol = Molecule(atoms=atoms, bonds=bonds, name="t")
    flags = np.zeros(len(atoms), dtype=bool)
    flags[0] = True                            # backbone pinned
    flags[a] = flags[a + 1] = True             # intruder is another backbone
    return mol, flags


def test_a_threaded_ring_is_freed():
    mol, flags = _threaded_cell()
    dims = np.array([60.0, 60.0, 60.0])
    n0, _ = count_speared_rings(mol, dims)
    assert n0 == 1, "the fixture must start threaded"
    freed, stuck = unthread_rings(mol, flags, dims)
    n1, _ = count_speared_rings(mol, dims)
    print(f"\n  threaded {n0} -> {n1}   (freed {freed}, stuck {stuck})")
    assert n1 == 0
    assert freed == 1 and stuck == 0


def test_the_ring_is_rotated_rigidly_not_deformed():
    mol, flags = _threaded_cell()
    dims = np.array([60.0, 60.0, 60.0])
    ring = list(range(1, 7))
    xyz0 = np.array([a.xyz.copy() for a in mol.atoms])
    d0 = [np.linalg.norm(xyz0[i] - xyz0[j]) for i in ring for j in ring]
    unthread_rings(mol, flags, dims)
    xyz1 = np.array([a.xyz for a in mol.atoms])
    d1 = [np.linalg.norm(xyz1[i] - xyz1[j]) for i in ring for j in ring]
    assert np.allclose(d0, d1, atol=1e-9), "internal ring geometry changed"
    # attachment bond length preserved too
    assert np.linalg.norm(xyz1[1] - xyz1[0]) == pytest.approx(
        np.linalg.norm(xyz0[1] - xyz0[0]), abs=1e-9)


def test_the_intruder_and_backbone_do_not_move():
    mol, flags = _threaded_cell()
    dims = np.array([60.0, 60.0, 60.0])
    fixed = [0, 7, 8]
    before = np.array([mol.atoms[i].xyz.copy() for i in fixed])
    unthread_rings(mol, flags, dims)
    after = np.array([mol.atoms[i].xyz for i in fixed])
    assert np.allclose(before, after, atol=1e-12)


def test_a_clean_cell_is_untouched():
    mol, flags = _threaded_cell()
    dims = np.array([60.0, 60.0, 60.0])
    # move the intruder far away first
    mol.atoms[7].xyz = np.array([30.0, 30.0, 30.0])
    mol.atoms[8].xyz = np.array([31.0, 30.0, 30.0])
    xyz0 = np.array([a.xyz.copy() for a in mol.atoms])
    freed, stuck = unthread_rings(mol, flags, dims)
    assert freed == 0 and stuck == 0
    assert np.allclose(xyz0, [a.xyz for a in mol.atoms])
