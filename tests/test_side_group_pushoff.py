"""Relieving side-group overlaps must be fast enough to actually be used.

The step built a neighbour grid and then ignored it: for every mobile atom it
measured the distance to ALL atoms, in a Python loop, up to 120 times. On a
10,560-atom PE/PS cell that is ~10^9 distance evaluations — slow enough to
look like a hang, so it got switched off.

Turning it off is what broke typing. Back-mapped side groups stayed
overlapping; DL_FIELD perceives bonds from geometry, read those contacts as
bonds, and reported cyclopropyl rings and alkenes in a saturated PE/PS blend
before stopping on an untypable "aliphatic" atom. One bug, two symptoms.

The contacts are now found with a periodic KD-tree in one call per iteration.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from paaf.cell.backmap import relax_substituents
from paaf.structure import Atom, Molecule


def _cell(n, edge, seed=0, clash_every=50):
    """Atoms on a loose lattice, with deliberate close pairs.

    ``clash_every=None`` makes a clean lattice — the slice trick still hits
    index 1 for any finite stride, which quietly gave the "no overlaps" case
    an overlap and made its assertion fail for the wrong reason.
    """
    g = int(round(n ** (1 / 3))) + 1
    pts = np.array([(i, j, k) for i in range(g) for j in range(g)
                    for k in range(g)])[:n]
    xyz = pts * (edge / g) + 0.01
    if clash_every:
        xyz[1::clash_every] = \
            xyz[0::clash_every][:len(xyz[1::clash_every])] + 0.6
    mol = Molecule(atoms=[Atom(index=i, element="C", xyz=xyz[i].copy())
                          for i in range(n)], bonds=[], name="t")
    return mol, np.array([edge] * 3, dtype=float)


def test_overlaps_are_actually_relieved():
    mol, dims = _cell(4000, 60.0)
    before, after = relax_substituents(mol, np.zeros(4000, bool), dims,
                                       target=2.2)
    print(f"\n  {before:.3f} A -> {after:.3f} A")
    assert before < 1.5, "the test did not create an overlap to fix"
    assert after >= 2.19, "overlaps were not relieved to the target"


def test_a_cell_the_size_of_a_real_one_finishes_quickly():
    """The measured failure was minutes-to-hours; this bounds it."""
    mol, dims = _cell(10560, 45.4)
    flags = np.zeros(10560, bool); flags[::4] = True
    t0 = time.time()
    relax_substituents(mol, flags, dims, target=2.2, iterations=120)
    elapsed = time.time() - t0
    print(f"\n  10,560 atoms, 120 iterations: {elapsed:.1f} s")
    assert elapsed < 60, f"still too slow to use ({elapsed:.0f} s)"


def test_the_backbone_does_not_move():
    """Its positions are what growth accepted; only side groups may shift."""
    mol, dims = _cell(2000, 40.0)
    flags = np.zeros(2000, bool); flags[::2] = True
    before = np.array([a.xyz.copy() for a in mol.atoms])
    relax_substituents(mol, flags, dims, target=2.2)
    after = np.array([a.xyz for a in mol.atoms])
    moved = np.abs(after[flags] - before[flags]).max()
    print(f"\n  largest pinned-atom movement: {moved:.2e} A")
    assert moved < 1e-9, "a pinned backbone atom moved"


def test_a_cell_with_no_overlaps_is_left_alone():
    mol, dims = _cell(1000, 80.0, clash_every=None)
    before_xyz = np.array([a.xyz.copy() for a in mol.atoms])
    before, after = relax_substituents(mol, np.zeros(1000, bool), dims,
                                       target=2.2)
    assert np.allclose(before_xyz, [a.xyz for a in mol.atoms])


def test_bonded_pairs_are_not_pushed_apart():
    """A bond is SUPPOSED to be shorter than the contact target."""
    n = 600
    xyz = np.zeros((n, 3))
    for i in range(n):
        xyz[i] = [(i // 2) * 4.0, 0.0, 0.0]
        if i % 2:
            xyz[i][1] = 1.09                      # an H on its C
    mol = Molecule(atoms=[Atom(index=i, element="C" if i % 2 == 0 else "H",
                               xyz=xyz[i].copy()) for i in range(n)],
                   bonds=[(i, i + 1, 1.0) for i in range(0, n - 1, 2)],
                   name="t")
    dims = np.array([1200.0, 60.0, 60.0])
    relax_substituents(mol, np.zeros(n, bool), dims, target=2.2)
    after = np.array([a.xyz for a in mol.atoms])
    lengths = np.linalg.norm(after[1::2] - after[0::2], axis=1)
    print(f"\n  bond lengths after: {lengths.min():.3f}-{lengths.max():.3f} A")
    assert lengths.max() < 1.3, "bonded C-H was stretched toward the target"


def test_exactly_coincident_atoms_are_separated():
    """Measured on a real cell: 333 H pairs at 0.000 A separation.

    Two atoms at the same point have no separation vector, so d/r is zero and
    the push is zero — they stay welded together however many iterations run.
    DL_FIELD perceives bonds from geometry and reported cyclopropyl rings and
    alkenes in a saturated PE/PS blend because of it. A direction has to be
    invented; any one will do.
    """
    n = 400
    xyz = np.zeros((n, 3))
    for i in range(n):
        xyz[i] = [(i // 2) * 6.0, 0.0, 0.0]        # pairs, exactly coincident
    mol = Molecule(atoms=[Atom(index=i, element="H", xyz=xyz[i].copy())
                          for i in range(n)], bonds=[], name="t")
    dims = np.array([1300.0, 60.0, 60.0])
    before, after = relax_substituents(mol, np.zeros(n, bool), dims,
                                       target=2.2)
    print(f"\n  {before:.3f} A -> {after:.3f} A")
    assert before < 1e-6, "the test did not create coincident atoms"
    assert after > 1.0, "coincident atoms were never separated"


def test_collapsed_bonds_are_not_all_restored_in_the_same_direction():
    """The bug that welded 333 hydrogen pairs together.

    ``_restore_bonds`` used a fixed [1, 0, 0] fallback whenever a bond had
    collapsed to zero length. A carbon whose two hydrogens had both collapsed
    onto it got both restored along +x — to exactly the same point. Measured
    on a real PE/PS cell: 333 pairs of same-carbon hydrogens at 0.000 A,
    which DL_FIELD read as bonds and reported as cyclopropyl rings in a
    saturated blend.
    """
    from paaf.cell.backmap import _restore_bonds

    # One carbon with two hydrogens, all three at the same point.
    xyz = np.zeros((3, 3))
    bi = np.array([0, 0]); bj = np.array([1, 2])
    target = np.array([1.09, 1.09])
    fixed = np.array([True, False, False])       # carbon pinned
    _restore_bonds(xyz, bi, bj, target, fixed, np.array([100.0] * 3))

    sep = np.linalg.norm(xyz[1] - xyz[2])
    print(f"\n  H-H separation after restore: {sep:.3f} A")
    assert np.linalg.norm(xyz[1] - xyz[0]) == pytest.approx(1.09, abs=1e-3)
    assert np.linalg.norm(xyz[2] - xyz[0]) == pytest.approx(1.09, abs=1e-3)
    assert sep > 0.1, "both hydrogens were restored to the same point"


def test_restore_is_reproducible():
    """Two identical inputs must give identical output, random or not."""
    from paaf.cell.backmap import _restore_bonds

    def run():
        xyz = np.zeros((3, 3))
        _restore_bonds(xyz, np.array([0, 0]), np.array([1, 2]),
                       np.array([1.09, 1.09]),
                       np.array([True, False, False]),
                       np.array([100.0] * 3))
        return xyz
    assert np.allclose(run(), run())
