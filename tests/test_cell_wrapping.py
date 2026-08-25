"""The exported cell must lie inside the box it declares.

Back-mapping UNWRAPS each chain so it is continuous across the periodic
boundary — a local frame cannot be built from three points if the chain jumps
a box length between them. But the file written afterwards has to be wrapped
back, and it was not.

Measured on a real 53.8 A cell: the coordinates spanned 78.7 x 99.4 x 87.9 A.
DL_FIELD applies the cell from the control file, folds those coordinates back
in, and atoms 99 A apart in the file land on top of one another. Carbons came
out with 5 to 10 neighbours and typing stopped with "atype = aliphatic".

This is why lowering the build density changed nothing: the atoms were never
inside the box to begin with, so there was no density at which they would be.
"""
from __future__ import annotations

import numpy as np
import pytest

from paaf.structure import Atom, Molecule


def _wrap(xyz, dims):
    """The rule the exporter applies, isolated so it can be checked."""
    return xyz - np.floor(xyz / dims) * dims


def test_wrapping_brings_every_atom_inside():
    dims = np.array([53.8, 53.8, 53.8])
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-60, 140, size=(2000, 3))
    w = _wrap(xyz, dims)
    print(f"\n  span {np.round(xyz.max(0) - xyz.min(0), 1)} -> "
          f"{np.round(w.max(0) - w.min(0), 1)}")
    assert (w >= 0).all() and (w < dims).all()


def test_wrapping_moves_atoms_by_whole_box_lengths_only():
    """Anything else would change the structure, not just its representation."""
    dims = np.array([53.8, 40.0, 61.0])
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-200, 200, size=(500, 3))
    shift = xyz - _wrap(xyz, dims)
    n = shift / dims
    assert np.allclose(n, np.round(n)), "a shift was not a whole box length"


def test_an_already_wrapped_cell_is_untouched():
    dims = np.array([50.0, 50.0, 50.0])
    rng = np.random.default_rng(2)
    xyz = rng.uniform(0, 50, size=(500, 3))
    assert np.allclose(_wrap(xyz, dims), xyz)


def test_bond_lengths_survive_under_minimum_image():
    """Wrapping must not tear a molecule apart — measured the right way.

    A wrapped bond looks enormous in raw coordinates and is correct under
    minimum image, which is how every MD code reads a periodic file.
    """
    dims = np.array([30.0, 30.0, 30.0])
    xyz = np.array([[29.5, 15.0, 15.0], [30.6, 15.0, 15.0]])   # 1.1 A bond
    w = _wrap(xyz, dims)
    d = w[1] - w[0]
    d -= dims * np.round(d / dims)
    print(f"\n  bond after wrap, minimum image: {np.linalg.norm(d):.3f} A")
    assert np.linalg.norm(d) == pytest.approx(1.1, abs=1e-6)
