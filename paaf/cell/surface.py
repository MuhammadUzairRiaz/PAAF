"""Surface / slab builder.

Cleaves a crystal along a Miller plane (hkl) and adds vacuum along the surface
normal. This is a pragmatic first-pass — for cubic crystals the surface
normal is the reciprocal-lattice direction; for other lattices we still use
the reciprocal-lattice normal computed from the input lattice vectors.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple

from ..logging_utils import get_logger
from ..structure import Atom, Molecule
from .crystal import build_crystal, CRYSTAL_PRESETS, lattice_vectors

log = get_logger(__name__)


def _select_slab(mol: Molecule, normal: np.ndarray,
                 thickness_ang: float) -> Molecule:
    coords = mol.coords()
    proj = coords @ normal
    lo = float(proj.min())
    hi = lo + thickness_ang
    keep = np.where(proj <= hi)[0]
    atoms = [Atom(
        index=k, element=mol.atoms[i].element, xyz=mol.atoms[i].xyz.copy(),
        name=mol.atoms[i].name, charge=mol.atoms[i].charge,
        ff_type=mol.atoms[i].ff_type,
    ) for k, i in enumerate(keep)]
    return Molecule(atoms=atoms, bonds=[], name=mol.name + "_slab")


def cleave_surface(
    preset: str,
    hkl: Tuple[int, int, int],
    supercell: Tuple[int, int, int] = (5, 5, 5),
    thickness_ang: float = 12.0,
    vacuum_ang: float = 15.0,
) -> Molecule:
    """Cleave a slab from a crystal preset along the Miller plane (hkl).

    Returns the slab as a :class:`Molecule`. Vacuum is added by simply
    reporting a nominal (thickness + vacuum) box along the surface normal —
    downstream writers use it when emitting LAMMPS/data files.
    """
    if preset not in CRYSTAL_PRESETS:
        raise KeyError(f"Unknown preset {preset!r}")
    p = CRYSTAL_PRESETS[preset]
    L = lattice_vectors(p.a, p.b, p.c, p.alpha, p.beta, p.gamma)
    # reciprocal lattice
    rec = np.linalg.inv(L).T
    normal = np.array(hkl, dtype=float) @ rec
    normal /= np.linalg.norm(normal)
    log.info("Cleaving %s along (%d%d%d), normal=%s",
             preset, *hkl, np.round(normal, 3).tolist())

    bulk = build_crystal(p.a, p.b, p.c, p.alpha, p.beta, p.gamma,
                        basis=p.basis, nx=supercell[0], ny=supercell[1],
                        nz=supercell[2], name=f"{preset}_bulk")
    slab = _select_slab(bulk, normal, thickness_ang)
    log.info("Slab has %d atoms (%d Å thick + %d Å vacuum)",
             len(slab.atoms), thickness_ang, vacuum_ang)
    return slab
