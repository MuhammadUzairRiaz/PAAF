"""Crystal builder — build periodic crystals from a Bravais lattice + basis.

Public API:

- :func:`build_crystal(a, b, c, alpha, beta, gamma, basis, nx, ny, nz)` — the
  generic constructor. Basis is a list of ``(element, fractional_xyz)`` tuples.
- :func:`build_preset(name, nx, ny, nz)` — build one of the shipped presets.
- :data:`CRYSTAL_PRESETS` — dict of preset descriptors.

Shipped presets (all common textbook lattices used in classical MD tutorials):

    Simple cubic (SC): "Po"
    BCC:              "Fe", "W"
    FCC:              "Cu", "Al", "Au"
    Diamond:          "C_diamond", "Si_diamond"
    Zinc blende:      "GaAs", "ZnS_zb"
    Wurtzite:         "ZnS_wurtzite"
    NaCl:             "NaCl"
    CsCl:             "CsCl"
    Graphene:         "graphene"
    Graphite:         "graphite"
    Alpha-quartz:     "quartz_alpha"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..logging_utils import get_logger
from ..structure import Atom, Molecule, write

log = get_logger(__name__)


# ============================================================ lattice math
def lattice_vectors(a: float, b: float, c: float,
                    alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Return the 3x3 matrix of Cartesian lattice vectors as rows."""
    ar, br, cr = a, b, c
    al = math.radians(alpha); be = math.radians(beta); ga = math.radians(gamma)
    ax = ar
    bx = br * math.cos(ga)
    by = br * math.sin(ga)
    cx = cr * math.cos(be)
    cy = cr * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(cr * cr - cx * cx - cy * cy, 0.0))
    return np.array([[ax, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]], dtype=float)


# ============================================================ generic build
def build_crystal(
    a: float, b: float, c: float,
    alpha: float = 90.0, beta: float = 90.0, gamma: float = 90.0,
    basis: Sequence[Tuple[str, Sequence[float]]] = (),
    nx: int = 1, ny: int = 1, nz: int = 1,
    name: str = "crystal",
) -> Molecule:
    """Build a supercell.

    Parameters
    ----------
    a, b, c        Unit-cell edge lengths (Å).
    alpha, beta, gamma  Unit-cell angles (degrees).
    basis          list of (element, [fx, fy, fz]) — fractional coordinates.
    nx, ny, nz     supercell repetitions along each lattice vector.
    """
    if not basis:
        raise ValueError("Basis must contain at least one atom")
    L = lattice_vectors(a, b, c, alpha, beta, gamma)
    atoms: List[Atom] = []
    idx = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                origin_frac = np.array([i, j, k], dtype=float)
                for element, frac in basis:
                    frac = np.array(frac, dtype=float)
                    cart = (origin_frac + frac) @ L
                    atoms.append(Atom(
                        index=idx, element=element, xyz=cart,
                        name=f"{element}{idx + 1}",
                    ))
                    idx += 1
    # Store the supercell edge as the box; downstream can use for LAMMPS box lines.
    mol = Molecule(atoms=atoms, bonds=[], name=name)
    mol.source_path = None
    return mol


# ============================================================ presets
@dataclass(frozen=True)
class CrystalPreset:
    name: str
    description: str
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    basis: Tuple[Tuple[str, Tuple[float, float, float]], ...]


CRYSTAL_PRESETS: Dict[str, CrystalPreset] = {
    "Po":     CrystalPreset("Po", "α-Polonium (simple cubic)",
                            3.359, 3.359, 3.359, 90, 90, 90,
                            (("Po", (0.0, 0.0, 0.0)),)),
    "Fe":     CrystalPreset("Fe", "α-Iron (BCC)",
                            2.8665, 2.8665, 2.8665, 90, 90, 90,
                            (("Fe", (0.0, 0.0, 0.0)), ("Fe", (0.5, 0.5, 0.5)))),
    "W":      CrystalPreset("W", "Tungsten (BCC)",
                            3.165, 3.165, 3.165, 90, 90, 90,
                            (("W", (0.0, 0.0, 0.0)), ("W", (0.5, 0.5, 0.5)))),
    "Cu":     CrystalPreset("Cu", "Copper (FCC)",
                            3.615, 3.615, 3.615, 90, 90, 90,
                            (("Cu", (0.0, 0.0, 0.0)), ("Cu", (0.5, 0.5, 0.0)),
                             ("Cu", (0.5, 0.0, 0.5)), ("Cu", (0.0, 0.5, 0.5)))),
    "Al":     CrystalPreset("Al", "Aluminium (FCC)",
                            4.046, 4.046, 4.046, 90, 90, 90,
                            (("Al", (0.0, 0.0, 0.0)), ("Al", (0.5, 0.5, 0.0)),
                             ("Al", (0.5, 0.0, 0.5)), ("Al", (0.0, 0.5, 0.5)))),
    "Au":     CrystalPreset("Au", "Gold (FCC)",
                            4.078, 4.078, 4.078, 90, 90, 90,
                            (("Au", (0.0, 0.0, 0.0)), ("Au", (0.5, 0.5, 0.0)),
                             ("Au", (0.5, 0.0, 0.5)), ("Au", (0.0, 0.5, 0.5)))),
    "C_diamond": CrystalPreset(
        "C_diamond", "Diamond cubic (carbon)",
        3.567, 3.567, 3.567, 90, 90, 90,
        (("C", (0.00, 0.00, 0.00)), ("C", (0.50, 0.50, 0.00)),
         ("C", (0.50, 0.00, 0.50)), ("C", (0.00, 0.50, 0.50)),
         ("C", (0.25, 0.25, 0.25)), ("C", (0.75, 0.75, 0.25)),
         ("C", (0.75, 0.25, 0.75)), ("C", (0.25, 0.75, 0.75))),
    ),
    "Si_diamond": CrystalPreset(
        "Si_diamond", "Silicon (diamond cubic)",
        5.431, 5.431, 5.431, 90, 90, 90,
        (("Si", (0.00, 0.00, 0.00)), ("Si", (0.50, 0.50, 0.00)),
         ("Si", (0.50, 0.00, 0.50)), ("Si", (0.00, 0.50, 0.50)),
         ("Si", (0.25, 0.25, 0.25)), ("Si", (0.75, 0.75, 0.25)),
         ("Si", (0.75, 0.25, 0.75)), ("Si", (0.25, 0.75, 0.75))),
    ),
    "GaAs": CrystalPreset(
        "GaAs", "Gallium arsenide (zinc-blende)",
        5.6533, 5.6533, 5.6533, 90, 90, 90,
        (("Ga", (0.00, 0.00, 0.00)), ("Ga", (0.50, 0.50, 0.00)),
         ("Ga", (0.50, 0.00, 0.50)), ("Ga", (0.00, 0.50, 0.50)),
         ("As", (0.25, 0.25, 0.25)), ("As", (0.75, 0.75, 0.25)),
         ("As", (0.75, 0.25, 0.75)), ("As", (0.25, 0.75, 0.75))),
    ),
    "ZnS_zb": CrystalPreset(
        "ZnS_zb", "Zinc sulfide (zinc-blende)",
        5.406, 5.406, 5.406, 90, 90, 90,
        (("Zn", (0.00, 0.00, 0.00)), ("Zn", (0.50, 0.50, 0.00)),
         ("Zn", (0.50, 0.00, 0.50)), ("Zn", (0.00, 0.50, 0.50)),
         ("S",  (0.25, 0.25, 0.25)), ("S",  (0.75, 0.75, 0.25)),
         ("S",  (0.75, 0.25, 0.75)), ("S",  (0.25, 0.75, 0.75))),
    ),
    "ZnS_wurtzite": CrystalPreset(
        "ZnS_wurtzite", "Zinc sulfide (wurtzite)",
        3.811, 3.811, 6.234, 90, 90, 120,
        (("Zn", (1/3, 2/3, 0.000)), ("Zn", (2/3, 1/3, 0.500)),
         ("S",  (1/3, 2/3, 0.375)), ("S",  (2/3, 1/3, 0.875))),
    ),
    "NaCl": CrystalPreset(
        "NaCl", "Rock salt (NaCl)",
        5.640, 5.640, 5.640, 90, 90, 90,
        (("Na", (0.0, 0.0, 0.0)), ("Na", (0.5, 0.5, 0.0)),
         ("Na", (0.5, 0.0, 0.5)), ("Na", (0.0, 0.5, 0.5)),
         ("Cl", (0.5, 0.5, 0.5)), ("Cl", (0.0, 0.0, 0.5)),
         ("Cl", (0.0, 0.5, 0.0)), ("Cl", (0.5, 0.0, 0.0))),
    ),
    "CsCl": CrystalPreset(
        "CsCl", "Caesium chloride",
        4.123, 4.123, 4.123, 90, 90, 90,
        (("Cs", (0.0, 0.0, 0.0)), ("Cl", (0.5, 0.5, 0.5))),
    ),
    "graphene": CrystalPreset(
        "graphene", "Graphene (single layer, hexagonal)",
        2.461, 2.461, 20.0, 90, 90, 120,
        (("C", (0.0, 0.0, 0.5)), ("C", (1/3, 2/3, 0.5))),
    ),
    "graphite": CrystalPreset(
        "graphite", "Graphite (AB stacking)",
        2.461, 2.461, 6.708, 90, 90, 120,
        (("C", (0.0, 0.0, 0.25)), ("C", (1/3, 2/3, 0.25)),
         ("C", (0.0, 0.0, 0.75)), ("C", (2/3, 1/3, 0.75))),
    ),
    "quartz_alpha": CrystalPreset(
        "quartz_alpha", "α-Quartz (hexagonal SiO2)",
        4.9134, 4.9134, 5.4052, 90, 90, 120,
        (("Si", (0.4697, 0.0000, 0.0000)),
         ("Si", (0.0000, 0.4697, 0.6667)),
         ("Si", (0.5303, 0.5303, 0.3333)),
         ("O",  (0.4133, 0.2672, 0.1188)),
         ("O",  (0.7328, 0.1461, 0.7855)),
         ("O",  (0.8539, 0.5867, 0.4522)),
         ("O",  (0.2672, 0.4133, 0.5478)),
         ("O",  (0.1461, 0.7328, 0.2145)),
         ("O",  (0.5867, 0.8539, 0.8812))),
    ),
}


def list_presets() -> List[CrystalPreset]:
    return list(CRYSTAL_PRESETS.values())


def build_preset(name: str, nx: int = 1, ny: int = 1, nz: int = 1) -> Molecule:
    """Build a supercell from a preset key."""
    if name not in CRYSTAL_PRESETS:
        raise KeyError(
            f"Unknown crystal preset {name!r}. Available: "
            + ", ".join(sorted(CRYSTAL_PRESETS))
        )
    p = CRYSTAL_PRESETS[name]
    log.info("Building %s crystal (%dx%dx%d supercell)", p.description, nx, ny, nz)
    return build_crystal(
        p.a, p.b, p.c, p.alpha, p.beta, p.gamma,
        basis=p.basis, nx=nx, ny=ny, nz=nz, name=name,
    )
