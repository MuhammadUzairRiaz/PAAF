"""Nanotube builder — armchair / zigzag / chiral single-wall nanotube.

Standard (n, m) construction from a graphene sheet, then optionally repeated
along the tube axis.
"""
from __future__ import annotations

import math
import numpy as np
from typing import List, Tuple

from ..logging_utils import get_logger
from ..structure import Atom, Molecule

log = get_logger(__name__)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def build_nanotube(
    n: int, m: int,
    length_ang: float = 20.0,
    element: str = "C",
    bond_length_ang: float = 1.42,
) -> Molecule:
    """Build a (n, m) single-wall nanotube of the specified length.

    Parameters
    ----------
    n, m           chirality indices. (n, 0) = zigzag; (n, n) = armchair.
    length_ang     minimum tube length (Å). Rounded up to a whole translation.
    element        element for every atom (C for CNT, B/N mix for BN nanotube).
    bond_length_ang  nearest-neighbor bond length (Å).
    """
    if n <= 0 or m < 0:
        raise ValueError("Require n > 0 and m >= 0")

    ac = bond_length_ang
    # Unit-cell vectors of graphene (a1, a2)
    a1 = ac * np.array([1.5, math.sqrt(3) / 2])
    a2 = ac * np.array([1.5, -math.sqrt(3) / 2])
    # Chiral vector C_h and its length -> circumference
    Ch = n * a1 + m * a2
    circ = np.linalg.norm(Ch)
    radius = circ / (2 * math.pi)
    # Translation vector T along tube axis
    dR = _gcd(2 * n + m, 2 * m + n)
    t1 = (2 * m + n) // dR
    t2 = -(2 * n + m) // dR
    T = t1 * a1 + t2 * a2
    T_len = np.linalg.norm(T)
    # Number of translation cells to reach at least `length_ang`
    n_axial = max(1, int(math.ceil(length_ang / T_len)))
    log.info("Building (%d,%d) tube: R=%.3f Å, |T|=%.3f Å, %d axial cells", n, m, radius, T_len, n_axial)

    # Basis atoms in the graphene unit cell (in same 2-D frame)
    basis = [np.zeros(2), ac * np.array([1.0, 0.0])]

    # Generate a large 2-D patch of graphene points, then keep the ones
    # inside the rectangle spanned by Ch and n_axial*T
    Ch_hat = Ch / circ
    T_hat = T / T_len
    # Grid range: coarse but generous
    span = int(max(abs(n), abs(m), t1, abs(t2))) * 3 + 4
    atoms: List[Atom] = []
    idx = 0
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            for b in basis:
                p = i * a1 + j * a2 + b
                u = float(np.dot(p, Ch_hat))    # around the tube (0..circ)
                v = float(np.dot(p, T_hat))     # along the axis (0..n_axial*T_len)
                if 0.0 <= u < circ and 0.0 <= v < n_axial * T_len:
                    theta = 2 * math.pi * (u / circ)
                    x = radius * math.cos(theta)
                    y = radius * math.sin(theta)
                    z = v
                    atoms.append(Atom(
                        index=idx, element=element,
                        xyz=np.array([x, y, z]), name=f"{element}{idx + 1}",
                    ))
                    idx += 1
    log.info("(%d,%d) nanotube: %d atoms", n, m, len(atoms))
    return Molecule(atoms=atoms, bonds=[], name=f"nanotube_{n}_{m}")
