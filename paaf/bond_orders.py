"""Recover bond orders that chain assembly threw away.

The loss
--------
mBuild's ``Compound`` records connectivity but not bond order, so both chain
builders wrote every bond as single::

    for a, b in comp.bonds():
        bonds.append((i, j, 1.0))        # the C=C is gone here

For polyisoprene that is not cosmetic. The SMARTS typer identifies alkene
carbons with patterns like ``[CX3]=[CX3]``; with no double bond left to match,
those rules never fire, every carbon falls through to the alkane rules, and a
20-unit chain comes out with 98 atoms typed 135 and not one typed 141.

Nothing downstream notices. The data file is well formed, the masses are
right, and only the chemistry is wrong.

Recovery
--------
The order is recoverable because the geometry still carries it: a C=C is
~1.33 Å where a C–C is ~1.53 Å, and the two carbons are planar with three
neighbours rather than four. Rather than trust distance alone — which
misreads a strained or unrelaxed bond — the two are combined:

* an atom's *available valence* (its element's normal valence minus its
  number of bonds) says how many extra orders it can accept;
* the bond length says which candidate pairs are close enough to be multiple.

A double bond is assigned only where both atoms have spare valence and the
distance supports it. That refuses to invent a double bond in a stretched
structure, which is the failure mode that matters: a spurious C=C would type
a saturated carbon as an alkene, which is exactly as wrong as the loss.

RDKit does this better when available, and is preferred; this runs when it is
not, and as a cross-check on what it returns.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["perceive_bond_orders", "restore_from_template", "order_summary"]

#: Normal valence, used to decide how many extra bond orders an atom can take.
_VALENCE: Dict[str, int] = {
    "H": 1, "F": 1, "Cl": 1, "Br": 1, "I": 1,
    "O": 2, "S": 2,
    "N": 3, "P": 3,
    "C": 4, "Si": 4,
}

#: Typical double-bond lengths, in Å. A pair is only promoted when it is
#: closer than the midpoint between its single and double values.
_DOUBLE: Dict[Tuple[str, str], float] = {
    ("C", "C"): 1.34, ("C", "O"): 1.23, ("C", "N"): 1.28,
    ("C", "S"): 1.60, ("N", "N"): 1.25, ("N", "O"): 1.22,
}
_SINGLE: Dict[Tuple[str, str], float] = {
    ("C", "C"): 1.54, ("C", "O"): 1.43, ("C", "N"): 1.47,
    ("C", "S"): 1.82, ("N", "N"): 1.45, ("N", "O"): 1.40,
}


def _pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def order_summary(mol) -> Dict[int, int]:
    """``{order: count}`` for the molecule's bonds, orders rounded."""
    out: Dict[int, int] = {}
    for bond in getattr(mol, "bonds", []):
        value = bond[2] if len(bond) > 2 else 1.0
        code = 4 if abs(float(value) - 1.5) < 0.01 else int(round(float(value)))
        out[code] = out.get(code, 0) + 1
    return out


# =====================================================================
def _rdkit_orders(mol) -> Optional[List[float]]:
    """Bond orders from RDKit's own perception, or ``None``."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except Exception:
        return None

    # Hand RDKit the geometry only and let it work out the bonding, then read
    # the orders back for the bonds we already know about.
    lines = [str(len(mol.atoms)), "chain"]
    for a in mol.atoms:
        lines.append(f"{a.element} {a.xyz[0]:.6f} {a.xyz[1]:.6f} {a.xyz[2]:.6f}")
    try:
        raw = Chem.MolFromXYZBlock("\n".join(lines) + "\n")
        if raw is None:
            return None
        rdDetermineBonds.DetermineBonds(raw, charge=0)
    except Exception as exc:
        log.info("RDKit bond perception declined (%s); using the "
                 "valence/geometry rule", exc)
        return None

    lookup: Dict[Tuple[int, int], float] = {}
    for b in raw.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        lookup[(min(i, j), max(i, j))] = float(b.GetBondTypeAsDouble())

    orders: List[float] = []
    for bond in mol.bonds:
        i, j = int(bond[0]), int(bond[1])
        orders.append(lookup.get((min(i, j), max(i, j)), 1.0))
    return orders


def perceive_bond_orders(mol, *, prefer_rdkit: bool = True) -> int:
    """Set bond orders on ``mol`` in place. Returns how many were raised.

    Safe to call on a molecule that already has correct orders: a bond is only
    promoted when both atoms have spare valence, so a fully-satisfied
    structure is left untouched.
    """
    bonds = list(getattr(mol, "bonds", []))
    if not bonds:
        return 0

    if prefer_rdkit:
        orders = _rdkit_orders(mol)
        if orders is not None:
            raised = 0
            for k, bond in enumerate(bonds):
                new = orders[k]
                if new > (bond[2] if len(bond) > 2 else 1.0) + 1e-6:
                    raised += 1
                mol.bonds[k] = (bond[0], bond[1], new)
            if raised:
                log.info("RDKit perceived %d multiple bonds", raised)
            return raised

    # ---- valence + geometry ------------------------------------------
    # Valence already used is the SUM of bond orders, not the number of bonds.
    # Counting bonds instead would leave an already-double bond looking as if
    # it still had spare valence, so a second call would "promote" it again —
    # harmless in the result, but it makes the reported count a lie and would
    # let a triple bond appear where a double belongs.
    used: Dict[int, float] = {}
    for bond in bonds:
        order = float(bond[2]) if len(bond) > 2 else 1.0
        used[int(bond[0])] = used.get(int(bond[0]), 0.0) + order
        used[int(bond[1])] = used.get(int(bond[1]), 0.0) + order

    spare: Dict[int, int] = {}
    for a in mol.atoms:
        normal = _VALENCE.get(a.element, 0)
        spare[a.index] = max(0, int(round(normal - used.get(a.index, 0.0))))

    # Shortest candidates first: the most convincing double bond wins the
    # spare valence before a marginal one can claim it.
    candidates: List[Tuple[float, int]] = []
    for k, bond in enumerate(bonds):
        i, j = int(bond[0]), int(bond[1])
        ei, ej = mol.atoms[i].element, mol.atoms[j].element
        key = _pair(ei, ej)
        if key not in _DOUBLE:
            continue
        d = float(np.linalg.norm(np.asarray(mol.atoms[i].xyz)
                                 - np.asarray(mol.atoms[j].xyz)))
        cutoff = 0.5 * (_DOUBLE[key] + _SINGLE[key])
        if d <= cutoff:
            candidates.append((d, k))
    candidates.sort()

    raised = 0
    for _d, k in candidates:
        bond = bonds[k]
        i, j = int(bond[0]), int(bond[1])
        if spare.get(i, 0) >= 1 and spare.get(j, 0) >= 1:
            mol.bonds[k] = (bond[0], bond[1], 2.0)
            spare[i] -= 1
            spare[j] -= 1
            raised += 1
    if raised:
        log.info("Perceived %d double bonds from valence and geometry", raised)
    return raised


# =====================================================================
def restore_from_template(chain, template, unit_size: int,
                          n_units: int, offset: int = 0) -> int:
    """Copy a repeat unit's bond orders onto every unit of a chain.

    Preferred when the template is trustworthy — it states the chemistry the
    user asked for rather than inferring it — but it needs the chain to be a
    clean repetition of the template, which capping and end groups break. The
    caller is expected to fall back to :func:`perceive_bond_orders`.

    Returns the number of bonds whose order was raised.
    """
    template_orders: Dict[Tuple[int, int], float] = {}
    for bond in getattr(template, "bonds", []):
        value = float(bond[2]) if len(bond) > 2 else 1.0
        if value > 1.0:
            i, j = int(bond[0]), int(bond[1])
            template_orders[(min(i, j), max(i, j))] = value
    if not template_orders:
        return 0

    raised = 0
    for unit in range(n_units):
        base = offset + unit * unit_size
        for (ti, tj), value in template_orders.items():
            gi, gj = base + ti, base + tj
            for k, bond in enumerate(chain.bonds):
                i, j = int(bond[0]), int(bond[1])
                if (min(i, j), max(i, j)) == (min(gi, gj), max(gi, gj)):
                    chain.bonds[k] = (bond[0], bond[1], value)
                    raised += 1
                    break
    return raised
