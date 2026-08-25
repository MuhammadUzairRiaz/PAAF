"""Which atoms are chemically the same, so they can be typed together.

A methyl group's three hydrogens are indistinguishable. So are the two on a
methylene. Typing them one row at a time is busywork, and worse, it is where
mistakes hide: fifteen rows that differ only by an index, and a single wrong
entry looks exactly like a right one.

Assigning by equivalence class removes the opportunity. Click one hydrogen of
a CH2, and both get the type.

How equivalence is decided
--------------------------
RDKit's canonical ranking with ``breakTies=False`` gives every atom a rank
that depends only on its environment — its element, bonds, and the
environments of its neighbours, recursively. Atoms with equal rank are
symmetry-equivalent under the molecular graph. That is the right notion here:
it is exactly "would a force field give these the same type".

Without RDKit a simpler substitute is used: element plus sorted neighbour
elements plus the neighbours' own neighbour elements. That catches methyl and
methylene hydrogens, which is the case that matters, while being honest that
it is coarser than the graph-canonical answer.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Set

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["equivalent_atoms", "equivalence_classes"]


def _rdkit_classes(mol) -> Dict[int, int]:
    """Canonical ranks from RDKit, or ``{}`` if it is unavailable."""
    try:
        from rdkit import Chem
    except Exception:
        return {}

    rw = Chem.RWMol()
    for a in mol.atoms:
        rw.AddAtom(Chem.Atom(a.element))
    order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
             3: Chem.BondType.TRIPLE}
    for bond in mol.bonds:
        i, j = int(bond[0]), int(bond[1])
        value = float(bond[2]) if len(bond) > 2 else 1.0
        btype = (Chem.BondType.AROMATIC if abs(value - 1.5) < 0.01
                 else order.get(int(round(value)), Chem.BondType.SINGLE))
        try:
            rw.AddBond(i, j, btype)
        except Exception:
            continue
    try:
        rdmol = rw.GetMol()
        Chem.SanitizeMol(rdmol, catchErrors=True)
        ranks = list(Chem.CanonicalRankAtoms(rdmol, breakTies=False))
    except Exception as exc:
        log.info("RDKit ranking unavailable (%s); using the graph fallback", exc)
        return {}
    return {i: int(r) for i, r in enumerate(ranks)}


def _graph_classes(mol) -> Dict[int, str]:
    """Element + neighbour shell, two deep. Coarser, but needs nothing."""
    adj: Dict[int, List[int]] = defaultdict(list)
    for bond in mol.bonds:
        i, j = int(bond[0]), int(bond[1])
        adj[i].append(j)
        adj[j].append(i)

    def shell(k: int) -> str:
        return "".join(sorted(mol.atoms[n].element for n in adj[k]))

    keys: Dict[int, str] = {}
    for a in mol.atoms:
        k = a.index
        neigh = sorted((mol.atoms[n].element, shell(n)) for n in adj[k])
        keys[k] = f"{a.element}|{shell(k)}|{neigh}"
    return keys


def equivalence_classes(mol, coarse: bool = False) -> Dict[int, List[int]]:
    """``{class_key: [atom indices]}`` grouping equivalent atoms.

    ``coarse=False`` uses RDKit's canonical ranking: exact graph symmetry.
    That is right for "click one methyl H, type all three".

    ``coarse=True`` uses the local environment only — element, neighbours,
    and the neighbours' neighbours. That is right for "the user typed one
    repeat unit, apply it to all twenty", because exact symmetry says an atom
    in the *end* unit is not equivalent to the same atom mid-chain. True, and
    useless here: they need the same force-field type.
    """
    if coarse:
        keys = _graph_classes(mol)
    else:
        ranks = _rdkit_classes(mol)
        keys = ranks if ranks else _graph_classes(mol)
    groups: Dict[object, List[int]] = defaultdict(list)
    for index, key in keys.items():
        groups[key].append(int(index))
    return {k: sorted(v) for k, v in groups.items()}


def equivalent_atoms(mol) -> Dict[int, List[int]]:
    """``{atom index: every index equivalent to it, including itself}``."""
    out: Dict[int, List[int]] = {}
    for members in equivalence_classes(mol).values():
        for index in members:
            out[index] = list(members)
    return out
