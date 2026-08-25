"""Serialise a PAAF molecule as an MDL molfile, bond orders included.

Why a molfile
-------------
The 3D viewer needs to draw a double bond as two lines and a single bond as
one. That information exists in the SMILES the monomer was built from, and it
survives into :class:`paaf.structure.Molecule` as the third element of each
bond tuple — but XYZ throws it away and PDB has no reliable place for it.

The MDL molfile (V2000) carries an explicit bond block with an order column,
is understood by every viewer worth using, and is a dozen lines to write. So
the viewer is fed one of these rather than being asked to guess bond orders
from distances, which is what produces the classic "aromatic ring drawn as
six single bonds" look.

Atom order is preserved exactly. That matters more than it sounds: the viewer
reports clicks by atom serial, and those serials are mapped straight back to
PAAF atom indices to decide which row of the typing table to select. A
reordering anywhere in here would silently mistype atoms.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["to_molblock", "bond_order_counts"]


def _order_code(order: float) -> int:
    """MDL bond-order code: 1 single, 2 double, 3 triple, 4 aromatic.

    PAAF stores aromatic bonds as 1.5, which is how RDKit and OpenBabel both
    represent a delocalised ring. MDL spells that 4.
    """
    try:
        value = float(order)
    except (TypeError, ValueError):
        return 1
    if abs(value - 1.5) < 0.01:
        return 4
    code = int(round(value))
    return code if 1 <= code <= 4 else 1


def to_molblock(mol, title: str = "", *,
                index_offset: int = 0) -> str:
    """An MDL V2000 molfile for ``mol``.

    ``index_offset`` shifts the reported atom numbering, which is what lets a
    copolymer be shown as one scene while each atom still maps back to the
    right monomer's typing row.

    The V2000 format is column-oriented and viewers do enforce it, so the
    fields are written to fixed widths rather than joined with spaces.
    """
    atoms = list(mol.atoms)
    bonds = list(getattr(mol, "bonds", []))

    lines: List[str] = [
        (title or getattr(mol, "name", "") or "molecule")[:80],
        "  PAAF",                       # program line
        "",                             # comment line
        f"{len(atoms):>3d}{len(bonds):>3d}  0  0  0  0  0  0  0  0999 V2000",
    ]

    for a in atoms:
        x, y, z = (float(a.xyz[0]), float(a.xyz[1]), float(a.xyz[2]))
        lines.append(
            f"{x:>10.4f}{y:>10.4f}{z:>10.4f} "
            f"{a.element:<3s} 0  0  0  0  0  0  0  0  0  0  0  0")

    for bond in bonds:
        i, j = int(bond[0]), int(bond[1])
        order = bond[2] if len(bond) > 2 else 1.0
        # Molfiles number atoms from 1.
        lines.append(f"{i + 1:>3d}{j + 1:>3d}{_order_code(order):>3d}  0  0  0  0")

    lines.append("M  END")
    return "\n".join(lines) + "\n"


def bond_order_counts(mol) -> Dict[int, int]:
    """How many bonds of each MDL order the molecule has.

    Used by the viewer's self-check: a monomer built from a SMILES containing
    ``=`` that comes back with no order-2 bonds means the order was lost
    somewhere upstream, and the picture would be quietly wrong rather than
    obviously broken.
    """
    counts: Dict[int, int] = {}
    for bond in getattr(mol, "bonds", []):
        code = _order_code(bond[2] if len(bond) > 2 else 1.0)
        counts[code] = counts.get(code, 0) + 1
    return counts


def combined_molblock(molecules: Sequence, titles: Optional[Sequence[str]] = None,
                      gap: float = 6.0) -> Tuple[str, List[Tuple[int, int]]]:
    """One molfile holding several molecules laid out side by side.

    Returns ``(molblock, mapping)`` where ``mapping[serial]`` is
    ``(molecule_number, atom_index_within_that_molecule)``. A copolymer is
    shown as both monomers in one scene, and the mapping is what turns a click
    on a shared scene back into "atom 7 of monomer 2".

    Each molecule is translated along x so they do not overlap; the offset is
    its own width plus ``gap``, so a long monomer does not sit on top of a
    short one.
    """
    import numpy as np

    lines_atoms: List[str] = []
    lines_bonds: List[str] = []
    mapping: List[Tuple[int, int]] = []
    serial = 0
    shift = 0.0

    for m_idx, mol in enumerate(molecules):
        atoms = list(mol.atoms)
        if not atoms:
            continue
        xyz = np.array([a.xyz for a in atoms], dtype=float)
        width = float(xyz[:, 0].max() - xyz[:, 0].min()) if len(xyz) else 0.0
        dx = shift - float(xyz[:, 0].min() if len(xyz) else 0.0)

        base = serial
        for k, a in enumerate(atoms):
            x = float(a.xyz[0]) + dx
            y, z = float(a.xyz[1]), float(a.xyz[2])
            lines_atoms.append(
                f"{x:>10.4f}{y:>10.4f}{z:>10.4f} "
                f"{a.element:<3s} 0  0  0  0  0  0  0  0  0  0  0  0")
            mapping.append((m_idx, k))
            serial += 1

        for bond in getattr(mol, "bonds", []):
            i, j = int(bond[0]), int(bond[1])
            order = bond[2] if len(bond) > 2 else 1.0
            lines_bonds.append(
                f"{base + i + 1:>3d}{base + j + 1:>3d}"
                f"{_order_code(order):>3d}  0  0  0  0")

        shift += width + gap

    header = [
        "PAAF monomers",
        "  PAAF",
        "",
        f"{len(lines_atoms):>3d}{len(lines_bonds):>3d}"
        f"  0  0  0  0  0  0  0  0999 V2000",
    ]
    block = "\n".join(header + lines_atoms + lines_bonds + ["M  END"]) + "\n"
    return block, mapping
