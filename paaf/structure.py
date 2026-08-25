"""Structure loading and conversion using OpenBabel.

Supported extensions: .xyz, .pdb, .mol, .mol2, .sdf, .cml, .smi (via SMILES).
The molecule is returned as a :class:`Molecule` container holding atoms
(element, xyz) and connectivity (bond list). Everything downstream (chain
builder, .lt writer, DL_FIELD atom-typer) operates on this container so
Moltemplate/mbuild are only imported when actually needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Atom:
    index: int          # 0-based internal index
    element: str        # chemical symbol
    xyz: np.ndarray     # (3,) angstrom
    name: str = ""      # optional PDB-style atom name
    charge: float = 0.0
    ff_type: Optional[str] = None  # assigned later by FF assigner


@dataclass
class Molecule:
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Tuple[int, int, float]] = field(default_factory=list)  # (i, j, order)
    name: str = "MOL"
    source_path: Optional[Path] = None

    # ------------------------------------------------------------------ IO
    def to_xyz(self, path: str | Path) -> Path:
        path = Path(path)
        with path.open("w") as f:
            f.write(f"{len(self.atoms)}\n{self.name}\n")
            for a in self.atoms:
                f.write(f"{a.element:>2s}  {a.xyz[0]:12.6f}  {a.xyz[1]:12.6f}  {a.xyz[2]:12.6f}\n")
        return path

    def coords(self) -> np.ndarray:
        return np.array([a.xyz for a in self.atoms], dtype=float)

    def elements(self) -> List[str]:
        return [a.element for a in self.atoms]

    def neighbors(self, i: int) -> List[int]:
        out = []
        for a, b, _ in self.bonds:
            if a == i:
                out.append(b)
            elif b == i:
                out.append(a)
        return out


# ---------------------------------------------------------------------- load
def load_structure(path: str | Path, name: Optional[str] = None) -> Molecule:
    """Load a molecule from any format OpenBabel can read.

    OpenBabel is optional at import time so the CLI/GUI can still be used
    for tasks that don't require it (e.g. only running the FF converter).
    """
    path = Path(path)
    try:
        from openbabel import openbabel  # noqa: F401
        from openbabel import pybel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "OpenBabel (python bindings) is required to read structures. "
            "Install with `conda install -c conda-forge openbabel`."
        ) from exc

    fmt = path.suffix.lstrip(".").lower()
    if fmt in {"", "smi", "smiles"}:
        raise ValueError("Use load_smiles() for SMILES input.")
    mol = next(pybel.readfile(fmt, str(path)))
    return _pybel_to_molecule(mol, name or path.stem, source=path)


def load_smiles(smiles: str, name: str = "MOL", add_h: bool = True, make_3d: bool = True) -> Molecule:
    from openbabel import pybel
    mol = pybel.readstring("smi", smiles)
    if add_h:
        mol.addh()
    if make_3d:
        mol.make3D(forcefield="mmff94", steps=100)
    return _pybel_to_molecule(mol, name)


_DUMMY_ELEMENTS = {"*", "Xx", "XX", "Du", "DU", ""}


def _pybel_to_molecule(mol, name: str, source: Optional[Path] = None) -> Molecule:
    from openbabel import openbabel
    raw_atoms: List[Atom] = []
    for i, atom in enumerate(mol.atoms):
        elem = openbabel.GetSymbol(atom.atomicnum)
        raw_atoms.append(
            Atom(
                index=i,
                element=elem,
                xyz=np.array(atom.coords, dtype=float),
                name=f"{elem}{i + 1}",
                charge=float(atom.partialcharge or 0.0),
            )
        )
    raw_bonds: List[Tuple[int, int, float]] = []
    ob = mol.OBMol
    for bidx in range(ob.NumBonds()):
        b = ob.GetBondById(bidx) or ob.GetBond(bidx)
        if b is None:
            continue
        i = b.GetBeginAtomIdx() - 1
        j = b.GetEndAtomIdx() - 1
        order = float(b.GetBondOrder())
        raw_bonds.append((i, j, order))

    # Strip dummy/wildcard atoms (*, Xx, Du...) that leak in from SMILES
    # polymerization sites. These make moltemplate crash later because their
    # "element" doesn't map to a valid mass or FF type.
    dummy_mask = {a.index for a in raw_atoms if a.element in _DUMMY_ELEMENTS
                  or atom_is_wildcard(mol.OBMol.GetAtom(a.index + 1))}
    if dummy_mask:
        import logging as _log
        _log.getLogger("paaf").info(
            "load_structure: stripped %d wildcard/dummy atom(s) "
            "(indices %s) from %s", len(dummy_mask), sorted(dummy_mask), name)
    kept = [a for a in raw_atoms if a.index not in dummy_mask]
    remap = {a.index: k for k, a in enumerate(kept)}
    for k, a in enumerate(kept):
        a.index = k; a.name = f"{a.element}{k + 1}"
    bonds = [(remap[i], remap[j], o) for (i, j, o) in raw_bonds
             if i in remap and j in remap]
    return Molecule(atoms=kept, bonds=bonds, name=name, source_path=source)


def atom_is_wildcard(ob_atom) -> bool:
    """OpenBabel returns atomicnum=0 for wildcard `*` atoms."""
    try:
        return int(ob_atom.GetAtomicNum()) == 0
    except Exception:
        return False


# ------------------------------------------------------------- convenience
def write(mol: Molecule, path: str | Path) -> Path:
    """Write a molecule to any OpenBabel-supported format.

    For plain-text formats (xyz, pdb) we can write natively without
    OpenBabel — this lets the cell/crystal/nanotube builders produce output
    on systems where OpenBabel is not installed.
    """
    path = Path(path)
    ext = path.suffix.lstrip(".").lower()

    if ext == "xyz":
        return mol.to_xyz(path)
    if ext == "pdb":
        return _write_pdb(mol, path)

    # Fall back to OpenBabel for the rest.
    try:
        from openbabel import openbabel, pybel
    except Exception as exc:
        raise RuntimeError(
            f"OpenBabel required to write {ext!r} files. "
            f"Install with `conda install -c conda-forge openbabel`, "
            f"or export as .xyz / .pdb which do not need OpenBabel."
        ) from exc

    ob = openbabel.OBMol()
    for a in mol.atoms:
        ob_atom = ob.NewAtom()
        ob_atom.SetAtomicNum(openbabel.GetAtomicNum(a.element))
        ob_atom.SetVector(*a.xyz.tolist())
    for i, j, order in mol.bonds:
        ob.AddBond(i + 1, j + 1, int(round(order)) or 1)
    pmol = pybel.Molecule(ob)
    pmol.title = mol.name
    pmol.write(ext, str(path), overwrite=True)
    return path


def _write_pdb(mol: Molecule, path: Path) -> Path:
    """Minimal PDB writer — enough to round-trip through Avogadro / OVITO."""
    lines = [f"REMARK   Generated by paaf", f"REMARK   Title: {mol.name}"]
    for i, a in enumerate(mol.atoms, start=1):
        # HETATM record; columns per PDB spec (approximate but valid).
        name = a.name[:4] if a.name else a.element
        lines.append(
            f"HETATM{i:5d}  {name:<4s}MOL     1    "
            f"{a.xyz[0]:8.3f}{a.xyz[1]:8.3f}{a.xyz[2]:8.3f}"
            f"  1.00  0.00          {a.element:>2s}"
        )
    for i, j, order in mol.bonds:
        lines.append(f"CONECT{i + 1:5d}{j + 1:5d}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path
