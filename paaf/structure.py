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
    if fmt in {"data", "lmp", "lammps"}:
        # OpenBabel cannot read LAMMPS data files, the natural input for a
        # packed periodic cell.
        return load_lammps_data(path, name=name)
    mol = next(pybel.readfile(fmt, str(path)))
    out = _pybel_to_molecule(mol, name or path.stem, source=path)
    cell = _read_cell(path, fmt)
    if cell is not None:
        # The periodic box, as edge lengths in Å (minimum-image distances).
        setattr(out, "cell", cell)
    return out


def _read_cell(path: Path, fmt: str) -> Optional[Tuple[float, float, float]]:
    """Orthorhombic box edges (Å) from a PDB CRYST1 record or a GRO box line."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    try:
        if fmt in {"pdb", "ent"}:
            for ln in lines:
                if ln.startswith("CRYST1"):
                    a, b, c = float(ln[6:15]), float(ln[15:24]), float(ln[24:33])
                    if a > 1.0 and b > 1.0 and c > 1.0:
                        return (a, b, c)
        elif fmt == "gro":
            last = [ln for ln in lines if ln.strip()][-1].split()
            a, b, c = (10.0 * float(x) for x in last[:3])       # nm -> Å
            if a > 0 and b > 0 and c > 0:
                return (a, b, c)
    except (ValueError, IndexError):
        return None
    return None


def load_lammps_data(path: str | Path, name: Optional[str] = None) -> Molecule:
    """Atoms, bonds and box from a LAMMPS data file.

    Elements come from the ``Masses`` section (nearest element mass; a
    comment naming an element wins). ``atom_style`` full, molecular, charge
    and atomic are recognised from the ``Atoms # style`` hint or the column
    count. Bond orders are not in the file and are set to 1.
    """
    from .type_guard import _element_from_mass

    path = Path(path)
    text = path.read_text(errors="replace").splitlines()
    bounds = {}
    sections: dict = {}
    current = None
    style_hint = ""
    headers = {"Masses", "Atoms", "Bonds", "Velocities", "Angles", "Dihedrals",
               "Impropers"}
    for raw in text[1:]:
        body = raw.split("#", 1)[0].strip()
        if not body:
            continue
        head = body.split()[0]
        if head in headers or body.endswith("Coeffs"):
            current = head if head in headers else None
            if head == "Atoms" and "#" in raw:
                style_hint = raw.split("#", 1)[1].strip().lower()
            if current:
                sections[current] = []
            continue
        toks = body.split()
        if current is None:
            if len(toks) >= 4 and toks[2].endswith("lo") and toks[3].endswith("hi"):
                bounds[toks[2][0]] = (float(toks[0]), float(toks[1]))
            continue
        sections[current].append((toks, raw))

    element_of_type = {}
    for toks, raw in sections.get("Masses", []):
        el = None
        if "#" in raw:
            word = raw.split("#", 1)[1].strip().split()
            if word and word[0].capitalize() in _ELEMENT_SYMBOLS:
                el = word[0].capitalize()
        element_of_type[toks[0]] = el or _element_from_mass(float(toks[1])) or "C"

    rows = sections.get("Atoms", [])
    if not rows:
        raise ValueError(f"{path.name}: no Atoms section")
    ncol = len(rows[0][0])
    style = style_hint or {7: "full", 10: "full", 6: "molecular", 9: "molecular",
                           5: "atomic", 8: "atomic"}.get(ncol, "full")
    t_col, q_col, x_col = {"full": (2, 3, 4), "molecular": (2, None, 3),
                           "charge": (1, 2, 3), "atomic": (1, None, 2)}.get(
                               style, (2, 3, 4))
    parsed = []
    for toks, _raw in rows:
        parsed.append((int(toks[0]), toks[t_col],
                       float(toks[q_col]) if q_col is not None else 0.0,
                       np.array([float(v) for v in toks[x_col:x_col + 3]])))
    parsed.sort(key=lambda r: r[0])
    index_of = {ident: k for k, (ident, *_r) in enumerate(parsed)}
    atoms = []
    for k, (_ident, t, q, xyz) in enumerate(parsed):
        el = element_of_type.get(t, "C")
        atoms.append(Atom(index=k, element=el, xyz=xyz, name=f"{el}{k + 1}",
                          charge=q))
    bonds = []
    for toks, _raw in sections.get("Bonds", []):
        i, j = int(toks[2]), int(toks[3])
        if i in index_of and j in index_of:
            bonds.append((index_of[i], index_of[j], 1.0))
    mol = Molecule(atoms=atoms, bonds=bonds, name=name or path.stem,
                   source_path=path)
    if all(ax in bounds for ax in "xyz"):
        setattr(mol, "cell", tuple(bounds[ax][1] - bounds[ax][0] for ax in "xyz"))
    return mol


_ELEMENT_SYMBOLS = {"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na",
                    "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Fe",
                    "Cu", "Zn", "Br", "I", "Se"}


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
    OpenBabel — this lets the cell builders produce output
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
    # 1.5 is an aromatic bond (RDKit's perception returns it). It must not
    # reach OpenBabel as a number: round(1.5) = 2 wrote rings as all-double
    # bonds, and 5 is stored by OpenBabel 3 as a literal bond order (mol2
    # type "5", invalid). Kekulise instead; OpenBabel re-perceives the ring.
    orders = _kekulised_orders(mol)
    aromatic_left = []
    for k, (i, j, order) in enumerate(mol.bonds):
        if orders is not None:
            ob_order = orders[k]
        elif abs(float(order) - 1.5) < 1e-6:
            ob_order = 1
            aromatic_left.append((i, j))
        else:
            ob_order = int(round(order)) or 1
        ob.AddBond(i + 1, j + 1, ob_order)
    for i, j in aromatic_left:
        # Kekulisation failed (no RDKit, or an unkekulisable ring): keep the
        # bond single and flag it aromatic, never a numeric code.
        b = ob.GetBond(i + 1, j + 1)
        if b is not None:
            b.SetAromatic()
            ob.GetAtom(i + 1).SetAromatic()
            ob.GetAtom(j + 1).SetAromatic()
    pmol = pybel.Molecule(ob)
    pmol.title = mol.name
    pmol.write(ext, str(path), overwrite=True)
    return path


def _kekulised_orders(mol: Molecule) -> Optional[List[int]]:
    """Integer bond orders (1/2/3) for ``mol.bonds``, aromatic 1.5 kekulised.

    ``None`` when there is nothing aromatic, or RDKit cannot kekulise it.
    """
    if not any(abs(float(o) - 1.5) < 1e-6 for _i, _j, o in mol.bonds):
        return None
    try:
        from rdkit import Chem
    except ImportError:
        return None
    try:
        rw = Chem.RWMol()
        for a in mol.atoms:
            atom = Chem.Atom(a.element)
            atom.SetNoImplicit(True)
            rw.AddAtom(atom)
        kinds = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                 3: Chem.BondType.TRIPLE}
        for i, j, o in mol.bonds:
            if abs(float(o) - 1.5) < 1e-6:
                rw.AddBond(int(i), int(j), Chem.BondType.AROMATIC)
                rw.GetBondBetweenAtoms(int(i), int(j)).SetIsAromatic(True)
                rw.GetAtomWithIdx(int(i)).SetIsAromatic(True)
                rw.GetAtomWithIdx(int(j)).SetIsAromatic(True)
            else:
                rw.AddBond(int(i), int(j),
                           kinds.get(int(round(float(o))), Chem.BondType.SINGLE))
        Chem.Kekulize(rw, clearAromaticFlags=True)
        out = []
        for i, j, _o in mol.bonds:
            bt = rw.GetBondBetweenAtoms(int(i), int(j)).GetBondType()
            out.append({Chem.BondType.DOUBLE: 2,
                        Chem.BondType.TRIPLE: 3}.get(bt, 1))
        return out
    except Exception:                       # unkekulisable -> caller flags
        return None


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
