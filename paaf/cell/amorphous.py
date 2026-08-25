"""Amorphous cell / blends builder — pack N copies of each molecule into a
periodic box at a target density.

Box shape
---------
`pack_cell` accepts a :class:`BoxShape` describing the periodic cell:

- ``cubic``       — single side length ``a``
- ``orthorhombic``— (a, b, c) with 90° angles
- ``triclinic``   — (a, b, c, α, β, γ) with arbitrary angles

`packmol` supports cubic and orthorhombic natively; triclinic packing is
performed inside the bounding orthorhombic box and the returned
:class:`Molecule` is decorated with the triclinic cell vectors so
downstream writers (LAMMPS ``xy xz yz`` tilt factors, GROMACS triclinic
box line) can emit the correct periodic cell.

Backends
--------
Tried in order (auto):

- **packmol** — external binary, most robust for blends.
- **mbuild.fill_box** — pure-Python, uses packmol under the hood if present.
- **grid fallback** — numpy-only, used for tests and when others are absent.
"""
from __future__ import annotations

import math
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..logging_utils import get_logger
from ..structure import Atom, Molecule, load_structure, write

log = get_logger(__name__)


# ============================================================ BoxShape
@dataclass
class BoxShape:
    """Periodic cell description.

    Attributes
    ----------
    shape : str
        "cubic" | "orthorhombic" | "triclinic".
    a, b, c : float
        Cell edge lengths in Å. For "cubic", only ``a`` is used.
    alpha, beta, gamma : float
        Cell angles in degrees. Only used for "triclinic".
    """
    shape: str = "cubic"                 # cubic | orthorhombic | triclinic
    a: float = 50.0
    b: float = 50.0
    c: float = 50.0
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

    def volume_ang3(self) -> float:
        if self.shape == "cubic":
            return self.a ** 3
        if self.shape == "orthorhombic":
            return self.a * self.b * self.c
        # triclinic: V = a b c sqrt(1 - cos²α - cos²β - cos²γ + 2 cosα cosβ cosγ)
        from math import cos, radians, sqrt
        ca, cb, cg = cos(radians(self.alpha)), cos(radians(self.beta)), cos(radians(self.gamma))
        v = 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg
        return self.a * self.b * self.c * sqrt(max(v, 0.0))

    def bounding_box(self) -> Tuple[float, float, float]:
        """Axis-aligned bounding box for packing purposes."""
        if self.shape == "cubic":
            return (self.a, self.a, self.a)
        if self.shape == "orthorhombic":
            return (self.a, self.b, self.c)
        # triclinic: compute Cartesian cell vectors and take extents
        from math import cos, radians, sin, sqrt
        al, be, ga = radians(self.alpha), radians(self.beta), radians(self.gamma)
        ax = self.a
        bx = self.b * cos(ga); by = self.b * sin(ga)
        cx = self.c * cos(be)
        cy = self.c * (cos(al) - cos(be) * cos(ga)) / sin(ga)
        cz = sqrt(max(self.c ** 2 - cx ** 2 - cy ** 2, 0.0))
        # Bounding box in Cartesian:
        xs = [0.0, ax, bx, ax + bx, cx, ax + cx, bx + cx, ax + bx + cx]
        ys = [0.0, 0.0, by, by, cy, cy, by + cy, by + cy]
        zs = [0.0, 0.0, 0.0, 0.0, cz, cz, cz, cz]
        return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))

    def scaled(self, factor: float) -> "BoxShape":
        """Uniform scale of edge lengths (angles preserved)."""
        return BoxShape(
            shape=self.shape,
            a=self.a * factor, b=self.b * factor, c=self.c * factor,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
        )

    @classmethod
    def from_any(cls, spec) -> "BoxShape":
        """Convert a scalar / tuple / dict / BoxShape into a :class:`BoxShape`."""
        if isinstance(spec, BoxShape):
            return spec
        if isinstance(spec, (int, float)):
            return cls(shape="cubic", a=float(spec), b=float(spec), c=float(spec))
        if isinstance(spec, (tuple, list)):
            if len(spec) == 3:
                a, b, c = map(float, spec)
                return cls(shape="orthorhombic", a=a, b=b, c=c)
            if len(spec) == 6:
                a, b, c, al, be, ga = map(float, spec)
                return cls(shape="triclinic", a=a, b=b, c=c,
                           alpha=al, beta=be, gamma=ga)
        if isinstance(spec, dict):
            return cls(**spec)
        raise TypeError(f"Cannot interpret {spec!r} as a BoxShape")

# molar mass of an element in g/mol (small set that covers organic MDs)
_MASS = {
    "H": 1.008, "He": 4.003, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990,
    "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06,
    "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078, "Fe": 55.845,
    "Cu": 63.546, "Zn": 65.38, "Br": 79.904, "I": 126.904,
}


# ============================================================ specs
@dataclass
class PackSpec:
    """One species in the amorphous cell.

    Either provide ``file`` (path to xyz/pdb/mol2/...) or ``molecule`` (a
    pre-loaded :class:`Molecule`). ``count`` chooses how many copies to pack.
    """
    file: Optional[str] = None
    molecule: Optional[Molecule] = None
    count: int = 1
    name: str = ""


# ============================================================ helpers
def _mass_of(mol: Molecule) -> float:
    return sum(_MASS.get(a.element, 12.0) for a in mol.atoms)


def _load_spec(spec: PackSpec) -> Molecule:
    if spec.molecule is not None:
        return spec.molecule
    if spec.file:
        return load_structure(spec.file, name=spec.name or Path(spec.file).stem)
    raise ValueError("PackSpec must have either file or molecule set")


_N_AVOGADRO = 6.02214076e23


def _box_from_density(
    specs: Sequence[PackSpec], mols: Sequence[Molecule], density_kg_m3: float
) -> float:
    """Return the cubic box side length (Å) that yields `density_kg_m3`.

    `_mass_of` returns molar mass in g/mol; convert to grams per system with
    Avogadro's number, then kg, then divide by density to get the volume.
    """
    total_g_per_mol = sum(spec.count * _mass_of(mol) for spec, mol in zip(specs, mols))
    total_mass_kg = (total_g_per_mol / _N_AVOGADRO) * 1e-3     # g/mol -> g -> kg
    volume_m3 = total_mass_kg / density_kg_m3
    volume_ang3 = volume_m3 * 1e30                              # m³ -> Å³
    return volume_ang3 ** (1.0 / 3.0)


def _shape_for_density(
    specs: Sequence[PackSpec], mols: Sequence[Molecule], density_kg_m3: float,
    template: Optional[BoxShape] = None,
) -> BoxShape:
    """Return a BoxShape that hits `density_kg_m3`.

    If `template` is None → cubic with the derived side length.
    Otherwise the template's aspect ratio and angles are preserved and edge
    lengths are uniformly scaled to hit the target volume.
    """
    total_g_per_mol = sum(spec.count * _mass_of(mol) for spec, mol in zip(specs, mols))
    total_mass_kg = (total_g_per_mol / _N_AVOGADRO) * 1e-3
    target_volume_ang3 = (total_mass_kg / density_kg_m3) * 1e30
    if template is None:
        side = target_volume_ang3 ** (1.0 / 3.0)
        return BoxShape(shape="cubic", a=side, b=side, c=side)
    current = template.volume_ang3()
    scale = (target_volume_ang3 / max(current, 1e-9)) ** (1.0 / 3.0)
    return template.scaled(scale)


def _merge(molecules: List[Molecule]) -> Molecule:
    """Concatenate a list of molecules into one, renumbering atom indices."""
    atoms: List[Atom] = []
    bonds: List[Tuple[int, int, float]] = []
    offset = 0
    for mol in molecules:
        for a in mol.atoms:
            atoms.append(Atom(
                index=a.index + offset, element=a.element,
                xyz=a.xyz.copy(), name=a.name, charge=a.charge, ff_type=a.ff_type,
            ))
        for i, j, o in mol.bonds:
            bonds.append((i + offset, j + offset, o))
        offset += len(mol.atoms)
    # Compact indices to 0..N-1
    for k, a in enumerate(atoms):
        a.index = k
    return Molecule(atoms=atoms, bonds=bonds, name="packed")


def _translate(mol: Molecule, delta: np.ndarray) -> Molecule:
    """Return a copy of `mol` with all atoms shifted by `delta`."""
    out = Molecule(
        atoms=[Atom(a.index, a.element, a.xyz + delta, a.name, a.charge, a.ff_type)
               for a in mol.atoms],
        bonds=list(mol.bonds),
        name=mol.name,
    )
    return out


def _random_rotation() -> np.ndarray:
    """Random 3x3 rotation matrix (uniform on SO(3))."""
    from math import cos, sin, pi
    q = np.random.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


# ============================================================ packmol backend
def _pack_with_packmol(
    specs: Sequence[PackSpec], mols: Sequence[Molecule], box: BoxShape,
    tolerance: float = 2.0, seed: int = 12345,
) -> Optional[Molecule]:
    exe = shutil.which("packmol")
    if exe is None:
        return None
    tmp = Path(tempfile.mkdtemp(prefix="mta_pack_"))
    inp = tmp / "pack.inp"
    files: list[Path] = []
    for i, (spec, mol) in enumerate(zip(specs, mols)):
        fp = tmp / f"species_{i}.pdb"
        write(mol, fp)
        files.append(fp)
    out = tmp / "packed.pdb"
    # Region syntax per packmol manual:
    #   inside cube x y z L
    #   inside box  x1 y1 z1 x2 y2 z2
    # For triclinic we pack inside the bounding orthorhombic box; the
    # returned Molecule carries the triclinic cell vectors as metadata.
    bx, by, bz = box.bounding_box()
    if box.shape == "cubic":
        region = f"inside cube 0. 0. 0. {box.a:.4f}"
    else:
        region = f"inside box 0. 0. 0. {bx:.4f} {by:.4f} {bz:.4f}"
    lines = [
        f"tolerance {tolerance}",
        f"seed {seed}",
        "filetype pdb",
        f"output {out}",
    ]
    for spec, fp in zip(specs, files):
        lines += [
            f"structure {fp}",
            f"  number {spec.count}",
            f"  {region}",
            "end structure",
        ]
    inp.write_text("\n".join(lines) + "\n")
    log.info("Running packmol (%s box)...", box.shape)
    with inp.open() as fh:
        proc = subprocess.run([exe], stdin=fh, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        log.warning("packmol failed:\n%s", proc.stdout + proc.stderr)
        return None
    return load_structure(out, name="packed")


# ============================================================ mbuild backend
def _pack_with_mbuild(
    specs: Sequence[PackSpec], mols: Sequence[Molecule], box: BoxShape,
    seed: int = 12345,
) -> Optional[Molecule]:
    try:
        import mbuild as mb
    except Exception:
        return None
    from ..chain_builder import _molecule_to_mbuild, _mbuild_to_molecule
    compounds = [_molecule_to_mbuild(m) for m in mols]
    counts = [spec.count for spec in specs]
    # mbuild.fill_box expects nm and an orthorhombic box.
    bx, by, bz = box.bounding_box()
    try:
        packed = mb.fill_box(
            compound=compounds if len(compounds) > 1 else compounds[0],
            n_compounds=counts if len(compounds) > 1 else counts[0],
            box=[bx * 0.1, by * 0.1, bz * 0.1],
            overlap=0.2, seed=seed,
        )
        return _mbuild_to_molecule(packed, name="packed")
    except Exception as exc:
        log.warning("mbuild.fill_box failed: %s", exc)
        return None


# ============================================================ grid fallback
def _pack_grid(
    specs: Sequence[PackSpec], mols: Sequence[Molecule], box: BoxShape,
    seed: int = 12345,
) -> Molecule:
    """Grid packer — no overlap checks; used as a fallback / for tests.

    Supports non-cubic orthorhombic boxes by using different grid step sizes
    per axis. For triclinic, we pack inside the bounding orthorhombic box.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    n_total = sum(s.count for s in specs)
    bx, by, bz = box.bounding_box()
    n_side = max(1, int(math.ceil(n_total ** (1.0 / 3.0))))
    step_x, step_y, step_z = bx / n_side, by / n_side, bz / n_side
    positions = [
        np.array([(i + 0.5) * step_x, (j + 0.5) * step_y, (k + 0.5) * step_z])
        for i in range(n_side) for j in range(n_side) for k in range(n_side)
    ]
    rng.shuffle(positions)
    placed: List[Molecule] = []
    idx = 0
    for spec, mol in zip(specs, mols):
        for _ in range(spec.count):
            center = mol.coords().mean(axis=0)
            rot = _random_rotation()
            xyz = (mol.coords() - center) @ rot.T
            copy = Molecule(
                atoms=[Atom(a.index, a.element, xyz[k].copy(), a.name, a.charge, a.ff_type)
                       for k, a in enumerate(mol.atoms)],
                bonds=list(mol.bonds), name=mol.name,
            )
            copy = _translate(copy, positions[idx])
            placed.append(copy)
            idx += 1
    return _merge(placed)


# ============================================================ top-level
def pack_cell(
    specs: Sequence[PackSpec],
    density_kg_m3: Optional[float] = None,
    box_ang: Optional[float] = None,
    shape: Optional[BoxShape | tuple | dict | float] = None,
    out_path: Optional[str | Path] = None,
    backend: str = "auto",
    seed: int = 12345,
) -> Tuple[Molecule, BoxShape]:
    """Pack molecules into a periodic box of any shape.

    Parameters
    ----------
    specs : list of PackSpec
        One entry per species (file/molecule + count).
    density_kg_m3 : float, optional
        Target density. If given, the box is sized to hit it. When combined
        with ``shape``, edge lengths are scaled uniformly to preserve the
        shape's aspect ratio and angles.
    box_ang : float or 3-tuple, optional
        Explicit cubic side length (single float) or orthorhombic edge
        lengths ``(a, b, c)``. Ignored when ``shape`` is given.
    shape : BoxShape / tuple / dict / float, optional
        Explicit box shape. Formats accepted:
          - ``BoxShape`` instance
          - ``float`` → cubic
          - ``(a, b, c)`` → orthorhombic
          - ``(a, b, c, α, β, γ)`` → triclinic
          - ``dict`` matching the ``BoxShape`` fields

    Returns
    -------
    (packed_molecule, box) : Molecule + BoxShape
    """
    if not specs:
        raise ValueError("At least one PackSpec is required")
    mols = [_load_spec(s) for s in specs]

    # Resolve the target box shape.
    template: Optional[BoxShape] = BoxShape.from_any(shape) if shape is not None else None
    if density_kg_m3 is not None:
        box = _shape_for_density(specs, mols, density_kg_m3, template=template)
        log.info("Box from target density %g kg/m³: %s (%.2f Å³)",
                 density_kg_m3, box.shape, box.volume_ang3())
    else:
        if template is not None:
            box = template
        elif box_ang is not None:
            box = BoxShape.from_any(box_ang)
        else:
            raise ValueError("Provide density_kg_m3, box_ang, or shape")

    log.info("Packing %d atoms into %s box (%.2f x %.2f x %.2f Å)",
             sum(s.count * len(m.atoms) for s, m in zip(specs, mols)),
             box.shape, *box.bounding_box())

    packed: Optional[Molecule] = None
    if backend in ("auto", "packmol"):
        packed = _pack_with_packmol(specs, mols, box, seed=seed)
    if packed is None and backend in ("auto", "mbuild"):
        packed = _pack_with_mbuild(specs, mols, box, seed=seed)
    if packed is None:
        log.info("Using grid fallback packer (install packmol / mbuild for production)")
        packed = _pack_grid(specs, mols, box, seed=seed)

    # Attach the (possibly triclinic) cell to the Molecule so downstream
    # writers can emit the correct LAMMPS/GROMACS box directives.
    setattr(packed, "cell", box)

    if out_path:
        write(packed, out_path)
        log.info("Wrote packed cell (%d atoms) -> %s", len(packed.atoms), out_path)
    return packed, box
