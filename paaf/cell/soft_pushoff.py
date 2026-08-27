"""Resolve close contacts in a back-mapped cell with a soft-core LAMMPS run.

Why the geometric push-off is not enough
----------------------------------------
The in-Python push-off moves atoms apart pairwise while restoring bond and
1-3 lengths. Measured on a real 10,560-atom PE/PS cell at 0.57 g/cm3 it
stalls: after 150 iterations, 1,167 non-bonded pairs remain closer than
1.65 Å — hydrogens trapped between PINNED backbone atoms have nowhere to go,
because the geometric scheme cannot move the backbone and cannot trade one
contact for another the way concerted motion can. DL_FIELD bonds anything
that close, sees five-coordinate carbons, and refuses to type the cell.

What this does instead
----------------------
The standard preparation-stage push-off (Kremer & Grest): a short LAMMPS run
with the purely repulsive ``pair_style soft``, whose amplitude is ramped from
almost nothing to a full push while ``fix nve/limit`` caps how far any atom
moves per step. Bonds are held by harmonic springs at their current per-class
lengths. No angles, no dihedrals, no charges — and crucially **no atom
types**, which is what breaks the circularity: the cell cannot be typed until
the contacts are gone, and this removes the contacts without needing types.

The backbone is NOT pinned here. That is deliberate: the bead positions were
decided by growth statistics, but a few Å of local adjustment under a soft
potential does not change chain topology or dimensions materially, and it is
exactly what lets the trapped hydrogens escape.

Everything is elementary LAMMPS (soft, harmonic, nve/limit), available in
any build.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..logging_utils import get_logger
from ..structure import Molecule

log = get_logger(__name__)

__all__ = ["soft_pushoff", "write_pushoff_inputs"]

#: Atomic masses for the data file; anything absent falls back to carbon.
_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
         "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Br": 79.904}


def _bond_classes(molecule: Molecule, dims: np.ndarray
                  ) -> Tuple[List[int], Dict[Tuple[str, str], int],
                             List[float]]:
    """One bond type per element pair, with r0 = that class's median length.

    Median, not mean: a handful of bonds stretched across a defect would
    otherwise drag the rest length of the whole class with them.
    """
    xyz = np.array([a.xyz for a in molecule.atoms], dtype=float)
    lengths: Dict[Tuple[str, str], List[float]] = {}
    per_bond_key: List[Tuple[str, str]] = []
    for i, j, _o in molecule.bonds:
        key = tuple(sorted((molecule.atoms[int(i)].element,
                            molecule.atoms[int(j)].element)))
        d = xyz[int(i)] - xyz[int(j)]
        d -= dims * np.round(d / dims)
        lengths.setdefault(key, []).append(float(np.linalg.norm(d)))
        per_bond_key.append(key)
    type_of_key = {k: n + 1 for n, k in enumerate(sorted(lengths))}
    r0 = [float(np.median(lengths[k])) for k in sorted(lengths)]
    bond_types = [type_of_key[k] for k in per_bond_key]
    return bond_types, type_of_key, r0


def _angle_classes(molecule: Molecule, dims: np.ndarray):
    """Angles (i-j-k, j centre) classed by element triple, theta0 = median.

    The input geometry is template-built, so its angles are correct; the
    median per class is immune to the handful the geometric stages bent.
    """
    xyz = np.array([a.xyz for a in molecule.atoms], dtype=float)
    adj: Dict[int, List[int]] = {}
    for i, j, _o in molecule.bonds:
        adj.setdefault(int(i), []).append(int(j))
        adj.setdefault(int(j), []).append(int(i))
    triples: List[Tuple[int, int, int]] = []
    keys: List[Tuple[str, str, str]] = []
    values: Dict[Tuple[str, str, str], List[float]] = {}
    for j, nbrs in adj.items():
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                i, k = nbrs[a], nbrs[b]
                e_i, e_k = sorted((molecule.atoms[i].element,
                                   molecule.atoms[k].element))
                key = (e_i, molecule.atoms[j].element, e_k)
                u = xyz[i] - xyz[j]
                u -= dims * np.round(u / dims)
                v = xyz[k] - xyz[j]
                v -= dims * np.round(v / dims)
                nu, nv = np.linalg.norm(u), np.linalg.norm(v)
                if nu < 1e-9 or nv < 1e-9:
                    continue
                ang = float(np.degrees(np.arccos(
                    np.clip((u @ v) / (nu * nv), -1.0, 1.0))))
                triples.append((i, j, k))
                keys.append(key)
                values.setdefault(key, []).append(ang)
    type_of = {k: n + 1 for n, k in enumerate(sorted(values))}
    theta0 = [float(np.median(values[k])) for k in sorted(values)]
    angle_types = [type_of[k] for k in keys]
    return triples, angle_types, theta0


def write_pushoff_inputs(molecule: Molecule, dims, work_dir: Path,
                         *, cutoff: float = 2.2, steps: int = 4000,
                         seed: int = 20260820) -> Tuple[Path, Path]:
    """Write ``pushoff.data`` and ``pushoff.in``; return their paths.

    The data file is ``atom_style molecular`` with one atom type per element
    and one bond type per element pair. It deliberately claims nothing about
    the force field — its only job is to carry positions and connectivity.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dims = np.asarray(dims, dtype=float)
    xyz = np.array([a.xyz for a in molecule.atoms], dtype=float)
    xyz = xyz - np.floor(xyz / dims) * dims          # inside the box

    elements = sorted({a.element for a in molecule.atoms})
    at_of = {e: n + 1 for n, e in enumerate(elements)}
    bond_types, _key_of, r0 = _bond_classes(molecule, dims)
    triples, angle_types, theta0 = _angle_classes(molecule, dims)

    lines = ["PAAF soft push-off (untyped by design)", "",
             f"{len(molecule.atoms)} atoms",
             f"{len(molecule.bonds)} bonds",
             f"{len(triples)} angles", "",
             f"{len(elements)} atom types",
             f"{len(r0)} bond types",
             f"{len(theta0)} angle types", "",
             f"0.0 {dims[0]:.6f} xlo xhi",
             f"0.0 {dims[1]:.6f} ylo yhi",
             f"0.0 {dims[2]:.6f} zlo zhi", "",
             "Masses", ""]
    lines += [f"{n} {_MASS.get(e, 12.011):.4f}  # {e}"
              for e, n in sorted(at_of.items(), key=lambda kv: kv[1])]
    lines += ["", "Atoms # molecular", ""]
    for k, a in enumerate(molecule.atoms):
        lines.append(f"{k + 1} 1 {at_of[a.element]} "
                     f"{xyz[k][0]:.6f} {xyz[k][1]:.6f} {xyz[k][2]:.6f}")
    lines += ["", "Bonds", ""]
    for b, (i, j, _o) in enumerate(molecule.bonds):
        lines.append(f"{b + 1} {bond_types[b]} {int(i) + 1} {int(j) + 1}")
    lines += ["", "Angles", ""]
    for m, (i, j, k) in enumerate(triples):
        lines.append(f"{m + 1} {angle_types[m]} {i + 1} {j + 1} {k + 1}")
    data = work_dir / "pushoff.data"
    data.write_text("\n".join(lines) + "\n")

    inp = work_dir / "pushoff.in"
    script = ["units real", "atom_style molecular", "boundary p p p",
              f"read_data {data.name}",
              "",
              f"pair_style soft {cutoff}",
              "pair_coeff * * 0.0",
              "# 1-2 excluded (the bond holds them). 1-3 excluded TOO, and",
              "# held by explicit angle terms instead: with 1-3 repulsion on",
              "# and no angles, the two hydrogens of every CH2 pushed each",
              "# other apart until H-C-H measured 127 degrees -- and DL_F",
              "# Notation reads a splayed angle as sp2, typing a perfect",
              "# methylene as an unsaturated carbon.",
              "special_bonds lj 0.0 0.0 1.0",
              ""]
    script += [f"bond_style harmonic"]
    script += [f"bond_coeff {n + 1} 300.0 {r:.4f}" for n, r in enumerate(r0)]
    script += ["", "angle_style harmonic"]
    script += [f"angle_coeff {n + 1} 60.0 {a0:.2f}"
               for n, a0 in enumerate(theta0)]
    script += ["",
               "# Ramp the repulsion from almost nothing to a full push while",
               "# nve/limit caps the per-step displacement — the standard",
               "# preparation-stage push-off (Kremer & Grest).",
               "variable prefactor equal ramp(1.0,100.0)",
               "fix push all adapt 1 pair soft a * * v_prefactor",
               "fix integrate all nve/limit 0.05",
               f"fix cool all langevin 300.0 300.0 100.0 {seed}",
               "",
               "thermo 500",
               f"run {steps}",
               "",
               "# Settle: dynamics ends mid-push with bonds slightly",
               "# stretched (C-H measured at 1.16-1.18 A against 1.09) and a",
               "# last contact or two not yet cleared — on a real cell,",
               "# exactly 2 pairs remained and DL_FIELD still refused. A",
               "# minimisation at full repulsion finishes both at once.",
               "unfix push",
               "unfix cool",
               "unfix integrate",
               "pair_coeff * * 100.0",
               "minimize 1.0e-4 1.0e-6 2000 20000",
               "",
               "# Tighten the bonds LAST, with the repulsion nearly off.",
               "# At full A the springs balance the push and every bond ends",
               "# ~3% long: C-C measured at 1.569-1.575 A against 1.53 -- and",
               "# DL_FIELD's C-C bond cutoff sits right there, so it read a",
               "# perfect CH2 as an unsaturated carbon (2 bonds) and refused",
               "# the cell. Soft is purely repulsive, so dropping A cannot",
               "# pull cleared contacts back together; the springs just win.",
               "pair_coeff * * 5.0",
               "minimize 1.0e-6 1.0e-8 4000 40000",
               "",
               "write_dump all custom pushoff_final.dump id x y z modify sort id"]
    inp.write_text("\n".join(script) + "\n")
    return data, inp


def soft_pushoff(molecule: Molecule, dims, work_dir: Path,
                 *, lammps_exe: str = "",
                 steps: int = 4000,
                 timeout_s: int = 900,
                 cancel=None,
                 emit: Optional[Callable[[str], None]] = None) -> bool:
    """Run the push-off and write the relaxed coordinates back in place.

    Returns ``True`` on success. On any failure the molecule is left exactly
    as it was and the reason is emitted — a cell with contacts is still
    exportable, it just will not type.
    """
    say = emit or (lambda m: None)
    from .relax import find_lammps

    exe = find_lammps(lammps_exe)
    if exe is None:
        say("  soft push-off skipped: no LAMMPS executable found")
        return False

    work_dir = Path(work_dir)
    _data, inp = write_pushoff_inputs(molecule, dims, work_dir, steps=steps)
    say(f"  soft push-off: {exe.name}, {steps} steps of pair_style soft …")
    from .packing import run_cancellable
    try:
        rc, out, err = run_cancellable([str(exe), "-in", inp.name],
                                       cwd=work_dir, timeout_s=timeout_s,
                                       cancel=cancel)
    except TimeoutError:
        say(f"  soft push-off timed out after {timeout_s}s; coordinates "
            f"unchanged")
        return False
    if rc != 0:
        tail = (err or out or "").strip().splitlines()[-4:]
        say("  soft push-off failed:\n    " + "\n    ".join(tail))
        return False

    dump = work_dir / "pushoff_final.dump"
    if not dump.exists():
        say("  soft push-off wrote no dump; coordinates unchanged")
        return False
    rows: List[Tuple[int, float, float, float]] = []
    in_atoms = False
    for line in dump.read_text().splitlines():
        if line.startswith("ITEM: ATOMS"):
            in_atoms = True
            continue
        if line.startswith("ITEM:"):
            in_atoms = False
            continue
        if in_atoms:
            p = line.split()
            rows.append((int(p[0]), float(p[1]), float(p[2]), float(p[3])))
    if len(rows) != len(molecule.atoms):
        say(f"  dump has {len(rows)} atoms, cell has "
            f"{len(molecule.atoms)}; coordinates unchanged")
        return False
    rows.sort()
    for (ident, x, y, z), atom in zip(rows, molecule.atoms):
        atom.xyz = np.array([x, y, z])
    say("  soft push-off done")
    return True
