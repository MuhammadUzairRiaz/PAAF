"""A stand-in for DL_FIELD that writes files shaped like the real thing.

It reads the .xyz it is given, perceives bonds from distance (as DL_FIELD
does), types atoms by element, and writes lammps1.data + lammps.in or
gromacs.gro + gromacs.top + gromacs1.itp with the same sections, comments
and numbering DL_FIELD 4.13 uses. Optional limits reproduce DL_FIELD's
compiled-in array sizes.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

LABEL = {"C": "CT", "H": "HC", "O": "OS", "N": "NT", "S": "SH"}
MASS = {"CT": 12.0115, "HC": 1.00797, "OS": 15.9994, "NT": 14.0067, "SH": 32.06}
CHARGE = {"CT": -0.12, "HC": 0.06, "OS": -0.4, "NT": -0.9, "SH": -0.3}
LJ = {"CT": (0.066, 3.5), "HC": (0.03, 2.5), "OS": (0.14, 2.9),
      "NT": (0.17, 3.25), "SH": (0.25, 3.55)}


class FakeDLField:
    def __init__(self, max_atoms: Optional[int] = None, drop_bond: bool = False):
        self.max_atoms = max_atoms
        self.drop_bond = drop_bond
        self.calls: List[dict] = []

    def __call__(self, structure, work_dir, ff_key, dl_dir, emit,
                 output_engine="lammps", box_ang=None, cancel=None):
        assert box_ang is not None, "the box must always reach DL_FIELD"
        work = Path(work_dir)
        out = work / "dlf_output1"
        out.mkdir(parents=True, exist_ok=True)
        rows = [l.split() for l in Path(structure).read_text().splitlines()[2:]
                if l.strip()]
        els = [r[0] for r in rows]
        xyz = np.array([[float(v) for v in r[1:4]] for r in rows])
        self.calls.append({"engine": output_engine, "n_atoms": len(els),
                           "box": box_ang})
        if self.max_atoms is not None and len(els) > self.max_atoms:
            (work / "dl_field.log").write_text(
                "Begin ATOM_TYPE determination using DL_F Notation...\n"
                "Error: too many C identified.\n")
            return None

        bonds = []
        for i in range(len(els)):
            d = np.linalg.norm(xyz[i + 1:] - xyz[i], axis=1)
            for k in np.nonzero(d < 1.75)[0]:
                j = i + 1 + int(k)
                if "H" in (els[i], els[j]) and d[k] > 1.25:
                    continue
                if els[i] == "H" and els[j] == "H":
                    continue
                bonds.append((i, j))
        if self.drop_bond and bonds:
            bonds = bonds[1:]
        adj = [[] for _ in els]
        for i, j in bonds:
            adj[i].append(j)
            adj[j].append(i)
        angles = [(i, j, k) for j in range(len(els))
                  for a, i in enumerate(adj[j]) for k in adj[j][a + 1:]]

        labels = [LABEL.get(e, e) for e in els]
        order: List[str] = []
        for lb in labels:
            if lb not in order:
                order.append(lb)
        tid = {lb: n + 1 for n, lb in enumerate(order)}
        bkeys: List[tuple] = []
        for i, j in bonds:
            k = tuple(sorted((labels[i], labels[j])))
            if k not in bkeys:
                bkeys.append(k)
        akeys: List[tuple] = []
        for i, j, k in angles:
            key = (min(labels[i], labels[k]), labels[j], max(labels[i], labels[k]))
            if key not in akeys:
                akeys.append(key)

        if output_engine == "lammps":
            L = [f"# LAMMPS data file. Produced from DL_FIELD 4.13", "",
                 f"{len(els)} atoms", f"{len(bonds)} bonds",
                 f"{len(angles)} angles", "0 dihedrals", "0 impropers",
                 f"{len(order)} atom types", f"{len(bkeys)} bond types",
                 f"{len(akeys)} angle types", "",
                 "-50 50 xlo xhi", "-50 50 ylo yhi", "-50 50 zlo zhi", "",
                 "Masses", ""]
            L += [f"   {tid[lb]}  {MASS[lb]:.6f}   # {lb}" for lb in order]
            L += ["", "Bond Coeffs", ""]
            L += [f"   {n + 1} harmonic  {300 + n:.6f}  {1.1 + 0.4 * ('CT' in k and k[0] == k[1]):.6f}"
                  for n, k in enumerate(bkeys)]
            L += ["", "Angle Coeffs", ""]
            L += [f"   {n + 1} harmonic  {35 + n:.6f}  109.500000"
                  for n, _ in enumerate(akeys)]
            L += ["", "Atoms", ""]
            L += [f"{i + 1} 1 {tid[labels[i]]} {CHARGE[labels[i]]:.6f} "
                  f"{xyz[i][0]:.6f} {xyz[i][1]:.6f} {xyz[i][2]:.6f} # {labels[i]}"
                  for i in range(len(els))]
            L += ["", "Bonds", ""]
            L += [f"{n + 1} {bkeys.index(tuple(sorted((labels[i], labels[j])))) + 1} "
                  f"{i + 1} {j + 1}" for n, (i, j) in enumerate(bonds)]
            if angles:
                L += ["", "Angles", ""]
                L += [f"{n + 1} {akeys.index((min(labels[i], labels[k]), labels[j], max(labels[i], labels[k]))) + 1} "
                      f"{i + 1} {j + 1} {k + 1}"
                      for n, (i, j, k) in enumerate(angles)]
            (out / "lammps1.data").write_text("\n".join(L) + "\n")
            pc = []
            for a in order:
                for b in order:
                    if tid[a] <= tid[b]:
                        e = (LJ[a][0] * LJ[b][0]) ** 0.5
                        s = (LJ[a][1] * LJ[b][1]) ** 0.5
                        pc.append(f"pair_coeff {tid[a]:5d} {tid[b]:5d}  "
                                  f"lj/cut/coul/long {e:.6f} {s:.6f} # {a} {b}")
            (out / "lammps.in").write_text(
                "units real\natom_style full\n"
                "bond_style      hybrid harmonic\n"
                "angle_style     hybrid harmonic\n"
                "pair_style      hybrid lj/cut/coul/long 12.0\n"
                "special_bonds   lj 0.0 0.0 0.5 coul 0.0 0.0 0.5\n\n"
                "read_data lammps1.data\n\n"
                f"group XYZ type {' '.join(str(tid[a]) for a in order)}\n\n"
                + "\n".join(pc) + "\n")
            return out / "lammps1.data"

        (out / "gromacs.gro").write_text(f"fake\n{len(els)}\n   5.0 5.0 5.0\n")
        top = ["[ defaults ]", "1 1 no 0.5 0.5", "", "[ atomtypes ]"]
        top += [f"{lb} 6 {MASS[lb]:.6f} 0.0 A 1.0e-03 1.0e-06" for lb in order]
        top += ["", '#include "gromacs1.itp"', "", "[ system ]", "fake", "",
                "[ molecules ]", "XYZ 1"]
        (out / "gromacs.top").write_text("\n".join(top) + "\n")
        itp = ["[ moleculetype ]", "XYZ 3", "", "[ atoms ]"]
        itp += [f"{i + 1} {labels[i]} 1 MOL {els[i]}{i + 1} {i + 1} "
                f"{CHARGE[labels[i]]:.6f} {MASS[labels[i]]:.6f}"
                for i in range(len(els))]
        itp += ["", "[ bonds ]"] + [f"{i + 1} {j + 1} 1 0.15 250000.0"
                                    for i, j in bonds]
        itp += ["", "[ angles ]"] + [f"{i + 1} {j + 1} {k + 1} 1 109.5 300.0"
                                     for i, j, k in angles]
        (out / "gromacs1.itp").write_text("\n".join(itp) + "\n")
        return out / "gromacs.gro"
