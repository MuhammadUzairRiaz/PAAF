"""Convert DL_FIELD 4.x .par files into Moltemplate .lt force-field libraries.

DL_FIELD .par files are section-based:

    UNIT kcal/mol
    POTENTIAL PCFF
    BOND      b0   K2   K3   K4    Remark
    atomA  atomB  1.5300  299.67 -501.77 679.81   ver 2.1
    ...
    END BOND
    ANGLE     ...
    END ANGLE
    DIHEDRAL  ...
    END DIHEDRAL
    IMPROPER  ...
    END IMPROPER
    VDW       ...            (or NONBOND)
    END VDW
    (plus cross terms for class-II FFs; those are emitted as comments unless
    the target LAMMPS pair_style supports them.)

The converter emits a Moltemplate .lt file of the form:

    <FFName> {
      write_once("In Init") {
        units           real
        atom_style      full
        bond_style      class2       # (or harmonic, morse, ...)
        angle_style     class2
        dihedral_style  class2
        improper_style  class2
        pair_style      lj/class2/coul/long 10.0
        pair_modify     mix arithmetic
        special_bonds   lj/coul 0.0 0.0 1.0
        kspace_style    pppm 1.0e-4
      }
      write_once("Data Masses") {
        @atom:X  mass
        ...
      }
      write_once("In Settings") {
        pair_coeff      @atom:X @atom:Y eps sigma
        bond_coeff      @bond:X-Y k b0 ...
        ...
      }
    }

The mapping from DL_FIELD potential styles to LAMMPS is:

  class-I (CVFF, OPLS, DREIDING, CHARMM, AMBER, GROMOS, TraPPE):
    bond_style      harmonic (or morse)
    angle_style     harmonic (or harmonic cos)
    dihedral_style  fourier or opls
    improper_style  cvff / harmonic
    pair_style      lj/cut/coul/long 10.0

  class-II (PCFF, COMPASS):
    bond_style      class2
    angle_style     class2
    dihedral_style  class2
    improper_style  class2
    pair_style      lj/class2/coul/long 10.0
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .logging_utils import get_logger

log = get_logger(__name__)


CLASS2_FFS = {"PCFF", "COMPASS"}


# ================================================================== parsing
@dataclass
class DLParams:
    ff: str = ""
    unit: str = "kcal/mol"
    atoms: List[Dict] = field(default_factory=list)   # from .sf, optional
    bonds: List[Dict] = field(default_factory=list)
    angles: List[Dict] = field(default_factory=list)
    dihedrals: List[Dict] = field(default_factory=list)
    impropers: List[Dict] = field(default_factory=list)
    vdw: List[Dict] = field(default_factory=list)


def parse_par(path: str | Path) -> DLParams:
    path = Path(path)
    dl = DLParams()
    section: Optional[str] = None
    header_seen = False
    with path.open() as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.split("!", 1)[0].split("#", 1)[0].strip()
            if not stripped:
                continue
            if stripped.startswith("UNIT"):
                parts = stripped.split()
                if len(parts) >= 2:
                    dl.unit = parts[1]
                continue
            if stripped.startswith("POTENTIAL"):
                parts = stripped.split()
                if len(parts) >= 2:
                    dl.ff = parts[1].upper()
                continue
            up = stripped.upper()
            if up.startswith("END "):
                section = None
                header_seen = False
                continue
            for tag in ("BOND", "ANGLE", "DIHEDRAL", "IMPROPER", "VDW", "NONBOND",
                        "ATOM_TYPE", "EQUIVALENCE", "AUTO", "BOND_INCREMENT"):
                if up.startswith(tag):
                    section = tag if tag != "NONBOND" else "VDW"
                    header_seen = False
                    break
            else:
                if section is None:
                    continue
                if section in ("EQUIVALENCE", "AUTO", "BOND_INCREMENT", "ATOM_TYPE"):
                    # not needed for LAMMPS coefficients; skip content
                    continue
                # skip the column header line just once
                if not header_seen and _looks_like_header(stripped):
                    header_seen = True
                    continue
                parts = stripped.split()
                if section == "BOND":
                    if len(parts) >= 4:
                        entry = {
                            "a": parts[0], "b": parts[1],
                            "b0": float(parts[2]),
                            "k2": float(parts[3]),
                        }
                        if len(parts) >= 6:
                            try:
                                entry["k3"] = float(parts[4])
                                entry["k4"] = float(parts[5])
                            except ValueError:
                                pass
                        dl.bonds.append(entry)
                elif section == "ANGLE":
                    if len(parts) >= 5:
                        entry = {
                            "a": parts[0], "b": parts[1], "c": parts[2],
                            "theta0": float(parts[3]),
                            "k2": float(parts[4]),
                        }
                        if len(parts) >= 7:
                            try:
                                entry["k3"] = float(parts[5])
                                entry["k4"] = float(parts[6])
                            except ValueError:
                                pass
                        dl.angles.append(entry)
                elif section == "DIHEDRAL":
                    if len(parts) >= 4:
                        try:
                            dl.dihedrals.append({
                                "a": parts[0], "b": parts[1], "c": parts[2], "d": parts[3],
                                "params": [float(x) for x in parts[4:] if _is_num(x)],
                            })
                        except ValueError:
                            pass
                elif section == "IMPROPER":
                    if len(parts) >= 4:
                        try:
                            dl.impropers.append({
                                "a": parts[0], "b": parts[1], "c": parts[2], "d": parts[3],
                                "params": [float(x) for x in parts[4:] if _is_num(x)],
                            })
                        except ValueError:
                            pass
                elif section == "VDW":
                    if len(parts) >= 3:
                        try:
                            dl.vdw.append({
                                "a": parts[0],
                                "eps": float(parts[1]),
                                "sigma": float(parts[2]),
                            })
                        except ValueError:
                            pass
    log.info(
        "Parsed %s: %d bonds, %d angles, %d dihedrals, %d impropers, %d vdw",
        path.name, len(dl.bonds), len(dl.angles), len(dl.dihedrals),
        len(dl.impropers), len(dl.vdw),
    )
    return dl


def _looks_like_header(line: str) -> bool:
    up = line.upper().split()
    if not up:
        return False
    return up[0] in {"B0", "THETA0", "TYPE", "K2", "K3", "K4"} or any(
        tok in {"REMARK", "B0", "K2", "THETA0", "K", "SIG", "EPS"} for tok in up
    )


def _is_num(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


# ================================================================ emit .lt
def to_moltemplate(
    dl: DLParams,
    out_path: str | Path,
    masses: Optional[Dict[str, float]] = None,
    ff_object_name: Optional[str] = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ff = ff_object_name or dl.ff or "GENERIC"
    class2 = dl.ff.upper() in CLASS2_FFS

    L: List[str] = []
    L.append(f"# Auto-generated by paaf.dlfield_converter")
    L.append(f"# Source force field: {dl.ff}   Units: {dl.unit}")
    L.append(f"# Class-II: {class2}")
    L.append("")
    L.append(f"{ff} {{")
    L.append("")
    L.append('  write_once("In Init") {')
    L.append("    units           real")
    L.append("    atom_style      full")
    if class2:
        L.append("    bond_style      class2")
        L.append("    angle_style     class2")
        L.append("    dihedral_style  class2")
        L.append("    improper_style  class2")
        L.append("    pair_style      lj/class2/coul/long 10.0")
    else:
        L.append("    bond_style      harmonic")
        L.append("    angle_style     harmonic")
        L.append("    dihedral_style  opls")
        L.append("    improper_style  cvff")
        L.append("    pair_style      lj/cut/coul/long 10.0 10.0")
    L.append("    pair_modify     mix arithmetic")
    L.append("    special_bonds   lj/coul 0.0 0.0 0.5")
    L.append("    kspace_style    pppm 1.0e-4")
    L.append("  }")
    L.append("")

    # masses -----------------------------------------------------------
    if masses:
        L.append('  write_once("Data Masses") {')
        for t, m in sorted(masses.items()):
            L.append(f"    @atom:{t}  {m}")
        L.append("  }")
        L.append("")

    # pair coeffs ------------------------------------------------------
    if dl.vdw:
        L.append('  write_once("In Settings") {')
        for e in dl.vdw:
            L.append(
                f"    pair_coeff  @atom:{e['a']}  @atom:{e['a']}  {e['eps']:.6f}  {e['sigma']:.6f}"
            )
        L.append("  }")
        L.append("")

    # bond coeffs ------------------------------------------------------
    if dl.bonds:
        L.append('  write_once("In Settings") {')
        for i, e in enumerate(dl.bonds):
            bname = f"{e['a']}-{e['b']}"
            if class2 and "k3" in e:
                L.append(
                    f"    bond_coeff  @bond:{bname}  {e['b0']:.4f}  {e['k2']:.4f}  {e.get('k3',0.0):.4f}  {e.get('k4',0.0):.4f}"
                )
            else:
                L.append(
                    f"    bond_coeff  @bond:{bname}  {e['k2']:.4f}  {e['b0']:.4f}"
                )
        L.append("  }")
        L.append("")

    # angle coeffs -----------------------------------------------------
    if dl.angles:
        L.append('  write_once("In Settings") {')
        for e in dl.angles:
            aname = f"{e['a']}-{e['b']}-{e['c']}"
            if class2 and "k3" in e:
                L.append(
                    f"    angle_coeff  @angle:{aname}  {e['theta0']:.4f}  {e['k2']:.4f}  {e.get('k3',0.0):.4f}  {e.get('k4',0.0):.4f}"
                )
            else:
                L.append(
                    f"    angle_coeff  @angle:{aname}  {e['k2']:.4f}  {e['theta0']:.4f}"
                )
        L.append("  }")
        L.append("")

    # dihedrals --------------------------------------------------------
    if dl.dihedrals:
        L.append('  write_once("In Settings") {')
        for e in dl.dihedrals:
            dname = f"{e['a']}-{e['b']}-{e['c']}-{e['d']}"
            p = " ".join(f"{v:.4f}" for v in e["params"])
            L.append(f"    dihedral_coeff  @dihedral:{dname}  {p}")
        L.append("  }")
        L.append("")

    # impropers --------------------------------------------------------
    if dl.impropers:
        L.append('  write_once("In Settings") {')
        for e in dl.impropers:
            dname = f"{e['a']}-{e['b']}-{e['c']}-{e['d']}"
            p = " ".join(f"{v:.4f}" for v in e["params"])
            L.append(f"    improper_coeff  @improper:{dname}  {p}")
        L.append("  }")
        L.append("")

    L.append(f"}} # {ff}")
    out_path.write_text("\n".join(L) + "\n")
    log.info("Wrote Moltemplate FF library: %s", out_path)
    return out_path


# ================================================================ atom masses
def parse_masses_from_par(path: str | Path) -> Dict[str, float]:
    """Extract atom key -> mass from an ATOM_TYPE table if present."""
    out: Dict[str, float] = {}
    in_at = False
    header_seen = False
    with Path(path).open() as f:
        for raw in f:
            line = raw.rstrip("\n")
            up = line.upper()
            if up.strip().startswith("ATOM_TYPE"):
                in_at = True
                header_seen = False
                continue
            if up.strip().startswith("END ATOM_TYPE"):
                in_at = False
                continue
            if not in_at:
                continue
            stripped = line.split("!", 1)[0].split("#", 1)[0].strip()
            if not stripped:
                continue
            parts = stripped.split()
            if not header_seen and (parts[0].upper() in {"ATOM_TYPE", "KEY", "TYPE"}):
                header_seen = True
                continue
            # heuristic: parts[1] = key, parts[3] = mass
            if len(parts) >= 4:
                try:
                    mass = float(parts[3])
                    key = parts[1]
                    out.setdefault(key, mass)
                except ValueError:
                    continue
    return out


# ================================================================ convenience
def convert_par(
    par_path: str | Path,
    out_lt: str | Path,
    ff_object_name: Optional[str] = None,
) -> Path:
    dl = parse_par(par_path)
    masses = parse_masses_from_par(par_path)
    return to_moltemplate(dl, out_lt, masses=masses, ff_object_name=ff_object_name)
