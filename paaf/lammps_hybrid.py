"""Convert DL_FIELD's *hybrid* LAMMPS output into plain (non-hybrid) styles.

DL_FIELD declares every style as a hybrid with exactly one sub-style::

    bond_style      hybrid harmonic
    pair_style      hybrid lj/cut/coul/long 12.0
    pair_coeff  1 1 lj/cut/coul/long 0.066 3.5

and every coefficient line in the data file repeats the sub-style name::

    Bond Coeffs
       1 harmonic   317.0  1.51

That is valid LAMMPS and runs. It is also the form that breaks the moment
two such systems are merged, that many analysis scripts and ``fix bond/react``
templates do not expect, and that the user cannot hand-edit as easily. This
module rewrites a matching ``.in`` + ``.data`` pair to::

    bond_style      harmonic
    pair_style      lj/cut/coul/long 12.0
    pair_coeff  1 1 0.066 3.5
    Bond Coeffs
       1 317.0  1.51

Only *single* sub-style hybrids are unwrapped. A genuine hybrid
(``hybrid harmonic morse``) is load-bearing and is left alone with a
warning — de-hybridising it would change which functional form each type
uses.

Nothing here is force-field specific: the sub-style names are read from the
files themselves, so PCFF's ``class2``/``quartic``, OPLS's ``harmonic``/
``opls``, CHARMM's ``charmm`` … all convert with the same code.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .blend_styles import (COEFF_SECTION_STYLE, dehybridise_coeff_lines,
                           dehybridise_pair_coeff, dehybridise_style)
from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["convert_input", "convert_data", "convert_to_nonhybrid",
           "is_hybrid_input"]

_STYLE_DIRECTIVES = ("pair_style", "bond_style", "angle_style",
                     "dihedral_style", "improper_style")

# Every header that starts a section in a LAMMPS data file. Needed so that
# the coefficient rewrite stops at the next section instead of eating into
# "Atoms".
_SECTION_HEADERS = {
    "Masses", "Pair Coeffs", "PairIJ Coeffs", "Bond Coeffs", "Angle Coeffs",
    "Dihedral Coeffs", "Improper Coeffs", "BondBond Coeffs", "BondAngle Coeffs",
    "MiddleBondTorsion Coeffs", "EndBondTorsion Coeffs", "AngleTorsion Coeffs",
    "AngleAngleTorsion Coeffs", "BondBond13 Coeffs", "AngleAngle Coeffs",
    "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers",
}


def is_hybrid_input(in_file: Path) -> bool:
    """True when any ``*_style`` line in ``in_file`` starts with ``hybrid``."""
    for line in Path(in_file).read_text(errors="replace").splitlines():
        parts = line.split("#")[0].split()
        if len(parts) >= 2 and parts[0] in _STYLE_DIRECTIVES and parts[1] == "hybrid":
            return True
    return False


def convert_input(src: Path, dst: Path,
                  read_data: Optional[str] = None) -> Dict[str, str]:
    """Rewrite ``src`` (a LAMMPS input) without single-sub-style hybrids.

    Returns ``{directive: substyle_removed}`` so the data file can be
    converted consistently. ``read_data`` optionally replaces the file name
    on the ``read_data`` line (e.g. point it at ``packed_box.data``).
    """
    src, dst = Path(src), Path(dst)
    removed: Dict[str, str] = {}
    out: List[str] = []
    for raw in src.read_text(errors="replace").splitlines():
        body = raw.split("#")[0]
        parts = body.split()
        if parts and parts[0] in _STYLE_DIRECTIVES:
            indent = raw[:len(raw) - len(raw.lstrip())]
            value = raw.lstrip()[len(parts[0]):].strip()
            new_value, sub = dehybridise_style(value)
            if sub:
                removed[parts[0]] = sub
                out.append(f"{indent}{parts[0]:<15} {new_value}")
                continue
            if "hybrid" in parts[1:2]:
                log.warning("%s: %r is a genuine multi-sub-style hybrid; "
                            "left unchanged", src.name, raw.strip())
        elif parts and parts[0] == "pair_coeff" and "pair_style" in removed:
            out.append(dehybridise_pair_coeff(raw, removed["pair_style"]).rstrip())
            continue
        elif parts and parts[0] == "read_data" and read_data:
            out.append(f"read_data {read_data}")
            continue
        out.append(raw.rstrip())
    dst.write_text("\n".join(out) + "\n")
    log.info("Wrote non-hybrid input %s (unwrapped: %s)", dst,
             ", ".join(f"{k}={v}" for k, v in removed.items()) or "nothing")
    return removed


def _section_of(line: str) -> Optional[str]:
    head = line.split("#")[0].strip()
    return head if head in _SECTION_HEADERS else None


def convert_data(src: Path, dst: Path,
                 substyles: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Strip sub-style tokens from the coefficient sections of a data file.

    ``substyles`` maps directive -> token as returned by :func:`convert_input`.
    When it is omitted the token is detected structurally per section (second
    column present and non-numeric), so the data file converts on its own.
    """
    src, dst = Path(src), Path(dst)
    substyles = dict(substyles or {})
    stripped: Dict[str, str] = {}
    lines = src.read_text(errors="replace").splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        sec = _section_of(line)
        out.append(line.rstrip())
        i += 1
        if sec is None or sec not in COEFF_SECTION_STYLE:
            continue
        # Collect this section's body up to the next header.
        body: List[str] = []
        while i < len(lines) and _section_of(lines[i]) is None:
            body.append(lines[i]); i += 1
        directive = COEFF_SECTION_STYLE[sec]
        token = substyles.get(directive, "")
        if not token:
            for b in body:
                parts = b.split("#")[0].split()
                if len(parts) >= 2 and not _is_number(parts[1]):
                    token = parts[1]; break
        if token:
            body = dehybridise_coeff_lines(body, token)
            stripped[sec] = token
        out.extend(b.rstrip() for b in body)
    dst.write_text("\n".join(out) + "\n")
    log.info("Wrote non-hybrid data %s (stripped: %s)", dst,
             ", ".join(f"{k}:{v}" for k, v in stripped.items()) or "nothing")
    return stripped


def _is_number(tok: str) -> bool:
    try:
        float(tok); return True
    except ValueError:
        return False


def convert_to_nonhybrid(in_file: Path, data_files: List[Path],
                         out_dir: Optional[Path] = None,
                         read_data: Optional[str] = None
                         ) -> Tuple[Path, List[Path]]:
    """Convert one input and any number of data files together.

    Files are written under ``out_dir`` (default: beside the originals) with
    the same names, so ``out_dir/lammps.in`` + ``out_dir/packed_box.data`` is
    a drop-in non-hybrid copy of the set. The originals are never touched.
    """
    in_file = Path(in_file)
    out_dir = Path(out_dir) if out_dir else in_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    new_in = out_dir / in_file.name
    removed = convert_input(in_file, new_in, read_data=read_data)
    new_data: List[Path] = []
    for d in data_files:
        d = Path(d)
        if not d.exists():
            continue
        nd = out_dir / d.name
        convert_data(d, nd, removed)
        new_data.append(nd)
    return new_in, new_data
