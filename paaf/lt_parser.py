"""Extract atom-type tables from any Moltemplate-native `.lt` force-field file.

Two sources, merged:

* ``write_once("In Charges")`` — OPLS-family files list every type there::

      set type @atom:80   charge      0.265 # C  - C3   | CH3 IN METHANOL "

* ``write_once("Data Masses")`` — every moltemplate force field has this
  (COMPASS, GAFF, DREIDING, TraPPE, SDK, MARTINI …), even when it has no
  per-type charges (COMPASS/DREIDING derive charges from bond increments)::

      @atom:c3a 12.01115  # c3a

  The element is read from the type name and confirmed against the mass
  (GAFF ``ca`` is aromatic carbon, not calcium: 12.01 amu decides).

Self-pair ``pair_coeff`` lines (``@atom:X @atom:X … eps sigma``) are attached
as the type's non-bonded parameters, so the picker can show ε/σ for every
force field, not just charges for OPLS.

Returns a list of :class:`AtomTypeInfo` records that the manual-typing GUI
dialog presents as a searchable table (id / element / key / description /
charge).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class AtomTypeInfo:
    ff_id: str              # numeric OPLS ID as string, e.g. "80"
    element: str            # "C", "H", "O", "N", ...
    key: str                # short chemistry key, e.g. "C3", "OS", "CT"
    charge: float           # partial charge in e
    description: str        # human-readable, e.g. "CH3 IN METHANOL"
    mass: float = 0.0       # amu, when the file lists it
    params: str = ""        # non-bonded parameters, e.g. "eps=0.066 sigma=3.50"

    def label(self) -> str:
        """Concise one-line summary suitable for a list widget."""
        return (f"@atom:{self.ff_id:<4s}  "
                f"{self.element:<2s} - {self.key:<5s}  "
                f"q={self.charge:+.3f}  {self.description}"
                + (f"  [{self.params}]" if self.params else ""))


_CHARGE_RE = re.compile(
    r"^\s*set\s+type\s+@atom:(\S+)\s+charge\s+(\S+)\s*(#.*)?$"
)
_COMMENT_RE = re.compile(r"([A-Za-z0-9\+\-'\*]+)\s*-\s*(\S+)\s*(?:\|\s*(.*))?")
_MASS_RE = re.compile(r"^\s*@atom:(\S+)\s+([0-9.eE+\-]+)\s*(#.*)?$")
_PAIR_RE = re.compile(r"^\s*pair_coeff\s+@atom:(\S+)\s+@atom:(\S+)\s+(.*?)\s*(#.*)?$")

_ELEMENT_MASS = {
    "H": 1.008, "He": 4.003, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.18, "Na": 22.99, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Ti": 47.867, "Fe": 55.845, "Cu": 63.546, "Zn": 65.38,
    "Br": 79.904, "I": 126.904,
}


def _element_from_name_and_mass(name: str, mass: float) -> str:
    """'c3a'/12.01 -> 'C'; 'ca'/12.01 -> 'C' (GAFF aromatic, not calcium);
    'si'/28.09 -> 'Si'; 'CH2'/14.03 -> 'C' (united atom); 'BP4'/72 -> '' (bead)."""
    stem = re.split(r"[^A-Za-z]", name, 1)[0]
    cands = []
    for k in (2, 1):
        sym = stem[:k].capitalize() if k == 2 else stem[:1].upper()
        if sym in _ELEMENT_MASS:
            cands.append(sym)
    if not cands:
        return ""
    if mass <= 0:
        return cands[0]
    best = min(cands, key=lambda e: abs(_ELEMENT_MASS[e] - mass))
    # United-atom groups (CH2 = 14.03) sit within a few amu of the heavy
    # atom; a coarse-grained bead (72 amu named "C1") does not.
    if mass > 1.3 * _ELEMENT_MASS[best] + 3.0:
        return ""
    return best


def _element_from_mass(mass: float) -> str:
    """Element by mass, united-atom aware: 14.027 is CH2 (C), not N (14.007);
    16.023 is NH2 (N), 15.999 is O. '' when nothing is within 0.06 amu of an
    element or element+kH (coarse-grained beads)."""
    if mass <= 0:
        return ""
    best, best_d = "", 1e9
    for e, me in _ELEMENT_MASS.items():
        kmax = 4 if e in ("C", "N", "O", "S", "Si", "P") else 0
        for k in range(kmax + 1):
            d = abs(me + k * _ELEMENT_MASS["H"] - mass)
            if d < best_d - 1e-9:
                best, best_d = e, d
    return best if best_d <= 0.06 else ""


def _pair_type_name(token: str) -> str:
    """The bare type name inside a pair_coeff atom token.

    ``80`` -> ``80``; ``*~pc3a~b*~a*~d*~i*`` (COMPASS) -> ``c3a``;
    ``BP4_bBP4_aBP4_dBP4_iBP4`` (MARTINI) -> ``BP4``."""
    if "~" in token:
        for seg in token.split("~"):
            if seg.startswith("p") and len(seg) > 1 and "*" not in seg:
                return seg[1:]
        return token
    m = re.match(r"^(.+?)_b[^_]*_a[^_]*_d[^_]*_i", token)
    return m.group(1) if m else token


@lru_cache(maxsize=32)
def parse_atom_types(path: str) -> List[AtomTypeInfo]:
    """Every atom type in a .lt file (In Charges ∪ Data Masses), with the
    self-pair non-bonded parameters attached where the file has them."""
    p = Path(path)
    if not p.exists():
        return []
    charges: dict[str, tuple[float, str]] = {}      # id -> (charge, comment)
    masses: dict[str, tuple[float, str]] = {}       # id -> (mass, comment)
    pairs: dict[str, str] = {}                      # id -> "eps=.. sigma=.."
    order: List[str] = []
    section = None
    for raw in p.read_text(errors="replace").splitlines():
        stripped = raw.strip()
        if "write_once(" in stripped or "write_once (" in stripped:
            if "In Charges" in stripped:
                section = "charges"
            elif "Data Masses" in stripped:
                section = "masses"
            elif "In Settings" in stripped:
                section = "settings"
            else:
                section = "other"
            continue
        if section and stripped.startswith("}"):
            section = None
            continue
        if section == "charges":
            m = _CHARGE_RE.match(raw)
            if m:
                try:
                    q = float(m.group(2))
                except ValueError:
                    continue
                tid = m.group(1)
                charges[tid] = (q, (m.group(3) or "").lstrip("# ").strip())
                if tid not in order:
                    order.append(tid)
        elif section == "masses":
            m = _MASS_RE.match(raw)
            if m:
                try:
                    mass = float(m.group(2))
                except ValueError:
                    continue
                tid = m.group(1)
                masses[tid] = (mass, (m.group(3) or "").lstrip("# ").strip())
                if tid not in order:
                    order.append(tid)
        elif section == "settings":
            m = _PAIR_RE.match(raw)
            if m:
                a, b = _pair_type_name(m.group(1)), _pair_type_name(m.group(2))
                if a != b or a in pairs or m.group(3).lstrip().startswith("hbond"):
                    continue
                nums = [t for t in m.group(3).split() if _is_num(t)]
                if len(nums) >= 2:
                    pairs[a] = f"eps={float(nums[0]):g} sigma={float(nums[1]):g}"
                elif nums:
                    pairs[a] = f"eps={float(nums[0]):g}"

    # Wildcard self-pairs (DREIDING writes ``@atom:C* @atom:C*`` for every
    # carbon type): expand onto the concrete ids they match, explicit first.
    import fnmatch
    for pat, val in list(pairs.items()):
        if any(ch in pat for ch in "*?"):
            for tid in order:
                if tid not in pairs and fnmatch.fnmatchcase(tid, pat):
                    pairs[tid] = val

    out: List[AtomTypeInfo] = []
    for tid in order:
        charge, ccomment = charges.get(tid, (0.0, ""))
        mass, mcomment = masses.get(tid, (0.0, ""))
        element = ""
        key = ""
        description = ccomment or mcomment
        cm = _COMMENT_RE.search(ccomment) if ccomment else None
        # "El - key | description" comments (oplsaa2024) name the element;
        # a free-text comment such as "Alkyl Fluoride C-F" (oplsaa2008) only
        # LOOKS like one. The mass line is authoritative: when it disagrees
        # with the comment's element, the comment was free text.
        _mass_el = _element_from_mass(mass) if mass > 0 else ""
        if (cm and cm.group(1).strip().capitalize() in _ELEMENT_MASS
                and (not _mass_el or _mass_el == cm.group(1).strip().capitalize()
                     or not ccomment.lstrip().startswith(cm.group(1)))):
            element = cm.group(1).strip().capitalize()
            key = cm.group(2).strip()
            description = (cm.group(3) or "").strip().strip('"').strip("'") or ccomment
            if _mass_el and _mass_el != element:
                element = _mass_el
        elif ccomment:
            # Comment is free text ("Acetic Acid >C=O (UA)"): element from mass.
            element = _element_from_mass(mass)
            key = tid
            description = ccomment.strip().strip('"')
        else:
            element = _element_from_name_and_mass(tid, mass)
            key = tid
            if mcomment and mcomment != tid:
                description = mcomment
            elif not description:
                description = tid
        out.append(AtomTypeInfo(
            ff_id=str(tid), element=element, key=key, charge=charge,
            description=description, mass=mass, params=pairs.get(tid, ""),
        ))
    return out


def _is_num(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def group_by_element(types: List[AtomTypeInfo]) -> dict[str, List[AtomTypeInfo]]:
    """Convenience: {element -> [types]} for filtering in the GUI."""
    out: dict[str, List[AtomTypeInfo]] = {}
    for t in types:
        out.setdefault(t.element or "?", []).append(t)
    for lst in out.values():
        lst.sort(key=lambda x: int(x.ff_id) if x.ff_id.isdigit() else 0)
    return out
