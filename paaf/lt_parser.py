"""Extract atom-type tables from any Moltemplate-native `.lt` force-field file.

Parses the ``write_once("In Charges")`` block, which for every atom-type ID
has a line like::

    set type @atom:80   charge      0.265 # C  - C3   | CH3 IN METHANOL "

Returns a list of :class:`AtomTypeInfo` records that the manual-typing GUI
dialog can present as a searchable table (id / element / key / description /
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

    def label(self) -> str:
        """Concise one-line summary suitable for a list widget."""
        return (f"@atom:{self.ff_id:<4s}  "
                f"{self.element:<2s} - {self.key:<5s}  "
                f"q={self.charge:+.3f}  {self.description}")


_CHARGE_RE = re.compile(
    r"^\s*set\s+type\s+@atom:(\S+)\s+charge\s+(\S+)\s*(#.*)?$"
)
_COMMENT_RE = re.compile(r"([A-Za-z0-9\+\-'\*]+)\s*-\s*(\S+)\s*(?:\|\s*(.*))?")


@lru_cache(maxsize=32)
def parse_atom_types(path: str) -> List[AtomTypeInfo]:
    """Extract every atom type from the ``In Charges`` section of a .lt file."""
    p = Path(path)
    if not p.exists():
        return []
    out: List[AtomTypeInfo] = []
    in_charges = False
    for raw in p.read_text(errors="replace").splitlines():
        stripped = raw.strip()
        if 'write_once("In Charges")' in stripped or "write_once('In Charges')" in stripped:
            in_charges = True
            continue
        if in_charges and stripped.startswith("}"):
            break
        if not in_charges:
            continue
        m = _CHARGE_RE.match(raw)
        if not m:
            continue
        ff_id, charge_str, comment = m.group(1), m.group(2), (m.group(3) or "").lstrip("# ").strip()
        try:
            charge = float(charge_str)
        except ValueError:
            continue
        element = ""
        key = ""
        description = comment
        cm = _COMMENT_RE.search(comment)
        if cm:
            element = cm.group(1).strip()
            key = cm.group(2).strip()
            description = (cm.group(3) or "").strip().strip('"').strip("'")
        out.append(AtomTypeInfo(
            ff_id=str(ff_id), element=element, key=key,
            charge=charge, description=description or comment,
        ))
    return out


def group_by_element(types: List[AtomTypeInfo]) -> dict[str, List[AtomTypeInfo]]:
    """Convenience: {element -> [types]} for filtering in the GUI."""
    out: dict[str, List[AtomTypeInfo]] = {}
    for t in types:
        out.setdefault(t.element or "?", []).append(t)
    for lst in out.values():
        lst.sort(key=lambda x: int(x.ff_id) if x.ff_id.isdigit() else 0)
    return out
