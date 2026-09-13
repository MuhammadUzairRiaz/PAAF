"""Refuse an atom type whose element is not the atom's element.

The failure this prevents
------------------------
A carbon type (OPLS 136, ``CT`` — CH2 alkane carbon) was assigned to a
hydrogen. PAAF accepted it and exported the hydrogen with mass 12.011. Nothing
downstream objected: the data file was well formed, LAMMPS would have read it,
and the error would have shown up as inexplicable dynamics much later.

The information needed to catch it was already on hand. Moltemplate's OPLS-AA
library states the element of every type in its charge table::

    set type @atom:136  charge -0.12  # C  - CT   | CH2 all-atom C: alkanes
    set type @atom:140  charge  0.06  # H  - HC   | H all-atom H: alkanes

So this module reads that, and an assignment whose elements disagree is
rejected rather than written.

Deliberately narrow
-------------------
Only element mismatches are refused. Whether a CH2 carbon should be 136 or 137
is chemistry the user is entitled to decide — and they were right to override
PAAF on exactly that, twice, in the run that prompted this. Blocking only the
category error keeps the guard from becoming something to fight.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["type_elements", "check_assignment", "check_all", "drop_unknown",
           "TypeMismatch"]


def drop_unknown(types_by_index: Dict[int, str],
                 known: Sequence[int]) -> Tuple[Dict[int, str], List[int]]:
    """Split assignments into those this structure has atoms for, and the rest.

    Returns ``(kept, dropped_indices)``.

    A typing session outlives the monomer it was started on. Assignments made
    against a 26-atom PBS are still in hand when a 12-atom butadiene is loaded,
    where they address atoms that do not exist — and being invisible, they
    cannot be corrected from the table. Dropping them is the only thing that
    can be done with them; the caller is expected to SAY so, because they were
    the user's assignments and they are being thrown away.
    """
    live = set(known)
    kept = {k: v for k, v in types_by_index.items() if k in live}
    dropped = sorted(k for k in types_by_index if k not in live)
    return kept, dropped

#: ``set type @atom:136  charge  -0.12  # C  - CT  | CH2 all-atom C: alkanes``
_CHARGE_LINE = re.compile(
    r"set\s+type\s+@atom:(\S+)\s+charge\s+\S+\s*#\s*([A-Za-z]{1,2})\b")


class TypeMismatch(ValueError):
    """An atom type was assigned to an atom of a different element."""


def _library_paths() -> List[Path]:
    root = Path(__file__).resolve().parent.parent / "ff_libraries" / "moltemplate"
    if not root.is_dir():
        return []
    # The default table uses the 2024 numbering. The 2008 libraries number
    # their types differently (2008 @atom:136 is an H); they are only read
    # when that force field is selected — see elements_for_ff().
    paths = sorted(root.glob("oplsaa*.lt")) + sorted(root.glob("loplsaa*.lt"))
    return [p for p in paths if "2008" not in p.name]


#: ``@atom:136 1.008  # 136`` inside ``write_once("Data Masses")``
_MASS_LINE = re.compile(r"^\s*@atom:(\S+)\s+([0-9]*\.?[0-9]+)\b", re.M)

_ELEMENT_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078, "Fe": 55.845,
    "Cu": 63.546, "Zn": 65.38, "Br": 79.904, "I": 126.904, "Li": 6.94,
    "B": 10.81, "Se": 78.971, "Cs": 132.905, "Rb": 85.468, "Sr": 87.62,
    "Ba": 137.327, "Ar": 39.948, "Ne": 20.180, "He": 4.0026, "Kr": 83.798,
    "Xe": 131.293,
}


def _element_from_mass(mass: float) -> Optional[str]:
    best, err = None, 1e9
    for el, m in _ELEMENT_MASSES.items():
        if abs(m - mass) < err:
            best, err = el, abs(m - mass)
    # United-atom beads (CH2 = 14.027, CH3 = 15.035) are not element masses.
    return best if err < 0.05 else None


_CACHE: Dict[str, Dict[str, str]] = {}


def elements_for_ff(ff_key: Optional[str]) -> Dict[str, str]:
    """The element table for the selected force field's bundled library.

    Falls back to the default (2024) table when the key is unknown or has no
    bundled library, so behaviour is unchanged for everything but the
    libraries with their own numbering.
    """
    if ff_key:
        try:
            from .ff_registry import get_ff
            ff = get_ff(ff_key)
            path = ff.bundled_path() if hasattr(ff, "bundled_path") else None
            if path is None and getattr(ff, "bundled_lt", None):
                from .ff_registry import BUNDLED_MOLTEMPLATE
                path = BUNDLED_MOLTEMPLATE / ff.bundled_lt
            if path is not None and Path(path).exists():
                table = type_elements(Path(path))
                if table:
                    return table
        except Exception as exc:                          # unknown key etc.
            log.debug("element table for %r unavailable (%s)", ff_key, exc)
    return type_elements()


def type_elements(lt_path: Optional[Path] = None) -> Dict[str, str]:
    """``{type_id: element}`` read from a Moltemplate force-field library.

    The element comes from the charge-table comment (``# C - CT``) and, for
    libraries whose comments carry no symbol (oplsaa2008.lt), from the type's
    mass in ``Data Masses``.

    Returns an empty mapping when no library can be read — in which case the
    guard stands down rather than blocking work on a guess.
    """
    paths = [Path(lt_path)] if lt_path else _library_paths()
    key = "|".join(str(p) for p in paths)
    if key in _CACHE:
        return _CACHE[key]

    mapping: Dict[str, str] = {}
    for path in paths:
        try:
            text = Path(path).read_text(errors="replace")
        except OSError:
            continue
        from_mass: Dict[str, str] = {}
        for m in _CHARGE_LINE.finditer(text):
            type_id, element = m.group(1), m.group(2)
            # First library wins; later ones only fill gaps.
            mapping.setdefault(type_id, element.strip())
        for m in _MASS_LINE.finditer(text):
            el = _element_from_mass(float(m.group(2)))
            if el:
                from_mass.setdefault(m.group(1), el)
        for type_id, el in from_mass.items():
            mapping.setdefault(type_id, el)
    if not mapping:
        log.info("No Moltemplate library readable; element guard is inactive.")
    _CACHE[key] = mapping
    return mapping


def check_assignment(atom_element: str, type_id: str,
                     elements: Optional[Dict[str, str]] = None
                     ) -> Optional[str]:
    """``None`` if the assignment is sound, else a message explaining why not.

    An unknown type is allowed through: the user may be working with a library
    PAAF has not read, and refusing everything unrecognised would make the
    guard worse than useless.
    """
    if not type_id:
        return None
    table = type_elements() if elements is None else elements
    expected = table.get(str(type_id).strip())
    if not expected:
        return None
    got = (atom_element or "").strip()
    if expected.lower() == got.lower():
        return None
    return (f"Type {type_id} is a {expected} type — it cannot be assigned to "
            f"a {got or '?'} atom. Assigning it would write this atom with "
            f"{expected}'s mass and charge.")


def check_all(elements_by_index: Dict[int, str],
              types_by_index: Dict[int, str],
              elements: Optional[Dict[str, str]] = None) -> List[Tuple[int, str]]:
    """Every mismatched assignment, as ``(atom_index, message)``.

    Run before export, so a mistake made in the table is caught even if the
    per-assignment check was bypassed.

    An assignment for an index that is not in ``elements_by_index`` is skipped,
    not reported. This guard compares an atom's element against its type's
    element; where there is no atom there is no element to compare, and
    ``elements_by_index.get(index, "")`` used to turn that into the message
    "cannot be assigned to a ? atom". Switching from a 26-atom monomer to a
    12-atom one left the older monomer's assignments in the map, every one of
    them produced that message, and Apply refused the whole thing — for a
    molecule whose own atoms were all typed correctly. Deciding what to do
    about a stale assignment belongs to the caller, which knows whether the
    index is stale or genuinely missing; see the dialog's ``_load_atoms``.
    """
    table = type_elements() if elements is None else elements
    problems: List[Tuple[int, str]] = []
    for index, type_id in sorted(types_by_index.items()):
        element = elements_by_index.get(index)
        if not element:
            continue
        message = check_assignment(element, type_id, table)
        if message:
            problems.append((int(index), message))
    return problems
