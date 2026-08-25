"""Polymers the user adds themselves, kept separately and kept forever.

Why not just edit the shipped CSV
---------------------------------
``paaf/data/polymer_database.csv`` lives inside the package. Adding a line to
it works until PAAF is reinstalled or updated, at which point the file is
replaced and the user's polymers are gone with no warning and no way back.
Nobody would think to keep a copy of a file they did not know they had edited.

So user entries live in their own file, under the user's home directory,
which nothing in an upgrade touches::

    ~/.paaf/custom_polymers.csv

The format is deliberately the same three columns the shipped database uses
for the fields that matter — ``pid``, ``smiles``, ``name`` — so a line can be
moved between the two by hand if anyone wants to.

What a valid entry is
---------------------
A polymerisation SMILES with exactly two ``[*]`` marks, each on an atom that
has a hydrogen to give up when the chain joins. That last condition is not
pedantry: linking removes one hydrogen per link atom, and a link atom with
none cannot be joined at all. Checking it here means the user is told when
they add the polymer, rather than several screens later when a build fails.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["CustomPolymer", "store_path", "load_custom", "save_custom",
           "add_custom", "update_custom", "delete_custom", "validate_entry",
           "CustomLibraryError"]


class CustomLibraryError(ValueError):
    """The entry cannot be stored as given."""


@dataclass(frozen=True)
class CustomPolymer:
    pid: str
    smiles: str
    name: str


#: Environment override, so a test (or a user with a shared home) can point
#: the store somewhere else without monkeypatching.
_ENV_VAR = "PAAF_CUSTOM_LIBRARY"

_PID_OK = re.compile(r"^[A-Za-z0-9_.-]+$")


def store_path() -> Path:
    """Where the user's own polymers are kept."""
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".paaf" / "custom_polymers.csv"


# ================================================================ validation
def validate_entry(pid: str, smiles: str, name: str,
                   existing: Optional[Dict[str, CustomPolymer]] = None,
                   allow_pid: Optional[str] = None) -> Tuple[str, str, str]:
    """Return the cleaned ``(pid, smiles, name)`` or raise.

    ``allow_pid`` is the entry being edited, whose own id is not a clash with
    itself.
    """
    pid = (pid or "").strip()
    smiles = (smiles or "").strip()
    name = (name or "").strip()

    if not pid:
        raise CustomLibraryError("Give the polymer an ID — it is how PAAF "
                                 "refers to it, e.g. MY_PIB.")
    if not _PID_OK.match(pid):
        raise CustomLibraryError(
            f"'{pid}' cannot be used as an ID. Letters, digits, '_', '-' and "
            f"'.' only — no spaces or commas, because the ID is also a "
            f"lookup key and a CSV field.")
    if not name:
        raise CustomLibraryError("Give the polymer a name — it is what you "
                                 "will see in the library list.")
    if "," in name:
        raise CustomLibraryError("The name cannot contain a comma.")
    if not smiles:
        raise CustomLibraryError("Give the polymerisation SMILES.")
    if smiles.count("[*]") != 2:
        raise CustomLibraryError(
            f"The SMILES needs exactly two [*] marks, showing where one unit "
            f"joins the next. This one has {smiles.count('[*]')}.\n\n"
            f"For example polyisobutylene is  [*]CC(C)(C)[*]")

    if existing and pid in existing and pid != allow_pid:
        raise CustomLibraryError(
            f"'{pid}' is already used by '{existing[pid].name}'. "
            f"Pick another ID, or edit that entry instead.")

    _check_chemistry(smiles)
    return pid, smiles, name


def _check_chemistry(smiles: str) -> None:
    """Refuse a SMILES that cannot be polymerised, and say why.

    Skipped silently when RDKit is unavailable — the guard is a courtesy, and
    refusing to save because a chemistry toolkit is missing would be worse
    than saving something that later turns out not to build.
    """
    try:
        from rdkit import Chem
    except Exception:                                 # pragma: no cover
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise CustomLibraryError(
            f"RDKit cannot read this SMILES:\n\n    {smiles}\n\n"
            f"Check the brackets and ring closures.")

    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        raise CustomLibraryError("The two [*] marks must each be a separate "
                                 "atom in the SMILES.")
    for dummy in dummies:
        if len(mol.GetAtomWithIdx(dummy.GetIdx()).GetNeighbors()) != 1:
            raise CustomLibraryError(
                "Each [*] must be attached to exactly one atom — it marks a "
                "single bond to the next unit.")

    # The real test, done by the code that will actually do it: replace each
    # wildcard with the hydrogen it stands for and see whether the result is a
    # sane molecule.
    #
    # An earlier version of this check counted hydrogens on the link atom and
    # refused polyisobutylene, [*]CC(C)(C)[*] — a real polymer PAAF builds
    # perfectly well. The mistake was reading the wildcard as something extra
    # attached to a carbon that was already full. It is not: the wildcard IS
    # the hydrogen the junction consumes, so a link atom never needs one of
    # its own.
    from .monomer import _capped_from_wildcards

    if _capped_from_wildcards(smiles) is None:
        raise CustomLibraryError(
            f"Capping the two [*] marks with hydrogen does not give a valid "
            f"molecule:\n\n    {smiles}\n\n"
            f"Usually this means a [*] sits on an atom whose valence is "
            f"already full.")


# =================================================================== storage
def load_custom(path: Optional[Path] = None) -> Dict[str, CustomPolymer]:
    """``{pid: CustomPolymer}``. Missing or unreadable file gives ``{}``."""
    path = Path(path) if path else store_path()
    if not path.exists():
        return {}
    out: Dict[str, CustomPolymer] = {}
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                pid = (row.get("pid") or "").strip()
                smiles = (row.get("smiles") or "").strip()
                name = (row.get("name") or "").strip()
                if pid and smiles and name:
                    out[pid] = CustomPolymer(pid, smiles, name)
    except Exception as exc:
        # Broad on purpose. A half-written or corrupt file raises csv.Error or
        # UnicodeError as readily as OSError, and none of them is a reason to
        # stop PAAF from starting — the shipped library is untouched and the
        # user can fix or delete the file.
        log.warning("Could not read your custom polymers from %s (%s); "
                    "continuing with the shipped library only.", path, exc)
        return {}
    log.info("Loaded %d custom polymer(s) from %s", len(out), path)
    return out


def save_custom(entries: Dict[str, CustomPolymer],
                path: Optional[Path] = None) -> Path:
    """Write the whole set, creating the directory if needed."""
    path = Path(path) if path else store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pid", "smiles", "name"])
        for pid in sorted(entries):
            e = entries[pid]
            writer.writerow([e.pid, e.smiles, e.name])
    log.info("Saved %d custom polymer(s) to %s", len(entries), path)
    return path


def add_custom(pid: str, smiles: str, name: str,
               path: Optional[Path] = None) -> CustomPolymer:
    entries = load_custom(path)
    pid, smiles, name = validate_entry(pid, smiles, name, existing=entries)
    entry = CustomPolymer(pid, smiles, name)
    entries[pid] = entry
    save_custom(entries, path)
    return entry


def update_custom(original_pid: str, pid: str, smiles: str, name: str,
                  path: Optional[Path] = None) -> CustomPolymer:
    """Edit an entry, possibly renaming its id."""
    entries = load_custom(path)
    if original_pid not in entries:
        raise CustomLibraryError(
            f"'{original_pid}' is not one of your custom polymers. Only your "
            f"own entries can be edited; the shipped ones are read-only.")
    pid, smiles, name = validate_entry(pid, smiles, name, existing=entries,
                                       allow_pid=original_pid)
    entries.pop(original_pid, None)
    entry = CustomPolymer(pid, smiles, name)
    entries[pid] = entry
    save_custom(entries, path)
    return entry


def delete_custom(pid: str, path: Optional[Path] = None) -> bool:
    """Remove an entry. ``False`` if it was not there to begin with."""
    entries = load_custom(path)
    if pid not in entries:
        return False
    entries.pop(pid)
    save_custom(entries, path)
    return True
