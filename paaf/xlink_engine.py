"""Apply a :class:`ReactionLibrary` to a packed / simulated system.

This is a pragmatic first-pass implementation:

- Reads a packed system as either a LAMMPS ``data`` file (via the user's
  existing ``xlink_engine_local_*.py`` format — atoms, bonds, angles,
  dihedrals, impropers), OR any structure OpenBabel can read.
- For every reaction template, scans the packed system for candidate
  reactive-atom pairs based on element + neighborhood signatures and
  spatial proximity (``cutoff`` Å).
- Applies the template edits (delete atoms/bonds, create bonds, change
  element labels) up to ``max_events`` times.
- Writes the edited system out as ``lammps.data`` or ``xyz``.

For heavy production runs users can hand the learned JSON library to their
existing ``xlink_engine_local_mpi_molecule_colors_fixed.py`` which already
knows how to iterate with LAMMPS relaxation between events.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .logging_utils import get_logger
from .reaction import ReactionLibrary, ReactionTemplate
from .structure import Atom, Molecule, load_structure, write

log = get_logger(__name__)


# =========================================================== data classes
@dataclass
class XlinkStats:
    events_applied: int = 0
    events_per_reaction: Dict[str, int] = field(default_factory=dict)
    bonds_created: int = 0
    bonds_deleted: int = 0
    atoms_deleted: int = 0

    def as_dict(self) -> dict:
        return {
            "events_applied": self.events_applied,
            "events_per_reaction": self.events_per_reaction,
            "bonds_created": self.bonds_created,
            "bonds_deleted": self.bonds_deleted,
            "atoms_deleted": self.atoms_deleted,
        }


# ======================================================== reactive scan
def _atom_signature(mol: Molecule, i: int) -> Tuple[str, int, tuple]:
    """(element, degree, sorted neighbor elements) — enough to match against a template."""
    return (
        mol.atoms[i].element,
        len(mol.neighbors(i)),
        tuple(sorted(mol.atoms[j].element for j in mol.neighbors(i))),
    )


def _find_candidate(mol: Molecule, template: ReactionTemplate, exclude: set[int]) -> Optional[Dict[int, int]]:
    """Search `mol` for atoms matching the reactive-site signatures.

    Returns a mapping ``template_reactant_index -> mol_atom_index`` (only for the
    reactive sites; not a full-graph mapping).
    """
    from .structure import load_structure  # noqa: F401  (kept for API clarity)
    # Rebuild reactant signatures once
    r_sig_needed = {}
    r_mol_cache = None
    try:
        r_mol_cache = load_structure(template.reactant)
    except Exception:
        return None
    for site in template.reactive_sites:
        idx = site["reactant_atom"]
        if idx >= len(r_mol_cache.atoms):
            continue
        r_sig_needed[idx] = _atom_signature(r_mol_cache, idx)

    # Greedy assign
    picked: Dict[int, int] = {}
    used_mol: set[int] = set(exclude)
    for r_idx, needed_sig in r_sig_needed.items():
        best = None
        for a in mol.atoms:
            if a.index in used_mol:
                continue
            if _atom_signature(mol, a.index) == needed_sig:
                best = a.index
                break
        if best is None:
            return None
        picked[r_idx] = best
        used_mol.add(best)
    return picked or None


def _distance(mol: Molecule, i: int, j: int) -> float:
    d = mol.atoms[i].xyz - mol.atoms[j].xyz
    return float(math.sqrt(float(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)))


# ============================================================ apply one
def _apply_template_once(
    mol: Molecule,
    template: ReactionTemplate,
    exclude: set[int],
    cutoff: float,
) -> Optional[Dict[str, int]]:
    picked = _find_candidate(mol, template, exclude)
    if not picked:
        return None

    # Check spatial proximity for bonds we are about to create.
    for a, b in template.created_bonds:
        if a in picked and b in picked:
            if _distance(mol, picked[a], picked[b]) > cutoff:
                return None

    # Apply deletions of bonds
    del_bond_pairs = set()
    for a, b in template.deleted_bonds:
        if a in picked and b in picked:
            del_bond_pairs.add(frozenset({picked[a], picked[b]}))
    mol.bonds = [
        (i, j, o) for (i, j, o) in mol.bonds if frozenset({i, j}) not in del_bond_pairs
    ]

    # Apply deletions of atoms (only those covered by the picked mapping)
    to_delete = {picked[a] for a in template.deleted_atoms if a in picked}
    if to_delete:
        keep = [a for a in mol.atoms if a.index not in to_delete]
        remap = {a.index: k for k, a in enumerate(keep)}
        for k, a in enumerate(keep):
            a.index = k
        mol.atoms = keep
        mol.bonds = [
            (remap[i], remap[j], o)
            for (i, j, o) in mol.bonds
            if i in remap and j in remap
        ]
        # translate picked as well
        picked = {r: remap[m] for r, m in picked.items() if m in remap}

    # Apply creation of bonds
    for a, b in template.created_bonds:
        if a in picked and b in picked:
            mol.bonds.append((picked[a], picked[b], 1.0))

    # Element changes
    for ch in template.element_changes:
        a = ch["reactant_atom"]
        if a in picked:
            mol.atoms[picked[a]].element = ch["to"]

    return {
        "bonds_created": sum(1 for a, b in template.created_bonds if a in picked and b in picked),
        "bonds_deleted": len(del_bond_pairs),
        "atoms_deleted": len(to_delete),
    }


# ============================================================ top-level
def apply_library(
    system_path: str | Path,
    library: ReactionLibrary,
    output_path: str | Path,
    max_events: int = 1_000_000,
    cutoff: float = 5.0,
) -> XlinkStats:
    """Apply every template in `library` to `system_path` and write result."""
    mol = load_structure(system_path)
    log.info("Loaded system %s (%d atoms, %d bonds)", system_path, len(mol.atoms), len(mol.bonds))
    stats = XlinkStats()
    for template in library.templates:
        stats.events_per_reaction.setdefault(template.name, 0)
        excluded: set[int] = set()
        for _ in range(max_events):
            info = _apply_template_once(mol, template, excluded, cutoff=cutoff)
            if not info:
                break
            stats.events_applied += 1
            stats.events_per_reaction[template.name] += 1
            stats.bonds_created += info["bonds_created"]
            stats.bonds_deleted += info["bonds_deleted"]
            stats.atoms_deleted += info["atoms_deleted"]
            log.info("Event %d (%s): +%d bonds, -%d bonds, -%d atoms",
                     stats.events_applied, template.name,
                     info["bonds_created"], info["bonds_deleted"], info["atoms_deleted"])
    write(mol, output_path)
    log.info("Wrote crosslinked system -> %s", output_path)
    return stats
