"""Apply a :class:`ReactionLibrary` to a packed / simulated system.

This is a pragmatic first-pass implementation:

- Reads a packed system as either a LAMMPS ``data`` file (via the user's
  existing ``xlink_engine_local_*.py`` format — atoms, bonds, angles,
  dihedrals, impropers), OR any structure OpenBabel can read.
- For every reaction template, scans the packed system for candidate
  reactive-atom pairs based on element + neighborhood signatures and
  spatial proximity (``cutoff`` Å, minimum-image when the system has a cell).
- Applies the template edits (delete atoms/bonds, create bonds, change bond
  orders and element labels) up to ``max_events`` times.
- Writes the edited system out as ``lammps.data`` or ``xyz``.

The output is TOPOLOGY ONLY: angles, dihedrals, atom types and charges are not
regenerated, so the result must be re-typed before MD.

For heavy production runs users can hand the learned JSON library to their
existing ``xlink_engine_local_mpi_molecule_colors_fixed.py`` which already
knows how to iterate with LAMMPS relaxation between events.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .logging_utils import get_logger
from .reaction import ReactionLibrary, ReactionTemplate
from .structure import Molecule, load_structure, write

log = get_logger(__name__)

RETYPE_WARNING = (
    "reactions-apply edits topology only: angles, dihedrals, atom types and "
    "charges were NOT regenerated. Re-type the output before running MD.")


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


def _reactant_graph(template: ReactionTemplate) -> Optional[Molecule]:
    """The reactant as a graph: from a file path, or from SMILES text."""
    text = (template.reactant or "").strip()
    if not text:
        return None
    try:
        if Path(text).exists():
            return load_structure(text)
    except (OSError, ValueError):
        pass
    try:
        from .reaction_smiles import _combine
        mol, _maps = _combine([s.strip() for s in text.split(" . ") if s.strip()])
        return mol
    except Exception as exc:
        log.warning("Reaction %s: reactant %r is neither a readable file nor "
                    "valid SMILES (%s)", template.name, text[:60], exc)
        return None


def _site_info(template: ReactionTemplate
               ) -> Optional[Tuple[Dict[int, tuple], Dict[int, List[int]]]]:
    """``({site: signature}, {site: bonded sites})`` for a template."""
    sigs: Dict[int, tuple] = {}
    bonded: Dict[int, List[int]] = {}
    if template.reactive_sites and all("signature" in s
                                       for s in template.reactive_sites):
        for s in template.reactive_sites:
            el, deg, nb = s["signature"]
            sigs[int(s["reactant_atom"])] = (el, int(deg), tuple(nb))
            bonded[int(s["reactant_atom"])] = [int(x) for x in s.get("bonded_sites", [])]
        return sigs, bonded
    r = _reactant_graph(template)
    if r is None:
        return None
    ids = {int(s["reactant_atom"]) for s in template.reactive_sites}
    for idx in ids:
        if idx < len(r.atoms):
            sigs[idx] = _atom_signature(r, idx)
            bonded[idx] = [j for j in r.neighbors(idx) if j in ids]
    return sigs, bonded


def _box_lengths(mol: Molecule) -> Optional[np.ndarray]:
    """Orthorhombic cell edges from ``mol.cell`` (BoxShape, float or 3-vector)."""
    cell = getattr(mol, "cell", None)
    if cell is None:
        return None
    try:
        if hasattr(cell, "a"):
            return np.array([float(cell.a), float(cell.b), float(cell.c)])
        arr = np.atleast_1d(np.asarray(cell, dtype=float)).ravel()
        if arr.size == 1:
            return np.repeat(arr, 3)
        if arr.size >= 3:
            return arr[:3]
    except (TypeError, ValueError):
        return None
    return None


def _distance(mol: Molecule, i: int, j: int, box: Optional[np.ndarray] = None) -> float:
    d = np.asarray(mol.atoms[i].xyz, dtype=float) - np.asarray(mol.atoms[j].xyz, dtype=float)
    if box is not None:
        d = d - box * np.round(d / box)            # minimum image
    return float(math.sqrt(float(d @ d)))


def _find_candidate(mol: Molecule, template: ReactionTemplate, exclude: set,
                    cutoff: float = 5.0,
                    box: Optional[np.ndarray] = None) -> Optional[Dict[int, int]]:
    """Map every reactive site to a system atom, nearest created-bond pair first.

    Returns ``{template_reactant_index: mol_atom_index}`` or None.
    """
    info = _site_info(template)
    if info is None:
        return None
    sigs, bonded = info
    if not sigs:
        return None

    atom_sig = {a.index: _atom_signature(mol, a.index)
                for a in mol.atoms if a.index not in exclude}
    cands: Dict[int, List[int]] = {s: [i for i, g in atom_sig.items() if g == sig]
                                   for s, sig in sigs.items()}
    if any(not c for c in cands.values()):
        return None

    xyz = np.array([a.xyz for a in mol.atoms], dtype=float)

    def extend(picked: Dict[int, int]) -> Optional[Dict[int, int]]:
        """Assign the remaining sites: bonded partners first, then nearest."""
        used = set(picked.values())
        pending = [s for s in sigs if s not in picked]
        guard = 0
        while pending and guard < 4 * len(sigs):
            guard += 1
            s = pending.pop(0)
            anchors = [picked[t] for t in bonded.get(s, []) if t in picked]
            options = [i for i in cands[s] if i not in used]
            if anchors:
                nb = set(mol.neighbors(anchors[0]))
                options = [i for i in options if i in nb]
            elif any(t in sigs and t not in picked for t in bonded.get(s, [])) and pending:
                pending.append(s)                  # wait for a bonded partner
                continue
            elif picked:
                ref = list(picked.values())
                options = [i for i in options
                           if min(_distance(mol, i, r, box) for r in ref) <= cutoff]
                options.sort(key=lambda i: min(_distance(mol, i, r, box) for r in ref))
            if not options:
                return None
            picked[s] = options[0]
            used.add(options[0])
        return picked if len(picked) == len(sigs) else None

    pairs = [(a, b) for a, b in template.created_bonds if a in sigs and b in sigs]
    if not pairs:
        return extend({})

    a_site, b_site = pairs[0]
    A = np.array(cands[a_site]); B = np.array(cands[b_site])
    try:
        from scipy.spatial import cKDTree
        if box is not None:
            wrapped = np.mod(xyz, box)
            tree = cKDTree(wrapped[B], boxsize=box)
            hits = tree.query_ball_point(wrapped[A], r=cutoff)
        else:
            tree = cKDTree(xyz[B])
            hits = tree.query_ball_point(xyz[A], r=cutoff)
        close = [(int(A[k]), int(B[j])) for k, js in enumerate(hits) for j in js]
    except ImportError:                                     # pragma: no cover
        close = [(int(i), int(j)) for i in A for j in B
                 if _distance(mol, int(i), int(j), box) <= cutoff]
    close = [(i, j) for i, j in close if i != j]
    close.sort(key=lambda p: _distance(mol, p[0], p[1], box))
    for i, j in close:
        got = extend({a_site: i, b_site: j})
        if got is None:
            continue
        # Every created bond must be in range, not just the first: a
        # condensation creates the ester C-O AND the leaving water's O-H.
        # Returning a mapping whose second bond is far made the caller reject
        # it and stop the whole template, with closer candidates untried.
        if all(_distance(mol, got[a], got[b], box) <= cutoff
               for a, b in pairs):
            return got
    return None


# ============================================================ apply one
def _apply_template_once(
    mol: Molecule,
    template: ReactionTemplate,
    exclude: set,
    cutoff: float,
) -> Optional[Dict[str, object]]:
    box = _box_lengths(mol)
    picked = _find_candidate(mol, template, exclude, cutoff=cutoff, box=box)
    if not picked:
        return None

    # Every bond about to be created must be within the cutoff.
    for a, b in template.created_bonds:
        if a in picked and b in picked:
            if _distance(mol, picked[a], picked[b], box) > cutoff:
                return None

    # Apply deletions of bonds
    del_bond_pairs = set()
    for a, b in template.deleted_bonds:
        if a in picked and b in picked:
            del_bond_pairs.add(frozenset({picked[a], picked[b]}))
    mol.bonds = [
        (i, j, o) for (i, j, o) in mol.bonds if frozenset({i, j}) not in del_bond_pairs
    ]

    # Bond-order changes on bonds the reaction keeps (C=C -> C-C ...)
    changed = {}
    for a, b, _old, new in template.changed_bonds:
        if a in picked and b in picked:
            changed[frozenset({picked[a], picked[b]})] = float(new)
    if changed:
        mol.bonds = [(i, j, changed.get(frozenset({i, j}), o)) for (i, j, o) in mol.bonds]

    # Apply deletions of atoms (only those covered by the picked mapping)
    remap = None
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
        picked = {r: remap[m] for r, m in picked.items() if m in remap}

    # Apply creation of bonds, with the product's bond order
    orders = list(template.created_bond_orders or [])
    n_created = 0
    for k, (a, b) in enumerate(template.created_bonds):
        if a in picked and b in picked:
            order = orders[k] if k < len(orders) else 1.0
            mol.bonds.append((picked[a], picked[b], float(order)))
            n_created += 1

    # Element changes
    for ch in template.element_changes:
        a = ch["reactant_atom"]
        if a in picked:
            mol.atoms[picked[a]].element = ch["to"]

    return {
        "bonds_created": n_created,
        "bonds_deleted": len(del_bond_pairs),
        "atoms_deleted": len(to_delete),
        "used": set(picked.values()),
        "remap": remap,
    }


# ============================================================ top-level
def apply_library(
    system_path: str | Path,
    library: ReactionLibrary,
    output_path: str | Path,
    max_events: int = 1_000_000,
    cutoff: float = 5.0,
    box: Optional[Sequence[float]] = None,
) -> XlinkStats:
    """Apply every template in `library` to `system_path` and write result.

    Distances use the minimum image when the system has a periodic cell:
    read from the file (LAMMPS data box, PDB CRYST1, GRO box line) or given
    as ``box`` (edge lengths, Å), which wins.
    """
    mol = load_structure(system_path)
    if box is not None:
        setattr(mol, "cell", tuple(float(x) for x in box))
    if getattr(mol, "cell", None) is None:
        log.info("%s carries no periodic box: distances are not minimum-imaged "
                 "(pass --box A B C for a periodic cell).", Path(system_path).name)
    log.info("Loaded system %s (%d atoms, %d bonds)", system_path, len(mol.atoms), len(mol.bonds))
    stats = XlinkStats()
    for template in library.templates:
        stats.events_per_reaction.setdefault(template.name, 0)
        if _site_info(template) is None:
            log.warning("Reaction %s: no reactive-site information could be "
                        "read; it was skipped.", template.name)
            continue
        excluded: set = set()
        for _ in range(max_events):
            info = _apply_template_once(mol, template, excluded, cutoff=cutoff)
            if not info:
                break
            remap = info["remap"]
            if remap is not None:
                excluded = {remap[i] for i in excluded if i in remap}
            # An atom that has reacted is not offered again for this template.
            excluded |= info["used"]
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
    if stats.events_applied:
        log.warning(RETYPE_WARNING)
    return stats
