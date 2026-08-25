"""Reaction / crosslinking learner and library.

Accepts reactant and product structures in **any format** OpenBabel can read
(xyz, pdb, mol2, mol, sdf, cml, SMILES). Learns the transformation as a
Weisfeiler-Lehman graph mapping (adapted from the user's `reaction_mapper.py`)
and stores it as a reaction *template*:

    {
      "name": "epoxide_opening",
      "reactant": "...",
      "product":  "...",
      "mapped_atoms": {r_index: p_index, ...},
      "deleted_atoms":     [r_index, ...],
      "new_product_atoms": [p_index, ...],
      "deleted_bonds":     [[r_i, r_j], ...],
      "created_bonds":     [[r_i, r_j], ...],
      "element_changes":   [{"reactant_atom": i, "from": "O", "to": "N"}, ...],
      "evidence":          {r_index: "WL_radius_3", ...}
    }

Multiple reactions form a *library*::

    Library.from_folders(root, reactions=[...])
    library.save("library.json")

The library can be handed to :mod:`paaf.xlink_engine` (or the
user's existing ``xlink_engine_local_mpi_*.py``) to apply templates to a
packed LAMMPS system.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .logging_utils import get_logger
from .structure import Molecule, load_structure

log = get_logger(__name__)


# ================================================================ helpers
def _wl_signatures(mol: Molecule, radius: int) -> Dict[int, tuple]:
    """Weisfeiler-Lehman node signatures up to `radius`."""
    sigs = {
        a.index: (a.element, len(mol.neighbors(a.index))) for a in mol.atoms
    }
    for _ in range(radius):
        sigs = {
            i: (sigs[i], tuple(sorted(sigs[j] for j in mol.neighbors(i))))
            for i in sigs
        }
    return sigs


def _initial_mapping(r: Molecule, p: Molecule, max_radius: int = 6):
    mapping: Dict[int, int] = {}
    used: set[int] = set()
    evidence: Dict[int, str] = {}
    for radius in range(max_radius, -1, -1):
        rs = _wl_signatures(r, radius)
        ps = _wl_signatures(p, radius)
        rb, pb = defaultdict(list), defaultdict(list)
        for i, s in rs.items():
            if i not in mapping:
                rb[s].append(i)
        for i, s in ps.items():
            if i not in used:
                pb[s].append(i)
        for sig, rnodes in rb.items():
            pnodes = pb.get(sig, [])
            if len(rnodes) == 1 and len(pnodes) == 1:
                mapping[rnodes[0]] = pnodes[0]
                used.add(pnodes[0])
                evidence[rnodes[0]] = f"unique_WL_radius_{radius}"
    return mapping, used, evidence


def _propagate(r: Molecule, p: Molecule, mapping, used, evidence):
    changed = True
    while changed:
        changed = False
        for ri in [a.index for a in r.atoms]:
            if ri in mapping:
                continue
            mapped_nbrs = [mapping[n] for n in r.neighbors(ri) if n in mapping]
            if not mapped_nbrs:
                continue
            candidates: list[tuple] = []
            for pj in [a.index for a in p.atoms]:
                if pj in used:
                    continue
                if r.atoms[ri].element != p.atoms[pj].element:
                    continue
                agree = sum(1 for m in mapped_nbrs if pj in p.neighbors(m))
                if agree:
                    degdiff = abs(len(r.neighbors(ri)) - len(p.neighbors(pj)))
                    candidates.append((agree, -degdiff, pj))
            candidates.sort(reverse=True)
            if candidates and (len(candidates) == 1 or candidates[0][:2] > candidates[1][:2]):
                pj = candidates[0][2]
                mapping[ri] = pj
                used.add(pj)
                evidence[ri] = "neighbor propagation"
                changed = True
    return mapping, used, evidence


def _assign_remaining(r: Molecule, p: Molecule, mapping, used, evidence):
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except Exception:
        # SciPy not installed; skip the global fallback assignment.
        return mapping, used, evidence

    rleft = [a.index for a in r.atoms if a.index not in mapping]
    pleft = [a.index for a in p.atoms if a.index not in used]
    if not rleft or not pleft:
        return mapping, used, evidence
    cost = np.full((len(rleft), len(pleft)), 1e6)
    for i, ri in enumerate(rleft):
        for j, pj in enumerate(pleft):
            ra, pa = r.atoms[ri], p.atoms[pj]
            if ra.element != pa.element:
                continue
            v = 4.0 * abs(len(r.neighbors(ri)) - len(p.neighbors(pj)))
            mapped_r = {mapping[n] for n in r.neighbors(ri) if n in mapping}
            p_nbrs = set(p.neighbors(pj))
            v += 15.0 * len(mapped_r - p_nbrs)
            v -= 8.0 * len(mapped_r & p_nbrs)
            cost[i, j] = v
    rows, cols = linear_sum_assignment(cost)
    for i, j in zip(rows, cols):
        if cost[i, j] < 1e5:
            ri, pj = rleft[i], pleft[j]
            mapping[ri] = pj
            used.add(pj)
            evidence[ri] = f"global assignment (cost={cost[i, j]:.2f})"
    return mapping, used, evidence


# ================================================================ template
@dataclass
class ReactionTemplate:
    name: str
    reactant: str
    product: str
    mapped_atoms: Dict[int, int] = field(default_factory=dict)
    deleted_atoms: List[int] = field(default_factory=list)
    new_product_atoms: List[int] = field(default_factory=list)
    deleted_bonds: List[Tuple[int, int]] = field(default_factory=list)
    created_bonds: List[Tuple[int, int]] = field(default_factory=list)
    element_changes: List[Dict] = field(default_factory=list)
    evidence: Dict[int, str] = field(default_factory=dict)
    # reactive-site markers, useful for the xlink engine
    reactive_sites: List[Dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.name}: mapped {len(self.mapped_atoms)}, "
            f"del atoms {len(self.deleted_atoms)}, new atoms {len(self.new_product_atoms)}, "
            f"del bonds {len(self.deleted_bonds)}, new bonds {len(self.created_bonds)}"
        )

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "reactant": self.reactant,
            "product": self.product,
            "mapped_atoms": {str(k): v for k, v in self.mapped_atoms.items()},
            "deleted_atoms": list(self.deleted_atoms),
            "new_product_atoms": list(self.new_product_atoms),
            "deleted_bonds": [list(b) for b in self.deleted_bonds],
            "created_bonds": [list(b) for b in self.created_bonds],
            "element_changes": list(self.element_changes),
            "evidence": {str(k): v for k, v in self.evidence.items()},
            "reactive_sites": list(self.reactive_sites),
        }

    @classmethod
    def from_json(cls, d: dict) -> "ReactionTemplate":
        return cls(
            name=d["name"],
            reactant=d["reactant"],
            product=d["product"],
            mapped_atoms={int(k): v for k, v in d.get("mapped_atoms", {}).items()},
            deleted_atoms=list(d.get("deleted_atoms", [])),
            new_product_atoms=list(d.get("new_product_atoms", [])),
            deleted_bonds=[tuple(x) for x in d.get("deleted_bonds", [])],
            created_bonds=[tuple(x) for x in d.get("created_bonds", [])],
            element_changes=list(d.get("element_changes", [])),
            evidence={int(k): v for k, v in d.get("evidence", {}).items()},
            reactive_sites=list(d.get("reactive_sites", [])),
        )


# ================================================================ mapping + extraction
def complete_mapping(
    r: Molecule,
    p: Molecule,
    seed: Optional[Dict[int, int]] = None,
    max_radius: int = 6,
):
    """Return ``(mapping, evidence)`` from reactant→product atom indices.

    If ``seed`` is given (e.g. an exact anchor from atom-map numbers) the
    Weisfeiler-Lehman initial pass is skipped and completion starts from the
    seed; otherwise the full WL initial mapping is used. In both cases the
    result is refined by neighbour propagation + a global (Hungarian) fallback.
    """
    if seed:
        mapping: Dict[int, int] = dict(seed)
        used: set[int] = set(seed.values())
        evidence: Dict[int, str] = {k: "atom-map anchor" for k in seed}
    else:
        mapping, used, evidence = _initial_mapping(r, p, max_radius=max_radius)
    mapping, used, evidence = _propagate(r, p, mapping, used, evidence)
    mapping, used, evidence = _assign_remaining(r, p, mapping, used, evidence)
    mapping, used, evidence = _propagate(r, p, mapping, used, evidence)
    return mapping, evidence


def extract_template(
    r: Molecule,
    p: Molecule,
    mapping: Dict[int, int],
    evidence: Dict[int, str],
    name: str,
    reactant: str,
    product: str,
) -> ReactionTemplate:
    """Build a :class:`ReactionTemplate` from a reactant/product pair and a
    completed reactant→product atom mapping (pure graph diff — no geometry)."""
    r_ids = {a.index for a in r.atoms}
    p_ids = {a.index for a in p.atoms}
    deleted_atoms = sorted(r_ids - set(mapping.keys()))
    new_product_atoms = sorted(p_ids - set(mapping.values()))

    inv = {pj: ri for ri, pj in mapping.items()}
    deleted_bonds: List[Tuple[int, int]] = []
    for a, b, _ in r.bonds:
        if a in mapping and b in mapping:
            pa, pb = mapping[a], mapping[b]
            if not any((pa == x and pb == y) or (pa == y and pb == x) for x, y, _ in p.bonds):
                deleted_bonds.append((a, b))
        elif a in deleted_atoms or b in deleted_atoms:
            deleted_bonds.append((a, b))

    created_bonds: List[Tuple[int, int]] = []
    for a, b, _ in p.bonds:
        if a in inv and b in inv:
            ra, rb = inv[a], inv[b]
            if not any((ra == x and rb == y) or (ra == y and rb == x) for x, y, _ in r.bonds):
                created_bonds.append((ra, rb))

    elem_changes: List[Dict] = []
    for ri, pj in sorted(mapping.items()):
        if r.atoms[ri].element != p.atoms[pj].element:
            elem_changes.append({
                "reactant_atom": ri,
                "product_atom": pj,
                "from": r.atoms[ri].element,
                "to":   p.atoms[pj].element,
            })

    # Reactive sites: atoms involved in bond making/breaking. Useful for the
    # crosslink engine when scanning a packed system for candidate pairs.
    site_atoms = set()
    for a, b in deleted_bonds + created_bonds:
        site_atoms.add(a); site_atoms.add(b)
    reactive_sites = [
        {"reactant_atom": a, "element": r.atoms[a].element,
         "n_neighbors": len(r.neighbors(a))}
        for a in sorted(site_atoms) if a < len(r.atoms)
    ]

    return ReactionTemplate(
        name=name,
        reactant=reactant,
        product=product,
        mapped_atoms=dict(mapping),
        deleted_atoms=deleted_atoms,
        new_product_atoms=new_product_atoms,
        deleted_bonds=sorted(deleted_bonds),
        created_bonds=sorted(created_bonds),
        element_changes=elem_changes,
        evidence=dict(evidence),
        reactive_sites=reactive_sites,
    )


# ================================================================ learner
def learn_reaction(
    reactant_path: str | Path,
    product_path: str | Path,
    name: Optional[str] = None,
    max_radius: int = 6,
) -> ReactionTemplate:
    """Learn a single reaction template from a reactant/product structure pair.

    Structures can be any format OpenBabel handles (xyz/pdb/mol2/mol/sdf/cml).
    """
    r = load_structure(reactant_path)
    p = load_structure(product_path)
    log.info(
        "Learning reaction %s: %s (%d atoms) -> %s (%d atoms)",
        name or Path(reactant_path).stem, Path(reactant_path).name, len(r.atoms),
        Path(product_path).name, len(p.atoms),
    )

    mapping, evidence = complete_mapping(r, p, seed=None, max_radius=max_radius)
    return extract_template(
        r, p, mapping, evidence,
        name=name or Path(reactant_path).parent.name or Path(reactant_path).stem,
        reactant=str(Path(reactant_path).resolve()),
        product=str(Path(product_path).resolve()),
    )


# ================================================================ library
@dataclass
class ReactionLibrary:
    templates: List[ReactionTemplate] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "paaf-reaction-library-v1",
            "n_reactions": len(self.templates),
            "reactions": [t.to_json() for t in self.templates],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        log.info("Wrote reaction library (%d templates) -> %s", len(self.templates), path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ReactionLibrary":
        d = json.loads(Path(path).read_text())
        return cls(templates=[ReactionTemplate.from_json(x) for x in d.get("reactions", [])])

    @classmethod
    def learn_many(cls, reactions: Sequence[dict]) -> "ReactionLibrary":
        """Learn a library from a list of reaction specs.

        Each spec: {"name": ..., "reactant": "path", "product": "path"}
        """
        lib = cls()
        for spec in reactions:
            t = learn_reaction(spec["reactant"], spec["product"], name=spec.get("name"))
            lib.templates.append(t)
            log.info("  " + t.summary())
        return lib

    @classmethod
    def from_folder(cls, root: str | Path,
                    reactant_names: Sequence[str] = ("reactant",),
                    product_names: Sequence[str] = ("product",)) -> "ReactionLibrary":
        """Discover reactions in ``root/<name>/`` following the layout used by
        the user's ``reaction_engine_local.py`` — a folder per reaction with
        ``reactant.<ext>`` and ``product.<ext>``.
        """
        root = Path(root)
        specs = []
        exts = ("xyz", "pdb", "mol2", "mol", "sdf", "data")
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            r_path = p_path = None
            for e in exts:
                for stem in reactant_names:
                    cand = folder / f"{stem}.{e}"
                    if cand.exists() and r_path is None:
                        r_path = cand
                for stem in product_names:
                    cand = folder / f"{stem}.{e}"
                    if cand.exists() and p_path is None:
                        p_path = cand
            if r_path and p_path:
                specs.append({"name": folder.name, "reactant": str(r_path), "product": str(p_path)})
        if not specs:
            raise RuntimeError(f"No reactant/product pairs discovered in {root}")
        return cls.learn_many(specs)

    def summary(self) -> str:
        return "\n".join(t.summary() for t in self.templates)
