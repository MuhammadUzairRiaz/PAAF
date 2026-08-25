"""Rebuild all-atom structures from a grown skeletal-bead cell.

Why this module has to exist
----------------------------
:mod:`paaf.cell.grow` builds chains one **skeletal atom** at a time. That is
the right granularity for the physics — RIS parameterises a skeletal bond, and
growing at that level is what makes chain dimensions and melt density come out
right. But no force-field typer will look at a chain of bare backbone beads
and produce anything useful: DL_FIELD and Moltemplate need elements,
hydrogens, side groups and bond orders.

So the beads have to be dressed. This module does that, and it can only do it
*because* a bead is a skeletal atom: the backbone of the atomistic chain is
already built, exactly, at positions the growth accepted. Nothing is
re-sampled and nothing moves. All that is added is the material hanging off
the backbone.

How
---
For each repeat unit PAAF embeds one 3D **template** — the repeat unit with
its two attachment points turned into hydrogens, minimised with MMFF. In that
template every backbone atom has a local frame defined by its two backbone
neighbours (a link hydrogen standing in at the unit's ends). The same frame is
built at the corresponding bead in the grown chain, and the rigid transform
between the two frames is applied to everything attached to that backbone
atom.

Because the transform is rigid, every substituent bond length and bond angle
survives exactly. What is *not* reproduced is side-group torsion: each unit
gets the template's rotamer. Growth never sampled those, so inventing them
here would be fabrication. They are what a subsequent minimisation and short
equilibration are for.

Tacticity
---------
Applying one template to every unit gives an **isotactic** chain. That is a
real stereochemical choice and it changes properties, so it is exposed rather
than hidden: ``"atactic"`` mirrors each unit's substituents through the local
backbone plane at random, ``"syndiotactic"`` alternates. Mirroring inverts the
configuration at the stereocentre, which is what tacticity is.

Limits, stated plainly
----------------------
* Side-group conformers are the template's, not sampled.
* Substituents of different chains can overlap; the reported worst contact
  says how badly, and minimisation is expected before any production run.
* Rings on the backbone itself (as opposed to pendant rings) are not handled —
  the backbone is assumed to be an acyclic path between attachment points.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..logging_utils import get_logger
from ..structure import Atom, Molecule
from .grow import (
    BackboneRingError, backbone_atoms_per_unit, backbone_ring_atoms,
    count_attachment_points, max_backbone_atoms_in_a_ring,
)

log = get_logger(__name__)

__all__ = ["UnitTemplate", "BackmapResult", "build_unit_template",
           "backmap_chain", "backmap_cell", "available", "find_rings",
           "count_speared_rings", "rotate_side_groups", "relax_substituents",
           "unthread_rings"]


def available() -> bool:
    """True when RDKit is importable (required to build templates)."""
    try:
        import rdkit  # noqa: F401
        return True
    except Exception:
        return False


# =====================================================================
# The template: one repeat unit in 3D, with its backbone identified
# =====================================================================
@dataclass
class UnitTemplate:
    """One repeat unit, embedded in 3D, with backbone bookkeeping.

    Attributes
    ----------
    elements :
        Element symbol per template atom.
    coords :
        (N, 3) template coordinates in Å.
    backbone :
        Template indices of the skeletal atoms, **in chain order** — the path
        from the head attachment point to the tail one.
    head_link, tail_link :
        Template indices of the two hydrogens that stand in for the bonds to
        the previous and next repeat unit. They are *not* emitted into the
        chain except at the two chain ends, where they become the real end
        caps.
    bonds :
        ``(i, j, order)`` over template indices.
    attached :
        For each backbone position, the template indices that ride with it —
        its substituent subtree, excluding backbone atoms and link hydrogens.
    """
    smiles: str
    elements: List[str] = field(default_factory=list)
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    backbone: List[int] = field(default_factory=list)
    head_link: int = -1
    tail_link: int = -1
    bonds: List[Tuple[int, int, float]] = field(default_factory=list)
    attached: List[List[int]] = field(default_factory=list)

    @property
    def n_backbone(self) -> int:
        return len(self.backbone)

    def atoms_per_unit(self) -> int:
        """Atoms this unit contributes to an interior position of the chain."""
        return len(self.backbone) + sum(len(a) for a in self.attached)


def _frame(p_prev: np.ndarray, p_self: np.ndarray,
           p_next: np.ndarray) -> np.ndarray:
    """Orthonormal 3x3 frame from three consecutive backbone points.

    Columns are (e1, e2, e3):
    ``e1`` along prev→next (the local chain direction), ``e2`` the component
    of the bisector perpendicular to it, ``e3`` their cross product. Using the
    prev→next axis rather than a single bond makes the frame insensitive to
    which of the two bonds is slightly off, and it is well defined for any
    bond angle that is not exactly 180°.
    """
    e1 = p_next - p_prev
    n1 = np.linalg.norm(e1)
    if n1 < 1e-9:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 = e1 / n1
    mid = p_self - 0.5 * (p_prev + p_next)
    e2 = mid - (mid @ e1) * e1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-9:
        # Collinear (all-trans and exactly straight): any perpendicular works.
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(e1 @ tmp) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e2 = tmp - (tmp @ e1) * e1
        e2 /= np.linalg.norm(e2)
    else:
        e2 = e2 / n2
    e3 = np.cross(e1, e2)
    return np.column_stack([e1, e2, e3])


def build_unit_template(smiles: str, seed: int = 0xC0FFEE) -> UnitTemplate:
    """Embed one repeat unit and work out its backbone and substituents.

    The two ``[*]`` attachment points are turned into hydrogens so the unit is
    a real, embeddable molecule. Crucially, converting in place (rather than
    re-parsing a capped SMILES) keeps every heavy atom's index, so the
    backbone path found on the starred molecule stays valid.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if count_attachment_points(smiles) != 2:
        raise ValueError(
            f"{smiles!r} must declare exactly two [*] attachment points to be "
            f"back-mapped as a linear repeat unit.")

    # A ring spanning ONE backbone bond (two skeletal atoms plus a bridge) is
    # fine: an epoxide wants that C-C at ~1.47 A and growth gives 1.53 A, a 4%
    # stretch the bridging atom absorbs and minimisation settles. A ring
    # spanning three or more bonds — a benzene in the chain, as in PET or
    # polycarbonate — cannot be reconciled with a path built at 1.53 A and
    # 112 degrees, and forcing it tears the ring open.
    n_ring = max_backbone_atoms_in_a_ring(smiles)
    if n_ring > 2:
        raise BackboneRingError(
            f"{smiles!r} has {n_ring} skeletal atom(s) inside a ring, i.e. the "
            f"backbone runs THROUGH a ring rather than past it. Poly(ethylene "
            f"terephthalate) and polycarbonate are of this kind; polystyrene, "
            f"whose phenyl is pendant, is not.\n\n"
            f"Such a unit cannot be back-mapped onto a chain grown with RIS "
            f"geometry: the ring's internal geometry is fixed by the ring, "
            f"while the grown path has the RIS bond length and angle, and "
            f"forcing one onto the other tears the ring's closure bond open. "
            f"PAAF refuses rather than writing a structure that looks "
            f"plausible and is chemically destroyed.\n\n"
            f"The bead-level cell is still valid and still exported: growth "
            f"itself is unaffected, because a bead is a skeletal atom either "
            f"way. Only the all-atom reconstruction is refused.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {smiles!r}")

    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    ends = []
    for s in stars:
        nbrs = mol.GetAtomWithIdx(s).GetNeighbors()
        if not nbrs:
            raise ValueError(f"{smiles!r}: an attachment point has no neighbour.")
        ends.append(nbrs[0].GetIdx())

    if ends[0] == ends[1]:
        # Both attachment points hang off the SAME atom, so the repeat unit
        # contributes exactly one skeletal atom. Poly(dimethylsiloxane) as the
        # library writes it, [*][Si]([*])(C)C, is of this form. RDKit's
        # GetShortestPath raises an Invariant Violation when asked for a path
        # from an atom to itself, so the single-atom case is handled directly.
        path = [ends[0]]
    else:
        path = list(Chem.GetShortestPath(mol, ends[0], ends[1]))
    if not path:
        raise ValueError(f"{smiles!r}: no backbone path between attachments.")

    # Turn the wildcards into hydrogens, in place, preserving indices.
    rw = Chem.RWMol(mol)
    for s in stars:
        a = rw.GetAtomWithIdx(s)
        a.SetAtomicNum(1)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
    capped = rw.GetMol()
    Chem.SanitizeMol(capped)
    capped = Chem.AddHs(capped)          # appends; existing indices unchanged

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(capped, params) != 0:
        if AllChem.EmbedMolecule(capped, randomSeed=seed,
                                 useRandomCoords=True) != 0:
            raise RuntimeError(f"3D embedding failed for the unit {smiles!r}")
    try:
        AllChem.MMFFOptimizeMolecule(capped, maxIters=2000)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(capped, maxIters=2000)
        except Exception:
            log.warning("Template cleanup failed for %s; using raw embed.",
                        smiles)

    conf = capped.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i))
                       for i in range(capped.GetNumAtoms())], dtype=float)
    elements = [a.GetSymbol() for a in capped.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
              float(b.GetBondTypeAsDouble())) for b in capped.GetBonds()]

    # Which star sits at which end of the path.
    head_link = stars[0] if ends[0] == path[0] else stars[1]
    tail_link = stars[1] if head_link == stars[0] else stars[0]

    # Substituent subtrees: flood out from each backbone atom without
    # crossing another backbone atom or a link hydrogen.
    blocked = set(path) | {head_link, tail_link}
    adj: Dict[int, List[int]] = {i: [] for i in range(capped.GetNumAtoms())}
    for i, j, _ in bonds:
        adj[i].append(j)
        adj[j].append(i)

    # Each non-backbone atom belongs to EXACTLY ONE backbone atom's group.
    # A bridging atom — the oxygen of an epoxide, reachable from both ring
    # carbons — would otherwise be flooded into both groups and emitted twice.
    # It is claimed by the first anchor along the path; the bond to the second
    # is still in `bonds`, so the ring closes correctly.
    claimed: set = set()
    attached: List[List[int]] = []
    for bb in path:
        group: List[int] = []
        stack = [n for n in adj[bb] if n not in blocked and n not in claimed]
        seen = set(stack)
        while stack:
            k = stack.pop()
            group.append(k)
            claimed.add(k)
            for n in adj[k]:
                if n not in seen and n not in blocked and n not in claimed:
                    seen.add(n)
                    stack.append(n)
        attached.append(sorted(group))

    return UnitTemplate(
        smiles=smiles, elements=elements, coords=coords, backbone=path,
        head_link=head_link, tail_link=tail_link, bonds=bonds,
        attached=attached,
    )


# =====================================================================
# Back-mapping one chain
# =====================================================================
@dataclass
class BackmapResult:
    molecule: Molecule
    n_chains: int = 0
    atoms_per_chain: List[int] = field(default_factory=list)
    worst_contact_a: float = float("inf")
    #: Bonds threading through a ring; -1 when the check could not run.
    #: A number, not just a note, because the caller regrows on it.
    n_speared: int = -1
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"back-mapped {self.n_chains} chain(s) to "
                 f"{len(self.molecule.atoms)} atoms",
                 f"closest non-bonded contact: {self.worst_contact_a:.2f} Å"]
        lines.extend(f"note: {n}" for n in self.notes)
        return "\n".join(lines)


def _unwrap(path_xyz: np.ndarray, dims: Optional[np.ndarray]) -> np.ndarray:
    """Undo periodic wrapping along a bonded path.

    Bonds must be measured in the chain's own continuous frame; a wrapped
    coordinate would put a 1.53 Å bond across the whole box.
    """
    if dims is None:
        return path_xyz.copy()
    out = path_xyz.copy()
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        out[i] -= dims * np.round(d / dims)
    return out


def backmap_chain(beads: np.ndarray, template, dp: int, *,
                  dims: Optional[np.ndarray] = None,
                  tacticity: str = "isotactic",
                  rng: Optional[np.random.Generator] = None,
                  name: str = "chain") -> Molecule:
    """Dress one grown backbone with its atoms.

    ``beads`` are the skeletal positions in growth order. They are used
    **as-is** for the backbone atoms — back-mapping never moves what growth
    decided.

    ``template`` is either one :class:`UnitTemplate` (a homopolymer) or a
    sequence of ``dp`` templates, one per repeat unit (a copolymer). The
    per-unit form is what lets a chain of isoprene and epoxidised isoprene be
    rebuilt: each unit is dressed with its own chemistry, in the order growth
    laid it down.
    """
    if isinstance(template, UnitTemplate):
        templates = [template] * dp
    else:
        templates = list(template)
        if len(templates) != dp:
            raise ValueError(
                f"{name}: {len(templates)} templates for {dp} repeat units.")

    expected = sum(t.n_backbone for t in templates)
    if len(beads) != expected:
        raise ValueError(
            f"{name}: got {len(beads)} beads but the sequence needs "
            f"{expected}.")

    rng = rng or np.random.default_rng(0)
    chain = _unwrap(np.asarray(beads, dtype=float), dims)

    # Virtual extensions so the first and last beads have a frame too. Reflect
    # the neighbour through the end bead: it keeps the bond length and gives a
    # straight local continuation, which is the least-assumption choice.
    ext_first = 2.0 * chain[0] - chain[1]
    ext_last = 2.0 * chain[-1] - chain[-2]

    def triad(i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        prev = ext_first if i == 0 else chain[i - 1]
        nxt = ext_last if i == len(chain) - 1 else chain[i + 1]
        return prev, chain[i], nxt

    atoms: List[Atom] = []
    bb_flags: List[bool] = []          # True for skeletal atoms — these pin
    bonds: List[Tuple[int, int, float]] = []
    # Map (unit index, template index) -> global atom index.
    gmap: Dict[Tuple[int, int], int] = {}

    def mirror_this_unit(u: int) -> bool:
        if tacticity == "isotactic":
            return False
        if tacticity == "syndiotactic":
            return u % 2 == 1
        if tacticity == "atactic":
            return bool(rng.integers(0, 2))
        raise ValueError(
            f"tacticity must be isotactic, syndiotactic or atactic, "
            f"not {tacticity!r}")

    bead0 = 0                          # first bead index of the current unit
    for u in range(dp):
        template = templates[u]
        t_coords = template.coords
        nb = template.n_backbone
        flip = mirror_this_unit(u)
        for b, t_bb in enumerate(template.backbone):
            gi = bead0 + b
            p_prev, p_self, p_next = triad(gi)

            # Template frame for this backbone atom: its neighbours along the
            # backbone, with the link hydrogens standing in at the unit ends.
            t_prev = template.head_link if b == 0 else template.backbone[b - 1]
            t_next = (template.tail_link if b == nb - 1
                      else template.backbone[b + 1])
            F_t = _frame(t_coords[t_prev], t_coords[t_bb], t_coords[t_next])
            F_x = _frame(p_prev, p_self, p_next)
            if flip:
                # Reflect through the local backbone plane (e1, e2): this
                # inverts the configuration at the stereocentre.
                F_x = F_x @ np.diag([1.0, 1.0, -1.0])
            R = F_x @ F_t.T

            gmap[(u, t_bb)] = len(atoms)
            bb_flags.append(True)
            atoms.append(Atom(index=len(atoms), element=template.elements[t_bb],
                              xyz=p_self,
                              name=f"{template.elements[t_bb]}{len(atoms) + 1}"))
            for t_sub in template.attached[b]:
                bb_flags.append(False)
                local = t_coords[t_sub] - t_coords[t_bb]
                gmap[(u, t_sub)] = len(atoms)
                atoms.append(Atom(
                    index=len(atoms), element=template.elements[t_sub],
                    xyz=p_self + R @ local,
                    name=f"{template.elements[t_sub]}{len(atoms) + 1}"))

            # End caps: the link hydrogens become real hydrogens, but only at
            # the two ends of the whole chain, not at every unit boundary.
            is_chain_head = (u == 0 and b == 0)
            is_chain_tail = (u == dp - 1 and b == template.n_backbone - 1)
            for t_link, want in ((template.head_link, is_chain_head),
                                 (template.tail_link, is_chain_tail)):
                if not want:
                    continue
                local = t_coords[t_link] - t_coords[t_bb]
                gmap[(u, t_link)] = len(atoms)
                bb_flags.append(False)
                atoms.append(Atom(index=len(atoms), element="H",
                                  xyz=p_self + R @ local,
                                  name=f"H{len(atoms) + 1}"))

        bead0 += nb

    # ---- bonds: template bonds inside each unit, plus the unit-unit links
    for u in range(dp):
        template = templates[u]
        for i, j, order in template.bonds:
            gi, gj = gmap.get((u, i)), gmap.get((u, j))
            if gi is not None and gj is not None:
                bonds.append((gi, gj, order))
        if u + 1 < dp:
            tail = gmap[(u, template.backbone[-1])]
            head = gmap[(u + 1, templates[u + 1].backbone[0])]
            bonds.append((tail, head, 1.0))

    mol = Molecule(atoms=atoms, bonds=bonds, name=name)
    setattr(mol, "backbone_flags", bb_flags)
    return mol


# =====================================================================
def backmap_cell(result, specs: Sequence, *, tacticity: str = "isotactic",
                 seed: int = 12345, wrap_into_box: bool = True,
                 push_off: bool = True, push_off_target: float = 2.2,
                 progress=None) -> BackmapResult:
    """Back-map a whole :class:`paaf.cell.grow.GrowthResult`.

    ``specs`` is the same list of :class:`paaf.cell.grow.GrowSpec` that was
    grown, in the same order, so each chain's SMILES and DP can be recovered.
    """
    if not available():
        raise RuntimeError(
            "RDKit is required to back-map beads to atoms. "
            "Install it with:  conda install -c conda-forge rdkit")

    emit = progress or (lambda _m: None)
    rng = np.random.default_rng(seed)
    dims = np.asarray(result.box.bounding_box(), dtype=float)
    coords = np.array([a.xyz for a in result.molecule.atoms], dtype=float)

    templates: Dict[str, UnitTemplate] = {}
    notes: List[str] = []
    all_atoms: List[Atom] = []
    all_bonds: List[Tuple[int, int, float]] = []
    all_flags: List[bool] = []
    per_chain: List[int] = []

    def template_for(smi: str, label: str) -> UnitTemplate:
        if smi not in templates:
            emit(f"  building 3D template for {label} …")
            templates[smi] = build_unit_template(smi)
            got = templates[smi].n_backbone
            want = backbone_atoms_per_unit(smi)
            if got != want:
                notes.append(
                    f"{label}: template backbone {got} atoms but growth "
                    f"assumed {want}; back-mapping used {got}.")
        return templates[smi]

    # Chains are stored in the caller's species order, so walking specs and
    # slicing the atom list stays valid — but a COPOLYMER chain needs its own
    # per-unit templates, taken from the sequence growth actually laid down.
    chain_stats = list(getattr(result, "chains", []))
    cursor = 0
    chain_i = 0
    for sp in specs:
        if sp.is_copolymer:
            for mono in sp.monomers:
                template_for(mono.smiles, mono.name or mono.smiles)
            tpl = templates[sp.monomers[0].smiles]
        else:
            tpl = template_for(sp.repeat_unit, sp.name or sp.repeat_unit)
        smi = sp.repeat_unit or (sp.monomers[0].smiles if sp.monomers else "")
        # Minimum image only works if nothing is longer than half the box. A
        # repeat unit with a long substituent — an octyloxy side chain is
        # 11 Å — can exceed that in a small cell, and then a bond measured
        # under minimum image is wrapped to the wrong periodic copy and reads
        # as 12 Å. The structure is fine; the cell is simply too small to
        # describe it. Measured, not hypothetical: this produced 12.29 Å bonds
        # at 2 chains of DP 6, and 1.53 Å for the same polymer at 4 x DP 20.
        cands = ([templates[m.smiles] for m in sp.monomers]
                 if sp.is_copolymer else [tpl])
        extent = max(float(np.max(t.coords.max(axis=0) - t.coords.min(axis=0)))
                     for t in cands)
        half = float(np.min(dims)) / 2.0
        if extent > half:
            notes.append(
                f"WARNING: {sp.name or smi} has a repeat unit {extent:.1f} Å "
                f"across, which exceeds half the box ({half:.1f} Å). The "
                f"minimum-image convention breaks down and bond lengths "
                f"measured across the boundary are meaningless. Build a "
                f"larger cell — more chains, or a higher degree of "
                f"polymerisation.")
        nb = tpl.n_backbone
        dp = int(sp.degree_of_polymerisation)

        for _ in range(int(sp.n_chains)):
            seq = getattr(chain_stats[chain_i], "sequence", None) \
                if chain_i < len(chain_stats) else None
            if seq is not None:
                per_unit = [templates[seq.monomers[u].smiles]
                            for u in seq.order]
                n_here = sum(t.n_backbone for t in per_unit)
                which = per_unit
            else:
                n_here = dp * nb
                which = tpl
            beads = coords[cursor:cursor + n_here]
            cursor += n_here
            mol = backmap_chain(beads, which, dp, dims=dims,
                                tacticity=tacticity, rng=rng,
                                name=f"{sp.name or 'chain'}_{chain_i + 1}")
            base = len(all_atoms)
            for a in mol.atoms:
                all_atoms.append(Atom(index=base + a.index, element=a.element,
                                      xyz=a.xyz, name=a.name))
            for i, j, o in mol.bonds:
                all_bonds.append((base + i, base + j, o))
            all_flags.extend(getattr(mol, "backbone_flags",
                                     [False] * len(mol.atoms)))
            per_chain.append(len(mol.atoms))
            chain_i += 1
            emit(f"  chain {chain_i}: {len(mol.atoms)} atoms")

    cell = Molecule(atoms=all_atoms, bonds=all_bonds, name="amorphous_cell")
    try:
        setattr(cell, "cell", result.box)
    except Exception:
        pass

    # Unthreading runs UNCONDITIONALLY, before the push-off gate. It is the
    # step that decides whether the cell can be typed at all — measured on a
    # real PE/PS cell: 35 threaded bonds -> 1 from this alone — and the
    # reseed probe calls this function with push_off=False, so gating it
    # would make the probe count raw threading and reject every seed.
    try:
        n_freed, n_stuck = unthread_rings(cell, all_flags, dims,
                                          n_angles=144, passes=3)
        if n_freed or n_stuck:
            emit(f"  unthreaded {n_freed} ring(s)"
                 + (f"; {n_stuck} still interlocked" if n_stuck else ""))
    except Exception as exc:                       # pragma: no cover
        emit(f"  ring unthreading skipped ({exc})")

    if push_off:
        emit("  relieving side-group overlaps (backbone pinned) …")
        start = _closest_nonbonded(
            np.array([a.xyz for a in all_atoms], dtype=float), all_bonds, dims)
        # Rigid rotation first: it is exact and it is the only move that can
        # help a ring. The per-atom push then cleans up whatever is left,
        # which for a polymer with only hydrogens on the backbone is all of it.
        n_rot = rotate_side_groups(cell, all_flags, dims)
        # Targeted unthreading AFTER the clash-driven rotation: a bond
        # through a ring's centre touches nothing, so the score above is
        # blind to it, and neither reseeding nor any push-off can undo an
        # interlock once it exists. This rotates exactly the threaded rings
        # until nothing passes through them.
        try:
            n_freed2, n_stuck2 = unthread_rings(cell, all_flags, dims,
                                                n_angles=144, passes=2)
            if n_freed2 or n_stuck2:
                emit(f"  re-unthreaded {n_freed2} ring(s)"
                     + (f"; {n_stuck2} still interlocked"
                        if n_stuck2 else ""))
        except Exception as exc:               # pragma: no cover
            emit(f"  ring unthreading skipped ({exc})")
        before, worst = relax_substituents(
            cell, all_flags, dims, target=push_off_target)
        notes.append(
            f"Push-off: closest non-bonded contact {start:.2f} Å -> "
            f"{worst:.2f} Å. {n_rot} side group(s) were rotated about their "
            f"own bond to the backbone — an exact rigid move that preserves "
            f"all internal geometry — and the remaining atoms were pushed "
            f"apart with every bond length restored. The backbone was pinned "
            f"throughout, so nothing growth validated was altered.")
    else:
        worst = _closest_nonbonded(
            np.array([a.xyz for a in all_atoms], dtype=float), all_bonds, dims)

    xyz = np.array([a.xyz for a in all_atoms], dtype=float)
    if wrap_into_box:
        # Wrap per chain by its centre of mass so molecules stay whole — a
        # per-atom wrap would tear bonds across the boundary.
        start = 0
        for n in per_chain:
            block = xyz[start:start + n]
            com = block.mean(axis=0)
            shift = dims * np.floor(com / dims)
            for k in range(start, start + n):
                all_atoms[k].xyz = xyz[k] - shift
            start += n

    bridged = sorted({sp.name or sp.repeat_unit for sp in specs
                      if max_backbone_atoms_in_a_ring(sp.repeat_unit) == 2}
                     | {m.name or m.smiles for sp in specs
                        for m in (sp.monomers or [])
                        if max_backbone_atoms_in_a_ring(m.smiles) == 2})
    if bridged:
        notes.append(
            f"Bridged backbone ring in: {', '.join(bridged)}. The ring spans "
            f"one backbone bond, which growth placed at the RIS length "
            f"(1.53 A) rather than the ring's own (~1.47 A for an epoxide). "
            f"The bridging atom absorbs the ~4% stretch; minimisation settles "
            f"it. Rings spanning more backbone bonds are refused outright.")

    notes.append(
        "Backbone atoms sit exactly where growth placed them. Side groups "
        "carry the template's rotamer — growth never sampled side-group "
        "torsions, so they are not reproduced here. Minimise before use.")
    if tacticity == "isotactic":
        notes.append(
            "Tacticity: ISOTACTIC (one template applied to every unit). Pass "
            "tacticity='atactic' for a random configuration at each "
            "stereocentre.")
    if worst < 1.2:
        notes.append(
            f"Closest non-bonded contact is still {worst:.2f} Å — severe. "
            f"Side groups of neighbouring chains have interpenetrated beyond "
            f"what the push-off can relieve. Rebuild at a lower density, or "
            f"raise the growth tolerance, before trusting this cell.")

    # Spearing can only be looked for once the rings exist, which is now.
    try:
        n_speared, offenders = count_speared_rings(cell, dims)
    except Exception as exc:                       # pragma: no cover
        n_speared, offenders = -1, []
        notes.append(f"Ring-spearing check could not run: {exc}")
    if n_speared > 0:
        notes.append(
            f"WARNING: {n_speared} bond(s) thread through a ring, e.g. atoms "
            f"{offenders[0]}. This is a topological defect — minimisation will "
            f"relax the strain and leave the two permanently interlocked. "
            f"Rebuild at lower density or with a different seed.")
    elif n_speared == 0:
        notes.append("Ring-spearing check: none found in the all-atom cell.")

    res = BackmapResult(molecule=cell, n_chains=len(per_chain),
                        atoms_per_chain=per_chain, worst_contact_a=worst,
                        n_speared=n_speared,
                        notes=notes)
    setattr(res, "backbone_flags", all_flags)
    setattr(res, "n_speared", n_speared)
    return res


def find_rings(molecule: Molecule, max_size: int = 8) -> List[List[int]]:
    """Smallest rings in the bond graph, by depth-limited search.

    For each bond, the shortest alternative path between its two ends is
    sought; if one exists within ``max_size`` steps, that path plus the bond
    is a ring. Duplicates are removed by frozen-set identity.
    """
    n = len(molecule.atoms)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i, j, _ in molecule.bonds:
        adj[i].append(j)
        adj[j].append(i)

    seen: set = set()
    rings: List[List[int]] = []
    for u, v, _ in molecule.bonds:
        # Shortest path u->v not using the direct bond.
        prev = {u: -1}
        frontier = [u]
        depth = 0
        found = False
        while frontier and depth < max_size and not found:
            nxt = []
            for k in frontier:
                for m in adj[k]:
                    if k == u and m == v:
                        continue          # the bond itself
                    if m == v:
                        prev[m] = k
                        found = True
                        break
                    if m not in prev:
                        prev[m] = k
                        nxt.append(m)
                if found:
                    break
            frontier = nxt
            depth += 1
        if not found:
            continue
        path = [v]
        while path[-1] != u:
            path.append(prev[path[-1]])
        key = frozenset(path)
        if len(path) <= max_size and key not in seen:
            seen.add(key)
            rings.append(path)
    return rings


def count_speared_rings(molecule: Molecule, dims: np.ndarray,
                        max_size: int = 8) -> Tuple[int, List[Tuple[int, int]]]:
    """Bonds that thread through a ring, in an all-atom cell.

    Why this lives here and not in the grower
    ----------------------------------------
    Spearing is a topological defect: a bond passing through a ring cannot be
    undone by energy minimisation, which will happily relax the strain and
    leave the two permanently interlocked. It is a classic silent failure of
    constructed cells.

    It cannot, however, be *detected* during growth in PAAF, because growth
    works at skeletal-bead resolution and a pendant ring is not represented at
    all until back-mapping puts it there. The predicate
    :func:`paaf.cell.grow._segment_intersects_disc` exists and is unit-tested,
    but at bead resolution it has nothing to act on. So the check is performed
    here instead, on the finished all-atom cell, and *reported* — by that point
    it is a diagnostic rather than a rejection criterion.

    Returns ``(count, offending_bonds)``.
    """
    from .grow import _segment_intersects_disc

    rings = find_rings(molecule, max_size=max_size)
    if not rings:
        return 0, []

    xyz = np.array([a.xyz for a in molecule.atoms], dtype=float)
    discs = []
    ring_atoms: List[set] = []
    for r in rings:
        pts = xyz[np.asarray(r, dtype=int)]
        # Unwrap the ring about its first atom before fitting a plane.
        ref = pts[0]
        d = pts - ref
        d -= dims * np.round(d / dims)
        pts = ref + d
        centre = pts.mean(axis=0)
        rel = pts - centre
        # Plane normal = smallest singular vector.
        _u, _s, vt = np.linalg.svd(rel, full_matrices=False)
        normal = vt[-1]
        radius = float(np.linalg.norm(rel, axis=1).mean())
        discs.append((centre, normal, radius))
        ring_atoms.append(set(r))

    offenders: List[Tuple[int, int]] = []
    for i, j, _ in molecule.bonds:
        p0, p1 = xyz[i], xyz[j]
        seg = p1 - p0
        seg -= dims * np.round(seg / dims)
        p1 = p0 + seg
        for k, (centre, normal, radius) in enumerate(discs):
            if i in ring_atoms[k] or j in ring_atoms[k]:
                continue                  # a ring's own bonds
            # Cheap reject before the exact test.
            mid = 0.5 * (p0 + p1)
            dm = mid - centre
            dm -= dims * np.round(dm / dims)
            if float(np.linalg.norm(dm)) > radius + 3.0:
                continue
            shift = np.round((mid - centre) / dims)
            if _segment_intersects_disc(p0 - dims * shift, p1 - dims * shift,
                                        centre, normal, radius):
                offenders.append((i, j))
                break
    return len(offenders), offenders


def unthread_rings(molecule: Molecule, backbone_flags: Sequence[bool],
                   dims: np.ndarray, *, n_angles: int = 72,
                   passes: int = 3) -> Tuple[int, int]:
    """Rotate threaded rings about their attachment bond until nothing
    passes through them. Returns ``(unthreaded, still_threaded)``.

    Why the clash-score rotation misses these
    -----------------------------------------
    :func:`rotate_side_groups` picks the rotation with the lowest CONTACT
    score. A bond threading cleanly through a ring's centre touches nothing —
    its atoms sit outside the contact cutoff on either side — so the score is
    zero and the rotation search is blind to it. Measured on a PE/PS cell:
    the score-driven rotation reduced threaded bonds from 29 to 6 by
    accident, and left 6 it could not see.

    This search tests the defect itself: for each ring that a foreign bond
    intersects, the ring (plus its hydrogens) is rotated about its own
    attachment bond — an exact rigid move — and an angle is chosen at which
    NO foreign bond intersects the ring disc. Rings whose intruder passes
    close to the rotation axis cannot be freed this way and are reported.

    Reseeding was tried first and lost to the arithmetic: with ~19,000 bonds
    and ~300 rings per cell, a few threads are near-certain at ANY seed and
    any workable density — eight seeds at two densities all threaded.
    """
    from .grow import _segment_intersects_disc

    xyz = np.array([a.xyz for a in molecule.atoms], dtype=float)
    n = len(xyz)
    flags = np.asarray(backbone_flags, dtype=bool)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i, j, _ in molecule.bonds:
        adj[i].append(j)
        adj[j].append(i)

    def _disc(ring_idx: np.ndarray):
        pts = xyz[ring_idx]
        ref = pts[0]
        d = pts - ref
        d -= dims * np.round(d / dims)
        pts = ref + d
        centre = pts.mean(axis=0)
        rel = pts - centre
        _u, _s, vt = np.linalg.svd(rel, full_matrices=False)
        return centre, vt[-1], float(np.linalg.norm(rel, axis=1).mean())

    # Bond endpoints as arrays, once — the intersection test per ring then
    # only walks a pre-filtered candidate list instead of all ~10,000 bonds.
    # (72 angles x 30 rings x every bond was ~23 million Python-level disc
    # tests and ran past any patience.)
    b_i = np.array([b[0] for b in molecule.bonds], dtype=int)
    b_j = np.array([b[1] for b in molecule.bonds], dtype=int)

    def _candidates(anchor: np.ndarray, reach: float, group: set):
        p0 = xyz[b_i]
        seg = xyz[b_j] - p0
        seg -= dims * np.round(seg / dims)
        mid = p0 + 0.5 * seg
        dm = mid - anchor
        dm -= dims * np.round(dm / dims)
        near = np.einsum("ij,ij->i", dm, dm) <= (reach + 3.0) ** 2
        out = []
        for k in np.where(near)[0]:
            i, j = int(b_i[k]), int(b_j[k])
            if i not in group and j not in group:
                out.append((i, j))
        return out

    def _intruders(ring_idx: np.ndarray, group: set, cand=None):
        centre, normal, radius = _disc(ring_idx)
        hits = []
        source = cand if cand is not None else _candidates(centre, radius,
                                                           group)
        for i, j in source:
            p0 = xyz[i]
            seg = xyz[j] - p0
            seg -= dims * np.round(seg / dims)
            p1 = p0 + seg
            mid = 0.5 * (p0 + p1)
            shift = np.round((mid - centre) / dims)
            if _segment_intersects_disc(p0 - dims * shift, p1 - dims * shift,
                                        centre, normal, radius):
                hits.append((i, j))
        return hits

    unthreaded = 0
    for _sweep in range(passes):
        rings = find_rings(molecule, max_size=8)
        threaded = []
        for r in rings:
            ridx = np.asarray(r, dtype=int)
            # The side group this ring belongs to: ring atoms plus anything
            # hanging off them that is not the backbone.
            group = set(int(x) for x in r)
            grew = True
            while grew:
                grew = False
                for k in list(group):
                    for m in adj[k]:
                        if m not in group and not flags[m]:
                            # only pull in non-ring hydrogens/substituents
                            if len(adj[m]) == 1:
                                group.add(m)
                                grew = True
            if _intruders(ridx, group):
                threaded.append((ridx, group))
        if not threaded:
            break

        for ridx, group in threaded:
            # Only PENDANT rings — exactly one bond out of the group — may be
            # rotated. An in-backbone ring (ENR's epoxide) has two: rotating
            # it about one attachment shears the other, and did — a backbone
            # bond measured at 3.09 A afterwards. Those rings are grown at
            # bead level where spearing IS checked, so they are not the ones
            # threading anyway.
            external = [(int(m), int(k))
                        for k in group for m in adj[int(k)]
                        if m not in group]
            if len(external) != 1:
                continue
            axis_pair = external[0]
            a, b = axis_pair
            axis = xyz[b] - xyz[a]
            axis -= dims * np.round(axis / dims)
            na = float(np.linalg.norm(axis))
            if na < 1e-9:
                continue
            axis = axis / na
            gidx = np.asarray(sorted(group), dtype=int)
            origin = xyz[a]
            local = xyz[gidx] - origin
            local -= dims * np.round(local / dims)
            saved = xyz[gidx].copy()
            # Every rotated position stays within this radius of the origin,
            # so one candidate list serves all angles.
            reach = float(np.linalg.norm(local, axis=1).max())
            orbit_cand = _candidates(origin, reach, group)

            freed = False
            for tdeg in np.linspace(0.0, 2.0 * np.pi, n_angles,
                                    endpoint=False)[1:]:
                c, s = np.cos(tdeg), np.sin(tdeg)
                rot = (local * c + np.cross(axis, local) * s
                       + np.outer(local @ axis, axis) * (1.0 - c))
                xyz[gidx] = origin + rot
                if not _intruders(ridx, group, cand=orbit_cand):
                    freed = True
                    break
            if freed:
                unthreaded += 1
            else:
                # Rotation about the attachment axis sweeps a cone; an
                # intruder passing close to that axis stays inside it at
                # every angle. The one rigid move left is REFLECTION through
                # the local backbone plane, which puts the whole side group
                # on the other side of the chain — a genuinely different
                # region of space. The backbone atom lies in the plane, so
                # every bond to it is preserved exactly. For a stereocentre
                # this flips the local configuration: one tacticity defect,
                # against a permanent interlock. Combined with a rotation
                # scan in the mirrored position.
                xyz[gidx] = saved
                bb_nbrs = [m for m in adj[a] if flags[m]]
                if len(bb_nbrs) >= 2:
                    e1 = xyz[bb_nbrs[0]] - xyz[a]
                    e1 -= dims * np.round(e1 / dims)
                    e2 = xyz[bb_nbrs[1]] - xyz[a]
                    e2 -= dims * np.round(e2 / dims)
                    pn = np.cross(e1, e2)
                    npn = float(np.linalg.norm(pn))
                    if npn > 1e-9:
                        pn = pn / npn
                        mirrored = local - 2.0 * np.outer(local @ pn, pn)
                        m_axis = mirrored[list(gidx).index(b)] \
                            if b in gidx else None
                        for tdeg in np.linspace(0.0, 2.0 * np.pi, n_angles,
                                                endpoint=False):
                            base = mirrored
                            if m_axis is not None:
                                ax2 = m_axis / max(
                                    float(np.linalg.norm(m_axis)), 1e-9)
                                c2, s2 = np.cos(tdeg), np.sin(tdeg)
                                base = (mirrored * c2
                                        + np.cross(ax2, mirrored) * s2
                                        + np.outer(mirrored @ ax2, ax2)
                                        * (1.0 - c2))
                            xyz[gidx] = origin + base
                            if not _intruders(ridx, group,
                                              cand=_candidates(
                                                  origin, reach, group)):
                                freed = True
                                break
                        if not freed:
                            xyz[gidx] = saved
                if freed:
                    unthreaded += 1

    for k, at in enumerate(molecule.atoms):
        at.xyz = xyz[k]
    # Recount what is left.
    still = 0
    rings = find_rings(molecule, max_size=8)
    for r in rings:
        ridx = np.asarray(r, dtype=int)
        group = set(int(x) for x in r)
        if _intruders(ridx, group):
            still += 1
    return unthreaded, still


def rotate_side_groups(molecule: Molecule, backbone_flags: Sequence[bool],
                       dims: np.ndarray, *, n_angles: int = 36,
                       min_group: int = 3,
                       contact: float = 3.0) -> int:
    """Relieve clashes by rotating each side group about its own bond.

    Why this is the right move for a ring
    -------------------------------------
    The per-atom push-off in :func:`relax_substituents` cannot help a phenyl
    group. A ring is geometrically rigid: any per-atom displacement that would
    move it out of a clash is immediately undone by the bond constraints, and
    the relaxation converges to a local limit with the ring still overlapping
    (measured for polystyrene: the closest contact stalls near 0.9 Å however
    many iterations are allowed).

    But a side group has one genuine degree of freedom that changes nothing
    else: rotation about the single bond joining it to the backbone. That
    rotation is *exact* — every bond length, every bond angle and every
    internal torsion of the group is preserved, because the whole group moves
    as a rigid body. It is precisely the side-group torsion that growth does
    not sample, recovered here where it costs almost nothing.

    Each rotatable group is scanned over ``n_angles`` uniform rotations and
    placed at whichever minimises a soft repulsive score against everything
    outside the group. Groups are treated greedily in order; this is a
    descent, not a global optimisation.

    Returns the number of groups that were rotated.
    """
    from .grow import _segment_intersects_disc

    xyz = np.array([a.xyz for a in molecule.atoms], dtype=float)
    n = len(xyz)
    flags = np.asarray(backbone_flags, dtype=bool)

    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i, j, _ in molecule.bonds:
        adj[i].append(j)
        adj[j].append(i)

    # Rings, so a rotation that would push one onto a bond can be VETOED.
    # The contact score alone cannot see that: a bond through a ring's
    # centre touches nothing and scores zero. Measured on a PE/PS cell, this
    # rotation used to RAISE the threaded-bond count from 19 to 32 — it was
    # un-clashing rings by threading them.
    _rings = find_rings(molecule, max_size=8)
    _ring_sets = [frozenset(int(x) for x in r) for r in _rings]
    b_i_all = np.array([bb[0] for bb in molecule.bonds], dtype=int)
    b_j_all = np.array([bb[1] for bb in molecule.bonds], dtype=int)

    def _would_thread(ring_local_idx: np.ndarray, group_set: set) -> bool:
        pts = xyz[ring_local_idx]
        ref = pts[0]
        d = pts - ref
        d -= dims * np.round(d / dims)
        pts = ref + d
        centre = pts.mean(axis=0)
        rel = pts - centre
        _u, _s, vt = np.linalg.svd(rel, full_matrices=False)
        normal = vt[-1]
        radius = float(np.linalg.norm(rel, axis=1).mean())
        p0 = xyz[b_i_all]
        seg = xyz[b_j_all] - p0
        seg -= dims * np.round(seg / dims)
        mid = p0 + 0.5 * seg
        dm = mid - centre
        dm -= dims * np.round(dm / dims)
        near = np.where(np.einsum("ij,ij->i", dm, dm)
                        <= (radius + 3.0) ** 2)[0]
        for k in near:
            i, j = int(b_i_all[k]), int(b_j_all[k])
            if i in group_set or j in group_set:
                continue
            q0 = xyz[i]
            sg = xyz[j] - q0
            sg -= dims * np.round(sg / dims)
            q1 = q0 + sg
            shift = np.round((0.5 * (q0 + q1) - centre) / dims)
            if _segment_intersects_disc(q0 - dims * shift, q1 - dims * shift,
                                        centre, normal, radius):
                return True
        return False

    rotated = 0
    for a in range(n):
        if not flags[a]:
            continue
        for b in adj[a]:
            if flags[b]:
                continue
            # The group is everything reachable from b without going back
            # through a. If that set reaches another backbone atom the bond
            # is not a terminal side link and must not be rotated.
            group: List[int] = []
            stack = [b]
            seen = {a, b}
            ok = True
            while stack:
                k = stack.pop()
                group.append(k)
                for m in adj[k]:
                    if m == a:
                        continue
                    if flags[m]:
                        ok = False
                        break
                    if m not in seen:
                        seen.add(m)
                        stack.append(m)
                if not ok:
                    break
            if not ok or len(group) < min_group:
                continue

            axis = xyz[b] - xyz[a]
            axis -= dims * np.round(axis / dims)
            na = float(np.linalg.norm(axis))
            if na < 1e-9:
                continue
            axis = axis / na

            gidx = np.asarray(group, dtype=int)
            outside = np.ones(n, dtype=bool)
            outside[gidx] = False
            outside[a] = False
            for m in adj[a]:
                outside[m] = False
            out_idx = np.where(outside)[0]
            if len(out_idx) == 0:
                continue

            origin = xyz[a]
            local = xyz[gidx] - origin
            local -= dims * np.round(local / dims)

            group_set = set(int(x) for x in gidx)
            group_rings = [np.asarray(sorted(rs), dtype=int)
                           for rs in _ring_sets if rs <= group_set]

            best_score, best_angle = None, 0.0
            for t in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False):
                c, s = np.cos(t), np.sin(t)
                # Rodrigues rotation of the whole group about the bond axis.
                rot = (local * c
                       + np.cross(axis, local) * s
                       + np.outer(local @ axis, axis) * (1.0 - c))
                pts = origin + rot
                d = pts[:, None, :] - xyz[None, out_idx, :]
                d -= dims * np.round(d / dims)
                r2 = np.einsum("ijk,ijk->ij", d, d)
                close = r2 < contact * contact
                if not close.any():
                    score = 0.0
                else:
                    r = np.sqrt(r2[close])
                    score = float(np.sum((contact - r) ** 2))
                if group_rings and (best_score is None or score < best_score
                                    or score == 0.0):
                    # Veto before accepting: place, test, restore.
                    saved_g = xyz[gidx].copy()
                    xyz[gidx] = pts
                    threaded = any(_would_thread(rl, group_set)
                                   for rl in group_rings)
                    xyz[gidx] = saved_g
                    if threaded:
                        continue
                if best_score is None or score < best_score:
                    best_score, best_angle = score, t
                if score == 0.0:
                    break

            if best_angle != 0.0:
                c, s = np.cos(best_angle), np.sin(best_angle)
                rot = (local * c
                       + np.cross(axis, local) * s
                       + np.outer(local @ axis, axis) * (1.0 - c))
                xyz[gidx] = origin + rot
                rotated += 1

    for k, at in enumerate(molecule.atoms):
        at.xyz = xyz[k]
    return rotated


def _restore_bonds(xyz: np.ndarray, bi: np.ndarray, bj: np.ndarray,
                   target_len: np.ndarray, fixed: np.ndarray,
                   dims: np.ndarray, sweeps: int = 24,
                   tol: float = 1e-4) -> None:
    """Restore every bond to its original length, in place (SHAKE-style).

    Iterative pairwise correction over the whole bond list. Each pass moves
    the two ends of a violated bond towards or away from one another along
    the bond vector, splitting the correction between them; a backbone atom
    is immobile and takes none of it, so its partner absorbs the whole
    correction.

    Unlike a parent-pointer projection this handles **cycles**, because a
    ring closure is just another entry in the bond list. Successive
    over-relaxation is not used — plain Gauss-Seidel converges quickly here
    since the displacements being corrected are small.
    """
    if len(bi) == 0:
        return
    for _ in range(sweeps):
        d = xyz[bi] - xyz[bj]
        d -= dims * np.round(d / dims)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        err = r - target_len
        if np.max(np.abs(err)) < tol:
            return
        bad = np.where(np.abs(err) > tol)[0]
        for k in bad:
            i, j = int(bi[k]), int(bj[k])
            fi, fj = bool(fixed[i]), bool(fixed[j])
            if fi and fj:
                continue                       # both pinned: nothing to do
            v = xyz[i] - xyz[j]
            v -= dims * np.round(v / dims)
            nv = float(np.linalg.norm(v))
            if nv < 1e-9:
                # A DIFFERENT direction for each collapsed bond.
                #
                # This used to be [1, 0, 0] for every one of them. A carbon
                # whose two hydrogens had both collapsed onto it therefore had
                # both restored along +x, to exactly the same point — measured
                # on a real PE/PS cell, 333 pairs of hydrogens on the same
                # carbon at 0.000 A separation. DL_FIELD perceives bonds from
                # geometry, read those zero-length H...H pairs as bonds, and
                # reported cyclopropyl rings and alkenes in a saturated blend
                # before stopping on an untypable atom.
                #
                # Any direction restores the bond length; they just must not
                # all be the same one. Derived from the atom indices so a
                # rebuild is reproducible.
                v = np.random.default_rng(
                    (i * 1000003 + j) & 0xFFFFFFFF).normal(size=3)
                nrm = float(np.linalg.norm(v))
                v = v / nrm if nrm > 1e-12 else np.array([1.0, 0.0, 0.0])
                nv = 1.0
            u = v / nv
            corr = (nv - target_len[k]) * u
            if fj:
                xyz[i] -= corr
            elif fi:
                xyz[j] += corr
            else:
                xyz[i] -= 0.5 * corr
                xyz[j] += 0.5 * corr


def relax_substituents(molecule: Molecule, backbone_flags: Sequence[bool],
                       dims: np.ndarray, *, target: float = 2.2,
                       iterations: int = 120,
                       step: float = 0.35) -> Tuple[float, float]:
    """Push overlapping side atoms apart while pinning the backbone.

    Why this is needed
    ------------------
    Growth places skeletal atoms with a hard-core guard of a couple of
    Ångström. That is the right guard for *beads*, but a hydrogen sits 1.09 Å
    off its carbon, so two backbone atoms at an acceptable separation can
    still leave their hydrogens nearly coincident. Handing a 0.3 Å contact to
    LAMMPS is how a minimisation explodes.

    What it does, and does not, change
    ----------------------------------
    Only non-backbone atoms move. The backbone is the part growth validated —
    its dimensions, its absence of spearing, its density — and none of that is
    put at risk here. After each displacement a constraint relaxation restores
    **every** bond to its original length, so bond lengths are preserved to
    within ``bond_tol``. Bond angles and side-group torsions do drift; that is
    the price, and it is the cheap part of the geometry for a real minimiser
    to fix afterwards.

    BUG HISTORY, kept as a warning. An earlier version pinned each mobile atom
    to a single parent and re-projected onto that one bond. A spanning tree
    cannot constrain a cycle, so the ring-closing bond of every pendant ring
    was left completely free: polystyrene phenyl rings were torn open to
    5.0 Å. Nothing caught it, because the only bond-length test used
    polyethylene, where every mobile atom is a hydrogen bonded directly to a
    pinned backbone carbon and a tree is therefore sufficient. Test the case
    with the cycle in it.

    This is a push-off, not a minimisation. It has no attractive term and no
    force field. It exists to make the structure safe to minimise properly.

    Returns ``(worst_before, worst_after)`` in Å.
    """
    xyz = np.array([a.xyz for a in molecule.atoms], dtype=float)
    n = len(xyz)
    flags = np.asarray(backbone_flags, dtype=bool)

    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i, j, _ in molecule.bonds:
        adj[i].append(j)
        adj[j].append(i)

    # Every bond gets a target length — including ring closures, which a
    # parent-pointer scheme structurally cannot represent.
    # Bonds AND 1-3 pairs, both held at their starting length.
    #
    # Holding only bonds fixes distances but leaves ANGLES free, and the two
    # hydrogens on a CH2 are a 1-3 pair — excluded from the repulsion below,
    # because 1-3 atoms are supposed to be close. So nothing kept them apart
    # while the restore was free to put both at 1.09 A from their carbon in
    # the same direction. Measured: 0 coincident atoms before this step, 367
    # after. Constraining bond lengths and 1-3 distances together is what
    # makes the local geometry rigid, which is what "the backbone is never
    # moved and side groups keep their shape" is supposed to mean.
    pair_i: List[int] = [int(b[0]) for b in molecule.bonds]
    pair_j: List[int] = [int(b[1]) for b in molecule.bonds]
    seen_pairs = {(min(a, b), max(a, b)) for a, b in zip(pair_i, pair_j)}
    # Only 1-3 pairs whose BOTH ends can move. A pair involving a pinned
    # backbone atom cannot collapse — the pinned end holds the geometry — and
    # constraining every 1-3 pair in a 10,000-atom cell makes the restore
    # sweep too slow to run. The mobile-mobile pairs are the two hydrogens on
    # a carbon, which is exactly the case that was collapsing.
    fixed_arr = flags
    for centre, nbrs in adj.items():
        nl = sorted(n for n in nbrs if not fixed_arr[n])
        for x in range(len(nl)):
            for y in range(x + 1, len(nl)):
                key = (nl[x], nl[y])
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    pair_i.append(key[0]); pair_j.append(key[1])
    bond_i = np.array(pair_i, dtype=int)
    bond_j = np.array(pair_j, dtype=int)
    if len(bond_i):
        bd = xyz[bond_i] - xyz[bond_j]
        bd -= dims * np.round(bd / dims)
        bond_len0 = np.sqrt(np.einsum("ij,ij->i", bd, bd))
    else:
        bond_len0 = np.zeros(0)

    # 1-2 pairs are excluded from repulsion outright — a bond is supposed
    # to be short. 1-3 pairs get a FLOOR instead of an exemption.
    #
    # They used to be exempt too, and that is how a hydrogen ended up 1.60 A
    # from the next backbone carbon: H-C(next) is a 1-3 pair, so no repulsion
    # applied, and the 1-3 length constraint skipped it because one end was
    # pinned. DL_FIELD bonds anything that close, saw a five-coordinate
    # carbon, and stopped ("Fail to decide type of unsaturated C atom").
    # The floor has to sit ABOVE DL_FIELD's bonding cutoff, or it protects
    # nothing: a first attempt used 1.55 A and DL_FIELD still bonded a pair
    # at 1.60. Real 1-3 separations start at 1.78 A (geminal H..H), so 1.72
    # clears the cutoff while leaving correct geometry untouched.
    excl: List[set] = []
    excl13: List[set] = []
    for i in range(n):
        bonded = set(adj[i])
        near13 = set()
        for j in adj[i]:
            near13 |= set(adj[j])
        near13 -= bonded
        near13.discard(i)
        excl.append(bonded)
        excl13.append(near13)
    target13 = min(1.72, target)

    before = _closest_nonbonded(xyz, molecule.bonds, dims)
    mobile = np.where(~flags)[0]
    if len(mobile) == 0:
        return before, before

    # Contacts found with a periodic KD-tree, not by scanning every atom.
    #
    # The previous loop built a neighbour grid and then ignored it: for each
    # mobile atom it computed the distance to ALL n atoms, inside a Python
    # loop, up to `iterations` times. On the 10,560-atom PE/PS cell that is
    # ~10^9 distance evaluations, and the step ran for so long it looked like
    # a hang — so it was switched off, and the cell went to DL_FIELD with its
    # side groups still overlapping. DL_FIELD perceives bonds from geometry,
    # read those contacts as bonds, and reported cyclopropyl rings and
    # alkenes in a saturated PE/PS blend before stopping on an untypable
    # `aliphatic` atom. The slow loop and the failed typing were one bug.
    #
    # cKDTree.query_pairs returns every pair closer than `target` in one
    # call, honouring the periodic box directly, so the cost is O(N log N)
    # in the pair search rather than O(N^2) in Python.
    from scipy.spatial import cKDTree

    mobile_set = set(int(i) for i in mobile)
    rng = np.random.default_rng(0xC0FFEE)
    for _ in range(iterations):
        # Wrap into the primary image: cKDTree's boxsize requires it, and a
        # coordinate that has drifted outside raises rather than wrapping.
        wrapped = xyz - np.floor(xyz / dims) * dims
        pairs = cKDTree(wrapped, boxsize=dims).query_pairs(
            r=target, output_type="ndarray")
        if len(pairs) == 0:
            break

        # Drop pairs that are meant to be close (bonded, 1-3) or that cannot
        # move (both atoms pinned to the backbone).
        keep = np.fromiter(
            ((int(a) in mobile_set or int(b) in mobile_set)
             and int(b) not in excl[int(a)]
             for a, b in pairs), dtype=bool, count=len(pairs))
        pairs = pairs[keep]
        if len(pairs) == 0:
            break

        ia, ib = pairs[:, 0], pairs[:, 1]
        is13 = np.fromiter((int(b) in excl13[int(a)]
                            for a, b in pairs), dtype=bool, count=len(pairs))
        pair_target = np.where(is13, target13, target)
        d = wrapped[ib] - wrapped[ia]
        d -= dims * np.round(d / dims)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))

        # Exactly coincident atoms need a direction invented for them.
        #
        # Measured on a real PE/PS cell: 333 pairs of hydrogens on the SAME
        # carbon sat at a separation of 0.000 A. Clamping r and dividing gives
        # d/r = 0, so the push is zero and they stay welded together forever —
        # and DL_FIELD, which perceives bonds from geometry, then reports
        # cyclopropyl rings and alkenes in a saturated blend. A random unit
        # vector is arbitrary but any direction separates them, after which
        # the normal term takes over.
        coincident = r < 1e-6
        if np.any(coincident):
            jitter = rng.normal(size=(int(coincident.sum()), 3))
            jitter /= np.linalg.norm(jitter, axis=1)[:, None]
            d[coincident] = jitter * 1e-3
            r[coincident] = 1e-3
        r = np.maximum(r, 1e-9)
        # Push each pair apart along its separation, share of the overlap
        # proportional to how far inside its own target it is. A 1-3 pair
        # already past its floor contributes nothing.
        mag = step * np.maximum(pair_target - r, 0.0) / target
        push = mag[:, None] * (d / r[:, None])

        disp = np.zeros_like(xyz)
        np.add.at(disp, ia, -push)
        np.add.at(disp, ib, +push)
        disp[flags] = 0.0            # backbone stays exactly where it was
        if not np.any(disp):
            break
        xyz += disp
        _restore_bonds(xyz, bond_i, bond_j, bond_len0, flags, dims)

    for k, a in enumerate(molecule.atoms):
        a.xyz = xyz[k]
    after = _closest_nonbonded(xyz, molecule.bonds, dims)
    return before, after


def _closest_nonbonded(xyz: np.ndarray, bonds: Sequence[Tuple[int, int, float]],
                       dims: np.ndarray, sample: int = 4000) -> float:
    """Closest 1-4-or-further contact under minimum image.

    Bonded and 1-3 pairs are excluded — they are *supposed* to be close. On
    large cells a random subsample of reference atoms is used, because the
    exact answer costs O(N²) and this is a diagnostic, not a criterion.
    """
    n = len(xyz)
    if n < 2:
        return float("inf")
    adj: Dict[int, set] = {i: set() for i in range(n)}
    for i, j, _ in bonds:
        adj[i].add(j)
        adj[j].add(i)
    excl: Dict[int, set] = {}
    for i in range(n):
        near = set(adj[i])
        for j in adj[i]:
            near |= adj[j]
        near.discard(i)
        excl[i] = near

    rng = np.random.default_rng(0)
    idx = (np.arange(n) if n <= sample
           else rng.choice(n, size=sample, replace=False))
    best = float("inf")
    for i in idx:
        d = xyz - xyz[i]
        d -= dims * np.round(d / dims)
        r2 = np.einsum("ij,ij->i", d, d)
        r2[i] = np.inf
        for j in excl[int(i)]:
            r2[j] = np.inf
        m = float(np.sqrt(r2.min()))
        if m < best:
            best = m
    return best
