"""Amorphous cell construction by chain growth (Theodorou–Suter).

References
----------
Theodorou, D. N. & Suter, U. W., *Macromolecules* **18** (1985) 1467 —
detailed molecular structure of a vinyl polymer glass; the bond-by-bond
growth method used here.
Meirovitch, H., *J. Chem. Phys.* **79** (1983) 502 — the scanning method.
Flory, P. J., *Statistical Mechanics of Chain Molecules*, 1969 — RIS.

Why growth rather than insertion
--------------------------------
The existing packers place whole rigid molecules at random positions and
orientations, rejecting overlaps. That is fine for solvents and short
oligomers and **cannot** reach melt density with real chains.

The reason is geometric, not a matter of tuning. A random coil pervades a
volume set by its radius of gyration, :math:`R_g \\propto N^{0.5}`, so the
pervaded volume grows as :math:`N^{1.5}` while the mass inside it grows only
as :math:`N`. At melt density that pervaded volume is mostly *other chains*.
Inserting a chain whole therefore requires finding a cavity shaped like an
entire coil, and the probability of that falls off roughly exponentially in
:math:`N`. Acceptance collapses somewhere around 20–50 repeat units.

Growing the chain bond by bond replaces one impossible question ("is there a
coil-shaped hole?") with many easy ones ("is there room for one more bead?").
Each step needs only local free volume.

The method, in the order it is applied per step
-----------------------------------------------
1. **Fixed bond lengths and angles.** Only the torsion varies. Nothing in the
   growth can distort the geometry, because each atom is placed by exact
   internal coordinates (see :func:`paaf.cell.ris.place_atom`).
2. **RIS weights.** Candidate torsions are the rotational isomeric states with
   their statistical weights, including the g\\ :sup:`+`\\ g\\ :sup:`-`
   correlation. This is what makes the coil the right *size*
   (:mod:`paaf.cell.ris`).
3. **Non-bonded bias.** Each candidate is further weighted by
   :math:`\\exp(-E^{nb}/k_BT)` against every atom already placed, under
   minimum image. This is what makes the coil avoid *other chains*.
4. **Scanning.** Optionally look ``scan_depth`` steps ahead and weight each
   candidate by the total surviving weight of its continuations, so the walk
   does not stroll into dead ends it cannot see.

   **This is off by default, and the reason is measured, not assumed.**
   Selecting a state in proportion to its look-ahead weight is a *biased*
   estimator — the Rosenbluth bias — because states that leave more room are
   over-chosen, and those are the extended ones. With ``scan_depth=1`` the
   characteristic ratio of dilute PE chains (2 x DP 40 at 0.10 g/cm3, 12
   seeds) rose from 5.66 at depth 0 to 7.40 at depth 1 and 7.52 at depth 2,
   against 6.78 for the same chain unperturbed. Correcting it requires
   carrying the Rosenbluth weight through and using it in every average,
   which is not implemented here. So: ``scan_depth=0`` when chain dimensions
   matter, ``scan_depth>0`` only when attrition would otherwise stop the cell
   being built at all.
5. **Spearing rejection.** Reject any step whose new bond passes through a
   ring. Threaded rings survive energy minimisation happily and are a classic
   silent failure of constructed cells.

Scope — read this before trusting a number
-------------------------------------------
Growth here operates on the **backbone**, with one bead per **skeletal atom**
carrying that atom's share of the repeat-unit mass, and an effective radius.
One growth step is therefore one skeletal bond, which is precisely what the
RIS model parameterises — so :math:`C_n` here is directly comparable with the
tabulated per-bond :math:`C_\\infty`. A repeat unit costs several steps:
:func:`backbone_atoms_per_unit` counts them from the SMILES (PE 2, PEO 3,
poly(butylene succinate) 10).

That is enough for the properties this module claims: chain dimensions
(:math:`C_\\infty`), absence of overlaps, absence of spearing, and bulk
density. It is **not** a full-atomistic construction: side-group torsions are
not sampled, and pendant atoms are not individually placed. Theodorou and
Suter do sample those for vinyl polymers. Extending to full atomistic detail
from arbitrary SMILES is a larger job and is *not* claimed here.

Materials Studio's Amorphous Cell is proprietary; Theodorou–Suter and
Meirovitch are the published basis. Anything beyond them in this module is
inference and is labelled where it occurs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..logging_utils import get_logger
from ..structure import Atom, Molecule
from .packing import (
    CancelToken, NeighbourGrid, PackFailed, ProgressFn, atomic_mass,
    check_cancel, emit, vdw_radius, wrap,
)
from .ris import (
    GAS_CONSTANT_KCAL, RISModel, build_backbone, characteristic_ratio,
    generic_ris, get_ris, place_atom, state_populations,
)

log = get_logger(__name__)

__all__ = ["GrowSpec", "ChainStats", "GrowthResult", "grow_amorphous_cell",
           "repeat_unit_mass", "molecular_formula", "backbone_atoms_per_unit",
           "count_attachment_points", "backbone_ring_atoms",
           "max_backbone_atoms_in_a_ring",
           "BackboneRingError"]

_N_AVOGADRO = 6.02214076e23


# =====================================================================
# Repeat-unit chemistry: mass and size from SMILES
# =====================================================================
def count_attachment_points(smiles: str) -> int:
    """How many ``[*]`` polymer connection points the SMILES declares."""
    from ..polymer_smiles import _WILDCARD
    return len(_WILDCARD.findall(smiles or ""))


def molecular_formula(smiles: str) -> Dict[str, int]:
    """Atom counts for one repeat unit, **including implicit hydrogens**.

    Implicit hydrogens matter enormously — ``CC`` written out is C2 **H6** —
    and a mass error here propagates straight into the cell density and into
    any weight-fraction calculation.

    There is a second, subtler correction that is easy to get wrong. Stripping
    the ``[*]`` markers leaves a *closed-shell* molecule: ``[*]CC[*]`` becomes
    ``CC``, and RDKit dutifully caps both carbons with hydrogen, giving ethane
    C2H6. But in a polymer those two carbons bond to the neighbouring repeat
    units, not to hydrogen. The repeat unit is C2H4.

    So one hydrogen is removed per attachment point. Without this every mass
    is high by 1.008 per connection — 2.016 g/mol for a linear unit, which is
    a 7% error for polyethylene and would quietly inflate every density the
    builder reports.
    """
    try:
        from rdkit import Chem
    except Exception as exc:                       # pragma: no cover
        raise RuntimeError(
            "RDKit is required to count atoms (including implicit hydrogens) "
            "from SMILES. Install with: conda install -c conda-forge rdkit"
        ) from exc

    from ..polymer_smiles import expand_if_polymer, is_polymer_smiles
    smi = smiles
    n_attach = 0
    if is_polymer_smiles(smi):
        n_attach = count_attachment_points(smi)
        smi = expand_if_polymer(smi, 1)            # one repeat unit
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    counts: Dict[str, int] = {}
    for a in mol.GetAtoms():
        counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1

    # Each attachment point is a bond to the next unit, not a C-H.
    if n_attach:
        have_h = counts.get("H", 0)
        if have_h < n_attach:
            raise ValueError(
                f"{smiles!r} declares {n_attach} attachment points but the "
                f"capped form has only {have_h} hydrogen(s) to remove.")
        counts["H"] = have_h - n_attach
        if counts["H"] == 0:
            del counts["H"]
    return counts


def repeat_unit_mass(smiles: str) -> float:
    """Molar mass of one repeat unit in g/mol, hydrogens included."""
    return sum(atomic_mass(el) * n
               for el, n in molecular_formula(smiles).items())


class BackboneRingError(ValueError):
    """The repeat unit has a ring *in* its backbone, which cannot be
    back-mapped onto an RIS-generated path. See :func:`backbone_ring_atoms`."""


def backbone_ring_atoms(smiles: str) -> int:
    """How many skeletal atoms of the repeat unit lie inside a ring.

    Zero for a normal chain. Non-zero for poly(ethylene terephthalate),
    polycarbonate, poly(butylene adipate-co-terephthalate) and anything else
    whose backbone runs *through* an aromatic or alicyclic ring rather than
    past it. A pendant ring — polystyrene's phenyl — gives zero, which is the
    distinction that matters.

    Why it matters
    --------------
    Growth is unaffected: a bead is a skeletal atom either way. Back-mapping
    is not. It reconstructs each unit by placing the backbone atoms at the
    grown positions and hanging rigid substituent groups off them, which
    assumes the substituents are *trees*. A backbone ring is a cycle that
    spans several backbone atoms at once, and its internal geometry is fixed
    by the ring, not by the RIS bond length and angle the growth used. The two
    geometries are incompatible: forcing the ring onto an RIS path stretches
    its closure bond.

    That failure used to be silent. Across the 124-polymer library it produced
    bonds of up to 11.9 Å in 24 cells — structures that look plausible in a
    viewer and are chemically destroyed. PAAF now refuses instead.
    """
    try:
        from rdkit import Chem
    except Exception:                              # pragma: no cover
        return 0
    if count_attachment_points(smiles) != 2:
        return 0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return 0
    try:
        ends = [mol.GetAtomWithIdx(s).GetNeighbors()[0].GetIdx() for s in stars]
        path = ([ends[0]] if ends[0] == ends[1]
                else list(Chem.GetShortestPath(mol, ends[0], ends[1])))
        ring_info = mol.GetRingInfo()
        return sum(1 for i in path if ring_info.NumAtomRings(i) > 0)
    except Exception:                              # pragma: no cover
        return 0


def max_backbone_atoms_in_a_ring(smiles: str) -> int:
    """The most skeletal atoms any single ring contains.

    This, not the total in :func:`backbone_ring_atoms`, is what decides
    whether a unit can be back-mapped. What matters is **how many backbone
    bonds a ring spans**, because those bonds' lengths and angles are fixed by
    the ring while growth places them at the RIS values.

    ``2`` — the ring spans exactly one backbone bond. An epoxide is the
        canonical case: two backbone carbons plus a bridging oxygen. The ring
        wants that C–C at about 1.47 Å and growth gives 1.53 Å, a 4%
        stretch — small enough that back-mapping is honest and minimisation
        will settle it.

    ``4 or more`` — the ring spans three or more backbone bonds, as in
        poly(ethylene terephthalate) or polycarbonate, where a benzene ring
        sits in the chain. A benzene's internal geometry cannot be reconciled
        with a path built at 1.53 Å and 112°, and forcing it produces a torn
        structure. These are refused.
    """
    try:
        from rdkit import Chem
    except Exception:                              # pragma: no cover
        return 0
    if count_attachment_points(smiles) != 2:
        return 0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return 0
    try:
        ends = [mol.GetAtomWithIdx(s).GetNeighbors()[0].GetIdx() for s in stars]
        path = set([ends[0]] if ends[0] == ends[1]
                   else Chem.GetShortestPath(mol, ends[0], ends[1]))
        worst = 0
        for ring in mol.GetRingInfo().AtomRings():
            worst = max(worst, len(path.intersection(ring)))
        return worst
    except Exception:                              # pragma: no cover
        return 0


def backbone_atoms_per_unit(smiles: str) -> int:
    """Skeletal atoms one repeat unit contributes to the backbone.

    This sets the *granularity of growth*, and getting it wrong is not a
    cosmetic error.

    The RIS model describes one **skeletal bond** at a time: bond length
    1.53 Å, bond angle 112°, three torsion states. So a growth step must be a
    skeletal bond. If instead one step is taken to be a whole repeat unit, the
    chain gets the right *mass* but the wrong *contour length* — a
    polyethylene unit ``[*]CC[*]`` spans two backbone bonds (~2.5 Å in trans),
    not one. Density would still come out correct, because density only knows
    about mass and volume, while every dimension derived from the chain path
    — :math:`R_g`, end-to-end distance, :math:`C_n` — would be too small by
    :math:`\\sqrt{\\text{atoms per unit}}`. For PEO that is √3; for the
    polyester ``[*]OCCCCOC(=O)CCC(=O)[*]`` it is √10.

    The count is the number of atoms on the shortest path between the two
    attachment points, which is exactly the number of skeletal bonds crossed
    in getting from one repeat unit to the next. Side groups are branches off
    that path and do not count::

        [*]CC[*]                    -> 2   (PE)
        [*]CC([*])c1ccccc1          -> 2   (PS; the phenyl is a branch)
        [*]OCC[*]                   -> 3   (PEO)
        [*]OCCCCOC(=O)CCC(=O)[*]    -> 10  (PBS)

    Returns 1 when the SMILES declares no attachment points or RDKit is
    unavailable, which degrades to the old behaviour rather than raising.
    """
    try:
        from rdkit import Chem
    except Exception:                              # pragma: no cover
        return 1
    if count_attachment_points(smiles) != 2:
        return 1
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 1
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return 1
    try:
        ends = []
        for s in stars:
            nbrs = mol.GetAtomWithIdx(s).GetNeighbors()
            if not nbrs:
                return 1
            ends.append(nbrs[0].GetIdx())
        if ends[0] == ends[1]:
            # Both attachment points on the same atom (e.g. a spiro link).
            return 1
        path = Chem.GetShortestPath(mol, ends[0], ends[1])
        return max(int(len(path)), 1)
    except Exception:                              # pragma: no cover
        return 1


def _effective_bead_radius(mass_amu: float) -> float:
    """Radius of a sphere holding one bead's mass at typical polymer density.

    ``mass_amu`` here is the mass of a single **skeletal atom's share** of the
    repeat unit, not the whole unit — see :func:`backbone_atoms_per_unit`.

    INFERENCE, not from the papers. A united-atom bead needs *some* size for
    the excluded-volume test. Taking a representative amorphous polymer
    density of 1.0 g/cm³, the volume per unit is ``M / (rho N_A)``; the radius
    follows from a sphere of that volume. It is deliberately a mild
    under-estimate of the true envelope, because the overlap criterion already
    applies a ``scale`` interpenetration factor on top.
    """
    volume_cm3 = mass_amu / (1.0 * _N_AVOGADRO)     # cm³ per unit at 1 g/cm³
    volume_a3 = volume_cm3 * 1.0e24
    return float((3.0 * volume_a3 / (4.0 * math.pi)) ** (1.0 / 3.0))


# =====================================================================
# Specs and results
# =====================================================================
@dataclass
class GrowSpec:
    """One polymer species to grow.

    Attributes
    ----------
    repeat_unit :
        Polymerisation SMILES with two attachment points, e.g.
        ``"[*]CC([*])c1ccccc1"``.
    n_chains :
        How many chains of this species.
    degree_of_polymerisation :
        Repeat units per chain (DP).
    name :
        Label used in progress messages and results.
    ris_key :
        Key into :data:`paaf.cell.ris.RIS_LIBRARY` (``"PE"``, ``"PS"``, …).
        When absent, the generic model is used and flagged in the result.
    mass_amu :
        Override for the repeat-unit mass. Normally computed from
        ``repeat_unit``; supplying it lets the grower run without RDKit.
    backbone_atoms :
        Skeletal atoms per repeat unit — the number of growth steps one unit
        costs. Normally derived from ``repeat_unit`` by
        :func:`backbone_atoms_per_unit`; override only to reproduce a
        specific granularity.
    monomers :
        For a COPOLYMER, the monomers and their fractions. When given it
        supersedes ``repeat_unit``: each chain gets its own sequence, and
        every bead takes its mass and radius from the monomer it actually
        belongs to rather than from a chain-wide average.
    arrangement :
        ``random`` (default), ``alternating``, ``block`` or ``exact``.
    sequence_seed :
        Separate from the growth seed, so the same monomer sequence can be
        grown into different conformations.
    """
    repeat_unit: str = ""
    n_chains: int = 1
    degree_of_polymerisation: int = 10
    name: str = ""
    ris_key: str = ""
    mass_amu: Optional[float] = None
    backbone_atoms: Optional[int] = None
    monomers: Optional[List[object]] = None      # List[sequence.Monomer]
    arrangement: str = "random"
    sequence_seed: int = 0

    # ------------------------------------------------------------------
    @property
    def is_copolymer(self) -> bool:
        return bool(self.monomers) and len(self.monomers) > 1

    def build_sequence(self, chain_index: int = 0):
        """The monomer ordering for one chain, or ``None`` for a homopolymer.

        Each chain gets a different sequence — that is what "random
        copolymer" means. Deriving the per-chain seed from
        ``sequence_seed`` keeps the whole cell reproducible.
        """
        if not self.monomers:
            return None
        from .sequence import build_sequence
        return build_sequence(
            self.monomers, int(self.degree_of_polymerisation),
            arrangement=self.arrangement,
            seed=self.sequence_seed + 7919 * chain_index)


@dataclass
class ChainStats:
    """Per-chain conformational measurements.

    ``n_beads`` is the number of skeletal atoms, which is also the number of
    entries this chain contributes to ``GrowthResult.molecule``.
    ``n_repeat_units`` is the DP. They differ by
    :func:`backbone_atoms_per_unit`.
    """
    species: str
    n_beads: int
    n_repeat_units: int
    r_end_to_end: float          # Å
    radius_of_gyration: float    # Å
    c_n: float                   # <R²>/(n l²), per SKELETAL BOND
    torsion_populations: Tuple[float, ...] = ()
    # The monomer ordering, for a copolymer chain. None for a homopolymer.
    # Back-mapping needs it to pick the right template per repeat unit.
    sequence: Optional[object] = None

    @property
    def n_units(self) -> int:
        """Deprecated alias for :attr:`n_beads`."""
        return self.n_beads


@dataclass
class GrowthResult:
    """Everything the caller needs to judge whether the cell is usable."""
    molecule: Molecule
    box: object
    chains: List[ChainStats] = field(default_factory=list)
    density_kg_m3: float = 0.0
    mean_c_n: float = 0.0
    torsion_populations: Tuple[float, ...] = ()
    n_rejected_overlap: int = 0
    n_rejected_spearing: int = 0
    n_restarts: int = 0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"grew {len(self.chains)} chain(s), "
            f"{len(self.molecule.atoms)} beads",
            f"density        : {self.density_kg_m3:.1f} kg/m³ "
            f"({self.density_kg_m3 / 1000:.3f} g/cm³)",
            f"mean C_n       : {self.mean_c_n:.2f}",
            f"torsion pops   : " + ", ".join(f"{p:.3f}" for p in self.torsion_populations),
            f"rejected       : {self.n_rejected_overlap} overlap, "
            f"{self.n_rejected_spearing} spearing, {self.n_restarts} restarts",
        ]
        lines.extend(f"note           : {n}" for n in self.notes)
        return "\n".join(lines)


# =====================================================================
# Spearing detection
# =====================================================================
def _segment_intersects_disc(p0: np.ndarray, p1: np.ndarray,
                             centre: np.ndarray, normal: np.ndarray,
                             radius: float) -> bool:
    """Does segment ``p0→p1`` cross the disc at ``centre``?

    A ring is approximated by the disc spanned by its atoms — centre, mean
    normal, and mean radius. Cheaper than a triangle fan and, for the
    near-planar rings that actually get speared (benzene, cyclohexane), it
    is the same test to within the ring's own pucker.

    Solve for the plane crossing, then check the crossing point lies inside
    the ring radius.
    """
    d = p1 - p0
    denom = float(normal @ d)
    if abs(denom) < 1e-9:
        return False                      # parallel to the ring plane
    t = float(normal @ (centre - p0)) / denom
    if t < 0.0 or t > 1.0:
        return False                      # crossing is outside the segment
    hit = p0 + t * d
    return bool(np.linalg.norm(hit - centre) <= radius)


@dataclass
class _Ring:
    centre: np.ndarray
    normal: np.ndarray
    radius: float


# =====================================================================
# Growth
# =====================================================================
def _rg(positions: np.ndarray) -> float:
    com = positions.mean(axis=0)
    d = positions - com
    return float(np.sqrt((d * d).sum() / len(positions)))


def _candidate_energy(grid: NeighbourGrid, point: np.ndarray, radius: float,
                      exclude: set) -> float:
    return grid.soft_energy(point, radius, exclude_slots=exclude)


def _lookahead_weight(model: RISModel, grid: NeighbourGrid, radius: float,
                      b: np.ndarray, c: np.ndarray, d: np.ndarray,
                      prev_state: int, depth: int, beta: float,
                      temperature: float, exclude: set,
                      dims: np.ndarray) -> float:
    """Meirovitch look-ahead: total surviving weight of continuations.

    Called with the frame ``(b, c, d)`` where ``d`` is the candidate just
    placed and ``prev_state`` is the torsion that placed it. Enumerates what
    could follow and returns the summed Boltzmann × RIS weight.

    Without look-ahead the walk commits to a torsion that may leave no
    acceptable continuation and the whole chain is lost — *attrition*. This
    weights each candidate by how much conformational room it actually leaves.

    BUG HISTORY, kept as a warning: an earlier version re-placed the candidate
    atom *inside* this function using the same state, so the current step's
    weight was counted twice. That did not fail loudly — it inflated the
    characteristic ratio by ~40% (C_n 9.0 against a 6.3 reference), which
    looks like a plausible number unless you check it against the isolated
    chain. Measure, do not assume.

    Cost is ``n_states ** depth``, so depth 2-3 is the practical range.
    """
    if depth <= 0:
        return 1.0
    l, theta = model.bond_length_a, model.bond_angle_deg
    angles = model.state_angles_deg
    u = model.u_matrix(temperature)
    total = 0.0
    for nxt in range(model.n_states):
        e = place_atom(b, c, d, l, theta, angles[nxt])
        ew = wrap(e[None], dims)[0]
        en = grid.soft_energy(ew, radius, exclude_slots=exclude)
        w = u[prev_state, nxt] * math.exp(-beta * min(en, 200.0))
        if w <= 0.0:
            continue
        total += w * _lookahead_weight(
            model, grid, radius, c, d, e, nxt, depth - 1, beta,
            temperature, exclude, dims)
    return total


def total_mass_amu_of(specs: Sequence[GrowSpec],
                      masses: Sequence[float]) -> float:
    return sum(m * int(s.n_chains) * int(s.degree_of_polymerisation)
               for m, s in zip(masses, specs))


def _growth_failure_message(label: str, chain_index: int, placed: int,
                            total_chains: int, dp: int, n_beads: int,
                            tolerance: float, scan_depth: int, scale: float,
                            max_restarts: int, density_g_cm3: float) -> str:
    """Explain a failed cell in terms of what the user can actually change.

    A bare "the box is too dense" is not much help when 27 of 30 chains went
    in fine. What matters is *how far it got* and which of the knobs is
    actually binding, so the message leads with the progress and puts the
    measured trade-off next to the suggestion.
    """
    lines = [
        f"Could not grow chain {chain_index + 1} of '{label}' after "
        f"{max_restarts} restarts.",
        "",
        f"{placed} of {total_chains} chains were placed before this one. "
        f"Chains go in one at a time, so the later ones grow into a box that "
        f"is already near its final density ({density_g_cm3:.3f} g/cm³) — "
        f"the hardest species is grown first for exactly this reason.",
        "",
        "In order of what usually helps:",
    ]
    if tolerance > 1.75:
        lines.append(
            f"  1. LOWER the overlap tolerance. It is {tolerance:.2f} Å, and "
            f"that is the binding constraint here. Measured on polyethylene "
            f"at melt density, 2.0 Å built 1 cell in 3 and 1.7 Å built 3 in "
            f"3. It is a guard against unrepairable overlap, not the physical "
            f"contact distance — the soft bias already handles that.")
    else:
        lines.append(
            f"  1. Lower the target density, or shorten the chains "
            f"(DP {dp} = {n_beads} skeletal atoms; every one of those steps "
            f"is a chance to dead-end).")
    lines.extend([
        f"  2. Raise the look-ahead depth from {scan_depth} to 1 or 2. It "
        f"defeats attrition, but it biases chain dimensions (Rosenbluth) — "
        f"use it to get a cell built, not to quote C_n from.",
        f"  3. Reduce the interpenetration scale (currently {scale}).",
        f"  4. Try a different random seed; failure is partly luck.",
    ])
    return "\n".join(lines)


def grow_amorphous_cell(
    specs: Sequence[GrowSpec],
    box,
    *,
    temperature: float = 300.0,
    ris: Optional[RISModel] = None,
    scan_depth: int = 0,
    tolerance: float = 1.7,
    scale: float = 0.55,
    seed: int = 12345,
    check_spearing: bool = True,
    max_restarts: int = 20,
    progress: Optional[ProgressFn] = None,
    cancel: Optional[CancelToken] = None,
) -> GrowthResult:
    """Grow chains into ``box`` bond by bond.

    Parameters mirror the existing packer where they mean the same thing
    (``tolerance``, ``scale``, ``seed``, ``progress``, ``cancel``) so a GUI
    can drive either.

    On ``tolerance``
    ----------------
    The hard minimum non-bonded approach, in Å. It is **not** the physical
    contact distance and should not be set to one. The soft LJ-like bias
    (:func:`paaf.cell.packing.soft_energy`) already discourages close
    approach; ``tolerance`` is only the guard against an overlap so severe
    that no subsequent minimisation could repair it. Setting it to the true
    united-atom contact distance double-counts the repulsion and stops the
    cell being built at all.

    The default was measured, not guessed. Growing 8 PE chains of DP 60 into
    a 29.7 Å box at 0.85 g/cm³ (the melt density at 413 K), unbiased
    (``scan_depth=0``), 3 seeds each::

        tolerance   cells built   mean C_n
          2.0 Å        1 / 3        5.68
          1.9 Å        2 / 3        6.69
          1.8 Å        2 / 3        7.60
          1.7 Å        3 / 3        5.74     <- default
          1.5 Å        3 / 3        5.27

    Raising ``max_restarts`` instead is a *worse* fix: at 2.0 Å going from 30
    to 150 restarts only reached 2/3 and pulled C_n down to 4.97, because
    surviving a restart selects for compact chains. That is a silent bias;
    losing a little hard-core radius is not.

    Raises
    ------
    PackFailed
        With a message naming what to change (lower the density, raise
        ``scan_depth``, shorten the chains).
    """
    if not specs:
        raise PackFailed("No species given — nothing to grow.")

    rng = np.random.default_rng(seed)
    beta = 1.0 / (GAS_CONSTANT_KCAL * temperature)
    dims = np.asarray(box.bounding_box(), dtype=float)
    volume_a3 = float(dims[0] * dims[1] * dims[2])

    notes: List[str] = []

    # ---- per-species chemistry -------------------------------------
    masses: List[float] = []          # per repeat unit
    bead_masses: List[float] = []     # per skeletal atom
    bpus: List[int] = []              # skeletal atoms per repeat unit
    radii: List[float] = []
    models: List[RISModel] = []
    for sp in specs:
        if sp.is_copolymer:
            # A copolymer has no single repeat-unit mass or granularity: both
            # depend on the sequence. Averages are used only for reporting and
            # for the difficulty ordering; every BEAD gets its own mass and
            # radius from the monomer it actually belongs to.
            for mono in sp.monomers:
                if not mono.mass_amu:
                    mono.resolve()
            weights = [mo.fraction for mo in sp.monomers]
            wsum = sum(weights) or 1.0
            m = sum(mo.mass_amu * w for mo, w in zip(sp.monomers, weights)) / wsum
            bpu = max(1, int(round(
                sum(mo.backbone_atoms * w
                    for mo, w in zip(sp.monomers, weights)) / wsum)))
        else:
            if sp.mass_amu is not None:
                m = float(sp.mass_amu)
            else:
                try:
                    m = repeat_unit_mass(sp.repeat_unit)
                except Exception as exc:
                    raise PackFailed(
                        f"Could not determine the repeat-unit mass for "
                        f"'{sp.name or sp.repeat_unit}': {exc}. Provide "
                        f"GrowSpec.mass_amu to bypass SMILES parsing.") from exc
            # One growth step is one SKELETAL BOND, because that is what the
            # RIS model parameterises. A repeat unit is worth several of them.
            if sp.backbone_atoms is not None:
                bpu = max(int(sp.backbone_atoms), 1)
            else:
                bpu = backbone_atoms_per_unit(sp.repeat_unit)
        masses.append(m)
        bpus.append(bpu)
        bead_masses.append(m / bpu)
        radii.append(_effective_bead_radius(m / bpu))

        model = ris or (get_ris(sp.ris_key) if sp.ris_key else generic_ris())
        if model.is_generic:
            notes.append(
                f"{sp.name or sp.repeat_unit}: GENERIC RIS parameters — chain "
                f"dimensions are qualitative, not a quantitative prediction.")
        models.append(model)

    total_chains = sum(int(s.n_chains) for s in specs)
    total_beads = sum(int(s.n_chains) * int(s.degree_of_polymerisation) * bpus[i]
                      for i, s in enumerate(specs))
    if total_beads == 0:
        raise PackFailed("Every species has zero chains or zero DP.")

    # ---- growth order: hardest species first ------------------------
    #
    # Chains go in one at a time, so whichever species is grown LAST faces a
    # box that is already at full density. Taking the species in the order the
    # caller listed them therefore makes success depend on an arbitrary
    # choice: a 70/30 PE/PS blend fails on the very first PS chain, because 27
    # PE chains have already filled the box and a polystyrene bead is
    # 2.74 Å against polyethylene's 1.77.
    #
    # So the order is chosen, not inherited. Difficulty is (beads per chain) x
    # (bead volume): long chains are harder because every one of their steps
    # can dead-end, and fat beads are harder because they exclude more of what
    # is left. The hardest species claims the empty box; the easiest fills the
    # gaps, which is what it is good at.
    order = sorted(
        range(len(specs)),
        key=lambda i: (int(specs[i].degree_of_polymerisation) * bpus[i]
                       * radii[i] ** 3),
        reverse=True)
    if len(specs) > 1 and order != list(range(len(specs))):
        names = " -> ".join(
            specs[i].name or specs[i].repeat_unit or f"species{i + 1}"
            for i in order)
        notes.append(
            f"Growth order was chosen by difficulty (chain length x bead "
            f"volume), not by the order given: {names}. The species grown "
            f"last has to fit into an already-full box, so the hardest one "
            f"goes first.")

    # Cutoff must cover the largest contact distance we ever test.
    max_r = max(radii)
    cutoff = max(tolerance, max_r * 2.0) + 1.0
    grid = NeighbourGrid(dims, cutoff, capacity=max(total_beads * 2, 64))

    # Ring list for the spearing test. It stays EMPTY at skeletal-bead
    # resolution, and that is not an oversight: a pendant ring is not
    # represented until back-mapping places it, so during growth there is
    # literally nothing to spear. The predicate is real and unit-tested, and
    # the check is performed for real on the finished all-atom cell by
    # paaf.cell.backmap.count_speared_rings. Keeping the hook here means a
    # future full-atomistic grower can populate it without restructuring.
    rings: List[_Ring] = []
    # Grown chains are held per species, because they are GROWN in difficulty
    # order but must be ASSEMBLED in the caller's order — back-mapping and
    # export both walk `specs` and slice the atom list, so a permuted atom
    # order would silently pair polystyrene beads with a polyethylene
    # template. Reordering the growth must stay invisible from outside.
    grown_by_species: List[List[tuple]] = [[] for _ in specs]
    n_overlap = 0
    n_spear = 0
    n_restart = 0
    placed = 0

    for si in order:
        sp = specs[si]
        model = models[si]
        radius = radii[si]
        dp = int(sp.degree_of_polymerisation)
        bpu = bpus[si]
        n_beads = dp * bpu
        label = sp.name or sp.repeat_unit or f"species{si + 1}"

        for ci in range(int(sp.n_chains)):
            check_cancel(cancel)

            # A copolymer chain gets its OWN sequence — that is what makes it
            # random rather than a repeating pattern — and every bead's mass
            # and radius follow the monomer it came from, not a chain average.
            seq = sp.build_sequence(ci)
            if seq is not None:
                n_beads = seq.n_beads
                bead_of = seq.bead_monomer_index()
                bead_radii = np.array(
                    [_effective_bead_radius(
                        seq.monomers[u].mass_amu
                        / max(seq.monomers[u].backbone_atoms, 1))
                     for u in bead_of])
                chain_mass = seq.chain_mass
            else:
                bead_radii = np.full(n_beads, radius)
                chain_mass = masses[si] * dp

            grown = None
            for attempt in range(max_restarts):
                out = _grow_one_chain(
                    model, grid, dims, bead_radii, n_beads, beta, temperature,
                    rng, tolerance, scale, scan_depth, check_spearing, rings)
                if out is not None:
                    grown, rej_o, rej_s, states = out
                    n_overlap += rej_o
                    n_spear += rej_s
                    break
                n_restart += 1
            if grown is None:
                raise PackFailed(_growth_failure_message(
                    label, ci, placed, total_chains, dp, n_beads,
                    tolerance, scan_depth, scale, max_restarts,
                    density_g_cm3=(total_mass_amu_of(specs, masses)
                                   / _N_AVOGADRO) / (volume_a3 * 1e-24)))

            wrapped = wrap(grown, dims)
            grid.add(wrapped, bead_radii[:len(wrapped)], mol_id=placed)
            # Statistics use the UNWRAPPED coordinates — wrapping would break
            # the end-to-end vector across the periodic boundary.
            stats = ChainStats(
                species=label,
                n_beads=len(grown),
                n_repeat_units=dp,
                sequence=seq,
                r_end_to_end=float(np.linalg.norm(grown[-1] - grown[0])),
                radius_of_gyration=_rg(grown),
                c_n=characteristic_ratio(grown, model.bond_length_a),
                torsion_populations=tuple(state_populations(
                    states, model.n_states)),
            )
            grown_by_species[si].append((wrapped, stats, label, seq,
                                         chain_mass))
            placed += 1
            emit(progress, placed=placed, total=total_chains,
                 attempts=n_overlap + n_spear, current_species=label,
                 message=f"grew {label} chain {ci + 1}/{sp.n_chains}",
                 fraction=placed / max(total_chains, 1))

    # ---- assemble in the caller's species order ---------------------
    all_atoms: List[Atom] = []
    all_bonds: List[Tuple[int, int, float]] = []
    chains: List[ChainStats] = []
    grown_mass = 0.0
    for si in range(len(specs)):
        for wrapped, stats, label, seq, chain_mass in grown_by_species[si]:
            grown_mass += chain_mass
            base = len(all_atoms)
            for k, p in enumerate(wrapped):
                all_atoms.append(Atom(index=base + k, element="C",
                                      xyz=p, name=f"{label[:3]}{base + k + 1}"))
            for k in range(len(wrapped) - 1):
                all_bonds.append((base + k, base + k + 1, 1.0))
            chains.append(stats)

    mol = Molecule(atoms=all_atoms, bonds=all_bonds, name="amorphous_cell")
    try:
        setattr(mol, "cell", box)
    except Exception:
        pass

    # Summed over the chains ACTUALLY grown, because a copolymer chain's mass
    # depends on the sequence that chain happened to get.
    total_mass_amu = grown_mass
    density = (total_mass_amu / _N_AVOGADRO) / (volume_a3 * 1.0e-24) * 1000.0

    mean_c = float(np.mean([c.c_n for c in chains])) if chains else 0.0
    pops = np.zeros(models[0].n_states)
    for c in chains:
        if c.torsion_populations:
            pops += np.asarray(c.torsion_populations)
    if chains:
        pops /= len(chains)

    gran = ", ".join(
        f"{sp.name or sp.repeat_unit} {bpus[i]}" for i, sp in enumerate(specs))
    notes.append(
        "Growth is backbone-level: one bead per SKELETAL ATOM, carrying that "
        "atom's share of the repeat-unit mass, so the contour length and "
        f"hence R_g and C_n are per backbone bond. Skeletal atoms per repeat "
        f"unit — {gran}. Side groups are not placed individually and their "
        "torsions are not sampled.")
    notes.append(
        "Chains are grown one after another into a box that starts empty, so "
        "later chains see free volume that is more open and more connected "
        "than a real melt's. The non-bonded weight exp(-E/kT) then favours "
        "the extended conformations that fit it. Measured for PE at "
        "0.85 g/cm3, DP 60: C_n 8.8 in the cell against 7.3 for the same "
        "chain grown in isolation, about +20%. Real melts screen excluded "
        "volume and sit near the unperturbed dimensions, so treat a cell's "
        "C_n as an upper bound and re-measure after equilibration.")
    if scan_depth > 0:
        notes.append(
            f"scan_depth={scan_depth}: look-ahead is ON. It defeats attrition "
            f"and lets denser cells be built, but selecting a torsion in "
            f"proportion to its look-ahead weight is a BIASED estimator "
            f"(Rosenbluth bias) — it over-samples extended conformations. "
            f"Measured on dilute PE: C_n 5.66 (depth 0) -> 7.40 (depth 1) "
            f"-> 7.52 (depth 2), against 6.78 for the unbiased "
            f"chain. Use scan_depth=0 when chain DIMENSIONS matter; use "
            f"scan_depth>0 when you need the cell built at all, and do not "
            f"quote C_n from it without a Rosenbluth-weight correction.")

    return GrowthResult(
        molecule=mol, box=box, chains=chains,
        density_kg_m3=density, mean_c_n=mean_c,
        torsion_populations=tuple(pops),
        n_rejected_overlap=n_overlap, n_rejected_spearing=n_spear,
        n_restarts=n_restart, notes=notes,
    )


def _grow_one_chain(model: RISModel, grid: NeighbourGrid, dims: np.ndarray,
                    radii: np.ndarray, n_beads: int, beta: float,
                    temperature: float,
                    rng: np.random.Generator, tolerance: float, scale: float,
                    scan_depth: int, check_spearing: bool,
                    rings: List[_Ring]):
    """Grow one chain of ``n_beads`` skeletal atoms.

    ``radii`` is per bead, not a single value: in a copolymer each bead takes
    its size from the monomer it belongs to, and an isoprene bead is not the
    same size as an epoxidised one.

    Returns ``(positions, n_overlap, n_spear, states)``.

    ``None`` means the chain died — the caller restarts it from a fresh seed
    position. Restarting a chain is cheap compared with a failed cell.
    """
    l, theta = model.bond_length_a, model.bond_angle_deg
    angles = np.asarray(model.state_angles_deg, dtype=float)
    n_states = model.n_states
    max_local_tries = 40

    # --- seed the first three beads somewhere with room
    start = None
    for _ in range(200):
        p0 = rng.random(3) * dims
        if not grid.overlaps(p0[None], radii[:1], tolerance, scale):
            start = p0
            break
    if start is None:
        return None

    pos = [start]
    # Second bead: random direction at the fixed bond length.
    for _ in range(max_local_tries):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        p1 = pos[0] + l * v
        if not grid.overlaps(wrap(p1[None], dims), radii[1:2],
                             tolerance, scale):
            pos.append(p1)
            break
    if len(pos) < 2:
        return None

    # Third bead: fixed bond angle, random azimuth.
    for _ in range(max_local_tries):
        ref = pos[0] + rng.normal(size=3)
        p2 = place_atom(ref, pos[0], pos[1], l, theta,
                        float(rng.uniform(-180, 180)))
        if not grid.overlaps(wrap(p2[None], dims), radii[2:3],
                             tolerance, scale):
            pos.append(p2)
            break
    if len(pos) < 3:
        return None

    own_slots: set = set()      # this chain is not yet in the grid, so empty
    states: List[int] = []
    n_overlap = 0
    n_spear = 0
    u = model.u_matrix(temperature)

    while len(pos) < n_beads:
        k = len(pos)                      # index of the bead being placed
        a, b, c = pos[-3], pos[-2], pos[-1]
        prev_state = states[-1] if states else 0

        # --- weight every rotational isomeric state
        weights = np.zeros(n_states)
        cands: List[Optional[np.ndarray]] = [None] * n_states
        for s in range(n_states):
            d = place_atom(a, b, c, l, theta, angles[s])
            dw = wrap(d[None], dims)[0]

            # Hard rejection first — cheaper than the energy, and a hard
            # overlap must never be reachable however favourable the RIS
            # weight is.
            if grid.overlaps(dw[None], radii[k:k + 1], tolerance, scale):
                n_overlap += 1
                continue
            # Self-avoidance within the chain: skip the three most recent
            # beads — the 1-2, 1-3 AND 1-4 partners — and test the rest.
            # The 1-4 pair is excluded because its distance is set by the
            # torsion being chosen, and a gauche state legitimately brings it
            # inside the tolerance.
            if len(pos) > 3:
                tail = np.asarray(pos[:-3])
                delta = tail - d
                delta -= dims * np.round(delta / dims)
                if np.any(np.einsum("ij,ij->i", delta, delta)
                          < (tolerance * tolerance)):
                    n_overlap += 1
                    continue
            if check_spearing and rings:
                if any(_segment_intersects_disc(c, d, r.centre, r.normal,
                                                r.radius) for r in rings):
                    n_spear += 1
                    continue

            cands[s] = d
            e = grid.soft_energy(dw, float(radii[k]),
                                 exclude_slots=own_slots)
            boltz = math.exp(-beta * min(e, 200.0))
            if scan_depth > 0:
                # Look ahead from the frame this candidate CREATES. The
                # candidate's own weight is applied once, below.
                boltz *= _lookahead_weight(
                    model, grid, float(radii[k]), b, c, d, s, scan_depth,
                    beta,
                    temperature, own_slots, dims)
            # P(phi) ∝ w_RIS · exp(-E_nb / kT)   — Theodorou–Suter eq. for the
            # conditional torsion probability.
            weights[s] = u[prev_state, s] * boltz

        total = weights.sum()
        if total <= 0.0:
            return None                      # dead end: caller restarts

        s = int(rng.choice(n_states, p=weights / total))
        pos.append(cands[s])
        states.append(s)

    return np.asarray(pos), n_overlap, n_spear, np.asarray(states, dtype=int)
