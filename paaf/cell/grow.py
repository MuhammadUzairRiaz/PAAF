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
   matter. Attrition no longer needs it — dead ends are backtracked (see
   :func:`grow_amorphous_cell`, "It never fails").
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

    from ..polymer_smiles import is_polymer_smiles
    smi = smiles
    n_attach = 0
    if is_polymer_smiles(smi):
        n_attach = count_attachment_points(smi)
        # Cap the attachment points with EXPLICIT hydrogens rather than
        # stripping the [*] markers. Stripping leaves any bracketed
        # neighbour ([Si], [N+], ...) still bracketed, so RDKit adds no
        # implicit hydrogens to it — and the attachment correction below
        # then subtracts H that were never counted. PDMS-type units came
        # out 2 H light (a 3.6% mass error) exactly this way.
        smi = smi.replace("[*]", "[H]")
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
    #: Dead ends escaped by retracting and regrowing a few beads.
    n_backtracks: int = 0
    #: Steps where every state was blocked even after backtracking and a
    #: fresh start, so the least-crowded one was taken. Relief then cleared
    #: the contacts; ``n_contacts`` says whether it fully could.
    n_forced: int = 0
    #: Non-bonded pairs still inside the overlap criterion in the returned
    #: cell. Zero unless the request was physically impossible to meet.
    n_contacts: int = 0
    min_nonbonded_distance: float = float("inf")
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
            f"dead ends      : {self.n_backtracks} backtracked, "
            f"{self.n_forced} forced",
            f"closest pair   : {self.min_nonbonded_distance:.2f} Å, "
            f"{self.n_contacts} contact(s) inside tolerance",
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
                      temperature: float, mol_id: int,
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
        en = grid.soft_energy(ew, radius, exclude_mol=mol_id)
        w = u[prev_state, nxt] * math.exp(-beta * min(en, 200.0))
        if w <= 0.0:
            continue
        total += w * _lookahead_weight(
            model, grid, radius, c, d, e, nxt, depth - 1, beta,
            temperature, mol_id, dims)
    return total


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
    max_restarts: int = 4,
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

    (Measured before dead ends were handled by recoil; the table is why the
    default is 1.7, not a statement about how often a build now fails.)

    It never fails
    --------------
    A build always returns a cell, whatever the density, chain count or
    tolerance. Three layers, each only reached when the one before runs out:

    1. **Recoil** inside a chain — a dead end retracts a few beads and
       regrows them (:func:`_grow_one_chain`).
    2. **Fresh start** — ``max_restarts`` new seeds for that chain, each
       placed in the roomiest of several random points.
    3. **Forced placement, then relief** — the last attempt takes the
       least-crowded state at a dead end instead of giving up, and
       :func:`paaf.cell.contact_relief.relieve_contacts` pushes the resulting
       contacts apart with bonds and angles held, then projects them back to
       their exact values.

    The result says which layers were needed (``n_backtracks``,
    ``n_restarts``, ``n_forced``) and how clean the cell is
    (``n_contacts``, ``min_nonbonded_distance``). A request that is
    physically impossible — beads that cannot fit at the tolerance at that
    density — still returns, with the remaining contacts counted and a note
    saying so, rather than a cell that silently is not what was asked.

    Raises
    ------
    PackFailed
        Only for input that describes nothing to build.
    PackCancelled
        When ``cancel`` is set.
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
    # One neighbour query per growth step covers every trial position, and
    # those lie a bond length from the tip — so a cell must span the largest
    # contact distance PLUS the longest bond (see _probe).
    max_r = max(radii)
    if any(sp.is_copolymer for sp in specs):
        max_r = max([max_r] + [
            _effective_bead_radius(mo.mass_amu / max(mo.backbone_atoms, 1))
            for sp in specs if sp.is_copolymer for mo in sp.monomers])
    max_bond = max(m.bond_length_a for m in models)
    cutoff = max(tolerance, max_r * max(scale, 1.0)) + max_bond + 0.25
    grid = NeighbourGrid(dims, cutoff, capacity=max(total_beads + 64, 64))

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
    n_backtrack = 0
    n_forced = 0
    n_forced_chains = 0
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

            # Strict attempts first; the last one may force a step rather
            # than fail, and relief below clears what it forced.
            grown = None
            n_attempts = max(int(max_restarts), 0) + 1
            for attempt in range(n_attempts):
                grown = _grow_one_chain(
                    model, grid, dims, bead_radii, n_beads, beta, temperature,
                    rng, tolerance, scale, scan_depth, check_spearing, rings,
                    mol_id=placed, force=attempt == n_attempts - 1,
                    cancel=cancel)
                if grown is not None:
                    break
                n_restart += 1
            n_overlap += grown.n_overlap
            n_spear += grown.n_spear
            n_backtrack += grown.n_backtracks
            if grown.n_forced:
                n_forced += grown.n_forced
                n_forced_chains += 1

            grown_by_species[si].append(
                [grown.positions, bead_radii[:len(grown.positions)], label,
                 seq, chain_mass, grown.states, model, dp])
            placed += 1
            emit(progress, placed=placed, total=total_chains,
                 attempts=n_overlap + n_spear, current_species=label,
                 message=f"grew {label} chain {ci + 1}/{sp.n_chains}",
                 fraction=placed / max(total_chains, 1))

    # ---- assemble in the caller's species order ---------------------
    records = [r for si in range(len(specs)) for r in grown_by_species[si]]
    from .contact_relief import ChainTopology, relieve_contacts
    topo = ChainTopology.from_chains(
        [len(r[0]) for r in records], [r[1] for r in records],
        [r[6].bond_length_a for r in records],
        [r[6].bond_angle_deg for r in records])
    flat = wrap(np.concatenate([r[0] for r in records]), dims)
    if n_forced:
        emit(progress, placed=placed, total=total_chains,
             attempts=n_overlap + n_spear, current_species="",
             message=f"relieving {n_forced} crowded step(s) "
                     f"in {n_forced_chains} chain(s) …",
             fraction=1.0)
    flat, contacts = relieve_contacts(flat, dims, topo, tolerance=tolerance,
                                      scale=scale, cancel=cancel)

    all_atoms: List[Atom] = []
    all_bonds: List[Tuple[int, int, float]] = []
    chains: List[ChainStats] = []
    grown_mass = 0.0
    base = 0
    for (unwrapped, _r, label, seq, chain_mass, states, model, dp) in records:
        n = len(unwrapped)
        wrapped = flat[base:base + n]
        if contacts.iterations:
            # Relief moved beads: rebuild the continuous chain from the moved
            # positions so the statistics describe the cell actually written.
            steps = np.diff(wrapped, axis=0)
            steps -= dims * np.round(steps / dims)
            unwrapped = np.vstack([wrapped[:1],
                                   wrapped[0] + np.cumsum(steps, axis=0)])
        grown_mass += chain_mass
        for k, p in enumerate(wrapped):
            all_atoms.append(Atom(index=base + k, element="C",
                                  xyz=p, name=f"{label[:3]}{base + k + 1}"))
        for k in range(n - 1):
            all_bonds.append((base + k, base + k + 1, 1.0))
        # Statistics use the UNWRAPPED coordinates — wrapping would break
        # the end-to-end vector across the periodic boundary.
        chains.append(ChainStats(
            species=label,
            n_beads=n,
            n_repeat_units=dp,
            sequence=seq,
            r_end_to_end=float(np.linalg.norm(unwrapped[-1] - unwrapped[0])),
            radius_of_gyration=_rg(unwrapped),
            c_n=characteristic_ratio(unwrapped, model.bond_length_a),
            torsion_populations=tuple(state_populations(
                states, model.n_states)),
        ))
        base += n

    mol = Molecule(atoms=all_atoms, bonds=all_bonds, name="amorphous_cell")
    try:
        setattr(mol, "cell", box)
    except Exception:
        log.debug("grow_amorphous_cell: ignored error", exc_info=True)

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
    if n_forced:
        notes.append(
            f"{n_forced} growth step(s) in {n_forced_chains} chain(s) had no "
            f"free rotational state even after backtracking, so the "
            f"least-crowded one was taken and the contacts were then pushed "
            f"apart with bond lengths and angles held "
            f"({contacts.iterations} relief iterations). Those chains' "
            f"torsions were adjusted by the relief and are slightly off the "
            f"RIS lattice; the rest of the cell is untouched.")
    if contacts.n_contacts:
        notes.append(
            f"{contacts.n_contacts} non-bonded pair(s) remain closer than the "
            f"overlap criterion (closest {contacts.min_distance:.2f} Å, "
            f"tolerance {tolerance:.2f} Å). The beads do not fit at this "
            f"density and tolerance — the cell was built anyway so it can be "
            f"relaxed, but lower 'Build at' or the tolerance for a clean "
            f"construction.")
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
        n_restarts=n_restart, n_backtracks=n_backtrack, n_forced=n_forced,
        n_contacts=contacts.n_contacts,
        min_nonbonded_distance=contacts.min_distance, notes=notes,
    )


def _probe(grid: NeighbourGrid, dims: np.ndarray, centre: np.ndarray,
           points: np.ndarray, radius: float, tolerance: float, scale: float,
           mol_id: int, n_recent: int):
    """Clash test, soft energy and clearance for several trial points at once.

    Every trial position for one growth step lies within one bond length of
    the chain tip, so a single neighbour query around ``centre`` (the wrapped
    tip) covers all of them — one query per step instead of two per state.
    The grid cutoff is sized for this in :func:`grow_amorphous_cell`.

    Beads of the chain being grown (``mol_id``) are in the grid too. Its
    ``n_recent`` newest beads — the 1-2, 1-3 and 1-4 partners of the bead
    being placed — are skipped; the rest are tested against ``tolerance``
    alone and carry no soft energy, which is exactly how the self-avoidance
    test worked before the chain was put into the grid.

    Returns ``(hard, energy, clearance)``, one entry per point. ``clearance``
    is min(r / criterion): below 1 is a hard overlap, and the largest value is
    the least-bad choice when every state is blocked.
    """
    s = len(points)
    hard = np.zeros(s, dtype=bool)
    energy = np.zeros(s)
    clearance = np.full(s, np.inf)
    slots = grid._candidates(centre)
    if slots is None:
        return hard, energy, clearance
    if n_recent:
        slots = slots[slots < grid.n_atoms - n_recent]
        if slots.size == 0:
            return hard, energy, clearance
    rad = grid._rad[slots]
    own = grid._mol[slots] == mol_id
    d = grid._pos[slots][None, :, :] - points[:, None, :]
    d -= dims * np.round(d / dims)
    r2 = np.einsum("smk,smk->sm", d, d)
    cut = np.where(own, tolerance,
                   np.maximum(tolerance, 0.5 * (rad + radius) * scale))
    ratio2 = r2 / (cut * cut)
    clearance = np.sqrt(ratio2.min(axis=1))
    hard = clearance < 1.0
    sigma2 = (0.5 * (rad + radius)) ** 2
    close = (r2 < sigma2) & ~own
    if np.any(close):
        frac = np.where(close, sigma2 / np.maximum(r2, 1e-6), 0.0)
        energy = _SOFT_EPSILON * np.minimum(frac ** 6, 1e6).sum(axis=1)
    return hard, energy, clearance


_SOFT_EPSILON = 0.2          # same as NeighbourGrid.soft_energy's default


def _state_offsets(model: RISModel) -> np.ndarray:
    """Local NeRF coordinates of the next atom for every torsion state.

    Row ``s`` is exactly the ``d2`` vector :func:`paaf.cell.ris.place_atom`
    builds for state ``s``; computing them once per chain instead of once per
    state per step is most of what makes growth fast.
    """
    theta = math.radians(model.bond_angle_deg)
    phi = np.radians(np.asarray(model.state_angles_deg, dtype=float))
    l = model.bond_length_a
    return np.stack([np.full_like(phi, -l * math.cos(theta)),
                     l * math.sin(theta) * np.cos(phi),
                     l * math.sin(theta) * np.sin(phi)], axis=1)


def _place_states(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                  offsets: np.ndarray) -> np.ndarray:
    """All candidate positions for one step — :func:`place_atom`, vectorised.

    Same frame, same convention: ``bc`` along the bond, ``n`` normal to the
    A-B-C plane, ``m = n x bc``. Written with scalar arithmetic because
    ``np.cross`` on single 3-vectors costs more than the whole rest of a step.
    """
    bx, by, bz = c[0] - b[0], c[1] - b[1], c[2] - b[2]
    inv = 1.0 / math.sqrt(bx * bx + by * by + bz * bz)
    bx, by, bz = bx * inv, by * inv, bz * inv
    ax, ay, az = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
    nn = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nn < 1e-9:
        return _place_states_collinear(a, b, c, offsets)
    nx, ny, nz = nx / nn, ny / nn, nz / nn
    frame = np.array([[bx, by, bz],
                      [ny * bz - nz * by, nz * bx - nx * bz, nx * by - ny * bx],
                      [nx, ny, nz]])
    return c + offsets @ frame


def _place_states_collinear(a, b, c, offsets):
    """A, B, C collinear: any perpendicular will do, as in place_atom."""
    bc = (c - b) / np.linalg.norm(c - b)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(bc @ helper) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    n = np.cross(bc, helper)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    return c + offsets @ np.stack([bc, m, n])


@dataclass
class _Grown:
    positions: np.ndarray
    states: np.ndarray
    n_overlap: int = 0
    n_spear: int = 0
    n_backtracks: int = 0
    n_forced: int = 0


def _grow_one_chain(model: RISModel, grid: NeighbourGrid, dims: np.ndarray,
                    radii: np.ndarray, n_beads: int, beta: float,
                    temperature: float,
                    rng: np.random.Generator, tolerance: float, scale: float,
                    scan_depth: int, check_spearing: bool,
                    rings: List[_Ring], *, mol_id: int = 0,
                    force: bool = False,
                    cancel: Optional[CancelToken] = None) -> Optional[_Grown]:
    """Grow one chain of ``n_beads`` skeletal atoms, adding it to ``grid``.

    ``radii`` is per bead, not a single value: in a copolymer each bead takes
    its size from the monomer it belongs to, and an isoprene bead is not the
    same size as an epoxidised one.

    Dead ends — every rotational state blocked — are handled by **recoil**
    (Consta, Wilding, Frenkel & Smit, *J. Chem. Phys.* **110** (1999) 3220):
    retract the last few beads and regrow them. Retracting two beads is
    usually enough; if the walk dead-ends again without getting past the
    same point, the retreat doubles, so a tip caged by its neighbours backs
    out of the cage instead of retrying the same trap. Losing a whole chain
    over one blocked step, as a plain restart does, is what made long chains
    in a filled box fail.

    When the dead-end budget runs out, a strict attempt removes its beads and
    returns ``None`` (the caller tries a fresh start). With ``force=True`` it
    instead takes the least-crowded state and counts it in ``n_forced``; the
    caller clears those contacts afterwards with
    :func:`paaf.cell.contact_relief.relieve_contacts`. A forced chain always
    completes.
    """
    l, theta = model.bond_length_a, model.bond_angle_deg
    angles = np.asarray(model.state_angles_deg, dtype=float)
    n_states = model.n_states
    u = model.u_matrix(temperature)
    offsets = _state_offsets(model)
    first_slot = grid.n_atoms
    out = _Grown(positions=np.zeros((0, 3)), states=np.zeros(0, dtype=int))

    def abandon():
        grid.remove_last(grid.n_atoms - first_slot)
        return None

    def put(p: np.ndarray, k: int) -> None:
        pos.append(p)
        grid.add(wrap(p[None], dims), radii[k:k + 1], mol_id=mol_id)

    pos: List[np.ndarray] = []

    # --- seed: the roomiest of a handful of random points. Starting in a
    # void rather than at the first legal point is what lets the last chains
    # into a nearly full box; the start is still uniformly random at low fill.
    best, best_c = None, -1.0
    for trial in range(200):
        pts = rng.random((8, 3)) * dims
        _, _, c = _probe(grid, dims, pts[0], pts[:1], float(radii[0]),
                         tolerance, scale, mol_id, 0)
        for p in pts[1:]:
            _, _, cc = _probe(grid, dims, p, p[None], float(radii[0]),
                              tolerance, scale, mol_id, 0)
            c = np.append(c, cc)
        j = int(np.argmax(c))
        if c[j] > best_c:
            best, best_c = pts[j], float(c[j])
        if best_c >= 1.0:
            break
    if best_c < 1.0 and not force:
        return None
    put(best, 0)

    # --- second bead: random direction; third: fixed angle, random azimuth.
    for k in (1, 2):
        if n_beads <= k:
            break
        trials = np.empty((24, 3))
        for t in range(24):
            if k == 1:
                v = rng.normal(size=3)
                trials[t] = pos[0] + l * v / np.linalg.norm(v)
            else:
                ref = pos[0] + rng.normal(size=3)
                trials[t] = place_atom(ref, pos[0], pos[1], l, theta,
                                       float(rng.uniform(-180, 180)))
        tw = wrap(trials, dims)
        _, _, c = _probe(grid, dims, wrap(pos[-1][None], dims)[0], tw,
                         float(radii[k]), tolerance, scale, mol_id, k)
        ok = np.flatnonzero(c >= 1.0)
        if ok.size:
            pick = int(ok[0])
        elif force:
            pick = int(np.argmax(c))
            out.n_forced += 1
        else:
            return abandon()
        put(trials[pick], k)

    states: List[int] = []
    budget = 60 + 2 * n_beads           # dead ends tolerated per attempt
    dead_ends = 0
    frontier, retreat = -1, 2

    while len(pos) < n_beads:
        k = len(pos)                      # index of the bead being placed
        if k % 512 == 0:
            check_cancel(cancel)
        a, b, c = pos[-3], pos[-2], pos[-1]
        prev_state = states[-1] if states else 0

        cands = _place_states(a, b, c, offsets)
        cw = wrap(cands, dims)
        rk = float(radii[k])
        hard, energy, clear = _probe(grid, dims, wrap(c[None], dims)[0], cw,
                                     rk, tolerance, scale, mol_id, 3)
        out.n_overlap += int(hard.sum())
        if check_spearing and rings:
            for s in range(n_states):
                if not hard[s] and any(
                        _segment_intersects_disc(c, cands[s], r.centre,
                                                 r.normal, r.radius)
                        for r in rings):
                    hard[s] = True
                    out.n_spear += 1

        # P(phi) ∝ w_RIS · exp(-E_nb / kT)   — Theodorou–Suter eq. for the
        # conditional torsion probability.
        weights = u[prev_state] * np.exp(-beta * np.minimum(energy, 200.0))
        weights[hard] = 0.0
        if scan_depth > 0:
            for s in np.flatnonzero(weights > 0.0):
                # Look ahead from the frame this candidate CREATES. The
                # candidate's own weight is applied once, above.
                weights[s] *= _lookahead_weight(
                    model, grid, rk, b, c, cands[s], int(s), scan_depth,
                    beta, temperature, mol_id, dims)

        total = float(weights.sum())
        if total > 0.0:
            s = int(rng.choice(n_states, p=weights / total))
        else:
            dead_ends += 1
            if dead_ends <= budget and k > 3:
                # Recoil. Escalate only while stuck at the same place.
                if k <= frontier:
                    retreat = min(retreat * 2, 64)
                else:
                    frontier, retreat = k, 2
                back = min(retreat, k - 3)
                del pos[-back:]
                del states[-back:]
                grid.remove_last(back)
                out.n_backtracks += 1
                continue
            if not force:
                return abandon()
            s = int(np.argmax(clear))
            out.n_forced += 1

        put(cands[s], k)
        states.append(s)

    out.positions = np.asarray(pos)
    out.states = np.asarray(states, dtype=int)
    return out
