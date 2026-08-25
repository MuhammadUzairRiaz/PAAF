"""Turn a composition specification into something the grower can build.

The user thinks in the units a polymer chemist thinks in — "70 wt% PE,
30 wt% PS, at 0.95 g/cm³, chains of DP 50, about 4000 atoms" — while
:func:`paaf.cell.grow.grow_amorphous_cell` needs integer chain counts and a
box edge in Ångström. This module is the translation, and it is deliberately
separate from both the GUI and the grower so the arithmetic can be tested on
its own.

Two directions, because both are legitimate
--------------------------------------------
``solve_from_weight_fractions``
    You state the target weight percentages and roughly how big the cell
    should be. PAAF picks integer chain counts. Because chain counts are
    integers, the realised composition is almost never exactly the requested
    one, and the result says so rather than quietly rounding.

``from_chain_counts``
    You state the chain counts yourself. PAAF computes the resulting weight
    percentages and the box edge. Nothing is rounded, nothing is inferred.

Both return the same :class:`CellComposition`, so the caller downstream does
not care which was used.

Where the masses come from
--------------------------
Every mass traces back to the SMILES via
:func:`paaf.cell.grow.repeat_unit_mass`, which counts implicit hydrogens and
subtracts one hydrogen per ``[*]`` attachment point. That correction is worth
2.016 g/mol on a linear repeat unit — 7% for polyethylene — and it lands
directly in both the density and the weight fractions, so it is not optional.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..logging_utils import get_logger
from .grow import GrowSpec, backbone_atoms_per_unit, repeat_unit_mass

log = get_logger(__name__)

__all__ = ["Component", "CellComposition", "solve_from_weight_fractions",
           "from_chain_counts", "box_edge_for_density"]

_N_AVOGADRO = 6.02214076e23


# =====================================================================
@dataclass
class Component:
    """One polymer in the cell, as the user specifies it."""
    name: str
    repeat_unit: str                    # polymerisation SMILES, [*]…[*]
    degree_of_polymerisation: int = 50
    weight_percent: float = 100.0       # requested; ignored in count mode
    n_chains: int = 0                   # used in count mode
    ris_key: str = ""

    # A COPOLYMER component: monomers with fractions, plus how they are
    # ordered. When present it supersedes `repeat_unit`, and the masses below
    # become composition-weighted averages — a copolymer has no single repeat
    # unit, so every per-bead quantity is resolved from the sequence instead.
    monomers: Optional[List[object]] = None      # List[sequence.Monomer]
    arrangement: str = "random"
    sequence_seed: int = 0

    # Filled in by the solver — never set these by hand.
    unit_mass: float = 0.0              # g/mol per repeat unit (mean)
    chain_mass: float = 0.0             # g/mol per chain
    backbone_atoms: int = 1             # skeletal atoms per repeat unit (mean)

    @property
    def is_copolymer(self) -> bool:
        return bool(self.monomers) and len(self.monomers) > 1

    def resolve_chemistry(self) -> None:
        """Compute masses and granularity from the SMILES or the monomers."""
        if self.is_copolymer:
            for mono in self.monomers:
                if not mono.mass_amu:
                    mono.resolve()
            w = [m.fraction for m in self.monomers]
            wsum = sum(w) or 1.0
            self.unit_mass = sum(m.mass_amu * f
                                 for m, f in zip(self.monomers, w)) / wsum
            self.backbone_atoms = max(1, int(round(
                sum(m.backbone_atoms * f
                    for m, f in zip(self.monomers, w)) / wsum)))
            self.chain_mass = self.unit_mass * int(self.degree_of_polymerisation)
            return
        self.unit_mass = repeat_unit_mass(self.repeat_unit)
        self.chain_mass = self.unit_mass * int(self.degree_of_polymerisation)
        self.backbone_atoms = backbone_atoms_per_unit(self.repeat_unit)

    @property
    def beads_per_chain(self) -> int:
        return int(self.degree_of_polymerisation) * int(self.backbone_atoms)


@dataclass
class CellComposition:
    """A fully determined cell: what to grow, and in what box."""
    components: List[Component] = field(default_factory=list)
    chain_counts: List[int] = field(default_factory=list)
    target_density_g_cm3: float = 1.0
    #: The density the cell is actually BUILT at. Growth places beads at a
    #: hard-core spacing and side groups are bolted on afterwards, so a cell
    #: constructed straight at bulk density has no room for them — measured
    #: on a PE/PS cell at 0.95 g/cm3: 14,528 non-bonded pairs under 2.0 A,
    #: which DL_FIELD reads as bonds and refuses to type. Build loose,
    #: compress with NPT. Same reasoning as the Blend page's "pack at %".
    build_density_g_cm3: float = 1.0
    box_edge_a: float = 0.0
    total_mass_amu: float = 0.0
    realised_weight_percent: List[float] = field(default_factory=list)
    requested_weight_percent: List[float] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # ---------------------------------------------------------- derived
    @property
    def total_chains(self) -> int:
        return int(sum(self.chain_counts))

    @property
    def total_beads(self) -> int:
        return int(sum(c.beads_per_chain * n
                       for c, n in zip(self.components, self.chain_counts)))

    @property
    def box_volume_a3(self) -> float:
        return self.box_edge_a ** 3

    @property
    def max_weight_percent_error(self) -> float:
        """Largest absolute deviation from the requested composition, in pp."""
        if not self.requested_weight_percent:
            return 0.0
        return max(abs(r - q) for r, q in
                   zip(self.realised_weight_percent,
                       self.requested_weight_percent))

    def grow_specs(self) -> List[GrowSpec]:
        """The list :func:`paaf.cell.grow.grow_amorphous_cell` consumes."""
        return [
            GrowSpec(repeat_unit=c.repeat_unit,
                     n_chains=int(n),
                     degree_of_polymerisation=int(c.degree_of_polymerisation),
                     name=c.name or c.repeat_unit,
                     ris_key=c.ris_key,
                     mass_amu=None if c.is_copolymer else (c.unit_mass or None),
                     backbone_atoms=None if c.is_copolymer else c.backbone_atoms,
                     monomers=c.monomers,
                     arrangement=c.arrangement,
                     sequence_seed=c.sequence_seed)
            for c, n in zip(self.components, self.chain_counts) if n > 0
        ]

    def box(self):
        """A :class:`paaf.cell.amorphous.BoxShape` for the solved edge."""
        from .amorphous import BoxShape
        return BoxShape(shape="cubic", a=self.box_edge_a)

    def summary(self) -> str:
        lines = [
            f"{self.total_chains} chains, {self.total_beads} skeletal beads",
            f"box            : {self.box_edge_a:.2f} Å cubic "
            f"(V = {self.box_volume_a3:,.0f} Å³)",
            f"density        : {self.target_density_g_cm3:.4f} g/cm³",
        ]
        for c, n, w in zip(self.components, self.chain_counts,
                           self.realised_weight_percent):
            lines.append(
                f"  {c.name:12s} {n:4d} x DP {c.degree_of_polymerisation:<4d} "
                f"chain {c.chain_mass:9,.0f} g/mol   {w:6.2f} wt%")
        lines.extend(f"note           : {n}" for n in self.notes)
        return "\n".join(lines)


# =====================================================================
def box_edge_for_density(total_mass_amu: float, density_g_cm3: float) -> float:
    """Cubic edge in Å whose volume gives ``density_g_cm3``.

    ``mass / (N_A · rho)`` is the volume in cm³; ``1e24`` converts to Å³.
    """
    if density_g_cm3 <= 0.0:
        raise ValueError("Density must be positive.")
    if total_mass_amu <= 0.0:
        raise ValueError("Total mass must be positive.")
    volume_cm3 = total_mass_amu / (_N_AVOGADRO * density_g_cm3)
    return float((volume_cm3 * 1.0e24) ** (1.0 / 3.0))


def _weight_percentages(components: Sequence[Component],
                        counts: Sequence[int]) -> Tuple[List[float], float]:
    total = sum(c.chain_mass * n for c, n in zip(components, counts))
    if total <= 0.0:
        return [0.0] * len(components), 0.0
    return ([100.0 * c.chain_mass * n / total
             for c, n in zip(components, counts)], total)


# =====================================================================
def solve_from_weight_fractions(
    components: Sequence[Component],
    target_density_g_cm3: float,
    *,
    target_beads: Optional[int] = None,
    target_chains: Optional[int] = None,
    build_fraction: float = 1.0,
) -> CellComposition:
    """Choose integer chain counts closest to the requested weight percentages.

    Exactly one size control must be given: ``target_beads`` (roughly how many
    skeletal atoms the cell should contain — this is what governs run time) or
    ``target_chains``.

    Why this is not just ``round(w * N)``
    -------------------------------------
    Weight fraction is set by *mass*, and chains of different species have very
    different masses — a DP-50 polystyrene chain is 5,208 g/mol against 1,403
    for a DP-50 polyethylene one. Allocating chains in proportion to weight
    without dividing by chain mass would give a cell that is nowhere near the
    requested composition.

    So the ideal (fractional) count for species *i* is

    .. math:: n_i = \\frac{w_i \\, M_\\text{total}}{m_i}

    and the integers are chosen by **largest remainder**: floor everything,
    then hand the leftover chains to whichever species is furthest below its
    target. Every species that was asked for a non-zero weight gets at least
    one chain — a "5 wt% additive" that rounds to zero chains is not a 5 wt%
    cell, and silently dropping it would be the worst outcome.

    The realised percentages are reported alongside the requested ones. With
    few chains they can differ a lot, and that is a fact about integers, not a
    bug: 3 chains cannot express 70/30.
    """
    comps = [c for c in components if c.weight_percent > 0.0]
    if not comps:
        raise ValueError("No component has a non-zero weight percent.")
    if (target_beads is None) == (target_chains is None):
        raise ValueError(
            "Give exactly one of target_beads or target_chains.")

    for c in comps:
        c.resolve_chemistry()
        if c.degree_of_polymerisation < 1:
            raise ValueError(f"{c.name}: degree of polymerisation must be >= 1.")

    notes: List[str] = []
    w_sum = sum(c.weight_percent for c in comps)
    if abs(w_sum - 100.0) > 1e-6:
        notes.append(
            f"Weight percentages summed to {w_sum:.3f}, not 100 — "
            f"normalised.")
    fracs = [c.weight_percent / w_sum for c in comps]

    # ---- how many chains in total?
    if target_chains is not None:
        n_total = int(target_chains)
        if n_total < len(comps):
            raise ValueError(
                f"{n_total} chains cannot represent {len(comps)} components; "
                f"ask for at least {len(comps)}.")
    else:
        # Beads per chain differs per species, so convert through the
        # weight-averaged beads-per-chain rather than assuming a common value.
        beads_per_gram = sum(
            f * c.beads_per_chain / c.chain_mass for f, c in zip(fracs, comps))
        if beads_per_gram <= 0:
            raise ValueError("Could not size the cell from target_beads.")
        mass_needed = float(target_beads) / beads_per_gram
        n_total = 0
        for f, c in zip(fracs, comps):
            n_total += max(1, int(round(f * mass_needed / c.chain_mass)))
        n_total = max(n_total, len(comps))

    # ---- ideal fractional counts at that total mass
    # Solve for the total mass M that yields n_total chains, then n_i = w_i M / m_i.
    inv = sum(f / c.chain_mass for f, c in zip(fracs, comps))
    mass_total = n_total / inv
    ideal = [f * mass_total / c.chain_mass for f, c in zip(fracs, comps)]

    # ---- largest-remainder rounding, with a floor of one chain each
    counts = [max(1, int(math.floor(x))) for x in ideal]
    deficit = n_total - sum(counts)
    if deficit > 0:
        order = sorted(range(len(comps)),
                       key=lambda i: ideal[i] - counts[i], reverse=True)
        for k in range(deficit):
            counts[order[k % len(order)]] += 1
    elif deficit < 0:
        # The one-chain floor pushed us over. Trim from the most over-supplied
        # species, never below one chain.
        order = sorted(range(len(comps)),
                       key=lambda i: counts[i] - ideal[i], reverse=True)
        k = 0
        while deficit < 0 and k < 10_000:
            i = order[k % len(order)]
            if counts[i] > 1:
                counts[i] -= 1
                deficit += 1
            k += 1

    realised, total_mass = _weight_percentages(comps, counts)
    requested = [100.0 * f for f in fracs]

    worst = max(abs(r - q) for r, q in zip(realised, requested))
    if worst > 1.0:
        notes.append(
            f"Integer chain counts give {worst:.1f} percentage points of "
            f"composition error. Raise the cell size to reduce it — weight "
            f"fractions are only as fine as one chain's mass.")
    for c, n, ideal_n in zip(comps, counts, ideal):
        if ideal_n < 0.5:
            notes.append(
                f"{c.name}: the requested weight fraction works out to "
                f"{ideal_n:.2f} chains. It was rounded UP to 1 rather than "
                f"dropped, so this component is over-represented.")

    try:
        frac = float(build_fraction)
    except (TypeError, ValueError):
        frac = 1.0
    frac = min(max(frac, 0.05), 1.0)
    build_density = float(target_density_g_cm3) * frac
    edge = box_edge_for_density(total_mass, build_density)
    if frac < 0.999:
        notes.append(
            f"Built at {build_density:.3f} g/cm3 ({frac:.0%} of the "
            f"{target_density_g_cm3:.3f} target) so the side groups have room "
            f"to be placed. Compress to the target with an NPT run before "
            f"measuring anything.")
    comp = CellComposition(
        components=list(comps), chain_counts=counts,
        target_density_g_cm3=float(target_density_g_cm3),
        build_density_g_cm3=build_density,
        box_edge_a=edge, total_mass_amu=total_mass,
        realised_weight_percent=realised, requested_weight_percent=requested,
        notes=notes,
    )
    log.info("Composition solved: %s", comp.summary().replace("\n", " | "))
    return comp


def from_chain_counts(components: Sequence[Component],
                      target_density_g_cm3: float,
                      build_fraction: float = 1.0) -> CellComposition:
    """Take the chain counts as given; report the composition they produce.

    Nothing is rounded here — the counts on each :class:`Component` are used
    verbatim, and the weight percentages are whatever they turn out to be.

    ``build_fraction`` loosens the BOX exactly as in
    :func:`solve_from_weight_fractions`. It was missing here, so the
    manual-counts route silently grew at the full target density no matter
    what "Build at" said — the weight-% route built at 45% while this one
    fought 0.95 g/cm3 and lost.
    """
    comps = [c for c in components if int(c.n_chains) > 0]
    if not comps:
        raise ValueError("Every component has zero chains.")
    for c in comps:
        c.resolve_chemistry()
        if c.degree_of_polymerisation < 1:
            raise ValueError(f"{c.name}: degree of polymerisation must be >= 1.")

    counts = [int(c.n_chains) for c in comps]
    realised, total_mass = _weight_percentages(comps, counts)
    try:
        frac = float(build_fraction)
    except (TypeError, ValueError):
        frac = 1.0
    frac = min(max(frac, 0.05), 1.0)
    build_density = float(target_density_g_cm3) * frac
    edge = box_edge_for_density(total_mass, build_density)
    notes = []
    if frac < 0.999:
        notes.append(
            f"Built at {build_density:.3f} g/cm3 ({frac:.0%} of the "
            f"{target_density_g_cm3:.3f} target) so the side groups have room "
            f"to be placed. Compress to the target with an NPT run before "
            f"measuring anything.")
    return CellComposition(
        components=list(comps), chain_counts=counts,
        target_density_g_cm3=float(target_density_g_cm3),
        build_density_g_cm3=build_density,
        box_edge_a=edge, total_mass_amu=total_mass,
        realised_weight_percent=realised,
        requested_weight_percent=list(realised),   # nothing was requested
        notes=notes,
    )
