"""Monomer sequences for copolymer chains.

Why this is its own module
--------------------------
A homopolymer chain is fully described by one SMILES and a degree of
polymerisation. A copolymer is not: you also need to say *which monomer sits
where*, and that ordering changes the physics. Epoxidised natural rubber is a
random copolymer of isoprene and epoxidised isoprene; a block copolymer of the
same two monomers at the same composition is a different material.

The ordering is separated from the growth code so it can be reasoned about and
tested on its own, and so a chain's realised composition can be reported
before anything is built.

Arrangements
------------
``random``
    Each position drawn independently from the requested fractions. This is
    what a free-radical copolymerisation with similar reactivity ratios
    produces, and it is what "random copolymer" means in practice. Because
    positions are drawn independently, a short chain's realised composition
    will scatter around the target — which is true of real chains too, and is
    reported rather than hidden.

``alternating``
    ABAB… Requires exactly two monomers. Reactivity ratios near zero give
    this.

``block``
    All of A, then all of B. Block lengths follow the requested fractions.

``exact``
    You supply the sequence yourself, as a list of monomer indices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["Monomer", "MonomerSequence", "build_sequence", "ARRANGEMENTS"]

ARRANGEMENTS = ("random", "alternating", "block", "exact")


@dataclass
class Monomer:
    """One monomer in a (co)polymer, with the chemistry PAAF derives from it."""
    smiles: str
    fraction: float = 1.0            # mole fraction, normalised on use
    name: str = ""

    # Filled by :meth:`resolve` — never set by hand.
    mass_amu: float = 0.0            # per repeat unit
    backbone_atoms: int = 1          # skeletal atoms this unit contributes
    backbone_ring_atoms: int = 0     # >0 means the backbone runs through a ring

    def resolve(self) -> None:
        from .grow import (
            backbone_atoms_per_unit, backbone_ring_atoms, repeat_unit_mass,
        )
        self.mass_amu = repeat_unit_mass(self.smiles)
        self.backbone_atoms = backbone_atoms_per_unit(self.smiles)
        self.backbone_ring_atoms = backbone_ring_atoms(self.smiles)


@dataclass
class MonomerSequence:
    """A concrete ordering of monomers along one chain."""
    monomers: List[Monomer] = field(default_factory=list)
    order: List[int] = field(default_factory=list)   # index into `monomers`
    arrangement: str = "random"

    # ---------------------------------------------------------- geometry
    @property
    def n_units(self) -> int:
        return len(self.order)

    @property
    def n_beads(self) -> int:
        """Skeletal atoms in the whole chain — the number of growth steps."""
        return sum(self.monomers[i].backbone_atoms for i in self.order)

    @property
    def chain_mass(self) -> float:
        return sum(self.monomers[i].mass_amu for i in self.order)

    def bead_monomer_index(self) -> List[int]:
        """Which monomer each skeletal bead belongs to, in growth order.

        This is what lets a copolymer be grown at all: mass and effective
        radius have to follow the monomer that bead actually came from, not a
        chain-wide average.
        """
        out: List[int] = []
        for u in self.order:
            out.extend([u] * self.monomers[u].backbone_atoms)
        return out

    def realised_fractions(self) -> List[float]:
        """Mole fraction of each monomer as actually placed."""
        if not self.order:
            return [0.0] * len(self.monomers)
        counts = [0] * len(self.monomers)
        for i in self.order:
            counts[i] += 1
        return [c / len(self.order) for c in counts]

    def realised_weight_fractions(self) -> List[float]:
        total = self.chain_mass or 1.0
        counts = [0] * len(self.monomers)
        for i in self.order:
            counts[i] += 1
        return [c * m.mass_amu / total
                for c, m in zip(counts, self.monomers)]

    def describe(self) -> str:
        bits = []
        for m, f, w in zip(self.monomers, self.realised_fractions(),
                           self.realised_weight_fractions()):
            bits.append(f"{m.name or m.smiles}: {100 * f:.1f} mol% "
                        f"/ {100 * w:.1f} wt%")
        return f"{self.arrangement}, {self.n_units} units — " + "; ".join(bits)


# =====================================================================
def build_sequence(monomers: Sequence[Monomer], n_units: int,
                   arrangement: str = "random",
                   seed: int = 0,
                   exact: Optional[Sequence[int]] = None) -> MonomerSequence:
    """Order ``n_units`` repeat units according to ``arrangement``.

    Fractions are normalised, so they may be given as percentages, mole
    fractions or any consistent ratio.
    """
    mons = [m for m in monomers if m.fraction > 0.0 or arrangement == "exact"]
    if not mons:
        raise ValueError("No monomer has a non-zero fraction.")
    for m in mons:
        if not m.mass_amu:
            m.resolve()
    if n_units < 1:
        raise ValueError("A chain needs at least one repeat unit.")

    arrangement = (arrangement or "random").lower()
    if arrangement not in ARRANGEMENTS:
        raise ValueError(
            f"arrangement must be one of {ARRANGEMENTS}, not {arrangement!r}")

    if len(mons) == 1:
        return MonomerSequence(monomers=list(mons), order=[0] * n_units,
                               arrangement="homopolymer")

    total = sum(m.fraction for m in mons)
    fracs = [m.fraction / total for m in mons]

    if arrangement == "exact":
        if exact is None:
            raise ValueError("arrangement='exact' needs an explicit sequence.")
        order = [int(i) for i in exact]
        if any(i < 0 or i >= len(mons) for i in order):
            raise ValueError("exact sequence indexes a monomer that does not "
                             "exist.")
    elif arrangement == "alternating":
        if len(mons) != 2:
            raise ValueError(
                f"'alternating' needs exactly two monomers, got {len(mons)}.")
        order = [i % 2 for i in range(n_units)]
    elif arrangement == "block":
        # Largest remainder, so the blocks sum to exactly n_units and no
        # requested monomer is dropped entirely.
        ideal = [f * n_units for f in fracs]
        counts = [max(1, int(np.floor(x))) for x in ideal]
        deficit = n_units - sum(counts)
        if deficit > 0:
            rank = sorted(range(len(mons)),
                          key=lambda i: ideal[i] - counts[i], reverse=True)
            for k in range(deficit):
                counts[rank[k % len(rank)]] += 1
        while sum(counts) > n_units:
            i = max(range(len(mons)), key=lambda k: counts[k])
            if counts[i] <= 1:
                break
            counts[i] -= 1
        order = []
        for i, c in enumerate(counts):
            order.extend([i] * c)
    else:                                            # random
        rng = np.random.default_rng(seed)
        order = list(rng.choice(len(mons), size=n_units, p=fracs))
        order = [int(i) for i in order]

    seq = MonomerSequence(monomers=list(mons), order=order,
                          arrangement=arrangement)
    log.info("Sequence: %s", seq.describe())
    return seq
