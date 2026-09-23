"""Copolymer sequence patterns shared by the chain Builder and the amorphous cell.

Two code paths build copolymer sequences — :func:`paaf.chain_builder.build_chain`
for single chains and :mod:`paaf.cell.sequence` for amorphous cells — and both
need the same arrangements. The pure ordering logic lives here so the two
cannot drift apart.

Monomers are named by letter in their table order: A is the first, B the
second, C the third, and so on. Every arrangement works for any number of
monomers (a terpolymer is just three rows).

Arrangements added here
-----------------------
``gradient``
    Composition drifts smoothly along the chain: mostly A at the start, mostly
    B at the end (with three monomers, A → B → C). Unit counts match the
    requested fractions exactly; only *where* each unit sits is drawn from the
    seed, so ``AAAABAABABBABBBB`` rather than a sharp ``AAAAAAAABBBBBBBB``.

``multiblock``
    A user-written block configuration, one section per block::

        AAAAA-BBBB-BBB-AAA        letters spelled out
        A5-B4-B3-A3               letter + length
        A5 B4 C3                  spaces or commas also separate sections
        (A5-B5)x3                 a repeated group

    If the pattern is shorter than the chain it is repeated (``repeat``) or
    every section is scaled in proportion (``stretch``).
"""
from __future__ import annotations

import random
import re
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "letter", "alternating_order", "block_counts", "gradient_order",
    "parse_block_pattern", "pattern_length", "multiblock_order",
    "run_length", "MULTIBLOCK_FILLS",
]

MULTIBLOCK_FILLS = ("repeat", "stretch")

#: Default spread of the gradient, as a fraction of the chain length. 0 gives
#: a sharp block copolymer; ~0.5 and above is close to random in the middle.
DEFAULT_GRADIENT_WIDTH = 0.2


def letter(i: int) -> str:
    """Monomer index → the letter the patterns use (0 → A, 25 → Z)."""
    return chr(ord("A") + int(i))


def run_length(order: Sequence[int], limit: int = 0) -> str:
    """``[0,0,0,1,1,0]`` → ``"A3-B2-A1"`` — a compact, readable sequence."""
    runs: List[str] = []
    prev, count = None, 0
    for m in order:
        if m == prev:
            count += 1
            continue
        if prev is not None:
            runs.append(f"{letter(prev)}{count}")
        prev, count = m, 1
    if prev is not None:
        runs.append(f"{letter(prev)}{count}")
    if limit and len(runs) > limit:
        return "-".join(runs[:limit]) + f"-… ({len(runs)} blocks)"
    return "-".join(runs)


# ----------------------------------------------------------------- simple
def alternating_order(n_monomers: int, n: int) -> List[int]:
    """ABAB… for two monomers, ABCABC… for three, and so on."""
    return [i % n_monomers for i in range(n)]


def block_counts(fractions: Sequence[float], n: int) -> List[int]:
    """Whole unit counts that sum to ``n`` and follow ``fractions``.

    Largest remainder, and no monomer with a non-zero fraction is rounded away
    entirely — a 2% component in a DP-20 chain still gets one unit.
    """
    total = sum(fractions) or 1.0
    ideal = [f / total * n for f in fractions]
    counts = [max(1 if f > 0 else 0, int(x)) for f, x in zip(fractions, ideal)]
    deficit = n - sum(counts)
    if deficit > 0:
        rank = sorted(range(len(counts)),
                      key=lambda i: ideal[i] - counts[i], reverse=True)
        for k in range(deficit):
            counts[rank[k % len(rank)]] += 1
    while sum(counts) > n:
        i = max(range(len(counts)), key=lambda k: counts[k])
        if counts[i] <= 1:
            break
        counts[i] -= 1
    return counts


# --------------------------------------------------------------- gradient
def gradient_order(fractions: Sequence[float], n: int,
                   seed: Optional[int] = None,
                   width: float = DEFAULT_GRADIENT_WIDTH) -> List[int]:
    """A gradient copolymer: A-rich start, B-rich end, smooth in between.

    Each unit of monomer *m* is given an ideal position inside that monomer's
    share of the chain (exactly as in a block copolymer), then jittered by a
    Gaussian of standard deviation ``width`` × chain length. Sorting by the
    jittered positions gives the sequence. Counts therefore match the
    fractions exactly, and the same seed gives the same chain.
    """
    counts = block_counts(fractions, n)
    rng = random.Random(seed)
    width = max(0.0, float(width))
    keyed: List[Tuple[float, int]] = []
    start = 0
    for m, c in enumerate(counts):
        for k in range(c):
            ideal = (start + k + 0.5) / n
            keyed.append((ideal + rng.gauss(0.0, width), m))
        start += c
    keyed.sort(key=lambda t: t[0])
    return [m for _key, m in keyed][:n]


# ------------------------------------------------------------- multiblock
_GROUP = re.compile(r"\(([^()]*)\)\s*[x×*]\s*(\d+)", re.IGNORECASE)


def parse_block_pattern(text: str, n_monomers: Optional[int] = None
                        ) -> List[Tuple[int, int]]:
    """Parse a block configuration into ``[(monomer_index, length), …]``.

    Raises ``ValueError`` with the offending section named, so a typo in the
    GUI is reported rather than silently building something else.
    """
    src = (text or "").strip()
    if not src:
        raise ValueError("The block pattern is empty — write sections such as "
                         "A5-B4-B3-A3 or AAAAA-BBBB-BBB-AAA.")
    # Expand repeated groups, innermost first: (A5-B5)x3 → A5-B5-A5-B5-A5-B5
    for _ in range(10):
        new = _GROUP.sub(lambda mt: "-".join([mt.group(1)] * int(mt.group(2))),
                         src)
        if new == src:
            break
        src = new
    if "(" in src or ")" in src:
        raise ValueError(f"Unbalanced or unrepeated brackets in {text!r}; "
                         f"write groups as (A5-B5)x3.")

    sections: List[Tuple[int, int]] = []
    for tok in re.split(r"[\s,;\-–|/]+", src):
        if not tok:
            continue
        mt = re.fullmatch(r"([A-Za-z])(?::?(\d+))?", tok)
        if mt:
            idx = ord(mt.group(1).upper()) - ord("A")
            length = int(mt.group(2)) if mt.group(2) else 1
        elif re.fullmatch(r"([A-Za-z])\1+", tok):
            idx = ord(tok[0].upper()) - ord("A")
            length = len(tok)
        else:
            raise ValueError(
                f"Cannot read block section {tok!r} in {text!r}. Use a letter "
                f"with a length (A5), or the letter repeated (AAAAA).")
        if length < 1:
            raise ValueError(f"Block section {tok!r} has zero length.")
        if n_monomers is not None and idx >= n_monomers:
            raise ValueError(
                f"Block section {tok!r} refers to monomer {letter(idx)}, but "
                f"only {n_monomers} monomer(s) are defined "
                f"(A–{letter(n_monomers - 1)}).")
        sections.append((idx, length))
    if not sections:
        raise ValueError(f"No block sections found in {text!r}.")
    return sections


def pattern_length(sections: Sequence[Tuple[int, int]]) -> int:
    return sum(length for _m, length in sections)


def multiblock_order(sections: Sequence[Tuple[int, int]], n: int,
                     fill: str = "repeat") -> List[int]:
    """Lay the block sections along a chain of ``n`` units.

    ``fill='repeat'`` tiles the pattern (and cuts the last copy at ``n``);
    ``fill='stretch'`` scales every section by ``n / pattern length`` so the
    chain holds the pattern exactly once, each section keeping at least one
    unit.
    """
    fill = (fill or "repeat").lower()
    if fill not in MULTIBLOCK_FILLS:
        raise ValueError(f"fill must be one of {MULTIBLOCK_FILLS}, not {fill!r}")
    total = pattern_length(sections)
    if fill == "stretch" and total != n:
        if n < len(sections):
            raise ValueError(
                f"The pattern has {len(sections)} sections but the chain only "
                f"{n} units; it cannot be stretched to fit. Use a longer "
                f"chain or fewer sections.")
        counts = block_counts([length for _m, length in sections], n)
        sections = [(m, c) for (m, _l), c in zip(sections, counts)]
    order: List[int] = []
    while len(order) < n:
        for m, length in sections:
            order.extend([m] * length)
    return order[:n]
