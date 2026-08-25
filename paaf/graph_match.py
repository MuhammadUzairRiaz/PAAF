"""Pair up the atoms of two molecules that are the same molecule.

Why a module of its own
-----------------------
Two separate problems in PAAF turned out to be this one problem.

*Which chain atom came from which monomer atom*, when the chain builder does
not emit atoms in monomer order — solved by matching the chain against a
reference chain whose provenance is known by construction.

*Which monomer atom is the link atom*, when all that is known is a
polymerisation SMILES like ``[*]CC(CC)[*]`` and a 3D file built from it
separately. The wildcards say exactly where the links are, but they say it in
the SMILES's numbering, and the 3D file has its own.

Both are "here are two atom orderings of one molecule; line them up".

How
---
Colour refinement (Weisfeiler-Leman). Every atom starts coloured by its
element; each round recolours it by its own colour plus the sorted colours of
its neighbours, compressed back to an integer so the tuples cannot grow. After
R rounds two atoms share a colour only if their R-hop surroundings are
identical.

The two graphs are refined TOGETHER as one disconnected graph. Refining them
separately would give two numberings that mean nothing to each other, since
colours are integers handed out in first-seen order.

What it does and does not promise
---------------------------------
Colour refinement is not a proof of isomorphism — a handful of famously
awkward graphs defeat it, and none of them is a polymer. What it does
guarantee is the direction that matters here: atoms that end up in the same
class are genuinely indistinguishable by any amount of looking at their
surroundings, so choosing between them is not a decision anyone could make
differently. Where it cannot separate two atoms, either answer is right.

Callers are expected to verify the result against whatever they actually care
about — an element, a fingerprint, a heavy-atom degree — and to treat ``None``
as "these are not the same molecule".
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["refine_colours", "match_graphs"]

#: Rounds of refinement. Each one extends an atom's view by one bond, so this
#: is a radius, not an iteration count: 60 bonds is far beyond any local
#: environment a force field distinguishes. Refinement also stops early once
#: it stops separating anything, which is the usual case well before 60.
DEFAULT_ROUNDS = 60


def refine_colours(nodes: Sequence[Hashable],
                   adjacency: Dict[Hashable, Sequence[Hashable]],
                   elements: Dict[Hashable, str],
                   rounds: int = DEFAULT_ROUNDS) -> Dict[Hashable, int]:
    """Colour each node by its neighbourhood, refining until stable."""
    colour: Dict[Hashable, object] = {n: elements[n] for n in nodes}
    distinct = len(set(colour.values()))
    for _ in range(rounds):
        signature = {n: (colour[n],
                         tuple(sorted(colour[m] for m in adjacency[n])))
                     for n in nodes}
        table: Dict[tuple, int] = {}
        for n in nodes:
            table.setdefault(signature[n], len(table))
        colour = {n: table[signature[n]] for n in nodes}
        if len(table) == distinct:
            break
        distinct = len(table)
    return {n: int(c) for n, c in colour.items()}


def _adjacency(nodes: Iterable[Hashable],
               bonds: Iterable[Tuple[Hashable, Hashable]]
               ) -> Dict[Hashable, List[Hashable]]:
    keep = set(nodes)
    adj: Dict[Hashable, List[Hashable]] = {n: [] for n in keep}
    for bond in bonds:
        i, j = bond[0], bond[1]
        if i in keep and j in keep:
            adj[i].append(j)
            adj[j].append(i)
    return adj


def match_graphs(elements_a: Dict[Hashable, str],
                 bonds_a: Iterable[Tuple[Hashable, Hashable]],
                 elements_b: Dict[Hashable, str],
                 bonds_b: Iterable[Tuple[Hashable, Hashable]],
                 rounds: int = DEFAULT_ROUNDS,
                 label: str = "") -> Optional[Dict[Hashable, Hashable]]:
    """``{node of A: node of B}``, or ``None`` if they are not the same graph.

    Where several atoms of A share an environment with several of B, they are
    paired in sorted order. That choice is arbitrary and has to be: nothing
    distinguishes them. It is at least reproducible.
    """
    if len(elements_a) != len(elements_b):
        log.debug("%sgraphs differ in size: %d vs %d",
                  f"{label}: " if label else "",
                  len(elements_a), len(elements_b))
        return None

    nodes = [("a", n) for n in elements_a] + [("b", n) for n in elements_b]
    joined_elements = {("a", n): e for n, e in elements_a.items()}
    joined_elements.update({("b", n): e for n, e in elements_b.items()})

    adj_a = _adjacency(elements_a, bonds_a)
    adj_b = _adjacency(elements_b, bonds_b)
    joined_adj: Dict[Hashable, List[Hashable]] = {}
    for n in elements_a:
        joined_adj[("a", n)] = [("a", m) for m in adj_a[n]]
    for n in elements_b:
        joined_adj[("b", n)] = [("b", m) for m in adj_b[n]]

    colour = refine_colours(nodes, joined_adj, joined_elements, rounds)

    by_colour_a: Dict[int, List[Hashable]] = defaultdict(list)
    by_colour_b: Dict[int, List[Hashable]] = defaultdict(list)
    for n in elements_a:
        by_colour_a[colour[("a", n)]].append(n)
    for n in elements_b:
        by_colour_b[colour[("b", n)]].append(n)

    if set(by_colour_a) != set(by_colour_b):
        log.debug("%s%d environment classes appear in only one of the two "
                  "molecules", f"{label}: " if label else "",
                  len(set(by_colour_a) ^ set(by_colour_b)))
        return None

    out: Dict[Hashable, Hashable] = {}
    for c, members_a in by_colour_a.items():
        members_b = by_colour_b[c]
        if len(members_a) != len(members_b):
            log.debug("%senvironment class %s has %d atoms in one molecule "
                      "and %d in the other", f"{label}: " if label else "",
                      c, len(members_a), len(members_b))
            return None
        for x, y in zip(sorted(members_a), sorted(members_b)):
            out[x] = y
    return out
