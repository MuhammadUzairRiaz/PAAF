"""Which monomer atom each chain atom came from — known, not guessed.

The failure this replaces
------------------------
The typing dialog hands back ``{monomer_atom_index: type}``. The pipeline
applied that map straight onto the *chain*::

    for i, t in manual_types.items():
        mol.atoms[i].ff_type = t          # i is a MONOMER index, mol is the CHAIN

Those are different numberings. Linking deletes a cap hydrogen at every
junction, so chain atom *k* is not monomer atom *k* beyond the first cap, and
every assignment past that point lands on the wrong atom.

A previous attempt papered over this by spreading types across chemical
equivalence classes. That made it worse: the element of the *source* atom was
read from the chain rather than the monomer, so a hydrogen's type could be
attached to a class of carbons and propagated — which is how two backbone CH2
carbons ended up typed 140 and written with mass 1.008.

The answer is not a cleverer guess. It is to keep the mapping.

How the mapping is obtained
---------------------------
A chain built from ``n`` copies of one monomer is periodic once the caps are
accounted for: the first unit keeps its head cap, the last keeps its tail cap,
and every junction consumes one from each side. That gives an exact
arithmetic map from chain atom to ``(unit, monomer atom)``.

Rather than trust the arithmetic, it is **verified atom by atom against the
element**. If a single chain atom's element disagrees with the monomer atom it
was mapped to, the whole mapping is rejected and ``None`` is returned. The
caller then applies nothing and says so, because a chain typed from a mapping
that is off by one is worse than a chain left to the automatic typer.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["unit_provenance", "expand_by_provenance", "ProvenanceError"]


class ProvenanceError(RuntimeError):
    """The chain could not be matched to its monomer."""


#: How many atoms a builder may add after assembly and still leave the chain
#: recognisable as a repetition of its monomer. Converting a terminal -C(=O)H
#: into -C(=O)OH adds one; a generous allowance is still far short of a unit.
_MAX_TERMINAL_EXTRA = 4

#: What such an end cap may be made of. A surplus carbon would mean the chain
#: is a different molecule, not a capped one.
_CAP_ELEMENTS = {"O", "H", "N"}

#: Refinement rounds for the graph match. Each round lets an atom "see" one
#: bond further, so after R rounds two atoms share a colour only if their
#: R-hop neighbourhoods are identical. Distinguishing a chain-end unit from an
#: interior one needs roughly the span of a unit; everything past that only
#: separates interior units from each other, which is wasted work because they
#: are typed identically. Capped so a 200-unit chain costs no more than a
#: 5-unit one.
_REFINE_ROUNDS = 60


def unit_provenance(chain, monomer, n_units: int
                    ) -> Optional[Dict[int, Tuple[int, int]]]:
    """``{chain atom: (unit, monomer atom)}``, or ``None`` if unverifiable.

    ``None`` is a real answer and callers must honour it: it means the chain
    is not a clean repetition of this monomer — a different backend, extra
    end-capping, a copolymer — and any positional mapping would be fiction.
    """
    if n_units < 1 or not getattr(monomer, "molecule", None):
        return None

    per_unit = len(monomer.molecule.atoms)
    heads = sorted(int(h) for h in (monomer.head_removes or []))
    tails = sorted(int(t) for t in (monomer.tail_removes or []))

    # Which of the monomer's own atoms survive in each unit.
    surviving: List[List[int]] = []
    for unit in range(n_units):
        dropped = set()
        if unit > 0:
            dropped.update(heads)          # head cap consumed by the junction
        if unit < n_units - 1:
            dropped.update(tails)          # tail cap consumed by the junction
        surviving.append([k for k in range(per_unit) if k not in dropped])

    expected = sum(len(s) for s in surviving)
    surplus = len(chain.atoms) - expected
    terminal_extra: List[int] = []
    if surplus:
        # A polyester chain is not quite a repetition of its monomer: the
        # builder converts the terminal -C(=O)H into -C(=O)OH once the chain is
        # finished, which adds an atom that belongs to no unit.
        #
        # This used to make the count disagree, so the mapping was refused and
        # manual atom types were silently dropped for EVERY polyester. PBS
        # typed by hand reached the chain not at all — the automatic typer did
        # the whole thing, and nothing said so.
        #
        # A small number of trailing O/H atoms is accepted as that end cap. It
        # is not taken on trust: the fingerprint check below still has to pass,
        # so a chain that differs for any other reason is still refused.
        candidates = list(range(expected, len(chain.atoms)))
        elements = {chain.atoms[i].element for i in candidates}
        if (0 < surplus <= _MAX_TERMINAL_EXTRA
                and elements <= _CAP_ELEMENTS):
            terminal_extra = candidates
            log.info("Chain has %d atoms, %d more than %d units of %s predict. "
                     "Treating the trailing %s as a terminal end cap, which the "
                     "builder adds after assembly.",
                     len(chain.atoms), surplus, n_units,
                     getattr(monomer, "name", "?"),
                     "/".join(sorted(elements)))
        else:
            log.info("Chain has %d atoms; %d units of %s predict %d. Not a "
                     "clean repetition, so no positional mapping is attempted.",
                     len(chain.atoms), n_units,
                     getattr(monomer, "name", "?"), expected)
            return None

    mapping: Dict[int, Tuple[int, int]] = {}
    chain_index = 0
    for unit, keep in enumerate(surviving):
        for monomer_index in keep:
            mapping[chain_index] = (unit, monomer_index)
            chain_index += 1

    # The end-cap atoms belong to no unit, but they ARE atoms in the chain and
    # the user can see and type them in the 3D view, so they need a place in
    # the map. They get the cap "unit" and their position within the cap.
    from .typing_context import CAP_UNIT

    for ordinal, index in enumerate(terminal_extra):
        mapping[index] = (CAP_UNIT, ordinal)

    # Verify on ENVIRONMENT, not just element.
    #
    # Element equality is far too weak: a methyl carbon and a backbone
    # methylene are both "C", so a mapping that swaps them passes an element
    # check and then writes 135 where 136 belongs, in every unit. That is
    # exactly what happened — the chain builder does not emit atoms in the
    # monomer's order, and positional mapping assumed it does.
    #
    # The fingerprint below distinguishes them: element, how many hydrogens,
    # and which heavy atoms it is bonded to. Two atoms with the same
    # fingerprint are interchangeable for typing purposes, which is the
    # property that matters.
    chain_fp = _fingerprints(chain)
    # The reference is the monomer AS IT SITS IN A CHAIN, not in isolation.
    #
    # In the capped monomer, isoprene's two link carbons and its methyl are
    # all ('C', 3, ('C',)) — three indistinguishable CH3 groups, because the
    # links still carry their cap. Only once the cap is replaced by a bond do
    # they become CH2 and separate from the methyl. Fingerprinting the bare
    # monomer therefore cannot tell the very atoms apart that this exists to
    # tell apart.
    #
    # A trimer supplies the answer for all three positions at once: unit 0 is
    # a chain head, unit 1 the repeat unit, unit 2 a tail.
    role_fp = _in_chain_fingerprints(monomer)
    if role_fp is None:
        log.warning("Could not work out the monomer's in-chain environment; "
                    "manual atom types will not be applied.")
        return None

    def _reference(unit: int) -> Dict[int, tuple]:
        if n_units == 1:
            # The only unit keeps both caps, so it is the bare monomer.
            return role_fp["single"]
        if unit == 0:
            return role_fp["head"]
        if unit == n_units - 1:
            return role_fp["tail"]
        return role_fp["middle"]

    monomer_fp = role_fp["middle"]

    # Atoms the end cap CONVERTED in place, and the atoms they are bonded to.
    #
    # The builder turns a terminal -C(=O)H into -C(=O)OH by changing that
    # HYDROGEN into the hydroxyl oxygen where it stands and appending the
    # acid's hydrogen at the end. Two atoms therefore stop matching the
    # monomer, and only one of them is trailing:
    #
    #   * the converted atom itself, H in the monomer and O in the chain;
    #   * the carbon it hangs off, which was an aldehyde carbon (one hydrogen)
    #     and is now an acid carbon (two oxygens).
    #
    # Neither is an error, and refusing the whole mapping over them is what
    # sent every hand-typed polyester down the environment-matching fallback
    # and lost its acid types. Both are recognised here: the converted atom
    # joins the cap, and its neighbour is checked on element only.
    #
    # Only worth looking for when the chain actually HAS surplus atoms. The
    # builder converts and appends together — an in-place conversion with
    # nothing appended is not something ``cap_carboxyl_end`` can produce — and
    # the test below reads each atom's element out of the positional mapping,
    # which on a reordered chain is comparing unrelated atoms. Without this
    # guard a shuffled POM chain (COCOC: no acid, no cap, no surplus) reported
    # "4 atoms were converted in place" and moved four ordinary backbone atoms
    # into the end cap, where nothing could type them.
    converted = [c for c, (u, m) in mapping.items()
                 if terminal_extra and u != CAP_UNIT
                 and chain.atoms[c].element != monomer.molecule.atoms[m].element
                 and chain.atoms[c].element in _CAP_ELEMENTS
                 and monomer.molecule.atoms[m].element in _CAP_ELEMENTS]
    if converted and len(converted) <= _MAX_TERMINAL_EXTRA:
        # Numbered by atom index, not by how they were found.
        #
        # The converted atom comes BEFORE the appended one in the chain, and
        # the typing view lists cap atoms in that same order (the -OH oxygen,
        # then its hydrogen). Numbering the appended atom first swapped the
        # two, and the element guard duly refused to write an oxygen type on a
        # hydrogen -- correctly, but the assignments were then lost.
        terminal_extra = sorted(set(list(terminal_extra) + converted))
        for ordinal, index in enumerate(terminal_extra):
            mapping[index] = (CAP_UNIT, ordinal)
        log.info("%d atom(s) were converted in place by the end cap (a "
                 "terminal -C(=O)H becoming -C(=O)OH changes a hydrogen into "
                 "an oxygen where it stands); counted as cap atoms.",
                 len(converted))

    # Recompute what touches the cap now that the converted atoms are in it.
    touching_cap = set()
    if terminal_extra:
        extra = set(terminal_extra)
        for bond in chain.bonds:
            a, b = int(bond[0]), int(bond[1])
            if a in extra and b not in extra:
                touching_cap.add(b)
            elif b in extra and a not in extra:
                touching_cap.add(a)

    mismatched = [(c, m) for c, (u, m) in mapping.items()
                  if c not in touching_cap and u != CAP_UNIT
                  and chain_fp[c] != _reference(u).get(m)]

    if mismatched:
        log.info("Positional mapping disagrees on %d atoms (e.g. chain %d is "
                 "%s, monomer %d is %s) — the builder did not preserve atom "
                 "order. Re-matching by chemical environment instead.",
                 len(mismatched), mismatched[0][0], chain_fp[mismatched[0][0]],
                 mismatched[0][1], monomer_fp.get(mismatched[0][1]))
        # Connectivity first. It makes no assumption about atom order at all,
        # where the environment repair below still reads each atom's UNIT out
        # of the positional mapping that has just been shown to be wrong.
        rebuilt = _match_by_graph(mapping, chain, monomer, n_units,
                                  chain_fp, _reference, exempt=touching_cap)
        if rebuilt is None:
            rebuilt = _match_by_environment(mapping, chain_fp, _reference,
                                            exempt=touching_cap)
        mapping = rebuilt
        if mapping is None:
            log.warning(
                "The chain could not be matched to its monomer by "
                "connectivity or by chemical environment, so manual atom "
                "types will NOT be applied rather than applied to the wrong "
                "atoms.")
            return None

    log.info("Chain provenance verified: %d atoms across %d units of %s",
             len(mapping), n_units, getattr(monomer, "name", "?"))
    return mapping


def _fingerprints(mol) -> Dict[int, tuple]:
    """A per-atom signature that distinguishes a methyl from a methylene.

    ``(element, hydrogen count, sorted heavy-neighbour elements)``. Two atoms
    sharing a fingerprint sit in the same local environment and therefore want
    the same force-field type — which makes it exactly the right granularity
    for deciding whether a mapping is sound.
    """
    from collections import defaultdict

    neighbours = defaultdict(list)
    for bond in mol.bonds:
        i, j = int(bond[0]), int(bond[1])
        neighbours[i].append(j)
        neighbours[j].append(i)

    out: Dict[int, tuple] = {}
    for a in mol.atoms:
        elements = [mol.atoms[n].element for n in neighbours[a.index]]
        out[a.index] = (a.element,
                        sum(1 for e in elements if e == "H"),
                        tuple(sorted(e for e in elements if e != "H")))
    return out


def _in_chain_fingerprints(monomer) -> Optional[Dict[str, Dict[int, tuple]]]:
    """What each monomer atom looks like once it is *in* a chain.

    Returns ``{"head"/"middle"/"tail": {monomer atom: fingerprint}}``.

    A monomer on its own is the wrong reference. Isoprene's capped repeat unit
    ``CC=C(C)C`` has three carbons that are all ``('C', 3, ('C',))`` — the two
    link carbons still carry their cap hydrogen, so they are indistinguishable
    from the methyl branch. Comparing a chain against that can only ever fail
    on exactly the atoms it exists to tell apart, which is why the methyl and
    the backbone methylene kept swapping types.

    A trimer resolves it without knowing any chemistry: at a junction the cap
    is replaced by a bond, so the middle unit's link carbons become CH2 and
    separate cleanly from the methyl, while the outer units keep their caps and
    describe the real chain ends.
    """
    from .typing_context import build_trimer

    try:
        # Without the terminal cap. This is the reference for the REPEATING
        # part of the chain; the cap is matched separately, and applying it
        # here would change the tail unit's fingerprints and make an uncapped
        # chain fail to match its own monomer.
        ctx = build_trimer([monomer], cap_carboxyl_end=False)
    except Exception as exc:                      # pragma: no cover - defensive
        log.info("Could not build a trimer for %s: %s",
                 getattr(monomer, "name", "?"), exc)
        return None

    trimer_fp = _fingerprints(ctx.molecule)
    role_of = {0: "head", 1: "middle", 2: "tail"}
    out: Dict[str, Dict[int, tuple]] = {"head": {}, "middle": {}, "tail": {}}
    for trimer_index, (unit, monomer_index) in ctx.provenance.items():
        out[role_of[unit]][monomer_index] = trimer_fp[trimer_index]

    # A one-unit chain is head and tail at once and keeps both caps, so the
    # bare monomer is the right reference there and only there.
    out["single"] = _fingerprints(monomer.molecule)
    return out


def _adjacency(bonds, nodes=None) -> Dict[object, List[object]]:
    """``{node: [neighbours]}`` from ``(i, j, order)`` triples.

    Node identity is left alone rather than coerced to ``int``: the chain's
    nodes are atom indices, but the reference graph's are ``(unit, monomer
    atom)`` pairs, and both go through here.
    """
    from collections import defaultdict

    adj: Dict[object, List[object]] = defaultdict(list)
    keep = None if nodes is None else set(nodes)
    for bond in bonds:
        i, j = bond[0], bond[1]
        if keep is not None and (i not in keep or j not in keep):
            continue
        adj[i].append(j)
        adj[j].append(i)
    if keep is not None:
        for n in keep:
            adj.setdefault(n, [])
    return adj


def _reference_topology(mapping: Dict[int, Tuple[int, int]], monomer,
                        n_units: int):
    """The chain's topology with provenance known by construction.

    Nodes are ``(unit, monomer atom)`` pairs, so every node already carries the
    answer we are looking for. Which pairs exist is taken from ``mapping``'s
    own values rather than recomputed: by the time this runs, the cap logic has
    moved any converted atom out of its unit, and the node set has to agree
    with that or the two graphs cannot be isomorphic.

    Bonds are the monomer's own, restricted to surviving atoms, plus one
    junction bond per pair of consecutive units. No geometry is involved —
    this is the connectivity a chain of ``n_units`` MUST have, whatever order
    the builder happened to emit its atoms in.
    """
    from .typing_context import CAP_UNIT

    present = {v for k, v in mapping.items() if v[0] != CAP_UNIT}
    elements = {node: monomer.molecule.atoms[node[1]].element
                for node in present}

    bonds: List[Tuple[object, object, float]] = []
    for unit in range(n_units):
        for bond in monomer.molecule.bonds:
            i, j = int(bond[0]), int(bond[1])
            if (unit, i) in present and (unit, j) in present:
                bonds.append(((unit, i), (unit, j), 1.0))

    head = int(getattr(monomer, "head_index", 0))
    tail = int(getattr(monomer, "tail_index", 0))
    for unit in range(n_units - 1):
        a, b = (unit, tail), (unit + 1, head)
        if a not in present or b not in present:
            return None, None                 # not the chain we think it is
        bonds.append((a, b, 1.0))

    return elements, bonds


def _refine(nodes, adjacency, elements, rounds: int) -> Dict[object, int]:
    """Colour every node by its neighbourhood, refining until stable.

    Round 0 is the element. Each further round replaces a node's colour with
    ``(own colour, sorted neighbour colours)``, compressed back to an integer
    so the tuples cannot grow without bound. This is the standard
    colour-refinement (Weisfeiler-Leman) procedure; all it needs from the
    graph is who is bonded to whom.

    Two nodes end up sharing a colour only if nothing about their surroundings
    tells them apart — the three hydrogens of a methyl, or two interior units
    of the same homopolymer. Those are exactly the cases where it does not
    matter which one you pick.
    """
    colour: Dict[object, object] = {n: elements[n] for n in nodes}
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
            break                             # refinement has converged
        distinct = len(table)
    return colour


def _match_by_graph(mapping: Dict[int, Tuple[int, int]], chain, monomer,
                    n_units: int, chain_fp: Dict[int, tuple], reference,
                    exempt=None) -> Optional[Dict[int, Tuple[int, int]]]:
    """Re-derive provenance from connectivity alone, ignoring atom order.

    Why this replaces the previous repair
    -------------------------------------
    ``_match_by_environment`` fixed up *which monomer atom* a chain atom came
    from, but it read *which unit* it belonged to straight out of the
    positional mapping — the same positional mapping whose failure had just
    triggered the repair. When a builder reorders atoms it does not politely
    keep them inside their own unit, so that unit number is fiction, and every
    atom was then matched against the wrong unit's reference. On a shuffled
    5-unit PBS it refused outright ("Unit 0: no monomer atom with environment
    ('C', 2, ('C', 'C'))"), and refusing means the user's hand-typing is
    dropped.

    Connectivity does not care about ordering. The reference chain is built as
    a graph whose every node already knows its ``(unit, monomer atom)``; both
    graphs are colour-refined; and nodes are paired within matching colours.
    Atom order never enters into it, so a builder is free to emit atoms in any
    order it likes.
    """
    from .typing_context import CAP_UNIT

    cap = {c: v for c, v in mapping.items() if v[0] == CAP_UNIT}
    body = [c for c in mapping if c not in cap]

    ref_elements, ref_bonds = _reference_topology(mapping, monomer, n_units)
    if ref_elements is None:
        log.info("The monomer's own link atoms are missing from the chain, so "
                 "no reference topology could be built.")
        return None
    if len(ref_elements) != len(body):
        log.info("Reference topology has %d atoms, the chain has %d outside "
                 "its end cap — not the same molecule.",
                 len(ref_elements), len(body))
        return None

    chain_elements = {c: chain.atoms[c].element for c in body}
    chain_adj = _adjacency(chain.bonds, nodes=body)
    ref_adj = _adjacency(ref_bonds, nodes=ref_elements)

    # Refined over the two graphs JOINED into one, not separately. Colours are
    # integers handed out in first-seen order, so refining each graph on its
    # own gives two numberings that mean nothing to each other. Run them
    # together — they share no bonds, so neither disturbs the other — and a
    # colour means the same thing on both sides, which is what makes the
    # classes directly comparable below.
    joined_nodes = [("c", c) for c in body] + [("r", n) for n in ref_elements]
    joined_elements = {("c", c): chain_elements[c] for c in body}
    joined_elements.update({("r", n): ref_elements[n] for n in ref_elements})
    joined_adj: Dict[object, List[object]] = {}
    for c in body:
        joined_adj[("c", c)] = [("c", m) for m in chain_adj[c]]
    for n in ref_elements:
        joined_adj[("r", n)] = [("r", m) for m in ref_adj[n]]
    joined = _refine(joined_nodes, joined_adj, joined_elements,
                     _REFINE_ROUNDS)

    from collections import defaultdict
    chain_by_colour: Dict[int, List[int]] = defaultdict(list)
    ref_by_colour: Dict[int, List[object]] = defaultdict(list)
    for c in body:
        chain_by_colour[joined[("c", c)]].append(c)
    for n in ref_elements:
        ref_by_colour[joined[("r", n)]].append(n)

    if set(chain_by_colour) != set(ref_by_colour):
        only = set(chain_by_colour) ^ set(ref_by_colour)
        example = next(iter(only))
        members = (chain_by_colour.get(example) or ref_by_colour.get(example))
        log.info("The chain and a %d-unit reference disagree on %d "
                 "environment classes (e.g. one with %d member(s)) — the "
                 "chain is not this monomer repeated %d times.",
                 n_units, len(only), len(members), n_units)
        return None

    rebuilt: Dict[int, Tuple[int, int]] = {}
    for colour, chain_members in chain_by_colour.items():
        ref_members = ref_by_colour[colour]
        if len(chain_members) != len(ref_members):
            log.info("Environment class %s has %d chain atoms but %d in the "
                     "reference — the chain is not a clean repetition.",
                     colour, len(chain_members), len(ref_members))
            return None
        # Within one class nothing distinguishes the members, so the pairing
        # is free. Sorting both sides only makes it reproducible.
        for c, node in zip(sorted(chain_members), sorted(ref_members)):
            rebuilt[c] = node

    rebuilt.update(cap)

    # Verified the same way the positional mapping is: every atom must look
    # like the monomer atom it was matched to, in the role its unit gives it.
    # Colour refinement is not a proof of isomorphism on its own, so this is
    # the check that decides whether the result is used.
    #
    # The atom the end cap hangs off is exempt, and has to be. The reference
    # has no cap — a terminal -C(=O)H that the builder later turns into
    # -C(=O)OH is still an aldehyde there — so that one carbon reads
    # ('C', 1, ('C', 'O')) in the reference and ('C', 1, ('C', 'O', 'O')) in
    # the real chain. Not a mismatch: the extra O is the cap, and the cap is
    # the thing we already accounted for. Without this the whole 5-unit match
    # was thrown away over a single atom that was matched correctly.
    skip = set(exempt or ())
    wrong = [(c, v) for c, v in rebuilt.items()
             if v[0] != CAP_UNIT and c not in skip
             and chain_fp[c] != reference(v[0]).get(v[1])]
    if wrong:
        c, (unit, m) = wrong[0]
        log.info("Graph match failed verification on %d atoms (e.g. chain %d "
                 "is %s but unit %d atom %d should be %s).",
                 len(wrong), c, chain_fp[c], unit, m,
                 reference(unit).get(m))
        return None

    log.info("Matched %d atoms to their monomer by connectivity (%d end-cap "
             "atoms kept as they were); the builder's atom order was not used.",
             len(rebuilt) - len(cap), len(cap))
    return rebuilt


def _match_by_environment(mapping: Dict[int, Tuple[int, int]],
                          chain_fp: Dict[int, tuple],
                          reference,
                          exempt: Optional[set] = None
                          ) -> Optional[Dict[int, Tuple[int, int]]]:
    """Re-pair each unit's atoms by fingerprint rather than by position.

    Within one unit, chain atoms and monomer atoms are matched on their
    environment signature. Atoms sharing a signature are interchangeable, so
    any pairing among them gives the same types. If a unit's signatures do not
    match as multisets, the chain is not what it claims to be and ``None`` is
    returned.
    """
    from collections import defaultdict

    from .typing_context import CAP_UNIT

    by_unit: Dict[int, List[int]] = defaultdict(list)
    monomer_for_unit: Dict[int, List[int]] = defaultdict(list)
    rebuilt_cap: Dict[int, Tuple[int, int]] = {}
    for c_index, (unit, m_index) in mapping.items():
        # Cap atoms are not part of any unit and were already numbered by
        # position. Re-matching them by environment asks for a reference that
        # does not exist -- the log said "Unit -1: no monomer atom with
        # environment ('O', 1, ('C',))" -- and one unmatchable atom rejected
        # the entire chain. They pass through untouched.
        if unit == CAP_UNIT:
            rebuilt_cap[c_index] = (unit, m_index)
            continue
        by_unit[unit].append(c_index)
        monomer_for_unit[unit].append(m_index)

    rebuilt: Dict[int, Tuple[int, int]] = {}
    for unit, chain_indices in by_unit.items():
        unit_fp = reference(unit)
        pool: Dict[tuple, List[int]] = defaultdict(list)
        for m_index in monomer_for_unit[unit]:
            signature = unit_fp.get(m_index)
            if signature is None:
                log.info("Unit %d: monomer atom %d has no in-chain "
                         "environment", unit, m_index)
                return None
            pool[signature].append(m_index)

        # Atoms the end cap touched are matched LAST and loosely.
        #
        # The cap changes the environment of the atom it hangs off -- an
        # aldehyde carbon becomes an acid carbon -- so its environment does not
        # exist anywhere in the monomer reference. Requiring an exact match
        # there made one atom reject the whole chain, which is how a 99-atom
        # PBS lost every hand-assigned type. Matching it by element among
        # whatever is left is enough: by then it is the only candidate of its
        # element in that unit.
        deferred = [c for c in sorted(chain_indices) if c in (exempt or set())]
        for c_index in sorted(chain_indices):
            if c_index in deferred:
                continue
            candidates = pool.get(chain_fp[c_index])
            if not candidates:
                log.info("Unit %d: no monomer atom with environment %s for "
                         "chain atom %d", unit, chain_fp[c_index], c_index)
                return None
            rebuilt[c_index] = (unit, candidates.pop())

        for c_index in deferred:
            element = chain_fp[c_index][0]
            leftover = [m for fp, ms in pool.items() if fp[0] == element
                        for m in ms]
            if not leftover:
                log.info("Unit %d: nothing left for cap-adjacent atom %d (%s)",
                         unit, c_index, element)
                return None
            chosen = leftover[0]
            for fp, ms in pool.items():
                if chosen in ms:
                    ms.remove(chosen)
                    break
            rebuilt[c_index] = (unit, chosen)

    rebuilt.update(rebuilt_cap)
    log.info("Re-matched %d atoms by chemical environment (%d end-cap atoms "
             "kept as they were)", len(rebuilt) - len(rebuilt_cap),
             len(rebuilt_cap))
    return rebuilt


def expand_by_provenance(chain, manual_types: Dict[int, str],
                         provenance: Dict[int, Tuple[int, int]],
                         *, type_elements: Optional[Dict[str, str]] = None,
                         head_types: Optional[Dict[int, str]] = None,
                         tail_types: Optional[Dict[int, str]] = None,
                         cap_types: Optional[Dict[int, str]] = None,
                         n_units: int = 0) -> Dict[int, str]:
    """Place a monomer's types onto every unit, keyed by chain atom.

    Every atom the map covers gets the type its monomer counterpart was
    given — exactly, with no inference. An assignment whose element does not
    match the atom it would land on is dropped and reported; that can only
    happen if the caller's map is inconsistent, and dropping is safer than
    writing a carbon type onto a hydrogen.
    """
    from .type_guard import check_assignment
    from .type_guard import type_elements as _library_elements
    from .typing_context import CAP_UNIT

    table = type_elements if type_elements is not None else _library_elements()

    if n_units <= 0 and provenance:
        units = [u for u, _k in provenance.values() if u != CAP_UNIT]
        n_units = (max(units) + 1) if units else 0

    out: Dict[int, str] = {}
    refused: List[str] = []
    for chain_index, (unit, monomer_index) in provenance.items():
        # A chain has three kinds of unit. The first keeps its head cap and the
        # last keeps its tail cap, so their link atoms have an extra hydrogen
        # and are a different force-field type from the repeat unit — a
        # terminal CH3 rather than a backbone CH2, say. Which set applies is
        # decided by position, not by chemistry, so this holds for any polymer.
        if unit == CAP_UNIT:
            # An end-cap atom. Its "monomer index" is its position in the cap,
            # matching the order the typing view listed them in.
            value = (cap_types or {}).get(monomer_index)
            if not value:
                continue
            element = chain.atoms[chain_index].element
            problem = check_assignment(element, value, table)
            if problem:
                refused.append(f"chain atom {chain_index} ({element}): {problem}")
                continue
            out[chain_index] = value
            continue

        table_for_unit = manual_types
        if unit == 0 and head_types:
            table_for_unit = head_types
        elif n_units > 1 and unit == n_units - 1 and tail_types:
            table_for_unit = tail_types
        value = table_for_unit.get(monomer_index) or manual_types.get(monomer_index)
        if not value:
            continue
        element = chain.atoms[chain_index].element
        problem = check_assignment(element, value, table)
        if problem:
            refused.append(f"chain atom {chain_index} ({element}): {problem}")
            continue
        out[chain_index] = value

    if refused:
        log.warning("Refused %d manual type assignments:\n  %s",
                    len(refused), "\n  ".join(refused[:5]))
    log.info("Applied manual atom types to %d of %d chain atoms "
             "(%d repeat-unit types%s)",
             len(out), len(chain.atoms), len(manual_types),
             "; separate head/tail sets for the chain ends"
             if (head_types or tail_types) else "")
    return out


def _deep_fingerprints(mol) -> Dict[int, tuple]:
    """A two-shell signature: the atom, plus each neighbour's own signature.

    The one-shell fingerprint cannot tell a carboxylic acid's -O-H from an
    alcohol's -O-H: both are ('O', 1 hydrogen, bonded to one carbon). That is
    exactly the pair that matters for a polyester, and matching on it gave a
    PBS chain the alcohol's types at BOTH ends -- the acid types the user
    assigned never appeared in the data file at all.

    Looking one bond further apart separates them, because the acid oxygen's
    carbon is a carbonyl (bonded to two oxygens) and the alcohol oxygen's is a
    CH2.
    """
    from collections import defaultdict

    shallow = _fingerprints(mol)
    neighbours = defaultdict(list)
    for bond in mol.bonds:
        i, j = int(bond[0]), int(bond[1])
        neighbours[i].append(j)
        neighbours[j].append(i)
    return {a.index: (shallow[a.index],
                      tuple(sorted(shallow[n] for n in neighbours[a.index])))
            for a in mol.atoms}


def types_by_environment(chain, monomer, by_role: Dict[str, Dict[int, str]],
                         *, type_elements: Optional[Dict[str, str]] = None
                         ) -> Dict[int, str]:
    """Type a chain by matching each atom's environment, with no unit mapping.

    The fallback for when the exact positional mapping is refused — a chain
    that is not a clean repetition of its monomer because the builder added a
    terminal group after assembly, say.

    The all-or-nothing alternative is worse than it sounds. The typing dialog
    seeds every atom from the automatic typer and the user overrides only the
    ones it got wrong, so the map handed over is COMPLETE. Discarding it
    discards the automatic types too, and the chain is retyped from scratch by
    the very typer whose answers were being corrected.

    Matching on environment cannot say which unit an atom is in, so it uses
    the repeat unit's types and leaves anything it cannot match to the caller.
    That is a real limitation: an atom whose environment exists only at a chain
    end gets nothing here. It is still far better than nothing, because the
    interior is almost all of a chain.
    """
    from .type_guard import check_assignment
    from .type_guard import type_elements as _library_elements

    table = type_elements if type_elements is not None else _library_elements()
    role_fp = _deep_in_chain_fingerprints(monomer)
    if role_fp is None:
        return {}

    # One lookup from environment to type. The ENDS go in first and the repeat
    # unit last, so where two roles genuinely share an environment the repeat
    # unit wins.
    #
    # That ordering is not arbitrary. Some environments are identical however
    # deep you look: an ester =O and a carboxylic acid =O are both an oxygen
    # double-bonded to a carbon that carries a carbon and two oxygens. Letting
    # the end win there put the acid's type on all six ester carbonyls of a
    # PBS trimer -- one atom's worth of end correctness bought at the cost of
    # six interior atoms. The repeat unit is the safer default because it is
    # the overwhelming majority of any chain.
    #
    # Environments that ARE distinguishable -- the acid's -OH oxygen and
    # carbon, which differ from the alcohol's and the ester's two bonds out --
    # still get their end types, because the repeat unit never claims them.
    wanted: Dict[tuple, str] = {}
    ambiguous: List[str] = []
    for role in ("head", "tail", "cap", "middle"):
        reference = role_fp.get(role) or {}
        for monomer_index, value in (by_role.get(role) or {}).items():
            signature = reference.get(monomer_index)
            if signature is None or not value:
                continue
            previous = wanted.get(signature)
            if previous is not None and previous != value and role == "middle":
                ambiguous.append(f"{previous} vs {value}")
            wanted[signature] = value
    if ambiguous:
        log.info("Some chain-end environments are indistinguishable from the "
                 "repeat unit's (%s); the repeat unit's type was used, so "
                 "check those atoms at the chain ends.",
                 ", ".join(sorted(set(ambiguous))[:4]))

    chain_fp = _deep_fingerprints(chain)
    out: Dict[int, str] = {}
    for atom in chain.atoms:
        value = wanted.get(chain_fp.get(atom.index))
        if not value:
            continue
        if check_assignment(atom.element, value, table):
            continue
        out[atom.index] = value

    log.warning(
        "The chain could not be mapped to its monomer exactly, so atom types "
        "were matched by chemical environment instead: %d of %d atoms took a "
        "type you assigned. The rest are left to the automatic typer. Check "
        "the chain ends in particular.",
        len(out), len(chain.atoms))
    return out


def _deep_in_chain_fingerprints(monomer) -> Optional[Dict[str, Dict[int, tuple]]]:
    """Two-shell in-chain signatures per role, INCLUDING the end cap.

    Two things the one-shell version could not do, both of which cost the user
    their acid end:

    * it could not tell an acid -O-H from an alcohol -O-H, so a PBS chain got
      the alcohol's types at both ends;
    * it had no ``cap`` entry at all, so types assigned to the terminal -COOH
      had nothing to match against and were dropped silently.

    The trimer is built WITH the carboxyl cap here, which is what supplies the
    cap signatures.
    """
    from .typing_context import CAP_UNIT, build_trimer

    try:
        ctx = build_trimer([monomer], cap_carboxyl_end=True)
    except Exception as exc:                      # pragma: no cover - defensive
        log.info("Could not build a trimer for %s: %s",
                 getattr(monomer, "name", "?"), exc)
        return None

    trimer_fp = _deep_fingerprints(ctx.molecule)
    role_of = {0: "head", 1: "middle", 2: "tail", CAP_UNIT: "cap"}
    out: Dict[str, Dict[int, tuple]] = {r: {} for r in role_of.values()}
    for trimer_index, (unit, monomer_index) in ctx.provenance.items():
        role = role_of.get(unit)
        if role is not None:
            out[role][monomer_index] = trimer_fp[trimer_index]
    return out
