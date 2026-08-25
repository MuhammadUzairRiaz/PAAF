"""Switching monomers must not make the typing dialog unusable.

The report
----------
A user typed PBS, then chose polybutadiene and pressed *Apply all
assignments*. Nothing was applied. Instead::

    These assignments would give an atom the mass and charge of a different
    element, so nothing has been applied:

    atom 13: Type 140 is a H type — it cannot be assigned to a ? atom.
    ...and 37 more.

Every visible row was typed correctly. The complaint was not about any of
them: butadiene's trimer has 32 atoms, and the offending indices ran to 18 in
a monomer numbered 0-11. They were left over from PBS, whose monomer has 26
atoms, and they were invisible — no row showed them, so no amount of fixing
the table could clear the message.

Two things were wrong, and both are pinned here.

``check_all`` compares an atom's element against its type's element. Where
there is no atom there is no element, and ``elements.get(index, "")`` turned
that absence into the string ``"?"`` and then into a mismatch. An index that
is not in the structure is not a wrong assignment; it is not an assignment.

The dialog kept ``_types_by_atom`` across a change of monomer. It should not:
those keys address a different molecule. They are dropped, and the user is
told, because they were the user's assignments.
"""
from __future__ import annotations

import pytest

from paaf.type_guard import check_all, drop_unknown              # noqa: E402

#: Polybutadiene, the molecule in the report. Twelve atoms, 0-11.
BUTADIENE = ("CC=CC", 0, 3)

#: What PBS left behind: indices valid in a 26-atom monomer, not in a 12-atom
#: one. 140 is a hydrogen type, so these would each read "? atom".
STALE = {13: "140", 14: "140", 15: "140", 16: "140", 17: "140", 18: "140"}


# ================================================ the guard, on its own
def test_an_index_with_no_atom_is_not_reported_as_a_mismatch():
    """The exact shape of the report: H types on indices that do not exist."""
    elements = {0: "C", 1: "H"}
    types = {0: "135", 1: "140", 13: "140", 14: "140", 18: "140"}
    problems = check_all(elements, types)
    print(f"\n  {len(problems)} problems from {len(types)} assignments")
    assert problems == [], \
        "assignments for atoms outside the structure were reported as errors"


def test_a_real_mismatch_is_still_reported():
    """The guard must not have been switched off to fix the above."""
    problems = check_all({0: "H"}, {0: "135"})       # 135 is a carbon type
    print(f"\n  {problems}")
    assert problems, "a carbon type on a hydrogen went unreported"
    assert problems[0][0] == 0


def test_a_mismatch_is_reported_even_beside_unknown_indices():
    """The stale keys must not mask a genuine error hiding among them."""
    problems = check_all({0: "H"}, {0: "135", 13: "140", 14: "140"})
    print(f"\n  {problems}")
    assert [i for i, _m in problems] == [0]


def test_an_empty_element_is_treated_as_unknown_not_as_a_clash():
    """An atom whose element could not be read cannot be judged either way."""
    assert check_all({0: ""}, {0: "135"}) == []


# ================================================ the pruning rule itself
def test_stale_assignments_are_separated_from_live_ones():
    live = {0: "143", 1: "142"}
    kept, dropped = drop_unknown({**STALE, **live}, known=range(12))
    print(f"\n  kept {sorted(kept)}, dropped {dropped}")
    assert kept == live
    assert dropped == sorted(STALE)


def test_nothing_is_dropped_when_every_key_has_an_atom():
    types = {0: "143", 5: "144"}
    kept, dropped = drop_unknown(types, known=range(12))
    assert kept == types and dropped == []


def test_the_original_map_is_not_modified_in_place():
    """The caller decides whether to commit the pruning, and says so first."""
    types = dict(STALE)
    drop_unknown(types, known=range(12))
    assert types == STALE, "drop_unknown mutated its argument"


def test_role_encoded_keys_survive_the_pruning():
    """Head and tail rows carry keys far above any monomer index.

    Treating a large key as stale because it exceeds the atom count would
    silently discard every assignment made on a chain end — the very rows
    that were added so the ends could be typed at all.
    """
    from paaf.typing_context import role_key

    head, tail = role_key("head", 4), role_key("tail", 4)
    known = [0, 1, 2, head, tail]
    kept, dropped = drop_unknown({head: "135", tail: "135", 99: "140"}, known)
    print(f"\n  kept {sorted(kept)}, dropped {dropped}")
    assert set(kept) == {head, tail}
    assert dropped == [99]
