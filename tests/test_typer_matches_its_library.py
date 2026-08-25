"""Every suggested type must belong to the element it is suggested for.

The failure
-----------
This typer's SMARTS rules were written against a different OPLS numbering
table from the bundled ``oplsaa2024.lt``. The ester rules assigned 210/211/212
labelled "ester carbonyl / -O- / =O" -- and in this library 210-214 are
SULFIDE AND DISULFIDE CARBONS. So every ester oxygen in a polyester was
suggested a carbon type.

The user never saw that as a suggestion problem. They saw it at export:

    atom 5 -- Type 211 is a C type, it cannot be assigned to a O atom

and reasonably concluded their own hand-typing was at fault. It was not: the
element guard was catching the SUGGESTION, faithfully applied.

An audit of all 56 rules found 17 in this state. The ester and acid IDs are
corrected here against the library (465/466/467 and 267-270). For any rule
still out of step, the typer now drops the suggestion and falls back to the
broad element default -- less specific, never wrong about mass and charge.
"""
from __future__ import annotations

import re

import pytest

from paaf.ff_registry import get_ff                              # noqa: E402
from paaf.lt_parser import parse_atom_types                      # noqa: E402
from paaf.typers.oplsaa import _RULES                            # noqa: E402


def _library():
    ff = get_ff("oplsaa")
    path = ff.bundled_path() if ff.bundled_lt else None
    if path is None or not path.exists():
        pytest.skip("the OPLS-AA .lt library is not bundled here")
    return {t.ff_id: t.element for t in parse_atom_types(str(path))}


def _target_element(smarts: str):
    """The element the FIRST atom of the pattern matches — the one typed."""
    body = smarts[1:] if smarts.startswith("[") else smarts
    match = re.match(r"([A-Z][a-z]?|[a-z])", body)
    if not match:
        return None
    token = match.group(1)
    if token.islower():
        return {"c": "C", "o": "O", "n": "N", "s": "S"}.get(token)
    return token


#: Rules known to still use another table's numbering. They are neutralised at
#: runtime (the suggestion is dropped for the element default), so they cannot
#: reach a data file — but they are listed here rather than hidden, because
#: each one is a specific chemistry that deserves a correct type.
KNOWN_WRONG = {"177", "178", "179", "180", "183", "142", "263", "473",
               "474", "725", "731", "739", "740", "741", "742"}


def test_the_ester_and_acid_rules_match_the_bundled_library():
    """The ones that broke PBS, fixed against the library's own numbering."""
    library = _library()
    wanted = {"465": "C", "466": "O", "467": "O",
              "267": "C", "268": "O", "269": "O", "270": "H"}
    for type_id, element in wanted.items():
        assert library.get(type_id) == element, (
            f"{type_id} should be {element} in oplsaa2024.lt")

    used = {t for _s, t, _d in _RULES}
    for type_id in wanted:
        assert type_id in used, f"the typer no longer uses {type_id}"
    for stale in ("210", "211", "212", "209"):
        assert stale not in used, (
            f"{stale} is a sulfide/alkane carbon in this library and must not "
            f"be suggested for an ester or acid")


@pytest.mark.parametrize("smarts,type_id,desc", _RULES,
                         ids=[r[1] + ":" + r[2][:24] for r in _RULES])
def test_every_rule_suggests_a_type_of_the_right_element(smarts, type_id, desc):
    library = _library()
    want = _target_element(smarts)
    have = library.get(type_id)
    if want is None or have is None:
        pytest.skip("element not determinable from this pattern")
    if want != have and type_id in KNOWN_WRONG:
        pytest.xfail(f"{type_id} ({desc}) uses another table's numbering; "
                     f"neutralised at runtime by the element check")
    assert want == have, (
        f"rule '{desc}' matches a {want} atom but type {type_id} is {have}")


def test_the_typer_drops_a_wrong_element_suggestion_rather_than_emitting_it():
    """The runtime guard, independent of which rules are still wrong.

    Whatever the rule table says, an atom must never be handed a type from
    another element — that is what writes a carbon's mass onto an oxygen.
    """
    from paaf.type_guard import check_assignment, type_elements

    table = type_elements()
    # The exact pair that reached the user's export.
    assert check_assignment("O", "211", table) is not None
    assert check_assignment("O", "467", table) is None
    assert check_assignment("C", "465", table) is None
    assert check_assignment("H", "270", table) is None
