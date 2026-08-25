"""Expand a polymerization SMILES into an n-mer.

The reaction scheme lets a user pick a polymer from the library — which is
stored as a *polymerization* SMILES with two connection points, e.g. PE's
``[*]CC[*]`` — and say how many repeat units the chain should have. This module
turns that into a plain n-mer SMILES that RDKit can parse.

**Why this exists.** Writing a 10-unit PBS chain by hand is unreasonable, and
worse, a reaction must use the *same* chain length on both sides: if the
reactant is a 10-mer, the product has to be that same 10-mer plus whatever
reacted onto it. Expanding both sides from one shared ``n`` makes that
automatic instead of something the user has to count.

**Atom maps in a repeat unit.** A reaction happens at *one* site, not once per
unit. If the user writes ``[*]CC(=O)[OH:1][*]`` and asks for 10 units, naively
repeating it would produce ten copies of map ``:1`` — a duplicate-map error.
So the maps are kept on exactly one unit (the last by default, i.e. the chain
end) and stripped from the rest.
"""
from __future__ import annotations

import re
from typing import List

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["is_polymer_smiles", "strip_atom_maps", "expand_repeat_unit",
           "expand_if_polymer"]


_WILDCARD = re.compile(r"\[\*\]|(?<![\[\w])\*")
# ``[OH:1]`` -> groups: inner "OH", map "1"
_MAPPED_ATOM = re.compile(r"\[([^\[\]]*?):(\d+)\]")


def is_polymer_smiles(smiles: str) -> bool:
    """True when the SMILES carries ``[*]`` polymer connection points."""
    return bool(_WILDCARD.search(smiles or ""))


def strip_atom_maps(smiles: str) -> str:
    """Remove ``:n`` map numbers, keeping the atom itself.

    ``[OH:1]`` -> ``[OH]``. Brackets are retained because dropping them can
    change the implied hydrogen count and therefore the molecule.

    >>> strip_atom_maps("CC(=O)[OH:1]")
    'CC(=O)[OH]'
    >>> strip_atom_maps("[C:2]CO")
    '[C]CO'
    """
    return _MAPPED_ATOM.sub(lambda m: f"[{m.group(1)}]", smiles or "")


# ``[*]CC([*])c1ccccc1`` — the second connection point sits in a BRANCH.
_BRANCH_FORM = re.compile(r"^\[\*\](?P<head>.*?)\(\[\*\]\)(?P<tail>.+)$")
_TERMINAL_FORM = re.compile(r"^\[\*\](?P<core>.*)\[\*\]$")


def _peel_leading_branches(tail: str) -> tuple:
    """Split ``tail`` into its leading balanced ``(...)`` groups and the rest.

    ``"(C)C(=O)OC"`` -> ``("(C)", "C(=O)OC")``;
    ``"c1ccccc1"``   -> ``("", "c1ccccc1")``;
    ``"(C)(C)"``     -> ``("(C)(C)", "")``.

    Bracket counting, not a regular expression: a branch may itself contain
    branches, and ``(C(=O)OC)`` must not be split at its inner parenthesis.
    """
    branches = []
    i = 0
    n = len(tail)
    while i < n and tail[i] == "(":
        depth = 0
        j = i
        while j < n:
            if tail[j] == "(":
                depth += 1
            elif tail[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j >= n:
            break                      # unbalanced; leave the rest alone
        branches.append(tail[i:j + 1])
        i = j + 1
    return "".join(branches), tail[i:]


def _repeat_core(poly_smiles: str) -> str:
    """The repeat unit, rewritten so that repeating it is chemically correct.

    Two layouts appear in the library and they must be treated differently:

    ``[*]CC[*]`` (terminal)
        The connection points are the first and last atoms, so the core is
        just the middle and ``core * n`` chains head-to-tail correctly.

    ``[*]CC([*])c1ccccc1`` (branch — vinyl polymers)
        The second connection point is a *branch* off the second carbon, and
        the phenyl is a substituent. Naively deleting the wildcards gives
        ``CCc1ccccc1``; repeating that would put the phenyl ring **in the
        backbone**. The substituent must be moved into parentheses so the
        chain continues from the substituted carbon:
        ``CC(c1ccccc1)`` → ``CC(c1ccccc1)CC(c1ccccc1)`` = polystyrene.

    Disubstituted units need more care
    ----------------------------------
    A quaternary backbone carbon carries *two* substituents, and the tail then
    already begins with a branch of its own. Poly(methyl methacrylate) is
    ``[*]CC([*])(C)C(=O)OC``, whose tail is ``(C)C(=O)OC``. Wrapping that
    whole string in another pair of parentheses gives ``CC((C)C(=O)OC)`` ---
    a doubled bracket that is not valid SMILES at all, so the unit cannot even
    be parsed. Poly(dimethylsiloxane) ``[*][Si]([*])(C)C`` fails the same way.

    The fix is to peel the leading balanced branch groups off the tail, keep
    them as the branches they already are, and parenthesise only what remains:
    ``CC`` + ``(C)`` + ``(C(=O)OC)``.
    """
    smiles = (poly_smiles or "").strip()

    m = _BRANCH_FORM.match(smiles)
    if m:
        head, tail = m.group("head"), m.group("tail")
        branches, rest = _peel_leading_branches(tail)
        if not rest:
            # Everything after the second [*] was already branches; the
            # backbone simply continues from the substituted atom.
            return f"{head}{branches}"
        return f"{head}{branches}({rest})"

    m = _TERMINAL_FORM.match(smiles)
    if m:
        return m.group("core")

    # Anything else: strip the markers and hope the wildcards were terminal.
    log.warning("Unrecognised polymerization SMILES layout %r; expanding by "
                "simple repetition, which may be wrong for branched units.",
                smiles)
    return _WILDCARD.sub("", smiles).replace("()", "")


def expand_repeat_unit(poly_smiles: str, n: int,
                       maps_on: str = "last") -> str:
    """Repeat a ``[*]…[*]`` unit ``n`` times into a plain SMILES.

    Parameters
    ----------
    poly_smiles :
        Polymerization SMILES, e.g. ``"[*]CC[*]"`` or ``"[*]OCCCCOC(=O)CCC(=O)[*]"``.
    n :
        Number of repeat units. ``n <= 1`` returns the single unit unchanged.
    maps_on :
        Which unit keeps its atom-map numbers — ``"last"`` (the growing chain
        end, the usual reactive site), ``"first"``, or ``"none"``.

    >>> expand_repeat_unit("[*]CC[*]", 3)
    'CCCCCC'
    >>> expand_repeat_unit("[*]CC(=O)[OH:1][*]", 2)
    'CC(=O)[OH]CC(=O)[OH:1]'
    """
    if n is None or n < 1:
        n = 1
    core = _repeat_core(poly_smiles)
    if not core:
        return poly_smiles or ""
    if n == 1:
        return core if maps_on != "none" else strip_atom_maps(core)

    bare = strip_atom_maps(core)
    if maps_on == "none":
        return bare * n
    units: List[str] = [bare] * n
    keep = n - 1 if maps_on == "last" else 0
    units[keep] = core
    return "".join(units)


def expand_if_polymer(smiles: str, n: int, maps_on: str = "last") -> str:
    """Expand only when the SMILES actually has ``[*]`` points.

    Small molecules (water, maleic anhydride) pass through untouched, so a
    single code path can handle both a polymer chain and its co-reactant.
    """
    if not is_polymer_smiles(smiles):
        return smiles or ""
    return expand_repeat_unit(smiles, n, maps_on=maps_on)
