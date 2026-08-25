"""Manual atom types must reach a polyester chain, cap and all.

The failure
-----------
A polyester is not quite a repetition of its monomer. PAAF links units by
deleting one hydrogen from each end, which needs the acid end presented as an
aldehyde ``-C(=O)H``; the real ``-C(=O)OH`` is restored **after** assembly by
``cap_carboxyl_end``. That adds atoms belonging to no unit.

``unit_provenance`` compared the chain's atom count against the count that N
units predict, found it one too many, and refused the mapping. Refusing is the
right instinct — a mapping that is off by one is worse than none — but here it
meant that for EVERY polyester the user's hand-typing was silently discarded
and the automatic typer did the whole chain. Measured on a five-unit PBS: 0 of
123 atoms took a manual type, and nothing in the output said so.

A small trailing run of O/H atoms is now recognised as the end cap. It is not
taken on trust: the fingerprint verification still has to pass, so a chain that
differs for any other reason is refused exactly as before.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from paaf.chain_builder import simple_backend                    # noqa: E402
from paaf.chain_provenance import (                              # noqa: E402
    _CAP_ELEMENTS, _MAX_TERMINAL_EXTRA, expand_by_provenance, unit_provenance,
)
from paaf.structure import Atom, Molecule                        # noqa: E402

rdkit = pytest.importorskip("rdkit", reason="needs RDKit")
from rdkit import Chem                                           # noqa: E402
from rdkit.Chem import AllChem                                   # noqa: E402

#: PBS as PAAF must present it for linking: alcohol at one end, aldehyde at
#: the other. The aldehyde H is the removable cap, not a chemistry claim.
PBS = "OCCCCOC(=O)CCC(=O)"


def _pbs_monomer():
    from paaf.monomer import Monomer

    m = Chem.AddHs(Chem.MolFromSmiles(PBS))
    AllChem.EmbedMolecule(m, randomSeed=1)
    AllChem.UFFOptimizeMolecule(m)
    conf = m.GetConformer()
    mol = Molecule(
        atoms=[Atom(index=i, element=a.GetSymbol(),
                    xyz=np.array([conf.GetAtomPosition(i).x,
                                  conf.GetAtomPosition(i).y,
                                  conf.GetAtomPosition(i).z]))
               for i, a in enumerate(m.GetAtoms())],
        bonds=[(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                b.GetBondTypeAsDouble()) for b in m.GetBonds()],
        name="pbs")

    def cap(k):
        return [n.GetIdx() for n in m.GetAtomWithIdx(k).GetNeighbors()
                if n.GetSymbol() == "H"][0]

    return Monomer(mol, 0, 10, [cap(0)], [cap(10)], name="pbs")


def _with_carboxyl_cap(chain: Molecule, extra=(("O", 1.34), ("H", 0.97))
                       ) -> Molecule:
    """What ``cap_carboxyl_end`` does: -C(=O)H becomes -C(=O)OH.

    Reproduced rather than called, because the real one needs mBuild. What
    matters here is the shape of the result: a couple of O/H atoms appended
    past the end of the repeating pattern.
    """
    out = Molecule(atoms=[Atom(index=a.index, element=a.element,
                               xyz=a.xyz.copy()) for a in chain.atoms],
                   bonds=list(chain.bonds), name="capped")
    anchor = max(i for i, a in enumerate(out.atoms) if a.element == "C")
    for element, offset in extra:
        new = len(out.atoms)
        out.atoms.append(Atom(index=new, element=element,
                              xyz=out.atoms[anchor].xyz
                              + np.array([offset, 0.0, 0.0])))
        out.bonds.append((anchor, new, 1.0))
        anchor = new
    return out


def _types(monomer):
    by_element = {"C": "136", "O": "108", "H": "140"}
    return {a.index: by_element[a.element] for a in monomer.molecule.atoms}


# =========================================== the cap no longer blocks typing
@pytest.mark.parametrize("n_units", [2, 5, 12])
def test_a_capped_polyester_chain_still_maps_to_its_monomer(n_units):
    mono = _pbs_monomer()
    capped = _with_carboxyl_cap(simple_backend([mono], [0] * n_units,
                                               conformer_seed=1))
    mapping = unit_provenance(capped, mono, n_units)
    print(f"\n  {n_units} units: {len(capped.atoms)} atoms -> "
          f"{len(mapping) if mapping else 0} mapped")
    assert mapping is not None, "the end cap made the chain unrecognisable"
    # Every atom is accounted for, cap included: the cap atoms belong to no
    # unit but they are still atoms the user can see and type.
    assert len(mapping) == len(capped.atoms)


def test_manual_types_reach_a_capped_pbs_chain():
    """The measured failure: 0 of 123 atoms took a manual type."""
    mono = _pbs_monomer()
    capped = _with_carboxyl_cap(simple_backend([mono], [0] * 5,
                                               conformer_seed=1))
    mapping = unit_provenance(capped, mono, 5)
    applied = expand_by_provenance(capped, _types(mono), mapping, n_units=5)
    print(f"\n  {len(applied)}/{len(capped.atoms)} atoms typed")
    assert len(applied) > 0, "manual types were dropped again"
    # The two cap atoms are typed only if cap types were supplied; none were.
    assert len(applied) == len(capped.atoms) - 2


def test_the_cap_atoms_themselves_are_left_untyped():
    """They have no monomer counterpart, so there is nothing to copy."""
    mono = _pbs_monomer()
    capped = _with_carboxyl_cap(simple_backend([mono], [0] * 4,
                                               conformer_seed=1))
    mapping = unit_provenance(capped, mono, 4)
    applied = expand_by_provenance(capped, _types(mono), mapping, n_units=4)
    cap_atoms = set(range(len(capped.atoms) - 2, len(capped.atoms)))
    assert not (cap_atoms & set(applied)), \
        "a cap atom was given a type it cannot have"


def test_no_atom_is_typed_as_the_wrong_element():
    from paaf.type_guard import check_all

    mono = _pbs_monomer()
    capped = _with_carboxyl_cap(simple_backend([mono], [0] * 4,
                                               conformer_seed=1))
    mapping = unit_provenance(capped, mono, 4)
    applied = expand_by_provenance(capped, _types(mono), mapping, n_units=4)
    problems = check_all({a.index: a.element for a in capped.atoms}, applied)
    print(f"\n  element problems: {problems or 'none'}")
    assert not problems


def test_the_atom_next_to_the_cap_does_not_invalidate_the_whole_mapping():
    """Its environment changed — that is what a cap IS — so it is exempt."""
    mono = _pbs_monomer()
    capped = _with_carboxyl_cap(simple_backend([mono], [0] * 3,
                                               conformer_seed=1))
    mapping = unit_provenance(capped, mono, 3)
    assert mapping is not None
    anchor = max(i for i, a in enumerate(capped.atoms)
                 if a.element == "C" and i in mapping)
    assert anchor in mapping, "the carbon bearing the cap was dropped"


# ================================= but a genuinely different chain is refused
def test_a_surplus_carbon_is_still_refused():
    """Only an O/H end cap is plausible; a carbon means another molecule."""
    mono = _pbs_monomer()
    chain = simple_backend([mono], [0] * 4, conformer_seed=1)
    wrong = _with_carboxyl_cap(chain, extra=(("C", 1.54),))
    print(f"\n  surplus carbon -> "
          f"{'mapped' if unit_provenance(wrong, mono, 4) else 'refused'}")
    assert unit_provenance(wrong, mono, 4) is None


def test_too_many_extra_atoms_are_refused():
    """A whole extra group is not an end cap."""
    mono = _pbs_monomer()
    chain = simple_backend([mono], [0] * 4, conformer_seed=1)
    many = _with_carboxyl_cap(
        chain, extra=tuple(("H", 1.0) for _ in range(_MAX_TERMINAL_EXTRA + 1)))
    assert unit_provenance(many, mono, 4) is None


def test_a_wrong_unit_count_is_still_refused():
    """The tolerance must not turn into 'close enough'."""
    mono = _pbs_monomer()
    capped = _with_carboxyl_cap(simple_backend([mono], [0] * 4,
                                               conformer_seed=1))
    assert unit_provenance(capped, mono, 9) is None
    assert unit_provenance(capped, mono, 3) is None


def test_the_allowance_is_small_and_made_of_cap_elements_only():
    """Guard the constants themselves: a lax value would hide real errors."""
    print(f"\n  up to {_MAX_TERMINAL_EXTRA} atoms of {sorted(_CAP_ELEMENTS)}")
    assert _MAX_TERMINAL_EXTRA <= 4
    assert "C" not in _CAP_ELEMENTS


def test_an_uncapped_chain_is_unaffected():
    """The common case must behave exactly as it did."""
    mono = _pbs_monomer()
    chain = simple_backend([mono], [0] * 5, conformer_seed=1)
    mapping = unit_provenance(chain, mono, 5)
    assert mapping is not None and len(mapping) == len(chain.atoms)


# ============================================ the library hands over a
# ============================================ polymerisable monomer
def test_the_library_never_offers_an_unpolymerisable_monomer():
    """PBS's curated entry stores the acid form, which cannot be linked."""
    from paaf.builder import LIBRARY, _polymerisable_form

    checked = 0
    for key, rec in LIBRARY.items():
        if "[*]" not in (rec.smiles or ""):
            continue
        form = _polymerisable_form(rec)
        assert "[*]" not in form, f"{key}: wildcards left in the built form"
        checked += 1
    print(f"\n  {checked} library entries")
    assert checked > 50


def test_pbs_is_offered_in_its_linkable_form_not_its_acid_form():
    from paaf.builder import LIBRARY, _polymerisable_form

    for key in ("PBS", "W01_P098"):
        rec = LIBRARY.get(key)
        if rec is None:
            continue
        form = _polymerisable_form(rec)
        print(f"\n  {key}: {form}")
        assert form == "OCCCCOC(=O)CCC(=O)", (
            f"{key} would be built in a form PAAF cannot polymerise")


# ================================ the view must show the molecule you will get
#
# The screenshot that prompted this: a PBS trimer in the typing view ending in
# -C(=O)H. PAAF has to present the acid end as an aldehyde so that linking has
# a hydrogen to remove, but that is a device for building, not the chain's
# chemistry — and typing against it means assigning a type to an atom that
# will not exist, while the real terminal -OH gets none.
def _smiles_of(mol) -> str:
    """Read a PAAF Molecule back out as SMILES, to check connectivity."""
    em = Chem.RWMol()
    for a in mol.atoms:
        em.AddAtom(Chem.Atom(a.element))
    kinds = {1.0: Chem.BondType.SINGLE, 2.0: Chem.BondType.DOUBLE,
             3.0: Chem.BondType.TRIPLE}
    for i, j, order in mol.bonds:
        em.AddBond(int(i), int(j), kinds.get(float(order), Chem.BondType.SINGLE))
    out = em.GetMol()
    Chem.SanitizeMol(out)
    return Chem.MolToSmiles(Chem.RemoveHs(out))


def _groups(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    count = lambda sma: len(m.GetSubstructMatches(Chem.MolFromSmarts(sma)))
    return {"acid": count("[CX3](=O)[OX2H1]"),
            "aldehyde": count("[CX3H1](=O)[#6]"),
            "ester": count("[CX3](=O)[OX2H0][#6]"),
            "alcohol": count("[OX2H1][CX4]")}


def test_the_typing_view_shows_real_pbs_not_an_aldehyde():
    """The whole complaint, as an assertion."""
    from paaf.typing_context import build_trimer

    ctx = build_trimer([_pbs_monomer()])
    smiles = _smiles_of(ctx.molecule)
    groups = _groups(smiles)
    print(f"\n  {smiles}\n  {groups}")
    assert groups["aldehyde"] == 0, "the chain end is still an aldehyde"
    assert groups["acid"] == 1, "the chain end is not a carboxylic acid"
    assert groups["alcohol"] == 1, "the chain start is not an alcohol"
    assert groups["ester"] == 5, "a junction ester is missing"


def test_the_trimer_matches_pbs_exactly():
    """Not just the right groups — the right molecule."""
    from paaf.typing_context import build_trimer

    ctx = build_trimer([_pbs_monomer()])
    want = Chem.MolToSmiles(Chem.MolFromSmiles(
        "OCCCCOC(=O)CCC(=O)" * 3 + "O"))
    print(f"\n  built: {_smiles_of(ctx.molecule)}\n  want : {want}")
    assert _smiles_of(ctx.molecule) == want


def test_the_cap_atoms_are_listed_and_typeable():
    """They are atoms in the chain, so they need rows and keys of their own."""
    from paaf.typing_context import CAP_UNIT, build_trimer

    ctx = build_trimer([_pbs_monomer()])
    print(f"\n  cap atoms {ctx.terminal_cap} = "
          f"{[ctx.molecule.atoms[i].element for i in ctx.terminal_cap]}")
    assert [ctx.molecule.atoms[i].element for i in ctx.terminal_cap] == ["O", "H"]
    for k, index in enumerate(ctx.terminal_cap):
        assert ctx.provenance[index] == (CAP_UNIT, k)


def test_cap_types_assigned_in_the_view_reach_the_chain():
    """The point of giving them rows: the types must land somewhere real."""
    from paaf.typing_context import merge_roles

    mono = _pbs_monomer()
    capped = _with_carboxyl_cap(simple_backend([mono], [0] * 4,
                                               conformer_seed=1))
    mapping = unit_provenance(capped, mono, 4)
    applied = expand_by_provenance(
        capped, _types(mono), mapping,
        cap_types={0: "108", 1: "140"},          # the -O- and its H
        n_units=4)
    cap_atoms = sorted(set(range(len(capped.atoms)))
                       - {i for i in applied if i < len(capped.atoms) - 2})
    print(f"\n  {len(applied)}/{len(capped.atoms)} typed, "
          f"cap atoms {[applied.get(i) for i in range(len(capped.atoms)-2, len(capped.atoms))]}")
    assert len(applied) == len(capped.atoms), "the cap was left untyped"
    assert applied[len(capped.atoms) - 2] == "108"
    assert applied[len(capped.atoms) - 1] == "140"


def test_a_non_polyester_gets_no_cap():
    """Polyethylene has no acid end to restore, so nothing must be added."""
    from paaf.monomer import Monomer
    from paaf.typing_context import build_trimer

    m = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    AllChem.UFFOptimizeMolecule(m)
    conf = m.GetConformer()
    mol = Molecule(
        atoms=[Atom(index=i, element=a.GetSymbol(),
                    xyz=np.array([conf.GetAtomPosition(i).x,
                                  conf.GetAtomPosition(i).y,
                                  conf.GetAtomPosition(i).z]))
               for i, a in enumerate(m.GetAtoms())],
        bonds=[(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                b.GetBondTypeAsDouble()) for b in m.GetBonds()], name="pe")
    cap = lambda k: [n.GetIdx() for n in m.GetAtomWithIdx(k).GetNeighbors()
                     if n.GetSymbol() == "H"][0]
    ctx = build_trimer([Monomer(mol, 0, 3, [cap(0)], [cap(3)], name="pe")])
    print(f"\n  polyethylene cap atoms: {ctx.terminal_cap}")
    assert ctx.terminal_cap == []
    assert _groups(_smiles_of(ctx.molecule))["acid"] == 0


# ============================== auto types AND manual overrides must survive
#
# The typing dialog seeds every atom from the automatic typer and the user
# overrides only the ones it got wrong. So the map handed to the pipeline is
# COMPLETE, and "the mapping failed, fall back to the automatic typer" throws
# away the automatic types as well — retyping the chain from scratch with the
# very typer whose answers were being corrected. That is what produced a chain
# with carbon types on eleven oxygens.
def test_types_are_recovered_by_environment_when_the_mapping_is_refused():
    from paaf.chain_provenance import types_by_environment
    from paaf.typing_context import merge_roles, split_by_role

    mono = _pbs_monomer()
    chain = simple_backend([mono], [0] * 4, conformer_seed=1)
    # A chain the positional mapping cannot handle: an extra carbon.
    odd = _with_carboxyl_cap(chain, extra=(("C", 1.54),))
    assert unit_provenance(odd, mono, 4) is None, "fixture must be unmappable"

    by_role = split_by_role(merge_roles(
        {"head": {}, "middle": _types(mono), "tail": {}, "cap": {}}))
    recovered = types_by_environment(odd, mono, by_role)
    print(f"\n  {len(recovered)}/{len(odd.atoms)} atoms recovered by "
          f"environment where the exact mapping gave 0")
    assert len(recovered) > 0.8 * len(odd.atoms), \
        "the fallback should recover most of the chain"


def test_the_fallback_never_types_an_atom_as_the_wrong_element():
    """It matches on environment, so the element guard still has to hold."""
    from paaf.chain_provenance import types_by_environment
    from paaf.type_guard import check_all
    from paaf.typing_context import merge_roles, split_by_role

    mono = _pbs_monomer()
    odd = _with_carboxyl_cap(
        simple_backend([mono], [0] * 3, conformer_seed=1),
        extra=(("C", 1.54),))
    by_role = split_by_role(merge_roles({"middle": _types(mono)}))
    recovered = types_by_environment(odd, mono, by_role)
    problems = check_all({a.index: a.element for a in odd.atoms}, recovered)
    print(f"\n  element problems: {problems or 'none'}")
    assert not problems


def test_the_exact_mapping_is_still_preferred_when_it_works():
    """The fallback is a fallback; it must not displace the precise answer."""
    mono = _pbs_monomer()
    capped = _with_carboxyl_cap(simple_backend([mono], [0] * 4,
                                               conformer_seed=1))
    mapping = unit_provenance(capped, mono, 4)
    assert mapping is not None, "this chain IS mappable, so no fallback needed"


# ====================================== reporting a problem must not crash
def test_element_problems_format_as_text_without_crashing():
    """check_all returns (index, message) PAIRS, not strings.

    The Apply-all button joined them directly, so the moment a single
    element mismatch existed the dialog raised TypeError and took the whole
    application down — an error path that crashes is worse than the error it
    was trying to report. This pins the shape of what it returns.
    """
    from paaf.type_guard import check_all

    problems = check_all({0: "O", 1: "C", 2: "H"},
                         {0: "211", 1: "136", 2: "136"})
    assert problems, "this fixture must produce mismatches"
    for entry in problems:
        assert isinstance(entry, tuple) and len(entry) == 2
        index, message = entry
        assert isinstance(index, int) and isinstance(message, str)

    # The formatting the dialog now uses.
    text = "\n\n".join(f"atom {i}: {m}" for i, m in problems[:8])
    print(f"\n  {len(problems)} problems formatted, {len(text)} chars")
    assert "atom 0:" in text and "211" in text


def test_a_clean_assignment_reports_no_problems():
    """The report must stay silent when there is nothing wrong."""
    from paaf.type_guard import check_all

    assert check_all({0: "C", 1: "H"}, {0: "136", 1: "140"}) == []


# ============================== the acid end must get the acid's own types
#
# Measured from a real export: the data file had 7 atom types --
# 136 140 154 155 465 466 467 -- and NONE of 267/268/269/270. The terminal
# -COOH came out with the ALCOHOL's types, because a one-shell fingerprint
# cannot tell an acid -O-H from an alcohol -O-H (both: oxygen, one hydrogen,
# bonded to one carbon) and there was no `cap` role to match against at all.
def _real_carboxyl_cap(chain: Molecule) -> Molecule:
    """The cap as the builder really makes it: the aldehyde H BECOMES the -OH.

    The earlier fixture appended O and H without removing the aldehyde
    hydrogen, which left that carbon with an extra bond and quietly made the
    test easier than reality.
    """
    out = Molecule(atoms=[Atom(index=a.index, element=a.element,
                               xyz=a.xyz.copy()) for a in chain.atoms],
                   bonds=list(chain.bonds), name="capped")
    nb = {}
    for i, j, _o in out.bonds:
        nb.setdefault(i, []).append(j)
        nb.setdefault(j, []).append(i)
    carbon = max(i for i, a in enumerate(out.atoms) if a.element == "C"
                 and any(out.atoms[n].element == "O" for n in nb.get(i, []))
                 and any(out.atoms[n].element == "H" for n in nb.get(i, [])))
    hydrogen = next(n for n in nb[carbon] if out.atoms[n].element == "H")
    out.atoms[hydrogen].element = "O"
    out.atoms[hydrogen].xyz = out.atoms[carbon].xyz + np.array([1.34, 0.0, 0.0])
    new = len(out.atoms)
    out.atoms.append(Atom(index=new, element="H",
                          xyz=out.atoms[hydrogen].xyz + np.array([0.97, 0, 0])))
    out.bonds.append((hydrogen, new, 1.0))
    return out, carbon, hydrogen, new


def _pbs_role_types():
    """Repeat/head/tail/cap types in the shape the dialog produces."""
    repeat = {0: "467", 5: "467", 6: "465", 7: "466", 10: "465", 11: "466"}
    repeat.update({k: "136" for k in (1, 2, 3, 4, 8, 9)})
    repeat.update({k: "140" for k in range(13, 25)})
    head = dict(repeat); head[0] = "154"; head[12] = "155"; head[1] = "135"
    tail = dict(repeat); tail[10] = "267"; tail[11] = "269"
    return {"head": head, "middle": repeat, "tail": tail,
            "cap": {0: "268", 1: "270"}}


def test_the_acid_end_gets_the_acid_types_not_the_alcohol_ones():
    from paaf.chain_provenance import types_by_environment
    from paaf.type_guard import check_all

    mono = _pbs_monomer()
    capped, acid_c, acid_o, acid_h = _real_carboxyl_cap(
        simple_backend([mono], [0] * 3, conformer_seed=1))
    got = types_by_environment(capped, mono, _pbs_role_types())

    print(f"\n  acid C {got.get(acid_c)}, -OH O {got.get(acid_o)}, "
          f"O-H {got.get(acid_h)}")
    assert got.get(acid_c) == "267", "the acid carbon kept an ester type"
    assert got.get(acid_o) == "268", "the acid -OH took the alcohol's type"
    assert got.get(acid_h) == "270", "the acid hydrogen took the alcohol's"
    assert not check_all({a.index: a.element for a in capped.atoms}, got)


def test_the_interior_esters_are_not_overwritten_by_the_acid_type():
    """The regression from letting an end type outrank the repeat unit.

    An ester =O and an acid =O are indistinguishable however deep you look, so
    preferring the end put the acid's type on all six ester carbonyls — one
    atom of end accuracy bought with six wrong interior atoms.
    """
    from collections import Counter

    from paaf.chain_provenance import types_by_environment

    mono = _pbs_monomer()
    capped, _c, _o, _h = _real_carboxyl_cap(
        simple_backend([mono], [0] * 3, conformer_seed=1))
    counts = Counter(types_by_environment(
        capped, mono, _pbs_role_types()).values())
    print(f"\n  466 ester=O x{counts.get('466', 0)}, "
          f"269 acid=O x{counts.get('269', 0)}")
    assert counts.get("466", 0) == 6, "ester carbonyls lost their type"
    assert counts.get("269", 0) <= 1, "an end type leaked into the interior"


def test_every_atom_of_a_capped_pbs_chain_is_typed():
    from paaf.chain_provenance import types_by_environment
    from paaf.type_guard import check_all

    mono = _pbs_monomer()
    capped, _c, _o, _h = _real_carboxyl_cap(
        simple_backend([mono], [0] * 3, conformer_seed=1))
    got = types_by_environment(capped, mono, _pbs_role_types())
    print(f"\n  {len(got)}/{len(capped.atoms)} typed")
    assert len(got) == len(capped.atoms)
    assert not check_all({a.index: a.element for a in capped.atoms}, got)


# ================= the EXACT mapping must survive a real in-place end cap
#
# The builder does not simply append the -OH: it changes the terminal
# aldehyde's HYDROGEN into the hydroxyl oxygen where it stands and appends the
# acid's hydrogen at the end. Two atoms therefore stop matching the monomer and
# only one of them is trailing, so treating "the cap" as a trailing run alone
# left the other failing the fingerprint check and the whole mapping was
# refused. Every hand-typed polyester then fell back to environment matching
# and lost its acid types -- measured on a real export, the data file had no
# 267/268/269/270 in it at all.
def test_the_exact_mapping_survives_an_in_place_carboxyl_cap():
    mono = _pbs_monomer()
    capped, acid_c, acid_o, acid_h = _real_carboxyl_cap(
        simple_backend([mono], [0] * 3, conformer_seed=1))
    mapping = unit_provenance(capped, mono, 3)
    print(f"\n  {len(capped.atoms)} atoms -> "
          f"{len(mapping) if mapping else 0} mapped")
    assert mapping is not None, "the in-place cap still breaks the mapping"
    assert len(mapping) == len(capped.atoms)


def test_the_acid_end_types_land_through_the_exact_mapping():
    from paaf.type_guard import check_all

    mono = _pbs_monomer()
    capped, acid_c, acid_o, acid_h = _real_carboxyl_cap(
        simple_backend([mono], [0] * 3, conformer_seed=1))
    roles = _pbs_role_types()
    applied = expand_by_provenance(
        capped, roles["middle"], unit_provenance(capped, mono, 3),
        head_types=roles["head"], tail_types=roles["tail"],
        cap_types=roles["cap"], n_units=3)

    print(f"\n  {len(applied)}/{len(capped.atoms)} typed; acid C "
          f"{applied.get(acid_c)}, -OH O {applied.get(acid_o)}, "
          f"O-H {applied.get(acid_h)}")
    assert len(applied) == len(capped.atoms), "some atoms were left untyped"
    assert applied[acid_c] == "267"
    assert applied[acid_o] == "268"
    assert applied[acid_h] == "270"
    assert not check_all({a.index: a.element for a in capped.atoms}, applied)


def test_the_cap_atoms_are_numbered_by_position_not_discovery_order():
    """The -OH oxygen is cap atom 0 and its hydrogen cap atom 1.

    The converted atom sits BEFORE the appended one in the chain, and the
    typing view lists them in that order. Numbering the appended atom first
    swapped them, and the element guard refused to write an oxygen type onto a
    hydrogen — right, but the two assignments were lost.
    """
    from paaf.typing_context import CAP_UNIT

    mono = _pbs_monomer()
    capped, _c, acid_o, acid_h = _real_carboxyl_cap(
        simple_backend([mono], [0] * 3, conformer_seed=1))
    mapping = unit_provenance(capped, mono, 3)
    print(f"\n  O {acid_o} -> {mapping[acid_o]}, H {acid_h} -> {mapping[acid_h]}")
    assert mapping[acid_o] == (CAP_UNIT, 0), "the -OH oxygen is not cap atom 0"
    assert mapping[acid_h] == (CAP_UNIT, 1), "the O-H hydrogen is not cap atom 1"
