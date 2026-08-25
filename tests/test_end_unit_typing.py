"""The chain ends must be typed too, and typed as ends.

Why this exists
---------------
Typing used to cover the repeat unit only. Everything in between two junctions
was the user's, and the first and last unit were left to the automatic typer —
on a chain whose bond orders had already been flattened. So a hand-typed
polyisoprene came out with its two terminal carbons typed as backbone CH2
(136) when they are terminal CH3 (135), because the monomer's link carbon is
the atom that changes identity depending on where in the chain it sits.

The rule is not about isoprene. At a junction the cap atom is consumed and
replaced by a bond; at a chain end it survives. So the first and last units
have one more atom on their link site than the interior does, and that is a
different force-field type for any polymer.

Three sets of types therefore have to travel from the dialog to the chain —
head, repeat, tail — through a config field, a text box and a pipeline that
all expect one ``{int: str}`` map. They do it by folding the unit's role into
the key. These tests pin that round trip and, more importantly, pin that the
ends actually come out different from the middle.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from paaf.chain_builder import simple_backend                   # noqa: E402
from paaf.chain_provenance import unit_provenance               # noqa: E402
from paaf.ff_assigner import expand_manual_types                # noqa: E402
from paaf.structure import Atom, Molecule                       # noqa: E402
from paaf.type_guard import check_all                           # noqa: E402
from paaf.typing_context import (                               # noqa: E402
    ROLES, build_trimer, merge_roles, role_key, split_by_role, split_role_key,
)

rdkit = pytest.importorskip("rdkit", reason="needs RDKit")
from rdkit import Chem                                          # noqa: E402
from rdkit.Chem import AllChem                                  # noqa: E402


def _monomer(name: str, smiles: str, head: int, tail: int):
    from paaf.monomer import Monomer

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=4)
    try:
        AllChem.UFFOptimizeMolecule(m)
    except Exception:
        pass
    conf = m.GetConformer()
    mol = Molecule(
        atoms=[Atom(index=i, element=a.GetSymbol(),
                    xyz=np.array([conf.GetAtomPosition(i).x,
                                  conf.GetAtomPosition(i).y,
                                  conf.GetAtomPosition(i).z]))
               for i, a in enumerate(m.GetAtoms())],
        bonds=[(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                b.GetBondTypeAsDouble()) for b in m.GetBonds()],
        name=name)

    def cap(k):
        return [n.GetIdx() for n in m.GetAtomWithIdx(k).GetNeighbors()
                if n.GetSymbol() == "H"][0]

    return Monomer(molecule=mol, head_index=head, tail_index=tail,
                   head_removes=[cap(head)], tail_removes=[cap(tail)],
                   name=name)


def _isoprene():
    return _monomer("isoprene", "CC=C(C)C", 0, 4)


def _isoprene_types(monomer):
    """The user's reference typing, with proper terminal types at the ends."""
    mol = monomer.molecule
    repeat = {0: "136", 1: "142", 2: "141", 3: "135", 4: "136"}
    alkene_h = [n for n in range(len(mol.atoms))
                if mol.atoms[n].element == "H"
                and any({i, j} == {n, 1} for i, j, _o in mol.bonds)]
    for a in mol.atoms:
        if a.element == "H":
            repeat[a.index] = "144" if a.index in alkene_h else "140"

    head = dict(repeat)
    head[monomer.head_index] = "135"      # keeps its cap: terminal CH3
    tail = dict(repeat)
    tail[monomer.tail_index] = "135"
    return {"head": head, "middle": repeat, "tail": tail}


# ================================================= the key encoding is sound
def test_a_repeat_unit_atom_keeps_its_plain_index():
    """Everything written before end typing existed must keep working."""
    assert role_key("middle", 7) == 7
    assert split_role_key(7) == ("middle", 7)


@pytest.mark.parametrize("role", ROLES)
def test_every_role_round_trips(role):
    for index in (0, 1, 42, 999):
        assert split_role_key(role_key(role, index)) == (role, index)


def test_head_and_tail_atom_four_are_not_the_same_key():
    """The bug this prevents: the units collapsing onto one set of types."""
    keys = {role_key(r, 4) for r in ROLES}
    print(f"\n  atom 4 in each role -> {sorted(keys)}")
    assert len(keys) == len(ROLES)


def test_an_old_style_map_reads_as_pure_repeat_unit():
    old = {0: "136", 3: "135", 9: "140"}
    split = split_by_role(old)
    assert split["middle"] == old
    assert not split["head"] and not split["tail"]


def test_the_three_sets_survive_being_flattened_and_unfolded():
    """They travel through one {int: str} field and must come back intact."""
    original = {"head": {0: "135", 1: "142"},
                "middle": {0: "136", 1: "142"},
                "tail": {4: "135"},
                "cap": {0: "154", 1: "155"}}     # a -COOH end cap
    assert split_by_role(merge_roles(original)) == original


def test_the_flattened_map_is_plain_integers_so_it_survives_a_text_box():
    """The GUI passes types as "key:value" lines; nothing may need escaping."""
    flat = merge_roles({"head": {4: "135"}, "middle": {4: "136"}, "tail": {}})
    text = "\n".join(f"{k}:{v}" for k, v in flat.items())
    parsed = {int(k): v for k, v in
              (line.split(":") for line in text.splitlines())}
    print(f"\n  {text!r}")
    assert parsed == flat


# ============================================== the ends really differ
def test_the_trimer_offers_all_three_units_for_typing():
    ctx = build_trimer([_isoprene()])
    units = Counter(u for u, _k in ctx.provenance.values())
    print(f"\n  atoms per unit: {dict(sorted(units.items()))}")
    assert set(units) == {0, 1, 2}, "every unit must be reachable"
    assert len(ctx.provenance) == len(ctx.molecule.atoms), \
        "every atom must belong to a unit, so every atom can be typed"


def test_the_end_units_keep_a_cap_the_middle_has_lost():
    """The chemical fact the whole feature rests on."""
    mono = _isoprene()
    ctx = build_trimer([mono])

    def hydrogens(k):
        return sum(1 for i, j, _o in ctx.molecule.bonds
                   for other in ([j] if i == k else [i] if j == k else [])
                   if ctx.molecule.atoms[other].element == "H")

    head = next(i for i, (u, m) in ctx.provenance.items()
                if u == 0 and m == mono.head_index)
    middle = next(i for i, (u, m) in ctx.provenance.items()
                  if u == 1 and m == mono.head_index)
    print(f"\n  head link carbon {hydrogens(head)} H, "
          f"middle link carbon {hydrogens(middle)} H")
    assert hydrogens(head) == hydrogens(middle) + 1


def test_types_assigned_on_the_trimer_split_by_role():
    ctx = build_trimer([_isoprene()])
    on_trimer = {}
    for trimer_index, (unit, _m) in ctx.provenance.items():
        on_trimer[trimer_index] = {0: "135", 1: "136", 2: "137"}[unit]
    by_role = ctx.types_by_role(on_trimer)
    assert set(by_role["head"].values()) == {"135"}
    assert set(by_role["middle"].values()) == {"136"}
    assert set(by_role["tail"].values()) == {"137"}


# ======================================= the types reach the right chain atoms
def test_the_chain_ends_get_terminal_types_and_the_interior_does_not():
    mono = _isoprene()
    chain = simple_backend([mono], [0] * 5, conformer_seed=1)
    flat = merge_roles(_isoprene_types(mono))
    applied = expand_manual_types(chain, flat, monomer=mono, n_units=5)
    counts = Counter(applied.values())
    print(f"\n  {dict(sorted(counts.items()))}")

    assert len(applied) == len(chain.atoms), "some atoms were left untyped"
    # One methyl branch per unit (5), plus one terminal CH3 at each end (2).
    assert counts["135"] == 7, "the chain ends were not typed as ends"
    # Interior link carbons: 10 link sites, less the 2 that are chain ends.
    assert counts["136"] == 8


def test_without_end_sets_the_repeat_types_still_cover_the_whole_chain():
    """Opt-in: a user who types only the repeat unit loses nothing."""
    mono = _isoprene()
    chain = simple_backend([mono], [0] * 5, conformer_seed=1)
    repeat = _isoprene_types(mono)["middle"]
    applied = expand_manual_types(chain, repeat, monomer=mono, n_units=5)
    print(f"\n  {len(applied)}/{len(chain.atoms)} typed with repeat types only")
    assert len(applied) == len(chain.atoms)
    assert Counter(applied.values())["135"] == 5     # methyls only


def test_no_end_type_lands_on_an_interior_atom():
    """A terminal type in the middle of a chain is a silent chemistry error."""
    mono = _isoprene()
    chain = simple_backend([mono], [0] * 6, conformer_seed=1)
    sets = _isoprene_types(mono)
    # Mark the ends with a type that appears nowhere else, so any leakage
    # into the interior is unmistakable.
    sets["head"][mono.head_index] = "148"
    sets["tail"][mono.tail_index] = "148"
    applied = expand_manual_types(chain, merge_roles(sets),
                                  monomer=mono, n_units=6)
    mapping = unit_provenance(chain, mono, 6)
    leaked = [c for c, t in applied.items()
              if t == "148" and mapping[c][0] not in (0, 5)]
    print(f"\n  marker type on {sum(1 for t in applied.values() if t == '148')} "
          f"atoms, {len(leaked)} of them interior")
    assert not leaked, "an end type was written onto an interior atom"


def test_end_types_are_element_checked_like_any_other():
    mono = _isoprene()
    chain = simple_backend([mono], [0] * 4, conformer_seed=1)
    applied = expand_manual_types(chain, merge_roles(_isoprene_types(mono)),
                                  monomer=mono, n_units=4)
    problems = check_all({a.index: a.element for a in chain.atoms}, applied)
    print(f"\n  element problems: {problems or 'none'}")
    assert not problems


def test_a_carbon_type_on_a_terminal_hydrogen_is_still_refused():
    mono = _isoprene()
    chain = simple_backend([mono], [0] * 3, conformer_seed=1)
    sets = _isoprene_types(mono)
    a_hydrogen = next(a.index for a in mono.molecule.atoms
                      if a.element == "H")
    sets["head"][a_hydrogen] = "136"          # carbon type on a hydrogen
    applied = expand_manual_types(chain, merge_roles(sets),
                                  monomer=mono, n_units=3)
    kept = {chain.atoms[i].element for i, t in applied.items() if t == "136"}
    print(f"\n  elements that kept 136: {kept}")
    assert kept <= {"C"}


def test_one_unit_chains_are_not_given_two_sets_of_end_types():
    """A single unit is head and tail at once and keeps both caps."""
    mono = _isoprene()
    chain = simple_backend([mono], [0], conformer_seed=1)
    applied = expand_manual_types(chain, merge_roles(_isoprene_types(mono)),
                                  monomer=mono, n_units=1)
    print(f"\n  n=1: {len(applied)}/{len(chain.atoms)} typed")
    assert len(applied) == len(chain.atoms)
    problems = check_all({a.index: a.element for a in chain.atoms}, applied)
    assert not problems


@pytest.mark.parametrize("name,smiles,head,tail", [
    ("polyethylene",  "CCCC",            0, 3),
    ("polypropylene", "CC(C)CC",         0, 4),
    ("polystyrene",   "CC(c1ccccc1)C",   0, 4),
    ("PVC",           "CC(Cl)C",         0, 3),
    ("PEG",           "COCCOC",          0, 5),
])
def test_end_typing_works_for_any_polymer(name, smiles, head, tail):
    """No part of this may depend on isoprene's shape."""
    mono = _monomer(name, smiles, head, tail)
    chain = simple_backend([mono], [0] * 4, conformer_seed=2)

    by_element = {"C": "136", "H": "140", "O": "108", "Cl": "264"}
    repeat = {a.index: by_element[a.element]
              for a in mono.molecule.atoms if a.element in by_element}
    ends = dict(repeat)
    ends[head] = ends[tail] = "135"

    applied = expand_manual_types(
        chain, merge_roles({"head": ends, "middle": repeat, "tail": ends}),
        monomer=mono, n_units=4)
    counts = Counter(applied.values())
    print(f"\n  {name}: {len(applied)}/{len(chain.atoms)} typed, "
          f"{counts['135']} terminal")
    assert len(applied) == len(chain.atoms)
    assert counts["135"] == 4, "each end unit's two link sites"
    assert not check_all({a.index: a.element for a in chain.atoms}, applied)


# ================================== what the dialog will actually put on screen
#
# The dialog itself needs Qt, which is not always available where the tests
# run. These exercise the exact building blocks it uses — build_trimer,
# role_key and the SMARTS typer — so the *content* of the three unit
# listings is pinned even where the widget cannot be constructed.
def _dialog_rows(monomer):
    """Mirror of AtomTypingDialog._unit_rows, without the widget."""
    pytest.importorskip("openbabel",
                        reason="the SMARTS typer needs OpenBabel")
    from paaf.chem_env import atom_group
    from paaf.typers.oplsaa import type_oplsaa

    ctx = build_trimer([monomer])
    guesses = type_oplsaa(ctx.molecule)
    role_of = {0: "head", 1: "middle", 2: "tail"}
    return [{"key": role_key(role_of[u], m), "monomer_index": m,
             "role": role_of[u], "source_index": t,
             "group": atom_group(ctx.molecule, t),
             "guess": guesses.get(t, "")}
            for t, (u, m) in sorted(ctx.provenance.items())]


def test_the_listing_covers_three_units_with_distinct_keys():
    rows = _dialog_rows(_isoprene())
    per_role = Counter(r["role"] for r in rows)
    print(f"\n  {dict(per_role)}")
    assert set(per_role) == set(ROLES)
    assert len({r["key"] for r in rows}) == len(rows), "two rows share a key"


def test_each_end_unit_lists_one_more_atom_than_the_repeat_unit():
    """It keeps the cap the interior gave up at its junction."""
    per_role = Counter(r["role"] for r in _dialog_rows(_isoprene()))
    assert per_role["head"] == per_role["middle"] + 1
    assert per_role["tail"] == per_role["middle"] + 1


def test_the_suggested_types_differ_between_the_ends_and_the_middle():
    """135 at the ends, 136 in the middle — offered, not left to be known."""
    mono = _isoprene()
    rows = {(r["role"], r["monomer_index"]): r["guess"]
            for r in _dialog_rows(mono)}
    head = rows[("head", mono.head_index)]
    middle = rows[("middle", mono.head_index)]
    tail = rows[("tail", mono.tail_index)]
    print(f"\n  head link C {head}, middle link C {middle}, tail link C {tail}")
    assert head == "135", "the chain start is a terminal CH3"
    assert middle == "136", "an interior link carbon is a backbone CH2"
    assert tail == "135", "the chain end is a terminal CH3"


def test_the_alkene_is_typed_the_same_in_every_unit():
    """Only the link sites change with position; the rest of the unit does not."""
    rows = _dialog_rows(_isoprene())
    alkene = {r["role"]: sorted(r["guess"] for r in rows
                                if r["guess"] in {"141", "142", "144"}
                                and r["role"] == role)
              for role in ROLES}
    print(f"\n  {alkene}")
    assert alkene["head"] == alkene["middle"] == alkene["tail"] != []


def test_the_whole_listing_can_be_flattened_and_placed_on_a_real_chain():
    """End to end: what the dialog collects is what reaches the chain."""
    mono = _isoprene()
    flat = {r["key"]: r["guess"] for r in _dialog_rows(mono) if r["guess"]}
    chain = simple_backend([mono], [0] * 4, conformer_seed=1)
    applied = expand_manual_types(chain, flat, monomer=mono, n_units=4)
    counts = Counter(applied.values())
    print(f"\n  {len(applied)}/{len(chain.atoms)} typed  "
          f"{dict(sorted(counts.items()))}")
    assert len(applied) == len(chain.atoms)
    assert counts["135"] == 6, "4 methyls + one terminal CH3 at each end"
    assert counts["141"] == counts["142"] == counts["144"] == 4
    assert not check_all({a.index: a.element for a in chain.atoms}, applied)
