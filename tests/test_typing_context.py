"""Typing a repeat unit in the environment it will actually have.

Isoprene's capped monomer is ``CC=C(C)C``. Its two link carbons carry a cap
hydrogen, so in isolation they really are CH3 — and an automatic typer says
135, correctly, about the molecule it was shown. In the chain that cap is
replaced by a bond and they become CH2, needing 136.

Typing the middle unit of a trimer removes the discrepancy: those carbons
have two heavy neighbours there, so 136 falls out without anyone having to
know to override it. The outer units are not padding — they are the real
chain ends, and a chain end does need different types.
"""
from __future__ import annotations

import numpy as np
import pytest

from paaf.structure import Atom, Molecule                       # noqa: E402
from paaf.typing_context import build_trimer                    # noqa: E402

rdkit = pytest.importorskip("rdkit", reason="needs RDKit")
from rdkit import Chem                                          # noqa: E402
from rdkit.Chem import AllChem                                  # noqa: E402


def _monomer(smiles: str, head: int, tail: int, name: str):
    """A Monomer whose link carbons carry one cap hydrogen each."""
    from paaf.monomer import Monomer

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
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
        name=name)

    def cap(k):
        return [n.GetIdx() for n in m.GetAtomWithIdx(k).GetNeighbors()
                if n.GetSymbol() == "H"][0]

    return Monomer(molecule=mol, head_index=head, tail_index=tail,
                   head_removes=[cap(head)], tail_removes=[cap(tail)],
                   name=name)


def _isoprene():
    return _monomer("CC=C(C)C", 0, 4, "isoprene")


def _hydrogens_on(mol, k: int) -> int:
    n = 0
    for i, j, _o in mol.bonds:
        other = j if i == k else i if j == k else None
        if other is not None and mol.atoms[other].element == "H":
            n += 1
    return n


def _heavy_on(mol, k: int) -> int:
    n = 0
    for i, j, _o in mol.bonds:
        other = j if i == k else i if j == k else None
        if other is not None and mol.atoms[other].element != "H":
            n += 1
    return n


def _trimer_atom(ctx, unit: int, monomer_index: int) -> int:
    return next(i for i, (u, k) in ctx.provenance.items()
                if u == unit and k == monomer_index)


# ======================================= the environment the user should see
def test_the_middle_units_link_carbons_are_CH2():
    """The whole point: 136, not 135, without anyone overriding anything."""
    ctx = build_trimer([_isoprene()])
    for monomer_index in (0, 4):
        k = _trimer_atom(ctx, 1, monomer_index)
        h = _hydrogens_on(ctx.molecule, k)
        heavy = _heavy_on(ctx.molecule, k)
        print(f"\n  middle unit, monomer atom {monomer_index}: "
              f"{heavy} heavy + {h} H")
        assert h == 2, "still looks like a CH3 — the cap was not consumed"
        assert heavy == 2, "not bonded into the chain on both sides"


def test_the_outer_units_keep_their_terminal_CH3():
    """Chain ends are genuinely different and must stay so."""
    ctx = build_trimer([_isoprene()])
    first = _trimer_atom(ctx, 0, 0)      # far end of the first unit
    last = _trimer_atom(ctx, 2, 4)       # far end of the last unit
    print(f"\n  first end: {_hydrogens_on(ctx.molecule, first)} H; "
          f"last end: {_hydrogens_on(ctx.molecule, last)} H")
    assert _hydrogens_on(ctx.molecule, first) == 3
    assert _hydrogens_on(ctx.molecule, last) == 3


def test_the_middle_unit_has_no_cap_hydrogens_left():
    """No cap atoms means no index drift against the real chain."""
    ctx = build_trimer([_isoprene()])
    assert not (set(ctx.caps) & set(ctx.middle))


def test_only_the_two_far_ends_keep_caps():
    ctx = build_trimer([_isoprene()])
    print(f"\n  {len(ctx.caps)} caps, all in the outer units: "
          f"{all(c in ctx.ends for c in ctx.caps)}")
    assert len(ctx.caps) == 2
    assert all(c in ctx.ends for c in ctx.caps)


# ================================================= structure of the trimer
def test_three_units_are_actually_joined():
    ctx = build_trimer([_isoprene()])
    print(f"\n  {ctx.summary()}")
    assert len(ctx.link_bonds) == 2, "the units are not connected"
    for i, j in ctx.link_bonds:
        # A junction joins two different units.
        assert ctx.provenance[i][0] != ctx.provenance[j][0]


def test_the_atom_count_is_three_units_minus_the_consumed_caps():
    mono = _isoprene()
    ctx = build_trimer([mono])
    per_unit = len(mono.molecule.atoms)
    # Two junctions, each consuming one cap from each side.
    print(f"\n  {len(ctx.molecule.atoms)} atoms "
          f"(3 x {per_unit} - 4 caps = {3 * per_unit - 4})")
    assert len(ctx.molecule.atoms) == 3 * per_unit - 4


def test_the_double_bond_survives_into_the_trimer():
    from paaf.bond_orders import order_summary

    ctx = build_trimer([_isoprene()])
    counts = order_summary(ctx.molecule)
    print(f"\n  {counts}")
    assert counts.get(2, 0) == 3, "one C=C per unit expected"


def test_every_atom_knows_where_it_came_from():
    ctx = build_trimer([_isoprene()])
    n = len(ctx.molecule.atoms)
    assert set(ctx.provenance) == set(range(n))
    # Middle and ends partition the trimer: no atom in both, none left out.
    assert not set(ctx.middle) & set(ctx.ends)
    assert set(ctx.middle) | set(ctx.ends) == set(range(n))


# ============================================ mapping types back and forth
def test_middle_unit_types_fold_back_onto_monomer_indices():
    ctx = build_trimer([_isoprene()])
    assigned = {k: "136" for k in ctx.middle[:2]}
    folded = ctx.types_by_monomer_index(assigned)
    print(f"\n  {assigned} -> {folded}")
    assert set(folded.values()) == {"136"}
    assert all(0 <= k < 15 for k in folded), "not monomer indices"


def test_end_unit_types_are_not_written_back_as_the_repeat_unit():
    """A chain end's types describe the ends, not the repeating type."""
    ctx = build_trimer([_isoprene()])
    end_only = {k: "135" for k in ctx.ends}
    assert ctx.types_by_monomer_index(end_only) == {}


def test_the_middle_map_covers_the_whole_middle_unit():
    ctx = build_trimer([_isoprene()])
    mapping = ctx.middle_to_monomer()
    assert set(mapping) == set(ctx.middle)
    assert len(set(mapping.values())) == len(ctx.middle), \
        "two middle atoms map to the same monomer atom"


# ==================================================== copolymer junctions
def test_a_copolymer_junction_can_be_built_and_is_labelled():
    """A-B-A is a different environment from A-A-A, and says so."""
    iso = _isoprene()
    epox = _monomer("CC1(C)OC1C", 0, 5, "epoxide")
    ctx = build_trimer([iso, epox, iso])
    print(f"\n  {ctx.junction_note}")
    assert "copolymer" in ctx.junction_note
    assert ctx.provenance[ctx.middle[0]][0] == 1
    # The middle unit is the epoxide, so its atoms outnumber isoprene's.
    assert len(ctx.middle) == len(epox.molecule.atoms) - 2


def test_a_homopolymer_junction_says_so():
    ctx = build_trimer([_isoprene()])
    assert "homopolymer" in ctx.junction_note


def test_wrong_number_of_monomers_is_refused():
    iso = _isoprene()
    with pytest.raises(ValueError):
        build_trimer([iso, iso])


# ============================================= what the 3D view shows
def test_the_page_dims_context_atoms_and_labels_only_the_typed_unit():
    """The junction must be visible, but unmistakably not the thing typed."""
    from paaf.gui.mol3d_view import build_page
    from paaf.molblock import to_molblock

    ctx = build_trimer([_isoprene()])
    page = build_page(to_molblock(ctx.molecule),
                      labels={i: str(i) for i in ctx.middle},
                      context=ctx.ends,
                      script_src="file:///tmp/3Dmol-min.js")
    print(f"\n  {len(ctx.middle)} typed atoms, {len(ctx.ends)} drawn as context")
    assert "DATA.context" in page, "context atoms would render solid"
    assert '"context":' in page
    # Every end atom is listed as context; no middle atom is.
    import json
    payload = json.loads(page.split("var DATA = ", 1)[1].split(";\nvar bridge", 1)[0])
    assert set(payload["context"]) == set(ctx.ends)
    assert not set(payload["context"]) & set(ctx.middle)
    assert set(int(k) for k in payload["labels"]) == set(ctx.middle)


def test_context_atoms_carry_no_assigned_colour():
    """Colour means "typed"; a context atom is not."""
    from paaf.gui.mol3d_view import colors_for_types

    ctx = build_trimer([_isoprene()])
    # Only middle-unit atoms ever reach colors_for_types.
    typed = {s: "136" for s in ctx.middle[:3]}
    colours = colors_for_types(typed)
    assert not set(colours) & set(ctx.ends)
