"""Typing must work for any polymer, not the one that exposed the bug.

Everything in this path was found and fixed against 1,4-polyisoprene. That is
exactly the situation where a fix quietly encodes one molecule's shape: an
alkene in the backbone, a methyl branch, one cap hydrogen at each end.

So these run the whole typing path over a spread of chemistries — saturated,
aromatic side group, ester backbone, halogen, ether, ring in the backbone,
and a heteroatom link — and assert the properties that must hold regardless:

* the chain maps back to its monomer, atom for atom, with every element agreeing;
* every atom ends up typed;
* no atom carries a type belonging to another element;
* the two chain ends are typed as ends, and the interior as repeat units.

None of these mention isoprene.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from paaf.chain_builder import simple_backend                   # noqa: E402
from paaf.chain_provenance import (                             # noqa: E402
    expand_by_provenance, unit_provenance,
)
from paaf.structure import Atom, Molecule                       # noqa: E402
from paaf.type_guard import check_all, type_elements            # noqa: E402
from paaf.typing_context import build_trimer                    # noqa: E402

rdkit = pytest.importorskip("rdkit", reason="needs RDKit")
from rdkit import Chem                                          # noqa: E402
from rdkit.Chem import AllChem                                  # noqa: E402


#: (name, capped repeat-unit SMILES, head atom, tail atom)
POLYMERS = [
    ("polyethylene",     "CCCC",                 0, 3),
    ("polypropylene",    "CC(C)CC",              0, 4),
    ("polyisoprene",     "CC=C(C)C",             0, 4),
    ("polystyrene",      "CC(c1ccccc1)C",        0, 4),
    ("PMMA",             "CC(C)(C(=O)OC)C",      0, 7),
    ("PVC",              "CC(Cl)C",              0, 3),
    ("polyoxymethylene", "COCOC",                0, 4),
    ("PEG",              "COCCOC",               0, 5),
    ("epoxidised unit",  "CC1(C)OC1C",           0, 5),
]


def _monomer(name: str, smiles: str, head: int, tail: int):
    """A Monomer with one cap hydrogen on each link atom."""
    from paaf.monomer import Monomer

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=3)
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
        hs = [n.GetIdx() for n in m.GetAtomWithIdx(k).GetNeighbors()
              if n.GetSymbol() == "H"]
        if not hs:
            pytest.skip(f"{name}: link atom {k} has no cap hydrogen")
        return hs[0]

    return Monomer(molecule=mol, head_index=head, tail_index=tail,
                   head_removes=[cap(head)], tail_removes=[cap(tail)],
                   name=name)


def _plausible_types(monomer):
    """A type per monomer atom, right element, chemistry-agnostic.

    The point of these tests is the machinery — mapping, coverage, element
    safety — not whether 136 is the chemically ideal choice. So each atom gets
    a type of the correct element and the assertions are about placement.
    """
    by_element = {"C": "136", "H": "140", "O": "108", "N": "739",
                  "Cl": "264", "S": "202", "F": "164"}
    out = {}
    for a in monomer.molecule.atoms:
        value = by_element.get(a.element)
        if value:
            out[a.index] = value
    return out


@pytest.mark.parametrize("name,smiles,head,tail", POLYMERS,
                         ids=[p[0] for p in POLYMERS])
def test_the_chain_maps_back_to_its_monomer(name, smiles, head, tail):
    """Element-verified, for every chemistry."""
    mono = _monomer(name, smiles, head, tail)
    chain = simple_backend([mono], [0] * 4, conformer_seed=2)
    mapping = unit_provenance(chain, mono, 4)
    print(f"\n  {name}: {len(chain.atoms)} chain atoms, "
          f"{len(mapping) if mapping else 0} mapped")
    assert mapping is not None, f"{name}: chain could not be matched"
    assert len(mapping) == len(chain.atoms)
    for c, (_u, m) in mapping.items():
        assert chain.atoms[c].element == mono.molecule.atoms[m].element


@pytest.mark.parametrize("name,smiles,head,tail", POLYMERS,
                         ids=[p[0] for p in POLYMERS])
def test_every_atom_of_every_unit_gets_typed(name, smiles, head, tail):
    mono = _monomer(name, smiles, head, tail)
    chain = simple_backend([mono], [0] * 5, conformer_seed=2)
    mapping = unit_provenance(chain, mono, 5)
    assert mapping is not None
    applied = expand_by_provenance(chain, _plausible_types(mono), mapping,
                                   n_units=5)
    print(f"\n  {name}: {len(applied)}/{len(chain.atoms)} typed")
    assert len(applied) == len(chain.atoms), f"{name}: atoms left untyped"


@pytest.mark.parametrize("name,smiles,head,tail", POLYMERS,
                         ids=[p[0] for p in POLYMERS])
def test_no_atom_gets_a_type_from_another_element(name, smiles, head, tail):
    """The mass-1.008-on-a-carbon failure, checked across chemistries."""
    mono = _monomer(name, smiles, head, tail)
    chain = simple_backend([mono], [0] * 4, conformer_seed=2)
    mapping = unit_provenance(chain, mono, 4)
    applied = expand_by_provenance(chain, _plausible_types(mono), mapping,
                                   n_units=4)
    problems = check_all({a.index: a.element for a in chain.atoms}, applied)
    print(f"\n  {name}: {len(problems)} element problems")
    assert not problems, f"{name}: {problems[:2]}"


@pytest.mark.parametrize("name,smiles,head,tail", POLYMERS,
                         ids=[p[0] for p in POLYMERS])
def test_a_trimer_can_be_built_for_every_polymer(name, smiles, head, tail):
    """The typing view needs one for any chemistry, not just isoprene."""
    mono = _monomer(name, smiles, head, tail)
    ctx = build_trimer([mono])
    per_unit = len(mono.molecule.atoms)
    print(f"\n  {name}: {ctx.summary()}")
    assert len(ctx.molecule.atoms) == 3 * per_unit - 4
    assert len(ctx.link_bonds) == 2
    assert ctx.middle and ctx.ends


@pytest.mark.parametrize("name,smiles,head,tail", POLYMERS,
                         ids=[p[0] for p in POLYMERS])
def test_the_middle_unit_has_lost_both_caps(name, smiles, head, tail):
    """The general rule behind 135-vs-136, stated without chemistry."""
    mono = _monomer(name, smiles, head, tail)
    ctx = build_trimer([mono])

    def hydrogens_on(mol, k):
        return sum(1 for i, j, _o in mol.bonds
                   for other in ([j] if i == k else [i] if j == k else [])
                   if mol.atoms[other].element == "H")

    for link in (mono.head_index, mono.tail_index):
        middle = next(i for i, (u, m) in ctx.provenance.items()
                      if u == 1 and m == link)
        end_unit = 0 if link == mono.head_index else 2
        end = next(i for i, (u, m) in ctx.provenance.items()
                   if u == end_unit and m == link)
        h_mid = hydrogens_on(ctx.molecule, middle)
        h_end = hydrogens_on(ctx.molecule, end)
        print(f"\n  {name}: link atom {link} — middle {h_mid} H, end {h_end} H")
        assert h_end == h_mid + 1, \
            "the chain end should keep exactly one more hydrogen"


# ============================================ head/tail get their own types
def test_the_chain_ends_are_typed_as_ends_not_as_repeat_units():
    """A terminal CH3 is not a backbone CH2, whatever the polymer."""
    mono = _monomer("polyethylene", "CCCC", 0, 3)
    chain = simple_backend([mono], [0] * 4, conformer_seed=1)
    mapping = unit_provenance(chain, mono, 4)

    middle = _plausible_types(mono)          # link carbons -> 136
    ends = dict(middle)
    ends[mono.head_index] = "135"            # terminal CH3
    ends[mono.tail_index] = "135"

    applied = expand_by_provenance(chain, middle, mapping,
                                   head_types=ends, tail_types=ends, n_units=4)
    counts = Counter(applied.values())
    print(f"\n  {dict(sorted(counts.items()))}")
    assert counts["135"] == 4, "the two end units should carry terminal types"
    assert counts["136"] > 0, "the interior should still be repeat-unit types"


def test_without_terminal_sets_every_unit_uses_the_repeat_types():
    """Opt-in: nothing changes for callers that do not supply them."""
    mono = _monomer("polyethylene", "CCCC", 0, 3)
    chain = simple_backend([mono], [0] * 4, conformer_seed=1)
    mapping = unit_provenance(chain, mono, 4)
    applied = expand_by_provenance(chain, _plausible_types(mono), mapping,
                                   n_units=4)
    assert "135" not in set(applied.values())


def test_a_single_unit_chain_is_not_treated_as_having_two_ends():
    """n=1 is head and tail at once; it must not double-apply."""
    mono = _monomer("polyethylene", "CCCC", 0, 3)
    chain = simple_backend([mono], [0], conformer_seed=1)
    mapping = unit_provenance(chain, mono, 1)
    assert mapping is not None
    applied = expand_by_provenance(chain, _plausible_types(mono), mapping,
                                   n_units=1)
    assert len(applied) == len(chain.atoms)
