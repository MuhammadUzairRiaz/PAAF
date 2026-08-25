"""Manual atom types must land correctly for ANY polymer at ANY chain length.

This is the requirement stated plainly: it does not matter how many monomers
or how many chains — when types are assigned by hand they must be applied to
the atoms they were assigned to, and to no others.

Everything in this path was found and fixed against one polymer at a time —
polyisoprene exposed the methyl/methylene swap, PBS exposed the polyester end
cap. Fixing a bug against a single molecule is exactly how a fix quietly
encodes that molecule's shape, so the guard against it is a matrix: every
chemistry crossed with every chain length, checked atom by atom.

How it checks
-------------
Each atom of each role gets a DISTINCT marker rather than a plausible force
field type — ``h7`` for atom 7 of the head unit, ``t7`` for atom 7 of the tail,
``m7`` for the repeat unit. A marker is meaningful only at the exact atom it
was assigned to, so any misplacement shows up immediately instead of hiding
behind a type that happens to be right for the neighbouring atom too. Ground
truth comes from the mapping itself, so the assertion is "every atom carries
the marker its own (unit, monomer atom) was given" — not a count, not a
sample.

n = 1 is included deliberately: a single unit is head and tail at once and
keeps both caps, which is the case most likely to be handled by accident.
"""
from __future__ import annotations

import numpy as np
import pytest

from paaf.chain_builder import simple_backend                   # noqa: E402
from paaf.chain_provenance import (                             # noqa: E402
    expand_by_provenance, unit_provenance,
)
from paaf.structure import Atom, Molecule                       # noqa: E402
from paaf.type_guard import check_all                           # noqa: E402

rdkit = pytest.importorskip("rdkit", reason="needs RDKit")
from rdkit import Chem                                          # noqa: E402
from rdkit.Chem import AllChem                                  # noqa: E402

#: (name, capped repeat-unit SMILES, head atom, tail atom)
POLYMERS = [
    ("PBS",           "OCCCCOC(=O)CCC(=O)", 0, 10),
    ("polyisoprene",  "CC=C(C)C",           0, 4),
    ("polyethylene",  "CCCC",               0, 3),
    ("polypropylene", "CC(C)CC",            0, 4),
    ("polystyrene",   "CC(c1ccccc1)C",      0, 4),
    ("PMMA",          "CC(C)(C(=O)OC)C",    0, 7),
    ("PVC",           "CC(Cl)C",            0, 3),
    ("PEG",           "COCCOC",             0, 5),
    ("POM",           "COCOC",              0, 4),
    ("epoxidised",    "CC1(C)OC1C",         0, 5),
    ("PTFE",          "C(F)(F)C(F)F",       0, 3),
]

#: One unit, a couple, a normal run, and a long one.
LENGTHS = [1, 2, 3, 5, 10, 25]


def _monomer(name: str, smiles: str, head: int, tail: int):
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

    return Monomer(mol, head, tail, [cap(head)], [cap(tail)], name=name)


def _markers(monomer):
    """A distinct marker per (role, monomer atom): h7 / m7 / t7."""
    n = len(monomer.molecule.atoms)
    return {role: {k: f"{role[0]}{k}" for k in range(n)}
            for role in ("head", "middle", "tail")}


def _expected(mapping, roles, n_units):
    """What every chain atom must end up with, straight from the mapping."""
    want = {}
    for chain_index, (unit, monomer_index) in mapping.items():
        if unit == 0:
            role = "head"
        elif n_units > 1 and unit == n_units - 1:
            role = "tail"
        else:
            role = "middle"
        want[chain_index] = roles[role][monomer_index]
    return want


@pytest.mark.parametrize("n_units", LENGTHS)
@pytest.mark.parametrize("name,smiles,head,tail", POLYMERS,
                         ids=[p[0] for p in POLYMERS])
def test_every_atom_gets_exactly_the_type_it_was_assigned(
        name, smiles, head, tail, n_units):
    mono = _monomer(name, smiles, head, tail)
    chain = simple_backend([mono], [0] * n_units, conformer_seed=2)
    mapping = unit_provenance(chain, mono, n_units)
    assert mapping is not None, f"{name} x{n_units}: the chain could not be mapped"

    roles = _markers(mono)
    applied = expand_by_provenance(
        chain, roles["middle"], mapping,
        head_types=roles["head"], tail_types=roles["tail"], n_units=n_units)
    want = _expected(mapping, roles, n_units)

    wrong = [(i, applied.get(i), t) for i, t in want.items()
             if applied.get(i) != t]
    print(f"\n  {name} x{n_units}: {len(applied)}/{len(chain.atoms)} typed, "
          f"{len(wrong)} misplaced")
    assert not wrong, f"{name} x{n_units}: first wrong {wrong[0]}"
    assert len(applied) == len(chain.atoms), "some atoms were left untyped"


@pytest.mark.parametrize("name,smiles,head,tail", POLYMERS,
                         ids=[p[0] for p in POLYMERS])
def test_no_atom_is_given_another_element_s_type(name, smiles, head, tail):
    """Whatever else happens, mass and charge must belong to the right element."""
    by_element = {"C": "136", "H": "140", "O": "108",
                  "Cl": "264", "F": "164", "N": "739"}
    mono = _monomer(name, smiles, head, tail)
    chain = simple_backend([mono], [0] * 6, conformer_seed=2)
    types = {a.index: by_element[a.element]
             for a in mono.molecule.atoms if a.element in by_element}
    applied = expand_by_provenance(
        chain, types, unit_provenance(chain, mono, 6), n_units=6)
    problems = check_all({a.index: a.element for a in chain.atoms}, applied)
    print(f"\n  {name}: {len(problems)} element problems")
    assert not problems


@pytest.mark.parametrize("name,smiles,head,tail", POLYMERS,
                         ids=[p[0] for p in POLYMERS])
def test_a_single_unit_is_both_ends_at_once(name, smiles, head, tail):
    """n=1 keeps both caps, so it is neither a plain head nor a plain tail."""
    mono = _monomer(name, smiles, head, tail)
    chain = simple_backend([mono], [0], conformer_seed=2)
    mapping = unit_provenance(chain, mono, 1)
    assert mapping is not None
    roles = _markers(mono)
    applied = expand_by_provenance(
        chain, roles["middle"], mapping, head_types=roles["head"],
        tail_types=roles["tail"], n_units=1)
    assert len(applied) == len(chain.atoms)


def test_the_marker_scheme_would_actually_catch_a_misplacement():
    """A test that cannot fail proves nothing — check it can.

    Swap two markers and the comparison must object; otherwise the matrix
    above is green for the wrong reason.
    """
    mono = _monomer("polyethylene", "CCCC", 0, 3)
    chain = simple_backend([mono], [0] * 3, conformer_seed=2)
    mapping = unit_provenance(chain, mono, 3)
    roles = _markers(mono)
    applied = expand_by_provenance(
        chain, roles["middle"], mapping, head_types=roles["head"],
        tail_types=roles["tail"], n_units=3)
    want = _expected(mapping, roles, 3)

    first, second = sorted(applied)[:2]
    applied[first], applied[second] = applied[second], applied[first]
    wrong = [i for i, t in want.items() if applied.get(i) != t]
    print(f"\n  after swapping two atoms: {len(wrong)} misplacements detected")
    assert len(wrong) == 2, "the check would not notice a swap"
