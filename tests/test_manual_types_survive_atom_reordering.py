"""Manual types must land correctly when the builder reorders atoms.

Why this exists
---------------
``test_manual_types_any_polymer_any_length.py`` builds its chains with
``simple_backend``, which emits atoms in monomer order. So the positional
mapping always succeeds there and the whole repair path underneath it — the
part that runs on the user's real chains — was never exercised. mBuild does
NOT preserve monomer order; the user's own log says so::

    Positional mapping disagrees on 92 atoms (e.g. chain 12 is ('H', 0, ('C',)))

Reordering is simulated here rather than assumed away, by shuffling the built
chain and rebuilding its bonds. Measured against the code as it stood, every
polyester refused outright from 5 units upward::

     5 units: MAPPING REFUSED
   100 units: MAPPING REFUSED

and refusing means the user's hand-typing is silently dropped and the
automatic typer does the whole chain.

The cause was structural. ``_match_by_environment`` repaired *which monomer
atom* each chain atom came from, but read *which unit* it belonged to out of
the positional mapping — the same mapping whose failure had triggered the
repair. A reordering builder does not keep atoms inside their own unit, so
that unit number was fiction and every atom was matched against the wrong
unit's reference. ``_match_by_graph`` replaces it: the reference chain is
built as a graph whose nodes already know their ``(unit, monomer atom)``, both
graphs are colour-refined, and atoms are paired within matching colours. Atom
order never enters into it.

What counts as correct
----------------------
Not "the same monomer atom index". Two hydrogens on one CH2 are the same
atom as far as any force field is concerned, and which of them a matcher picks
is not a fact about the chain. The same is true of the two ends of a symmetric
chain like polyethylene, where head and tail are genuinely interchangeable.

So the assertion is that each atom receives a type belonging to an atom in the
**same two-shell chemical environment** — element, neighbours, and neighbours'
neighbours. That is strictly stronger than what the code checks internally
(one shell), so it cannot pass merely by agreeing with the implementation, and
it is exactly the granularity at which a force-field type is defined.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from paaf.chain_builder import simple_backend                   # noqa: E402
from paaf.chain_provenance import (                             # noqa: E402
    _deep_fingerprints, _deep_in_chain_fingerprints, unit_provenance,
)
from paaf.structure import Atom, Molecule                       # noqa: E402

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

#: One unit, a short chain, a normal one, and the length the user asked about.
LENGTHS = [1, 5, 25, 100]

CAP_UNIT = -1


def _monomer(name, smiles, head, tail):
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


def _reorder(chain, seed, keep_last=0):
    """The same molecule with its atoms emitted in a different order.

    ``keep_last`` leaves a trailing run in place, which is where a builder
    that caps the chain after assembly puts the atoms it appends.

    Returns the reordered molecule and ``{old index: new index}``, which is
    what lets the test say where each atom went.
    """
    n = len(chain.atoms)
    body = list(range(n - keep_last))
    random.Random(seed).shuffle(body)
    order = body + list(range(n - keep_last, n))
    new_of = {old: new for new, old in enumerate(order)}
    return Molecule(
        atoms=[Atom(index=new, element=chain.atoms[old].element,
                    xyz=chain.atoms[old].xyz.copy())
               for new, old in enumerate(order)],
        bonds=[(new_of[int(i)], new_of[int(j)], o) for i, j, o in chain.bonds],
        name=chain.name), new_of


def _role(unit, n_units):
    if n_units == 1:
        return "single"          # one unit keeps BOTH caps: the bare monomer
    if unit == 0:
        return "head"
    if unit == n_units - 1:
        return "tail"
    return "middle"


def _environments(monomer):
    """Two-shell environment of every monomer atom, per chain role."""
    env = _deep_in_chain_fingerprints(monomer)
    assert env is not None, "could not build the reference trimer"
    env["single"] = _deep_fingerprints(monomer.molecule)
    return env


def _misassigned(mapping, truth, env, n_units):
    """Atoms given a type from a genuinely different chemical environment."""
    wrong = []
    for chain_index, (unit, monomer_index) in mapping.items():
        if unit == CAP_UNIT:
            continue
        got = env[_role(unit, n_units)].get(monomer_index)
        want_role, want_index = truth[chain_index]
        want = env[want_role].get(want_index)
        if got != want:
            wrong.append((chain_index, got, want))
    return wrong


# ============================================ the matrix: chemistry x length
@pytest.mark.parametrize("n_units", LENGTHS)
@pytest.mark.parametrize("name,smiles,head,tail", POLYMERS,
                         ids=[p[0] for p in POLYMERS])
def test_types_land_correctly_however_the_builder_orders_its_atoms(
        name, smiles, head, tail, n_units):
    mono = _monomer(name, smiles, head, tail)
    ordered = simple_backend([mono], [0] * n_units, conformer_seed=2)

    # Ground truth from the ordered chain, where the positional map is exact.
    base = unit_provenance(ordered, mono, n_units)
    assert base is not None, f"{name} x{n_units}: the ordered chain refused"

    shuffled, new_of = _reorder(ordered, seed=n_units)
    mapping = unit_provenance(shuffled, mono, n_units)
    assert mapping is not None, (
        f"{name} x{n_units}: reordering the atoms made the chain "
        f"unrecognisable, so every manual type would be dropped")

    env = _environments(mono)
    truth = {new_of[c]: (_role(u, n_units), m) for c, (u, m) in base.items()}
    wrong = _misassigned(mapping, truth, env, n_units)
    print(f"\n  {name} x{n_units}: {len(shuffled.atoms)} atoms, "
          f"{len(wrong)} in the wrong environment")
    assert not wrong, f"{name} x{n_units}: first wrong {wrong[0]}"


# ================================================ the length the user asked
@pytest.mark.parametrize("n_units", [100, 200])
def test_a_long_pbs_chain_is_mapped_atom_for_atom(n_units):
    """"if i want to build 100 monomers pbs chain it should work the same way"."""
    mono = _monomer("PBS", "OCCCCOC(=O)CCC(=O)", 0, 10)
    ordered = simple_backend([mono], [0] * n_units, conformer_seed=2)
    base = unit_provenance(ordered, mono, n_units)
    shuffled, new_of = _reorder(ordered, seed=1)
    mapping = unit_provenance(shuffled, mono, n_units)
    assert mapping is not None
    assert len(mapping) == len(shuffled.atoms), "some atoms went unmapped"

    env = _environments(mono)
    truth = {new_of[c]: (_role(u, n_units), m) for c, (u, m) in base.items()}
    wrong = _misassigned(mapping, truth, env, n_units)
    print(f"\n  PBS x{n_units}: {len(shuffled.atoms)} atoms, {len(wrong)} wrong")
    assert not wrong


def test_the_chain_ends_keep_their_own_types_at_length():
    """The ends are the whole reason units are tracked separately.

    A hundred units is ninety-eight identical interiors and two ends. If the
    ends were quietly folded into the interior nothing above would notice,
    because a wrong end atom is 1% of the chain.
    """
    n_units = 100
    mono = _monomer("PBS", "OCCCCOC(=O)CCC(=O)", 0, 10)
    ordered = simple_backend([mono], [0] * n_units, conformer_seed=2)
    shuffled, _ = _reorder(ordered, seed=1)
    mapping = unit_provenance(shuffled, mono, n_units)
    units = {u for u, _m in mapping.values()}
    per_unit = {u: sum(1 for v in mapping.values() if v[0] == u)
                for u in (0, 1, n_units - 1)}
    print(f"\n  {len(units)} distinct units; head {per_unit[0]} atoms, "
          f"interior {per_unit[1]}, tail {per_unit[n_units - 1]}")
    assert 0 in units and n_units - 1 in units, "an end unit vanished"
    # Each end keeps a cap the interior gave up at its junction.
    assert per_unit[0] == per_unit[1] + 1
    assert per_unit[n_units - 1] == per_unit[1] + 1


# ======================================================== negative controls
def test_the_environment_check_would_notice_a_real_misplacement():
    """A test that cannot fail proves nothing.

    Swapping two atoms of genuinely different environments must be reported;
    otherwise the matrix above is green for the wrong reason.
    """
    mono = _monomer("PBS", "OCCCCOC(=O)CCC(=O)", 0, 10)
    ordered = simple_backend([mono], [0] * 5, conformer_seed=2)
    base = unit_provenance(ordered, mono, 5)
    shuffled, new_of = _reorder(ordered, seed=5)
    mapping = dict(unit_provenance(shuffled, mono, 5))
    env = _environments(mono)
    truth = {new_of[c]: (_role(u, 5), m) for c, (u, m) in base.items()}
    assert not _misassigned(mapping, truth, env, 5), "should start clean"

    # Find a carbon and an oxygen and give the carbon the oxygen's source.
    carbon = next(c for c, (u, m) in mapping.items()
                  if u != CAP_UNIT and shuffled.atoms[c].element == "C")
    oxygen = next(c for c, (u, m) in mapping.items()
                  if u != CAP_UNIT and shuffled.atoms[c].element == "O")
    mapping[carbon] = mapping[oxygen]
    wrong = _misassigned(mapping, truth, env, 5)
    print(f"\n  after giving a carbon an oxygen's provenance: "
          f"{len(wrong)} detected")
    assert wrong, "a carbon typed as an oxygen went unnoticed"


def test_a_chain_of_the_wrong_monomer_is_still_refused():
    """Robustness must not become credulity.

    The matcher no longer trusts atom order — it must still refuse a chain
    that is not this monomer repeated, rather than finding some mapping.
    """
    pbs = _monomer("PBS", "OCCCCOC(=O)CCC(=O)", 0, 10)
    peg = _monomer("PEG", "COCCOC", 0, 5)
    chain, _ = _reorder(simple_backend([peg], [0] * 5, conformer_seed=2),
                        seed=3)
    result = unit_provenance(chain, pbs, 5)
    print(f"\n  PEG chain against a PBS monomer -> "
          f"{'MAPPED' if result else 'refused'}")
    assert result is None, "a PEG chain was accepted as five units of PBS"


def test_a_reordered_chain_of_the_wrong_length_is_refused():
    mono = _monomer("PBS", "OCCCCOC(=O)CCC(=O)", 0, 10)
    chain, _ = _reorder(simple_backend([mono], [0] * 6, conformer_seed=2),
                        seed=3)
    assert unit_provenance(chain, mono, 5) is None, \
        "a six-unit chain was accepted as five units"


def test_reordering_is_reproducible_but_not_a_no_op():
    """Guard the test's own instrument.

    If ``_reorder`` quietly returned the chain unchanged, every assertion
    above would be testing the ordered path all over again.
    """
    mono = _monomer("polyethylene", "CCCC", 0, 3)
    chain = simple_backend([mono], [0] * 4, conformer_seed=2)
    shuffled, new_of = _reorder(chain, seed=1)
    moved = sum(1 for old, new in new_of.items() if old != new)
    print(f"\n  {moved}/{len(chain.atoms)} atoms changed position")
    assert moved > len(chain.atoms) // 2, "the shuffle barely moved anything"
    again, _ = _reorder(chain, seed=1)
    assert [a.element for a in again.atoms] == \
        [a.element for a in shuffled.atoms], "the shuffle is not reproducible"
    # Same molecule: same bond count, same element census.
    assert len(shuffled.bonds) == len(chain.bonds)
    assert sorted(a.element for a in shuffled.atoms) == \
        sorted(a.element for a in chain.atoms)
