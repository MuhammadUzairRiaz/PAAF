"""Why a 20-unit polyisoprene came out with one alkene carbon in it.

The exported chain contained 98 atoms typed 135 (CH3 alkane carbon), one
typed 142, one typed 144, and **none** typed 141 — for a polymer with twenty
carbon-carbon double bonds. Two independent faults combined:

1. Both chain builders wrote every bond as ``1.0``, because mBuild's Compound
   carries connectivity but not bond order. The C=C was destroyed at that
   conversion, so the SMARTS typer's alkene patterns had nothing to match and
   every carbon fell through to the alkane rules.

2. ``manual_types`` is keyed by *monomer* atom index and was written straight
   onto *chain* indices, so the user's typing landed on the first fifteen
   atoms and nowhere else.

Either alone produces a plausible-looking file that is chemically wrong, and
nothing downstream objects: the data file is well formed and the masses are
right.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from paaf.bond_orders import order_summary, perceive_bond_orders  # noqa: E402
from paaf.ff_assigner import expand_manual_types                  # noqa: E402
from paaf.structure import Atom, Molecule                         # noqa: E402


def _polyisoprene(n_units: int = 6) -> Molecule:
    """``-CH2-C(CH3)=CH-CH2-`` repeated, with every bond flattened to single.

    This is the shape the chain builders actually produced.
    """
    atoms, bonds = [], []
    # Real bond lengths, because that is what carries the bond order: a C=C is
    # ~1.34 A where a C-C is ~1.53. A fixture with uniform spacing would make
    # perception look broken when it is the fixture that is wrong.
    for u in range(n_units):
        b = u * 13
        origin = np.array([u * 9.0, 0.0, 0.0])
        zig = np.array([0.87, 0.79, 0.0])          # |.| ~ 1.18, scaled below

        def step(base, length, sign=1.0):
            d = zig * np.array([1.0, sign, 1.0])
            return base + d / np.linalg.norm(d) * length

        p0 = origin
        p1 = step(p0, 1.53, +1)
        p2 = step(p1, 1.34, -1)                    # the C=C
        p3 = step(p1, 1.51, +1) + np.array([0, 0, 1.2])   # methyl, off-plane
        p4 = step(p2, 1.51, +1)
        centres = [p0, p1, p2, p3, p4]
        for k, p in enumerate(centres):
            atoms.append(Atom(index=b + k, element="C", xyz=p))
        # Hydrogens at 1.09 A on their own carbons: 2 on C0, 1 on C2,
        # 3 on C3 (methyl), 2 on C4.
        owners = [0, 0, 2, 3, 3, 3, 4, 4]
        for k, owner in enumerate(owners):
            offset = np.array([0.0, 0.63 if k % 2 else -0.63,
                               0.89 if k % 3 else -0.89])
            h = centres[owner] + offset / np.linalg.norm(offset) * 1.09
            atoms.append(Atom(index=b + 5 + k, element="H", xyz=h))
        bonds += [(b + 0, b + 1, 1.0), (b + 1, b + 2, 1.0),
                  (b + 1, b + 3, 1.0), (b + 2, b + 4, 1.0)]
        bonds += [(b + 0, b + 5, 1.0), (b + 0, b + 6, 1.0), (b + 2, b + 7, 1.0),
                  (b + 3, b + 8, 1.0), (b + 3, b + 9, 1.0), (b + 3, b + 10, 1.0),
                  (b + 4, b + 11, 1.0), (b + 4, b + 12, 1.0)]
        if u:
            bonds.append((b - 13 + 4, b + 0, 1.0))
    return Molecule(atoms=atoms, bonds=bonds, name="polyisoprene")


#: One repeat unit typed the way the user's hand-built reference does.
ONE_UNIT = {0: "136", 1: "141", 2: "142", 3: "135", 4: "136",
            5: "140", 6: "140", 7: "144", 8: "140",
            9: "140", 10: "140", 11: "140", 12: "140"}


# ================================================ bond orders were destroyed
def test_the_builders_hand_over_a_chain_with_no_double_bonds():
    """The starting point: mBuild gives connectivity only."""
    chain = _polyisoprene()
    orders = order_summary(chain)
    print(f"\n  as built: {orders}")
    assert orders.get(2, 0) == 0, "this fixture is meant to start flat"


@pytest.mark.parametrize("smiles,expected", [
    ("CC=C(C)C", 1),          # isoprene repeat unit
    ("CCCC", 0),              # saturated: nothing to find
    ("CC(=O)OC", 1),          # ester carbonyl
    ("C=CC=C", 2),            # butadiene
])
def test_perception_recovers_exactly_the_right_number(smiles, expected):
    """Neither losing a double bond nor inventing one."""
    rdkit = pytest.importorskip("rdkit")
    from rdkit import Chem
    from rdkit.Chem import AllChem

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
        bonds=[(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), 1.0)   # flattened
               for b in m.GetBonds()],
        name=smiles)
    perceive_bond_orders(mol, prefer_rdkit=False)
    got = order_summary(mol).get(2, 0)
    print(f"\n  {smiles}: found {got} double bonds (expected {expected})")
    assert got == expected


def test_a_saturated_chain_gains_nothing():
    """Inventing a C=C would type a saturated carbon as an alkene."""
    n = 12
    mol = Molecule(
        atoms=[Atom(index=i, element="C",
                    xyz=np.array([1.53 * i, 0.0, 0.0])) for i in range(n)],
        bonds=[(i, i + 1, 1.0) for i in range(n - 1)], name="alkane")
    raised = perceive_bond_orders(mol, prefer_rdkit=False)
    print(f"\n  raised {raised} bonds on a 1.53 Å backbone")
    assert raised == 0


def test_perception_is_idempotent():
    """Running it twice must not keep promoting bonds."""
    rdkit = pytest.importorskip("rdkit")
    chain = _polyisoprene(3)
    first = perceive_bond_orders(chain, prefer_rdkit=False)
    second = perceive_bond_orders(chain, prefer_rdkit=False)
    print(f"\n  first pass {first}, second pass {second}")
    assert second == 0


# ==================================== the typing reached only one unit
#
# These used to assert that types were spread across chemical equivalence
# classes. That mechanism has been removed: it guessed, and its guess put a
# hydrogen type onto two backbone carbons. The contract now is an exact,
# element-verified mapping — or nothing at all.
#
# The positive case (types reaching every unit, correctly) is covered against a
# real monomer in test_manual_types_reach_the_chain.py. What belongs here is
# the refusal.
def test_nothing_is_applied_without_the_monomer_to_map_against():
    """Silence beats typing the wrong atoms."""
    chain = _polyisoprene(6)
    applied = expand_manual_types(chain, ONE_UNIT)
    print(f"\n  {len(ONE_UNIT)} types offered, {len(applied)} applied")
    assert applied == {}, "types were placed without a verified mapping"


def test_nothing_is_applied_when_the_unit_count_cannot_be_verified():
    chain = _polyisoprene(6)

    class _NotOurMonomer:
        molecule = _polyisoprene(1)
        head_removes = [0]
        tail_removes = [1]
        name = "wrong"

    applied = expand_manual_types(chain, ONE_UNIT,
                                  monomer=_NotOurMonomer(), n_units=6)
    print(f"\n  mismatched monomer -> {len(applied)} applied")
    assert applied == {}


def test_an_already_complete_map_is_left_alone():
    chain = _polyisoprene(2)
    full = {a.index: ("140" if a.element == "H" else "135")
            for a in chain.atoms}
    assert expand_manual_types(chain, full) == full


# ============================================== bond orders, still required
def test_the_chain_regains_its_double_bonds():
    """Independent of typing: the C=C must survive assembly."""
    chain = _polyisoprene(6)
    perceive_bond_orders(chain, prefer_rdkit=False)
    orders = order_summary(chain)
    print(f"\n  {orders}")
    assert orders.get(2, 0) >= 6, "one C=C per unit expected"
