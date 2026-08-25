"""The user's manual atom types must land on the atoms they were meant for.

What went wrong
---------------
The typing dialog returns ``{monomer_atom_index: type}``. The pipeline wrote
that straight onto the chain, whose numbering differs because a cap hydrogen
is deleted at every junction. So the types landed on the first fifteen chain
atoms and nowhere else.

A first attempt to fix it spread types across chemical equivalence classes and
made things worse: it read the source atom's element from the *chain* rather
than the monomer, so a hydrogen's type could be attached to a class of carbons
and propagated. Two backbone CH2 carbons were written with type 140 and mass
1.008 — a carbon masquerading as a hydrogen, in a file LAMMPS would have read
without complaint.

The fix is an exact, element-verified mapping, and a refusal to apply anything
when that mapping cannot be verified. These tests pin both halves, using the
user's own 1,4-isoprene reference as ground truth.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from paaf.chain_provenance import (                            # noqa: E402
    expand_by_provenance, unit_provenance,
)
from paaf.ff_assigner import expand_manual_types                # noqa: E402
from paaf.structure import Atom, Molecule                       # noqa: E402

rdkit = pytest.importorskip("rdkit", reason="needs RDKit")
from rdkit import Chem                                          # noqa: E402
from rdkit.Chem import AllChem                                  # noqa: E402


def _isoprene_monomer():
    """Capped 1,4-isoprene repeat unit; links at C0 and C4."""
    from paaf.monomer import Monomer

    m = Chem.AddHs(Chem.MolFromSmiles("CC=C(C)C"))
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
        name="isoprene")

    def cap(k):
        return [n.GetIdx() for n in m.GetAtomWithIdx(k).GetNeighbors()
                if n.GetSymbol() == "H"][0]

    return Monomer(molecule=mol, head_index=0, tail_index=4,
                   head_removes=[cap(0)], tail_removes=[cap(4)],
                   name="isoprene")


def _chain(monomer, n_units: int):
    from paaf.chain_builder import simple_backend

    return simple_backend([monomer], [0] * n_units, conformer_seed=1)


def _reference_types(monomer):
    """The user's hand-built assignment, keyed by monomer atom index."""
    mol = monomer.molecule
    types = {0: "136", 1: "142", 2: "141", 3: "135", 4: "136"}
    alkene_h = [n for n in range(len(mol.atoms))
                if mol.atoms[n].element == "H"
                and any({i, j} == {n, 1} for i, j, _o in mol.bonds)]
    for a in mol.atoms:
        if a.element == "H":
            types[a.index] = "144" if a.index in alkene_h else "140"
    return types


# =============================================== the mapping is exact
def test_the_chain_maps_back_to_the_monomer_atom_by_atom():
    mono = _isoprene_monomer()
    chain = _chain(mono, 3)
    mapping = unit_provenance(chain, mono, 3)
    print(f"\n  {len(chain.atoms)} chain atoms -> "
          f"{len(mapping) if mapping else 0} mapped")
    assert mapping is not None, "the chain could not be matched to its monomer"
    assert len(mapping) == len(chain.atoms)


def test_every_mapped_atom_has_the_right_element():
    """One disagreement invalidates the mapping; there should be none."""
    mono = _isoprene_monomer()
    chain = _chain(mono, 4)
    mapping = unit_provenance(chain, mono, 4)
    assert mapping is not None
    for c_index, (_unit, m_index) in mapping.items():
        assert (chain.atoms[c_index].element
                == mono.molecule.atoms[m_index].element)


def test_the_mapping_covers_every_unit():
    mono = _isoprene_monomer()
    chain = _chain(mono, 5)
    mapping = unit_provenance(chain, mono, 5)
    units = Counter(u for u, _k in mapping.values())
    print(f"\n  atoms per unit: {dict(sorted(units.items()))}")
    assert set(units) == {0, 1, 2, 3, 4}


def test_a_mismatched_chain_is_refused_not_guessed():
    """Refusing beats typing the wrong atoms."""
    mono = _isoprene_monomer()
    chain = _chain(mono, 3)
    # Claim a unit count the chain cannot possibly have.
    assert unit_provenance(chain, mono, 7) is None


# ================================== the user's types reach the whole chain
def test_manual_types_are_applied_to_every_unit():
    mono = _isoprene_monomer()
    chain = _chain(mono, 3)
    applied = expand_manual_types(chain, _reference_types(mono),
                                  monomer=mono, n_units=3)
    print(f"\n  {len(_reference_types(mono))} monomer types -> "
          f"{len(applied)} of {len(chain.atoms)} chain atoms")
    assert len(applied) == len(chain.atoms), "some atoms were left untyped"


def test_the_exported_composition_is_right():
    """The numbers that exposed the bug: 3 units of isoprene."""
    mono = _isoprene_monomer()
    chain = _chain(mono, 3)
    applied = expand_manual_types(chain, _reference_types(mono),
                                  monomer=mono, n_units=3)
    counts = Counter(applied.values())
    print(f"\n  {dict(sorted(counts.items()))}")

    assert counts["141"] == 3, "one alkene carbon per unit"
    assert counts["142"] == 3, "one alkene CH per unit"
    assert counts["144"] == 3, "one alkene H per unit"
    # 3 methyls + the two chain-end CH2s, which the monomer calls 136.
    assert counts["135"] + counts["136"] == 9


def test_no_carbon_is_ever_given_a_hydrogen_type():
    """The exact corruption from the broken export: mass 1.008 on a carbon."""
    from paaf.type_guard import check_all

    mono = _isoprene_monomer()
    chain = _chain(mono, 3)
    applied = expand_manual_types(chain, _reference_types(mono),
                                  monomer=mono, n_units=3)
    problems = check_all({a.index: a.element for a in chain.atoms}, applied)
    print(f"\n  element problems: {problems or 'none'}")
    assert not problems


def test_the_carbon_and_hydrogen_counts_survive():
    """13 C / 28 H was the signature of two carbons typed as hydrogens."""
    from paaf.type_guard import type_elements

    mono = _isoprene_monomer()
    chain = _chain(mono, 3)
    applied = expand_manual_types(chain, _reference_types(mono),
                                  monomer=mono, n_units=3)
    table = type_elements()
    by_type = Counter(table.get(t, "?") for t in applied.values())
    by_atom = Counter(a.element for a in chain.atoms)
    print(f"\n  from the types: {dict(by_type)}")
    print(f"  from the atoms: {dict(by_atom)}")
    assert by_type["C"] == by_atom["C"], "a carbon was typed as something else"
    assert by_type["H"] == by_atom["H"]


# ============================================= refusing rather than guessing
def test_nothing_is_applied_without_a_monomer():
    mono = _isoprene_monomer()
    chain = _chain(mono, 3)
    assert expand_manual_types(chain, _reference_types(mono)) == {}


def test_nothing_is_applied_when_the_unit_count_is_wrong():
    mono = _isoprene_monomer()
    chain = _chain(mono, 3)
    assert expand_manual_types(chain, _reference_types(mono),
                               monomer=mono, n_units=9) == {}


def test_a_complete_map_is_passed_through_untouched():
    mono = _isoprene_monomer()
    chain = _chain(mono, 2)
    full = {a.index: ("140" if a.element == "H" else "135")
            for a in chain.atoms}
    assert expand_manual_types(chain, full, monomer=mono, n_units=2) == full


def test_an_assignment_with_the_wrong_element_is_dropped():
    """Even with a valid mapping, a bad type must not be written."""
    mono = _isoprene_monomer()
    chain = _chain(mono, 3)
    mapping = unit_provenance(chain, mono, 3)
    broken = dict(_reference_types(mono))
    # Put a carbon type on a monomer hydrogen.
    a_hydrogen = next(a.index for a in mono.molecule.atoms
                      if a.element == "H" and a.index in broken)
    broken[a_hydrogen] = "136"
    applied = expand_by_provenance(chain, broken, mapping)
    kept = {chain.atoms[i].element for i, t in applied.items() if t == "136"}
    print(f"\n  elements that kept 136: {kept}")
    assert kept <= {"C"}, "a hydrogen was given a carbon type"


def test_scaling_to_twenty_units_still_types_everything():
    mono = _isoprene_monomer()
    chain = _chain(mono, 20)
    applied = expand_manual_types(chain, _reference_types(mono),
                                  monomer=mono, n_units=20)
    counts = Counter(applied.values())
    print(f"\n  20 units: {len(applied)}/{len(chain.atoms)} typed; "
          f"141={counts['141']} 142={counts['142']} 144={counts['144']}")
    assert len(applied) == len(chain.atoms)
    assert counts["141"] == 20 and counts["142"] == 20 and counts["144"] == 20
