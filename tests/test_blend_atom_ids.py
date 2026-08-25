"""Atom ids must not collide when components are merged.

The bug this pins: ``_replicate_component_atoms`` offset atom ids only
*within* a component, while ``_replicate_component_topology`` offset them
across components too. So the second component's atoms restarted at 1 and
overwrote the first component's numbering, while its bonds correctly pointed
past the end — at atoms that no longer existed.

Every summary of the broken file looked right. The atom count was right, the
molecule count was right, every chain was present. Only the ids were wrong,
and nothing checked them until OVITO refused to open the file.
"""
from __future__ import annotations

import pytest

from paaf.blend_replicator import (                              # noqa: E402
    BlendComponent, _replicate_component_atoms,
    _replicate_component_topology, validate_merged_topology,
)


def _component(name: str, natoms: int, nbonds: int, count: int
               ) -> BlendComponent:
    """A component with `natoms` atoms/chain, numbered 1..natoms."""
    atoms = [f"{i} 1 {(i % 3) + 1} 0.0 {i}.0 0.0 0.0"
             for i in range(1, natoms + 1)]
    bonds = [f"{i} 1 {i} {i + 1}" for i in range(1, nbonds + 1)]
    comp = BlendComponent(name=name, data_file=None, count=count)
    comp.natoms_chain = natoms
    comp.sections = {"Atoms": atoms, "Bonds": bonds}
    comp.ntypes = {"atom": 3, "bond": 1, "angle": 0,
                   "dihedral": 0, "improper": 0}
    return comp


def _merge(components):
    """Mimic replicate_blend's merge loop and return (atoms, bonds)."""
    atoms, bonds = [], []
    running_atomid = running_molid = running_bondid = 0
    for comp in components:
        coords = [(0.0, 0.0, 0.0)] * (comp.count * comp.natoms_chain)
        atoms.extend(_replicate_component_atoms(
            comp, coords, running_atomid, 0, running_molid))
        bonds.extend(_replicate_component_topology(
            comp, "Bonds", running_atomid, 0, running_bondid))
        running_atomid += comp.count * comp.natoms_chain
        running_bondid += comp.count * len(comp.sections["Bonds"])
        running_molid += comp.count
    return atoms, bonds


# ============================================================= the bug itself
def test_two_components_never_share_an_atom_id():
    """ENR (27 atoms) + PBS (136 atoms), 20 chains each — the reported case."""
    atoms, _bonds = _merge([_component("PBS", 136, 135, 20),
                            _component("ENR", 27, 26, 20)])
    ids = [int(l.split()[0]) for l in atoms]
    print(f"\n  {len(atoms)} atoms, {len(set(ids))} unique ids, "
          f"range {min(ids)}–{max(ids)}")
    assert len(ids) == 3260, "wrong atom count"
    assert len(set(ids)) == 3260, (
        f"{len(ids) - len(set(ids))} duplicated atom ids — the second "
        f"component restarted numbering at 1")
    assert sorted(ids) == list(range(1, 3261)), "ids are not 1..N contiguous"


def test_every_bond_points_at_an_atom_that_exists():
    """The condition OVITO reports as 'Nonexistent atom ID'."""
    atoms, bonds = _merge([_component("PBS", 136, 135, 20),
                           _component("ENR", 27, 26, 20)])
    ids = {int(l.split()[0]) for l in atoms}
    dangling = [b for b in bonds
                if any(int(t) not in ids for t in b.split()[2:])]
    print(f"\n  {len(bonds)} bonds, {len(dangling)} dangling")
    assert not dangling, f"e.g. {dangling[:3]}"


def test_the_second_component_starts_where_the_first_ended():
    atoms, _ = _merge([_component("A", 10, 9, 3),
                       _component("B", 4, 3, 2)])
    first = [int(l.split()[0]) for l in atoms[:30]]
    second = [int(l.split()[0]) for l in atoms[30:]]
    print(f"\n  A ends at {max(first)}, B runs {min(second)}–{max(second)}")
    assert max(first) == 30
    assert min(second) == 31 and max(second) == 38


def test_molecule_ids_stay_correct_too():
    """Molecule numbering was already right — it must stay right."""
    atoms, _ = _merge([_component("A", 10, 9, 3),
                       _component("B", 4, 3, 2)])
    from collections import Counter
    per_mol = Counter(int(l.split()[1]) for l in atoms)
    print(f"\n  {dict(per_mol)}")
    assert dict(per_mol) == {1: 10, 2: 10, 3: 10, 4: 4, 5: 4}


def test_atom_type_and_id_offsets_are_independent():
    """Type ids and atom ids are offset by different amounts; don't conflate."""
    comp = _component("B", 4, 3, 1)
    lines = _replicate_component_atoms(comp, [(0.0, 0.0, 0.0)] * 4,
                                       atom_id_offset=100,
                                       atom_type_offset=7, molid_offset=2)
    got = [(int(l.split()[0]), int(l.split()[1]), int(l.split()[2]))
           for l in lines]
    print(f"\n  (id, mol, type): {got}")
    assert [g[0] for g in got] == [101, 102, 103, 104]
    assert all(g[1] == 3 for g in got)
    assert [g[2] for g in got] == [9, 10, 8, 9]      # (i%3)+1 + 7


# =============================================================== the guard
def test_the_guard_catches_duplicated_atom_ids():
    """What should have fired instead of writing a broken file."""
    atoms = ["1 1 1 0.0 0 0 0", "2 1 1 0.0 0 0 0", "1 2 1 0.0 0 0 0"]
    with pytest.raises(RuntimeError) as e:
        validate_merged_topology(atoms, {})
    print(f"\n  {e.value}")
    assert "duplicated atom id" in str(e.value)
    assert "has not been written" in str(e.value)


def test_the_guard_catches_a_bond_to_a_missing_atom():
    atoms = ["1 1 1 0.0 0 0 0", "2 1 1 0.0 0 0 0"]
    with pytest.raises(RuntimeError) as e:
        validate_merged_topology(atoms, {"Bonds": ["1 1 2 9"]})
    print(f"\n  {e.value}")
    assert "Nonexistent atom ID" in str(e.value)
    assert "highest real id is 2" in str(e.value)


def test_the_guard_names_the_section_at_fault():
    atoms = ["1 1 1 0.0 0 0 0", "2 1 1 0.0 0 0 0"]
    with pytest.raises(RuntimeError) as e:
        validate_merged_topology(
            atoms, {"Bonds": ["1 1 1 2"], "Angles": ["1 1 1 2 77"]})
    assert "Angles" in str(e.value)


def test_a_sound_merge_passes_the_guard():
    atoms, bonds = _merge([_component("PBS", 136, 135, 20),
                           _component("ENR", 27, 26, 20)])
    validate_merged_topology(atoms, {"Bonds": bonds})    # must not raise
    print(f"\n  {len(atoms)} atoms / {len(bonds)} bonds accepted")
