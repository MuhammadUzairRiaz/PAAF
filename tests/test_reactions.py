"""Tests for the reaction learner + builder that don't need OpenBabel."""
from __future__ import annotations

import numpy as np
import pytest

from paaf.reaction import (
    ReactionLibrary, ReactionTemplate, _initial_mapping, _propagate,
    _assign_remaining, learn_reaction,
)
from paaf.structure import Atom, Molecule


def _ethanol():
    # CH3CH2OH (indices 0..8): C0,C1,O2,H3,H4,H5,H6,H7,H8
    atoms = [
        Atom(0, "C", np.array([0.0, 0.0, 0.0])),
        Atom(1, "C", np.array([1.5, 0.0, 0.0])),
        Atom(2, "O", np.array([2.2, 1.2, 0.0])),
        Atom(3, "H", np.array([-0.5, 0.9, 0.5])),
        Atom(4, "H", np.array([-0.5, -0.9, 0.5])),
        Atom(5, "H", np.array([-0.5, 0.0, -0.9])),
        Atom(6, "H", np.array([1.8, -0.5, 0.9])),
        Atom(7, "H", np.array([1.8, -0.5, -0.9])),
        Atom(8, "H", np.array([3.1, 1.2, 0.0])),
    ]
    bonds = [(0, 1, 1), (1, 2, 1), (0, 3, 1), (0, 4, 1), (0, 5, 1), (1, 6, 1), (1, 7, 1), (2, 8, 1)]
    return Molecule(atoms=atoms, bonds=bonds, name="EtOH")


def _ethoxide():  # remove the -OH hydrogen: CH3CH2O
    atoms = [
        Atom(0, "C", np.array([0.0, 0.0, 0.0])),
        Atom(1, "C", np.array([1.5, 0.0, 0.0])),
        Atom(2, "O", np.array([2.2, 1.2, 0.0])),
        Atom(3, "H", np.array([-0.5, 0.9, 0.5])),
        Atom(4, "H", np.array([-0.5, -0.9, 0.5])),
        Atom(5, "H", np.array([-0.5, 0.0, -0.9])),
        Atom(6, "H", np.array([1.8, -0.5, 0.9])),
        Atom(7, "H", np.array([1.8, -0.5, -0.9])),
    ]
    bonds = [(0, 1, 1), (1, 2, 1), (0, 3, 1), (0, 4, 1), (0, 5, 1), (1, 6, 1), (1, 7, 1)]
    return Molecule(atoms=atoms, bonds=bonds, name="EtO")


def test_wl_mapping_and_deletion_detected():
    r = _ethanol()
    p = _ethoxide()
    m, used, ev = _initial_mapping(r, p, max_radius=4)
    m, used, ev = _propagate(r, p, m, used, ev)
    m, used, ev = _assign_remaining(r, p, m, used, ev)

    # C and O atoms should be mapped; the -OH hydrogen (index 8) should not.
    assert 0 in m and 1 in m and 2 in m
    assert 8 not in m


def test_reaction_template_roundtrip(tmp_path):
    t = ReactionTemplate(
        name="dehydration",
        reactant="r.xyz",
        product="p.xyz",
        mapped_atoms={0: 0, 1: 1, 2: 2},
        deleted_atoms=[8],
        deleted_bonds=[(2, 8)],
        created_bonds=[],
        element_changes=[],
        evidence={0: "unique_WL_radius_3"},
    )
    lib = ReactionLibrary(templates=[t])
    path = lib.save(tmp_path / "lib.json")
    reloaded = ReactionLibrary.load(path)
    assert len(reloaded.templates) == 1
    assert reloaded.templates[0].deleted_atoms == [8]
    assert reloaded.templates[0].name == "dehydration"


def test_builder_library_lookup():
    from paaf.builder import get_recipe, list_library, LIBRARY
    assert len(list_library()) >= 15
    pbs = get_recipe("pbs")
    assert "OCCCCOC(=O)CCC" in pbs.smiles
    with pytest.raises(KeyError):
        get_recipe("not_a_polymer")


def test_polymer_database_loaded():
    """The 103-polymer database CSV must load and be searchable."""
    from paaf.builder import get_recipe, list_library, search_library

    all_rows = list_library("all")
    curated = list_library("curated")
    db = list_library("database")
    assert len(db) >= 100, f"expected >=100 DB entries, got {len(db)}"
    assert len(curated) >= 15
    # Total should be curated + db (deduped)
    assert len(all_rows) >= len(db)

    # Look-up by PID
    pe = get_recipe("W01_P001")
    assert pe.description == "Polyethylene"
    assert pe.tg_k is not None
    assert pe.density_kg_m3 is not None

    # Look-up by full polymer name
    ps = get_recipe("Polystyrene")
    assert ps.pid == "W01_P013"

    # Search should find Nylons
    nylons = search_library("nylon")
    assert len(nylons) >= 4
