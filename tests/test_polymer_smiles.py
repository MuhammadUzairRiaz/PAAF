"""Expanding a polymerization SMILES into an n-mer.

The reaction scheme uses this so a user can pick a polymer from the library,
set one chain length, and have BOTH sides of the reaction expand to the same
number of repeat units.
"""
from __future__ import annotations

import pytest

from paaf.polymer_smiles import (
    expand_if_polymer, expand_repeat_unit, is_polymer_smiles, strip_atom_maps,
)


# ------------------------------------------------------------- detection
def test_detects_polymerization_smiles():
    assert is_polymer_smiles("[*]CC[*]")
    assert is_polymer_smiles("[*]CC([*])c1ccccc1")
    assert not is_polymer_smiles("CCO")
    assert not is_polymer_smiles("O")


def test_strip_atom_maps_keeps_the_atom():
    assert strip_atom_maps("CC(=O)[OH:1]") == "CC(=O)[OH]"
    assert strip_atom_maps("[C:2]CO") == "[C]CO"
    assert strip_atom_maps("CCO") == "CCO"


# ------------------------------------------------------------- expansion
def test_terminal_form_repeats_head_to_tail():
    assert expand_repeat_unit("[*]CC[*]", 1) == "CC"
    assert expand_repeat_unit("[*]CC[*]", 3) == "CCCCCC"
    assert expand_repeat_unit("[*]OCCCCOC(=O)CCC(=O)[*]", 2) == (
        "OCCCCOC(=O)CCC(=O)OCCCCOC(=O)CCC(=O)")


def test_branch_form_keeps_the_substituent_off_the_backbone():
    """Regression: naive wildcard deletion put the phenyl IN the backbone.

    ``[*]CC([*])c1ccccc1`` is polystyrene — the ring is a substituent on the
    second carbon, not a backbone member. Repeating ``CCc1ccccc1`` would chain
    through the ring, which is a different (and wrong) molecule.
    """
    assert expand_repeat_unit("[*]CC([*])c1ccccc1", 2) == (
        "CC(c1ccccc1)CC(c1ccccc1)")
    assert expand_repeat_unit("[*]CC([*])Cl", 3) == "CC(Cl)CC(Cl)CC(Cl)"
    # The ring must never be immediately followed by a backbone carbon.
    assert "c1ccccc1C" not in expand_repeat_unit("[*]CC([*])c1ccccc1", 2)


def test_atom_maps_survive_on_exactly_one_unit():
    """A reaction happens at ONE site; repeating the maps would collide."""
    out = expand_repeat_unit("[*]CC(=O)[OH:1][*]", 3)
    assert out.count(":1") == 1
    assert out.endswith("[OH:1]")          # default: the chain end

    first = expand_repeat_unit("[*]CC(=O)[OH:1][*]", 3, maps_on="first")
    assert first.count(":1") == 1
    assert first.startswith("CC(=O)[OH:1]")

    none = expand_repeat_unit("[*]CC(=O)[OH:1][*]", 3, maps_on="none")
    assert ":1" not in none


@pytest.mark.parametrize("n", [0, -5, None])
def test_degenerate_counts_give_a_single_unit(n):
    assert expand_repeat_unit("[*]CC[*]", n) == "CC"


def test_small_molecules_pass_through_untouched():
    """Water and other co-reactants are not polymers."""
    assert expand_if_polymer("O", 10) == "O"
    assert expand_if_polymer("CC(=O)[OH:1]", 10) == "CC(=O)[OH:1]"
    assert expand_if_polymer("[*]CC[*]", 2) == "CCCC"


def test_same_n_gives_the_same_backbone_on_both_sides():
    """The point of the feature: one n keeps reactant and product matched."""
    n = 5
    reactant = expand_if_polymer("[*]OCCCCOC(=O)CCC(=O)[*]", n)
    product = expand_if_polymer("[*]OCCCCOC(=O)CCC(=O)[*]", n)
    assert reactant == product
    assert reactant.count("OCCCCOC(=O)CCC(=O)") == n
