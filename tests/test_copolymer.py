"""Tests for the shipped copolymer library."""
from __future__ import annotations


def test_copolymer_library_loads_enr_preset():
    from paaf.copolymer_library import (
        get_copolymer, list_copolymers, COPOLYMER_LIBRARY,
    )
    rows = list_copolymers()
    assert len(rows) >= 1
    r = get_copolymer("W01_P0ENR")
    assert r.smiles_a.startswith("[*]") and r.smiles_b.startswith("[*]")
    assert 0.0 <= r.fraction_b <= 1.0
    assert abs(r.fraction_a + r.fraction_b - 1.0) < 1e-9
    assert r.sequence_mode in ("random", "alternating", "block")


def test_copolymer_recipe_computes_fractions():
    from paaf.copolymer_library import CopolymerRecipe
    r = CopolymerRecipe(pid="X", smiles_a="[*]C[*]", smiles_b="[*]O[*]",
                        fraction_b=0.3, sequence_mode="random", random_seed=1)
    assert abs(r.fraction_a - 0.7) < 1e-9
    assert "70%" in r.description and "30%" in r.description
