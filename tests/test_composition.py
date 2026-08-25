"""Acceptance tests for the composition solver.

The solver turns what a chemist specifies (weight percent, target density,
DP) into what the grower needs (integer chain counts, a box edge). Every
measured value is printed.

The load-bearing property is that weight fraction is set by *mass*: chains of
different species have very different masses, so allocating chains in
proportion to weight without dividing by chain mass gives a cell that is
nowhere near the requested composition.
"""
from __future__ import annotations

import pytest

rdkit = pytest.importorskip("rdkit", reason="composition needs RDKit for masses")

from paaf.cell.composition import (                       # noqa: E402
    Component, box_edge_for_density, from_chain_counts,
    solve_from_weight_fractions,
)

PE = "[*]CC[*]"
PS = "[*]CC([*])c1ccccc1"
N_AVOGADRO = 6.02214076e23


def _pair(dp_pe=50, dp_ps=50, w_pe=70.0):
    return [
        Component(name="PE", repeat_unit=PE, degree_of_polymerisation=dp_pe,
                  weight_percent=w_pe, ris_key="PE"),
        Component(name="PS", repeat_unit=PS, degree_of_polymerisation=dp_ps,
                  weight_percent=100.0 - w_pe, ris_key="PS"),
    ]


# ------------------------------------------------------------ box arithmetic
def test_box_edge_reproduces_the_requested_density():
    """The inverse calculation must return the density that was asked for."""
    mass, want = 500_000.0, 0.95
    edge = box_edge_for_density(mass, want)
    got = (mass / N_AVOGADRO) / (edge ** 3 * 1e-24)
    print(f"\n  mass {mass:,.0f} amu at {want} g/cm³ -> edge {edge:.3f} Å")
    print(f"  back-calculated density: {got:.6f} g/cm³")
    assert abs(got - want) < 1e-9


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_density_is_refused(bad):
    with pytest.raises(ValueError):
        box_edge_for_density(1000.0, bad)


# ------------------------------------------------- weight fractions are mass
def test_weight_fraction_accounts_for_chain_mass_not_chain_count():
    """A DP-50 PS chain is 3.7x the mass of a DP-50 PE chain.

    If the solver allocated chains in proportion to weight percent directly,
    70 wt% PE would give 7 PE : 3 PS and a realised composition near 39 wt%
    PE. Dividing by chain mass is what makes it come out right.
    """
    comp = solve_from_weight_fractions(_pair(), 0.95, target_beads=4000)
    pe, ps = comp.chain_counts
    naive_pe_wt = (100.0 * 7 * comp.components[0].chain_mass
                   / (7 * comp.components[0].chain_mass
                      + 3 * comp.components[1].chain_mass))
    print(f"\n  chain masses  : PE {comp.components[0].chain_mass:,.0f}, "
          f"PS {comp.components[1].chain_mass:,.0f} g/mol")
    print(f"  solved counts : {pe} PE : {ps} PS")
    print(f"  realised      : {comp.realised_weight_percent[0]:.2f} wt% PE "
          f"(requested 70.00)")
    print(f"  a naive 7:3 split would give {naive_pe_wt:.2f} wt% PE")
    assert pe > ps
    assert abs(comp.realised_weight_percent[0] - 70.0) < 5.0


def test_larger_cells_track_the_requested_composition_more_closely():
    """Composition error is a property of integers, and must shrink with size."""
    print()
    errs = []
    for beads in (400, 2000, 10_000, 40_000):
        comp = solve_from_weight_fractions(_pair(), 0.95, target_beads=beads)
        errs.append(comp.max_weight_percent_error)
        print(f"  target {beads:6,d} beads -> {comp.total_chains:4d} chains, "
              f"{comp.realised_weight_percent[0]:6.2f} wt% PE, "
              f"error {comp.max_weight_percent_error:5.2f} pp")
    assert errs[-1] < errs[0], "more chains must resolve composition better"
    assert errs[-1] < 1.0


def test_rounding_error_is_reported_not_hidden():
    """A small cell cannot express 70/30, and must say so."""
    comp = solve_from_weight_fractions(_pair(), 0.95, target_chains=3)
    print(f"\n  3 chains: counts {comp.chain_counts}, realised "
          f"{[f'{w:.1f}' for w in comp.realised_weight_percent]} wt%")
    for n in comp.notes:
        print(f"    note: {n}")
    assert comp.max_weight_percent_error > 1.0
    assert any("percentage points" in n for n in comp.notes)


def test_a_minor_component_is_never_silently_dropped():
    """1 wt% of a heavy polymer rounds to <1 chain — it must survive."""
    comps = [
        Component(name="PE", repeat_unit=PE, degree_of_polymerisation=50,
                  weight_percent=99.0, ris_key="PE"),
        Component(name="PS", repeat_unit=PS, degree_of_polymerisation=50,
                  weight_percent=1.0, ris_key="PS"),
    ]
    comp = solve_from_weight_fractions(comps, 0.95, target_chains=6)
    print(f"\n  counts {comp.chain_counts}, realised "
          f"{[f'{w:.1f}' for w in comp.realised_weight_percent]} wt%")
    for n in comp.notes:
        print(f"    note: {n}")
    assert min(comp.chain_counts) >= 1, "the minor component vanished"
    assert any("over-represented" in n or "percentage points" in n
               for n in comp.notes)


def test_weight_percentages_that_do_not_sum_to_100_are_normalised_and_flagged():
    comps = _pair(w_pe=60.0)
    comps[1].weight_percent = 30.0            # 60 + 30 = 90
    comp = solve_from_weight_fractions(comps, 0.95, target_beads=4000)
    print(f"\n  requested {[f'{w:.2f}' for w in comp.requested_weight_percent]}"
          f"  (input summed to 90)")
    assert abs(sum(comp.requested_weight_percent) - 100.0) < 1e-6
    assert any("normalised" in n for n in comp.notes)


# ------------------------------------------------------------- count mode
def test_chain_count_mode_reports_the_composition_it_produces():
    comps = [
        Component(name="PE", repeat_unit=PE, degree_of_polymerisation=40,
                  n_chains=5, ris_key="PE"),
        Component(name="PS", repeat_unit=PS, degree_of_polymerisation=40,
                  n_chains=3, ris_key="PS"),
    ]
    comp = from_chain_counts(comps, 0.95)
    print(f"\n{comp.summary()}")
    assert comp.chain_counts == [5, 3]
    assert abs(sum(comp.realised_weight_percent) - 100.0) < 1e-6
    # PS chains are much heavier, so 3 of them outweigh 5 PE chains.
    assert comp.realised_weight_percent[1] > comp.realised_weight_percent[0]


def test_the_solved_box_actually_gives_the_target_density():
    """End to end: the box handed to the grower must be the right size."""
    print()
    for want in (0.85, 0.95, 1.05):
        comp = solve_from_weight_fractions(_pair(), want, target_beads=3000)
        got = (comp.total_mass_amu / N_AVOGADRO) / (comp.box_volume_a3 * 1e-24)
        print(f"  target {want:.2f} -> edge {comp.box_edge_a:6.2f} Å, "
              f"realised {got:.6f} g/cm³")
        assert abs(got - want) < 1e-9


def test_bead_count_matches_the_skeletal_granularity():
    """total_beads must be DP x skeletal atoms per unit, not just DP."""
    from paaf.cell.grow import backbone_atoms_per_unit
    comp = from_chain_counts(
        [Component(name="PEO", repeat_unit="[*]OCC[*]",
                   degree_of_polymerisation=10, n_chains=2)], 1.1)
    bpu = backbone_atoms_per_unit("[*]OCC[*]")
    print(f"\n  PEO skeletal atoms per unit: {bpu} (O, C, C)")
    print(f"  2 chains x DP 10 -> {comp.total_beads} beads")
    assert bpu == 3
    assert comp.total_beads == 2 * 10 * 3


def test_grow_specs_carry_the_resolved_chemistry():
    comp = solve_from_weight_fractions(_pair(), 0.95, target_beads=2000)
    specs = comp.grow_specs()
    print()
    for s in specs:
        print(f"  {s.name:4s} n={s.n_chains:3d} DP={s.degree_of_polymerisation}"
              f"  mass={s.mass_amu:7.2f}  backbone_atoms={s.backbone_atoms}")
    assert all(s.mass_amu and s.mass_amu > 0 for s in specs)
    assert all(s.backbone_atoms >= 1 for s in specs)


# --------------------------------------------------------------- guard rails
def test_exactly_one_size_control_is_required():
    with pytest.raises(ValueError):
        solve_from_weight_fractions(_pair(), 0.95)
    with pytest.raises(ValueError):
        solve_from_weight_fractions(_pair(), 0.95, target_beads=100,
                                    target_chains=5)


def test_too_few_chains_for_the_components_is_refused():
    with pytest.raises(ValueError):
        solve_from_weight_fractions(_pair(), 0.95, target_chains=1)


def test_all_zero_weights_is_refused():
    comps = _pair()
    for c in comps:
        c.weight_percent = 0.0
    with pytest.raises(ValueError):
        solve_from_weight_fractions(comps, 0.95, target_beads=1000)
