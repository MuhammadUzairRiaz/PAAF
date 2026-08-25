"""A cell must be BUILT looser than the density it is meant to end at.

Growth places beads at a hard-core spacing; side groups are bolted on
afterwards by back-mapping and need room that was never reserved for them.
Measured on a PE/PS cell constructed straight at 0.95 g/cm3: 14,528
non-bonded pairs closer than 2.0 A. DL_FIELD perceives bonds by distance, so
carbons came out with 5-10 neighbours and it refused to type them —
"Unexpected error in detect_new_key(). atype = aliphatic".

No amount of pushing atoms apart fixes that: at bulk density there is nowhere
to push them to. The cell has to be grown in a larger box and compressed with
NPT afterwards, which is what the Blend page has always done for packmol.
"""
from __future__ import annotations

import pytest

from paaf.cell.composition import Component, solve_from_weight_fractions


def _components():
    return [Component(name="PE", repeat_unit="[*]CC[*]",
                      degree_of_polymerisation=50, weight_percent=70.0),
            Component(name="PS", repeat_unit="[*]CC([*])c1ccccc1",
                      degree_of_polymerisation=50, weight_percent=30.0)]


def _solve(fraction):
    return solve_from_weight_fractions(_components(), 0.95, target_beads=3000,
                                       build_fraction=fraction)


def test_a_lower_build_fraction_gives_a_bigger_box():
    full, loose = _solve(1.0), _solve(0.6)
    print(f"\n  100%: {full.box_edge_a:.1f} A   60%: {loose.box_edge_a:.1f} A")
    assert loose.box_edge_a > full.box_edge_a
    # Volume scales as 1/fraction.
    assert (loose.box_edge_a / full.box_edge_a) ** 3 == pytest.approx(
        1 / 0.6, rel=0.01)


def test_the_target_density_is_remembered_not_overwritten():
    """The cell is BUILT loose but still knows where it must end up."""
    c = _solve(0.6)
    assert c.target_density_g_cm3 == pytest.approx(0.95)
    assert c.build_density_g_cm3 == pytest.approx(0.57)


def test_full_density_is_unchanged():
    """The old behaviour is still available and still the same numbers."""
    c = _solve(1.0)
    assert c.build_density_g_cm3 == pytest.approx(c.target_density_g_cm3)


def test_the_user_is_told_to_compress():
    """A loose cell is not a result; saying so is part of the job."""
    notes = " ".join(_solve(0.6).notes).lower()
    print(f"\n  {notes[-160:]}")
    assert "npt" in notes and "compress" in notes


def test_no_note_when_building_at_full_density():
    assert not any("compress" in n.lower() for n in _solve(1.0).notes)


@pytest.mark.parametrize("bad,expected", [(0.0, 0.05), (5.0, 1.0),
                                          (-1.0, 0.05)])
def test_the_fraction_is_clamped_to_something_buildable(bad, expected):
    """A zero or negative fraction would ask for an infinite box."""
    c = _solve(bad)
    assert c.build_density_g_cm3 == pytest.approx(0.95 * expected)


def _counted_pe():
    c = Component(name="PE", repeat_unit="[*]CC[*]",
                  degree_of_polymerisation=50, weight_percent=100.0)
    c.n_chains = 2
    return c


def test_manual_chain_counts_also_honour_build_at():
    """"Enter chain counts myself" grew at the FULL target density no matter
    what Build-at said — the dialog even printed "already near its final
    density (0.950 g/cm3)" with Build at 45% on screen. The loose box must
    apply to both composition routes."""
    from paaf.cell.composition import from_chain_counts

    full = from_chain_counts([_counted_pe()], 0.95, build_fraction=1.0)
    loose = from_chain_counts([_counted_pe()], 0.95, build_fraction=0.45)
    print(f"\n  full edge {full.box_edge_a:.2f} A, "
          f"loose edge {loose.box_edge_a:.2f} A")
    assert loose.build_density_g_cm3 == pytest.approx(0.95 * 0.45)
    assert loose.box_edge_a > full.box_edge_a * 1.25
    assert loose.target_density_g_cm3 == 0.95
    assert any("NPT" in n for n in loose.notes)


def test_manual_chain_counts_default_is_the_old_behaviour():
    from paaf.cell.composition import from_chain_counts

    comp = from_chain_counts([_counted_pe()], 0.95)
    assert comp.build_density_g_cm3 == pytest.approx(0.95)
    assert comp.notes == []
