"""Acceptance tests for the RIS model.

These check the *physics*, not that the code runs. Every measured value is
printed so the numbers can be read, not just the pass/fail.

The load-bearing test is the characteristic ratio. Three regimes must be
distinguishable:

    freely jointed        C = 1     (no geometry at all)
    freely rotating       C ≈ 2.2   (bond angle, unweighted torsions)
    RIS                   C ≈ 6.7   (+ torsional statistics)   <- PE

If the RIS implementation were decorative rather than real, the measured value
would sit at the freely-rotating baseline.
"""
from __future__ import annotations

import numpy as np
import pytest

from paaf.cell import ris


# --------------------------------------------------------------- geometry
def test_backbone_geometry_is_exact():
    """Fixed bond lengths and angles must not drift over a long chain."""
    m = ris.RIS_LIBRARY["PE"]
    rng = np.random.default_rng(1)
    states = ris.sample_torsions(m, 300, 413.0, rng)
    pos = ris.build_backbone(m, states)

    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    v1 = pos[:-2] - pos[1:-1]
    v2 = pos[2:] - pos[1:-1]
    cos = np.einsum("ij,ij->i", v1, v2) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))

    print(f"\n  bond length  : {d.mean():.6f} ± {d.std():.2e} Å "
          f"(ideal {m.bond_length_a})")
    print(f"  bond angle   : {ang.mean():.4f} ± {ang.std():.2e}° "
          f"(ideal {m.bond_angle_deg})")
    assert abs(d.mean() - m.bond_length_a) < 1e-9
    assert d.std() < 1e-9
    assert abs(ang.mean() - m.bond_angle_deg) < 1e-6
    assert ang.std() < 1e-6


def test_freely_rotating_baseline_is_reproduced():
    """Unweighted torsions must land on the analytic freely-rotating value.

    This isolates the geometry from the statistics: if this fails, the chain
    builder is wrong and any C_inf agreement would be luck.
    """
    m = ris.RIS_LIBRARY["PE"]
    analytic = ris.freely_rotating_c_infinity(m.bond_angle_deg)
    measured, err = ris.measure_c_infinity(m, 400, 413.0, n_chains=400,
                                           seed=3, random_torsions=True)
    print(f"\n  freely rotating: analytic {analytic:.3f}, "
          f"measured {measured:.3f} ± {err:.3f}")
    assert abs(measured - analytic) < 0.25


# ------------------------------------------------------- characteristic ratio
@pytest.mark.parametrize("key", ["PE", "PP", "PS", "PMMA"])
def test_characteristic_ratio_matches_literature(key):
    """C_inf within 20% of the tabulated value for each parameterised polymer."""
    m = ris.RIS_LIBRARY[key]
    t = m.reference_c_inf_temp_k or 413.0
    measured, err = ris.measure_c_infinity(m, 500, t, n_chains=400, seed=5)
    ref = m.reference_c_inf
    dev = 100.0 * (measured - ref) / ref
    print(f"\n  {key:5s} T={t:.0f} K  measured C = {measured:.2f} ± {err:.2f}"
          f"   reference {ref}   deviation {dev:+.1f}%")
    assert abs(dev) <= 20.0, f"{key}: C_inf off by {dev:.1f}%"


def test_ris_is_far_from_the_random_control():
    """The whole point: statistics must expand the coil well beyond geometry."""
    m = ris.RIS_LIBRARY["PE"]
    with_ris, _ = ris.measure_c_infinity(m, 400, 413.0, n_chains=300, seed=9)
    without, _ = ris.measure_c_infinity(m, 400, 413.0, n_chains=300, seed=9,
                                        random_torsions=True)
    print(f"\n  PE with RIS {with_ris:.2f}  vs random torsions {without:.2f}"
          f"   ratio {with_ris / without:.2f}x")
    assert with_ris > 2.5 * without


def test_c_n_grows_towards_the_asymptote():
    """C_n must increase with chain length and level off — not be flat."""
    m = ris.RIS_LIBRARY["PE"]
    vals = []
    print()
    for n in (20, 50, 100, 200, 400):
        c, e = ris.measure_c_infinity(m, n, 413.0, n_chains=250, seed=4)
        vals.append(c)
        print(f"  n_bonds {n:4d}   C_n = {c:.2f} ± {e:.2f}")
    assert vals[0] < vals[-1]
    assert abs(vals[-1] - vals[-2]) < 0.6 * abs(vals[1] - vals[0])


# ------------------------------------------------------------- populations
@pytest.mark.parametrize("key", ["PE", "PS"])
def test_torsion_populations_match_the_transfer_matrix(key):
    """Realised state fractions must equal the RIS prediction.

    The reference is the transfer-matrix (principal eigenvector) result, which
    is the exact long-chain limit of the model being sampled.
    """
    m = ris.RIS_LIBRARY[key]
    rng = np.random.default_rng(17)
    states = np.concatenate(
        [ris.sample_torsions(m, 500, 413.0, rng) for _ in range(80)])
    got = ris.state_populations(states, m.n_states)
    want = ris.expected_populations(m, 413.0)
    print(f"\n  {key}")
    for lbl, g, w in zip(m.state_labels, got, want):
        print(f"    {lbl:3s} realised {g:.4f}  predicted {w:.4f}  Δ {g - w:+.4f}")
    assert np.max(np.abs(got - want)) < 0.02


def test_gauche_fraction_falls_as_temperature_drops():
    """Trans is the low-energy state, so cooling must favour it."""
    m = ris.RIS_LIBRARY["PE"]
    print()
    last = None
    for t in (500.0, 400.0, 300.0):
        p = ris.expected_populations(m, t)
        print(f"  T = {t:.0f} K   trans {p[0]:.3f}   gauche {p[1] + p[2]:.3f}")
        if last is not None:
            assert p[0] > last
        last = p[0]


def test_pentane_effect_is_present():
    """g+g- must be suppressed relative to g+g+ in the weight matrix."""
    m = ris.RIS_LIBRARY["PE"]
    u = m.u_matrix(413.0)
    # rows/cols are (t, g+, g-)
    print(f"\n  U[g+,g+] = {u[1, 1]:.4f}   U[g+,g-] = {u[1, 2]:.4f}"
          f"   ratio {u[1, 1] / u[1, 2]:.1f}x")
    assert u[1, 2] < u[1, 1]
    assert u[2, 1] < u[2, 2]


# ------------------------------------------------------------- provenance
def test_generic_model_is_labelled_not_disguised():
    g = ris.generic_ris()
    assert g.is_generic
    assert "GENERIC" in g.provenance.upper()
    assert ris.get_ris("PBS").is_generic          # unknown polymer
    assert not ris.get_ris("PE").is_generic
    print(f"\n  unknown key -> {ris.get_ris('PBS').name} "
          f"(is_generic={ris.get_ris('PBS').is_generic})")


def test_every_library_entry_records_where_it_came_from():
    for key, m in ris.RIS_LIBRARY.items():
        assert m.provenance, f"{key} has no provenance"
        assert m.reference_c_inf is not None
        assert m.reference_c_inf_temp_k is not None


def test_sampling_is_reproducible():
    m = ris.RIS_LIBRARY["PE"]
    a = ris.sample_torsions(m, 200, 413.0, np.random.default_rng(42))
    b = ris.sample_torsions(m, 200, 413.0, np.random.default_rng(42))
    assert np.array_equal(a, b)
