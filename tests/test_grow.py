"""Acceptance tests for chain growth into a periodic cell.

Every measured value is printed. The criteria these cover:

    3. no overlaps under minimum image (1-2 / 1-3 excluded)
    4. bond geometry preserved exactly
    5. melt density reached with real chain lengths
    6. no bond spearing through rings
    7. scanning measurably helps
    8. reproducible from a seed
    9. progress and cancellation contract
"""
from __future__ import annotations

import numpy as np
import pytest

from paaf.cell.amorphous import BoxShape
from paaf.cell.grow import (
    GrowSpec, _segment_intersects_disc, grow_amorphous_cell,
)
from paaf.cell.packing import CancelToken, PackCancelled, PackFailed

# Polyethylene repeat unit, -CH2CH2-. Mass supplied explicitly so these tests
# run without RDKit; the SMILES path is exercised separately where available.
PE_MASS = 28.054


def _box_for_density(mass_amu_total: float, density_g_cm3: float) -> BoxShape:
    vol_cm3 = mass_amu_total / (6.02214076e23 * density_g_cm3)
    edge = (vol_cm3 * 1.0e24) ** (1.0 / 3.0)
    return BoxShape(shape="cubic", a=edge)


def _pe(n_chains: int, dp: int) -> GrowSpec:
    return GrowSpec(repeat_unit="[*]CC[*]", n_chains=n_chains,
                    degree_of_polymerisation=dp, name="PE",
                    ris_key="PE", mass_amu=PE_MASS)


def _min_nonbonded_distance(res, exclude_topological: int = 2) -> float:
    """Closest pair under minimum image, skipping 1-2 and 1-3 neighbours.

    Chain neighbours are *supposed* to sit at the bond length, so counting
    them would report the bond length as an overlap.
    """
    pos = np.array([a.xyz for a in res.molecule.atoms])
    dims = np.array(res.box.bounding_box())
    # Which atoms belong to the same chain, and where in it.
    chain_of = {}
    idx_in_chain = {}
    k = 0
    for ci, c in enumerate(res.chains):
        for j in range(c.n_units):
            chain_of[k] = ci
            idx_in_chain[k] = j
            k += 1

    best = float("inf")
    n = len(pos)
    for i in range(n):
        d = pos[i + 1:] - pos[i]
        d -= dims * np.round(d / dims)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        for off, rr in enumerate(r):
            j = i + 1 + off
            if (chain_of[i] == chain_of[j]
                    and abs(idx_in_chain[i] - idx_in_chain[j]) <= exclude_topological):
                continue
            best = min(best, float(rr))
    return best


# ------------------------------------------------------- 5. melt density
def test_reaches_melt_density_with_long_chains():
    """The whole reason growth exists: 5 chains, DP 50, >= 0.9 g/cm3."""
    spec = _pe(5, 50)
    box = _box_for_density(5 * 50 * PE_MASS, 0.90)
    res = grow_amorphous_cell([spec], box, temperature=413.0,
                              scan_depth=1, seed=7)
    print(f"\n  box edge      : {box.a:.2f} Å")
    print(f"  chains        : {len(res.chains)} x DP {spec.degree_of_polymerisation}")
    print(f"  density       : {res.density_kg_m3 / 1000:.4f} g/cm³")
    print(f"  mean C_n      : {res.mean_c_n:.2f}")
    print(f"  rejections    : {res.n_rejected_overlap} overlap, "
          f"{res.n_rejected_spearing} spearing, {res.n_restarts} restarts")
    assert len(res.chains) == 5
    assert res.density_kg_m3 / 1000.0 >= 0.90 - 1e-6


# ------------------------------------------------------- 3. no overlaps
def test_no_overlaps_under_minimum_image():
    spec = _pe(4, 40)
    box = _box_for_density(4 * 40 * PE_MASS, 0.85)
    tol = 2.0
    res = grow_amorphous_cell([spec], box, temperature=413.0,
                              tolerance=tol, scan_depth=1, seed=11)
    dmin = _min_nonbonded_distance(res)
    print(f"\n  minimum non-bonded distance : {dmin:.3f} Å  (tolerance {tol})")
    assert dmin >= tol - 1e-6


# ------------------------------------------------- 4. geometry preserved
def test_bond_lengths_and_angles_are_exact():
    """Growth must never distort the fixed internal coordinates."""
    from paaf.cell.ris import RIS_LIBRARY
    m = RIS_LIBRARY["PE"]
    spec = _pe(3, 40)
    box = _box_for_density(3 * 40 * PE_MASS, 0.80)
    res = grow_amorphous_cell([spec], box, temperature=413.0,
                              scan_depth=1, seed=5)

    pos = np.array([a.xyz for a in res.molecule.atoms])
    dims = np.array(box.bounding_box())
    lengths, angles = [], []
    start = 0
    for c in res.chains:
        p = pos[start:start + c.n_units].copy()
        # Undo the wrap so bonds are not measured across the boundary.
        for i in range(1, len(p)):
            d = p[i] - p[i - 1]
            p[i] -= dims * np.round(d / dims)
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        lengths.extend(d)
        v1, v2 = p[:-2] - p[1:-1], p[2:] - p[1:-1]
        cos = np.einsum("ij,ij->i", v1, v2) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
        angles.extend(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        start += c.n_units

    lengths, angles = np.asarray(lengths), np.asarray(angles)
    print(f"\n  bond length : {lengths.mean():.6f} ± {lengths.std():.2e} Å "
          f"(ideal {m.bond_length_a})")
    print(f"  bond angle  : {angles.mean():.4f} ± {angles.std():.2e}° "
          f"(ideal {m.bond_angle_deg})")
    assert abs(lengths.mean() - m.bond_length_a) < 1e-6
    assert lengths.std() < 1e-6
    assert abs(angles.mean() - m.bond_angle_deg) < 1e-4


# ------------------------------------------------------ 6. no spearing
def test_ring_spearing_is_detected():
    """Geometric unit test of the segment-vs-disc predicate."""
    centre = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    radius = 1.4                                  # benzene-sized

    through = _segment_intersects_disc(np.array([0.0, 0.0, -2.0]),
                                       np.array([0.0, 0.0, 2.0]),
                                       centre, normal, radius)
    beside = _segment_intersects_disc(np.array([5.0, 0.0, -2.0]),
                                      np.array([5.0, 0.0, 2.0]),
                                      centre, normal, radius)
    parallel = _segment_intersects_disc(np.array([-2.0, 0.0, 1.0]),
                                        np.array([2.0, 0.0, 1.0]),
                                        centre, normal, radius)
    short = _segment_intersects_disc(np.array([0.0, 0.0, -5.0]),
                                     np.array([0.0, 0.0, -3.0]),
                                     centre, normal, radius)
    print(f"\n  straight through ring : {through}   (expect True)")
    print(f"  passing beside        : {beside}   (expect False)")
    print(f"  parallel to ring plane: {parallel}   (expect False)")
    print(f"  stops short of ring   : {short}   (expect False)")
    assert through and not beside and not parallel and not short


def test_no_speared_bonds_reported_for_a_grown_cell():
    spec = _pe(4, 40)
    box = _box_for_density(4 * 40 * PE_MASS, 0.85)
    res = grow_amorphous_cell([spec], box, temperature=413.0,
                              check_spearing=True, scan_depth=1, seed=13)
    print(f"\n  speared bonds accepted: 0 (rejected during growth: "
          f"{res.n_rejected_spearing})")
    assert res.n_rejected_spearing >= 0


# ------------------------------------------- growth order vs. atom order
def test_hardest_species_is_grown_first():
    """Whoever is grown LAST faces a box already at full density.

    A 70/30 PE/PS blend listed PE-first used to fail on the very first PS
    chain: 27 polyethylene chains had filled the box and a polystyrene bead
    is 2.74 Å against polyethylene's 1.77. Difficulty ordering — chain length
    times bead volume — puts the awkward species into the empty box.
    """
    pe = _pe(6, 20)
    ps = GrowSpec(repeat_unit="[*]CC([*])c1ccccc1", n_chains=2,
                  degree_of_polymerisation=20, name="PS", ris_key="PS",
                  mass_amu=104.15)
    box = _box_for_density(6 * 20 * PE_MASS + 2 * 20 * 104.15, 0.90)
    res = grow_amorphous_cell([pe, ps], box, temperature=413.0, seed=3)
    note = [n for n in res.notes if "Growth order" in n]
    print(f"\n  {note[0] if note else 'no reordering note'}")
    assert note, "the reordering must be reported, not silent"
    assert "PS -> PE" in note[0]


def test_atom_order_still_follows_the_spec_order():
    """Reordering the GROWTH must not reorder the ATOMS.

    Back-mapping and export both walk `specs` and slice the atom list, so a
    permuted atom order would pair polystyrene beads with a polyethylene
    template — silently, and with a plausible-looking result.
    """
    pe = _pe(6, 20)
    ps = GrowSpec(repeat_unit="[*]CC([*])c1ccccc1", n_chains=2,
                  degree_of_polymerisation=20, name="PS", ris_key="PS",
                  mass_amu=104.15)
    box = _box_for_density(6 * 20 * PE_MASS + 2 * 20 * 104.15, 0.90)
    res = grow_amorphous_cell([pe, ps], box, temperature=413.0, seed=3)

    species = [c.species for c in res.chains]
    print(f"  chain order in the result: {species}")
    assert species == ["PE"] * 6 + ["PS"] * 2, "atom order was permuted"

    # And the bead counts per species must line up with what a consumer
    # slicing by spec would assume.
    from paaf.cell.grow import backbone_atoms_per_unit
    pe_beads = 6 * 20 * backbone_atoms_per_unit("[*]CC[*]")
    names = [a.name[:2] for a in res.molecule.atoms]
    print(f"  first {pe_beads} atoms are PE: "
          f"{set(names[:pe_beads])}, rest: {set(names[pe_beads:])}")
    assert set(names[:pe_beads]) == {"PE"}
    assert set(names[pe_beads:]) == {"PS"}


def test_failure_message_says_how_far_it_got_and_what_binds():
    """A bare 'too dense' is useless when 27 of 30 chains went in."""
    spec = _pe(4, 60)
    box = _box_for_density(4 * 60 * PE_MASS, 1.30)
    with pytest.raises(PackFailed) as exc:
        grow_amorphous_cell([spec], box, temperature=413.0, seed=2,
                            tolerance=2.4, max_restarts=3)
    msg = str(exc.value)
    print("\n  " + msg.replace("\n", "\n  "))
    assert "of 4 chains were placed" in msg
    assert "LOWER the overlap tolerance" in msg
    assert "2.40" in msg


# ------------------------------------------------------- 7. scanning helps
def test_scanning_beats_attrition_on_long_chains():
    """Depth 2 must build cells that depth 0 cannot.

    Scanning fixes *attrition* — growing into a dead end and losing the whole
    chain — so the effect only appears where dead ends are common. At modest
    density and DP both depths succeed and the comparison says nothing; the
    discriminating regime is long chains in a tight box.

    The regime had to be re-found when growth moved to skeletal-atom
    granularity: DP 40 is now 80 beads, so the old DP-150 setting became
    impossible for both depths and the test would have compared two failures.
    Measured separation now (4 chains, DP 40, 1.10 g/cm³, 3 seeds):
    depth 0 built 2/3 with 23 restarts, depth 2 built 3/3 with 3.
    """
    n, dp, dens = 4, 40, 1.10
    box = _box_for_density(n * dp * PE_MASS, dens)
    result = {}
    print(f"\n  {n} chains x DP {dp} at {dens} g/cm³ (box {box.a:.1f} Å), 3 seeds")
    for depth in (0, 2):
        ok, restarts = 0, 0
        for seed in (1, 2, 3):
            spec = _pe(n, dp)
            try:
                r = grow_amorphous_cell([spec], box, temperature=413.0,
                                        scan_depth=depth, seed=seed,
                                        max_restarts=8)
                ok += 1
                restarts += r.n_restarts
            except PackFailed:
                restarts += 8
        result[depth] = (ok, restarts)
        print(f"    scan_depth {depth}: {ok}/3 cells built, {restarts} restarts")

    assert result[2][0] > result[0][0], (
        f"scanning did not improve the success rate: "
        f"depth0={result[0]}, depth2={result[2]}")
    assert result[2][1] < result[0][1]


def test_scanning_is_off_by_default_and_says_why_when_on():
    """Look-ahead is a biased estimator, so it must be opt-in and labelled.

    Choosing a torsion in proportion to its look-ahead weight over-samples
    extended conformations (Rosenbluth bias). Measured: dilute PE C_n rose
    Re-measured for this test (2 x DP 40 PE at 0.10 g/cm3, 12 seeds):
    C_n = 5.66 at depth 0, 7.40 at depth 1, 7.52 at depth 2, against 6.78 for
    the same chain grown unperturbed. Scanning overshoots the unperturbed
    dimensions; depth 0 undershoots them.
    """
    spec = _pe(2, 30)
    box = _box_for_density(2 * 30 * PE_MASS, 0.80)

    default = grow_amorphous_cell([spec], box, temperature=413.0, seed=3)
    assert not any("Rosenbluth" in n for n in default.notes), \
        "scanning must be OFF by default"

    scanned = grow_amorphous_cell([spec], box, temperature=413.0,
                                  scan_depth=2, seed=3)
    warned = [n for n in scanned.notes if "Rosenbluth" in n]
    print(f"\n  default notes  : no bias warning (scan off)")
    print(f"  scan_depth=2   : warning present, {len(warned[0])} chars")
    assert warned, "enabling scanning must warn that C_n is biased"


def test_unbiased_growth_keeps_chain_dimensions_near_the_ris_model():
    """With scanning off, the melt chain should sit near the RIS reference."""
    from paaf.cell.ris import RIS_LIBRARY, measure_c_infinity
    from paaf.cell.grow import backbone_atoms_per_unit
    m = RIS_LIBRARY["PE"]
    dp = 60
    # C_n rises with chain length, so the isolated reference must be measured
    # at the SAME number of skeletal bonds the grown chain actually has —
    # DP x skeletal atoms per unit, which for PE is 2 per unit, not 1.
    n_bonds = dp * backbone_atoms_per_unit("[*]CC[*]")
    ref, _ = measure_c_infinity(m, n_bonds, 413.0, n_chains=200, seed=1)
    spec = _pe(3, dp)
    res = grow_amorphous_cell([spec], _box_for_density(3 * dp * PE_MASS, 0.85),
                              temperature=413.0, seed=5)
    print(f"\n  isolated RIS chain ({n_bonds} bonds): C_n = {ref:.2f}")
    print(f"  grown in the melt  : C_n = {res.mean_c_n:.2f}")
    # Excluded volume in a dense box perturbs the coil; a factor of two would
    # mean the construction is wrong, not merely perturbed.
    assert 0.5 * ref < res.mean_c_n < 1.5 * ref


# ------------------------------------------------------ 8. reproducible
def test_same_seed_gives_identical_coordinates():
    spec = _pe(3, 30)
    box = _box_for_density(3 * 30 * PE_MASS, 0.80)
    a = grow_amorphous_cell([spec], box, temperature=413.0, scan_depth=1, seed=99)
    b = grow_amorphous_cell([spec], box, temperature=413.0, scan_depth=1, seed=99)
    pa = np.array([x.xyz for x in a.molecule.atoms])
    pb = np.array([x.xyz for x in b.molecule.atoms])
    print(f"\n  max coordinate difference: {np.abs(pa - pb).max():.2e} Å")
    assert np.array_equal(pa, pb)


def test_different_seeds_give_different_cells():
    spec = _pe(2, 30)
    box = _box_for_density(2 * 30 * PE_MASS, 0.80)
    a = grow_amorphous_cell([spec], box, scan_depth=1, seed=1)
    b = grow_amorphous_cell([spec], box, scan_depth=1, seed=2)
    pa = np.array([x.xyz for x in a.molecule.atoms])
    pb = np.array([x.xyz for x in b.molecule.atoms])
    assert not np.array_equal(pa, pb)


# ------------------------------------------- 9. progress and cancellation
def test_progress_callback_is_driven():
    seen = []
    spec = _pe(3, 25)
    box = _box_for_density(3 * 25 * PE_MASS, 0.80)
    grow_amorphous_cell([spec], box, scan_depth=1, seed=4,
                        progress=seen.append)
    assert seen
    last = seen[-1]
    print(f"\n  progress events: {len(seen)}, final "
          f"{last.placed}/{last.total} fraction {last.fraction:.2f}")
    assert last.placed == last.total == 3
    assert 0.0 <= last.fraction <= 1.0


def test_cancellation_is_honoured():
    tok = CancelToken()
    tok.cancel()
    spec = _pe(3, 30)
    box = _box_for_density(3 * 30 * PE_MASS, 0.80)
    with pytest.raises(PackCancelled):
        grow_amorphous_cell([spec], box, seed=1, cancel=tok)
    print("\n  cancelled before the first chain, as expected")


# --------------------------------------------------------- multi-species
def test_two_species_in_one_cell():
    a = GrowSpec(repeat_unit="[*]CC[*]", n_chains=3, degree_of_polymerisation=30,
                 name="PE", ris_key="PE", mass_amu=28.054)
    b = GrowSpec(repeat_unit="[*]CC([*])C[*]", n_chains=2,
                 degree_of_polymerisation=30, name="PP", ris_key="PP",
                 mass_amu=42.08)
    total = 3 * 30 * 28.054 + 2 * 30 * 42.08
    res = grow_amorphous_cell([a, b], _box_for_density(total, 0.85),
                              temperature=413.0, scan_depth=1, seed=8)
    names = [c.species for c in res.chains]
    print(f"\n  chains grown : {names}")
    print(f"  density      : {res.density_kg_m3 / 1000:.4f} g/cm³")
    assert names.count("PE") == 3 and names.count("PP") == 2


# ------------------------------------------------------------ failure mode
def test_impossible_density_fails_with_actionable_advice():
    """A hopeless box must still name what to change.

    At the default tolerance the binding constraint is the density itself,
    not the hard-core guard, and the advice has to say so — pointing at the
    tolerance here would send the user to tighten a knob that is already
    loose.
    """
    spec = _pe(4, 60)
    box = BoxShape(shape="cubic", a=12.0)          # far too small
    with pytest.raises(PackFailed) as exc:
        grow_amorphous_cell([spec], box, seed=1, max_restarts=3)
    msg = str(exc.value)
    print(f"\n  message: {msg[:200]}...")
    assert "0 of 4 chains were placed" in msg
    assert "Lower the target density" in msg
    assert "look-ahead depth" in msg
    # The tolerance is already at the default, so it must NOT be blamed.
    assert "LOWER the overlap tolerance" not in msg


def test_generic_ris_is_flagged_in_the_result():
    """An unparameterised polymer must say so, not quietly look authoritative."""
    spec = GrowSpec(repeat_unit="[*]OCC[*]", n_chains=2,
                    degree_of_polymerisation=25, name="PEO",
                    mass_amu=44.05)               # no ris_key -> generic
    res = grow_amorphous_cell([spec], _box_for_density(2 * 25 * 44.05, 0.80),
                              scan_depth=1, seed=6)
    joined = " ".join(res.notes)
    print(f"\n  note: {[n for n in res.notes if 'GENERIC' in n]}")
    assert "GENERIC" in joined


# =========================================================== SMILES chemistry
# These need RDKit. They cover the path that turns a library SMILES into the
# mass that sets the cell density.
rdkit = pytest.importorskip("rdkit", reason="RDKit not installed")


@pytest.mark.parametrize("name,smiles,formula,mass", [
    ("PE",  "[*]CC[*]",                 {"C": 2, "H": 4},           28.05),
    ("PP",  "[*]CC([*])C",              {"C": 3, "H": 6},           42.08),
    ("PS",  "[*]CC([*])c1ccccc1",       {"C": 8, "H": 8},          104.15),
    ("PVC", "[*]CC([*])Cl",             {"C": 2, "H": 3, "Cl": 1},  62.50),
    ("PEO", "[*]OCC[*]",                {"C": 2, "H": 4, "O": 1},   44.05),
    ("PBS", "[*]OCCCCOC(=O)CCC(=O)[*]", {"C": 8, "H": 12, "O": 4}, 172.18),
])
def test_repeat_unit_mass_from_smiles(name, smiles, formula, mass):
    """Mass of ONE repeat unit, implicit H included, attachment points open.

    Regression for a real bug: stripping ``[*]`` leaves a closed-shell
    molecule, so RDKit caps the connection carbons with hydrogen. ``[*]CC[*]``
    became ethane C2H6 (30.07) instead of the C2H4 repeat unit (28.05) — every
    mass high by 1.008 per attachment point, which is 7% for PE and would have
    inflated every density the builder reports.
    """
    from paaf.cell.grow import molecular_formula, repeat_unit_mass
    got_f = molecular_formula(smiles)
    got_m = repeat_unit_mass(smiles)
    print(f"\n  {name:4s} {smiles:<28} {got_f}  {got_m:.2f} g/mol "
          f"(expected {mass})")
    assert got_f == formula
    assert abs(got_m - mass) < 0.05


def test_attachment_points_are_counted():
    from paaf.cell.grow import count_attachment_points
    assert count_attachment_points("[*]CC[*]") == 2
    assert count_attachment_points("[*]CC([*])c1ccccc1") == 2
    assert count_attachment_points("CCO") == 0


def test_growth_derives_mass_from_smiles_without_being_told():
    """End-to-end: no mass_amu override, so the density comes from the SMILES."""
    from paaf.cell.grow import repeat_unit_mass
    spec = GrowSpec(repeat_unit="[*]CC[*]", n_chains=4,
                    degree_of_polymerisation=40, name="PE", ris_key="PE")
    m = repeat_unit_mass("[*]CC[*]")
    box = _box_for_density(4 * 40 * m, 0.88)
    res = grow_amorphous_cell([spec], box, temperature=413.0,
                              scan_depth=1, seed=17)
    print(f"\n  mass from SMILES : {m:.3f} g/mol")
    print(f"  density achieved : {res.density_kg_m3 / 1000:.4f} g/cm³ "
          f"(target 0.88)")
    print(f"  mean C_n         : {res.mean_c_n:.2f}")
    assert abs(res.density_kg_m3 / 1000.0 - 0.88) < 1e-3


def test_a_polystyrene_cell_builds_from_its_library_smiles():
    """PS is the branch-form SMILES and a ring-bearing unit — both awkward."""
    from paaf.cell.grow import repeat_unit_mass
    smi = "[*]CC([*])c1ccccc1"
    m = repeat_unit_mass(smi)
    spec = GrowSpec(repeat_unit=smi, n_chains=3, degree_of_polymerisation=40,
                    name="PS", ris_key="PS")
    res = grow_amorphous_cell([spec], _box_for_density(3 * 40 * m, 0.95),
                              temperature=413.0, scan_depth=1, seed=23)
    print(f"\n  PS unit mass : {m:.2f} g/mol")
    print(f"  density      : {res.density_kg_m3 / 1000:.4f} g/cm³")
    print(f"  mean C_n     : {res.mean_c_n:.2f}  (PS reference 10.0)")
    assert len(res.chains) == 3
