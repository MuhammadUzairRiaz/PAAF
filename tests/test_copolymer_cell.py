"""Copolymer chains in the amorphous cell.

Epoxidised natural rubber is a random copolymer of isoprene and epoxidised
isoprene, and the epoxide unit has a ring *in* its backbone. Both of those
were blockers, so both are tested here — separately, and then together
alongside a homopolymer (PBS) in one cell.

The load-bearing check is the atom count. A copolymer's formula depends on the
sequence each chain happened to get, so it is derived from the sequences that
were actually grown rather than from the requested composition.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit", reason="needs RDKit")

from paaf.cell.amorphous import BoxShape                     # noqa: E402
from paaf.cell.backmap import backmap_cell, build_unit_template  # noqa: E402
from paaf.cell.grow import (                                 # noqa: E402
    BackboneRingError, GrowSpec, backbone_ring_atoms, grow_amorphous_cell,
    max_backbone_atoms_in_a_ring, molecular_formula,
)
from paaf.cell.sequence import Monomer, build_sequence       # noqa: E402

ISOPRENE = "[*]C/C=C(C)\\C[*]"
EPOXIDE = "[*]CC1(C)OC1C[*]"
PBS = "[*]OCCCCOC(=O)CCC(=O)[*]"


def _enr(fraction_epoxide=0.25, seed=1):
    return [Monomer(ISOPRENE, 1.0 - fraction_epoxide, "isoprene"),
            Monomer(EPOXIDE, fraction_epoxide, "epoxide")]


def _box(mass_amu, density=0.95):
    v = mass_amu / (6.02214076e23 * density)
    return BoxShape(shape="cubic", a=(v * 1e24) ** (1 / 3))


# ====================================================== the sequence itself
def test_random_sequence_scatters_around_the_target():
    """Each chain gets its own draw, which is what "random copolymer" means.

    A short chain's realised composition does NOT equal the target, and it
    should not: real chains scatter too. What must hold is that the average
    over many chains converges.
    """
    fracs = []
    for chain in range(40):
        seq = build_sequence(_enr(0.25), 20, "random", seed=chain)
        fracs.append(seq.realised_fractions()[1])
    fracs = np.asarray(fracs)
    print(f"\n  epoxide mole fraction over 40 chains of DP 20:")
    print(f"    mean {fracs.mean():.3f} (target 0.25), "
          f"spread {fracs.min():.2f}–{fracs.max():.2f}")
    assert abs(fracs.mean() - 0.25) < 0.04
    assert fracs.std() > 0.02, "no scatter at all — this is not random"


def test_alternating_and_block_do_what_they_say():
    alt = build_sequence(_enr(0.5), 8, "alternating")
    blk = build_sequence(_enr(0.5), 8, "block")
    print(f"\n  alternating: {alt.order}")
    print(f"  block      : {blk.order}")
    assert alt.order == [0, 1, 0, 1, 0, 1, 0, 1]
    assert blk.order == [0, 0, 0, 0, 1, 1, 1, 1]


def test_alternating_refuses_more_than_two_monomers():
    mons = _enr(0.34) + [Monomer("[*]CC[*]", 0.33, "PE")]
    with pytest.raises(ValueError):
        build_sequence(mons, 9, "alternating")


def test_block_never_drops_a_requested_monomer():
    """A 2% component in a DP-20 chain rounds below one unit — keep it."""
    seq = build_sequence(_enr(0.02), 20, "block")
    print(f"\n  2% epoxide, DP 20 -> {seq.order}")
    assert 1 in seq.order


def test_sequence_knows_its_bead_count_and_mass():
    seq = build_sequence(_enr(0.5), 10, "alternating")
    # 5 isoprene + 5 epoxide, 4 skeletal atoms each.
    print(f"\n  {seq.describe()}")
    print(f"  beads {seq.n_beads}, chain mass {seq.chain_mass:.2f}")
    assert seq.n_beads == 40
    assert abs(seq.chain_mass - 5 * 68.12 - 5 * 84.12) < 0.1
    assert len(seq.bead_monomer_index()) == seq.n_beads


# ================================================== the epoxide backbone ring
def test_a_ring_spanning_one_backbone_bond_is_allowed():
    """The epoxide: two skeletal carbons plus a bridging oxygen."""
    print(f"\n  epoxide: {backbone_ring_atoms(EPOXIDE)} skeletal atoms in a "
          f"ring, worst single ring {max_backbone_atoms_in_a_ring(EPOXIDE)}")
    assert max_backbone_atoms_in_a_ring(EPOXIDE) == 2
    t = build_unit_template(EPOXIDE)          # must not raise
    assert t.n_backbone == 4


@pytest.mark.parametrize("smiles,name", [
    ("[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]", "PET"),
    ("[*]OC(=O)Oc1ccc(cc1)C(C)(C)c1ccc(cc1)[*]", "polycarbonate"),
])
def test_a_ring_spanning_several_backbone_bonds_is_still_refused(smiles, name):
    """A benzene in the chain cannot be reconciled with RIS spacing."""
    n = max_backbone_atoms_in_a_ring(smiles)
    print(f"\n  {name}: worst single ring holds {n} skeletal atoms")
    assert n > 2
    with pytest.raises(BackboneRingError):
        build_unit_template(smiles)


def test_the_epoxide_bridge_lands_at_a_sane_bond_length():
    t = build_unit_template(EPOXIDE)
    c = t.coords
    lengths = {}
    for i, j, _o in t.bonds:
        if t.elements[i] != "H" and t.elements[j] != "H":
            lengths[f"{t.elements[i]}{i}-{t.elements[j]}{j}"] = float(
                np.linalg.norm(c[i] - c[j]))
    print("\n  " + "  ".join(f"{k} {v:.3f}" for k, v in lengths.items()))
    # The two ring carbons are backbone neighbours; growth will place them at
    # 1.53 A, so the template's value must be close or the ring is strained.
    ring_cc = [v for k, v in lengths.items() if k.startswith("C2-C5")
               or k.startswith("C5-C2")]
    assert ring_cc and abs(ring_cc[0] - 1.53) < 0.10


# ======================================================= a whole ENR/PBS cell
def test_enr_and_pbs_build_together_with_the_right_formula():
    """The end-to-end case: a copolymer and a homopolymer in one cell.

    The expected formula is derived from the sequences that were actually
    grown. Comparing against the *requested* 75/25 would be wrong — each chain
    gets its own draw.
    """
    specs = [
        GrowSpec(name="ENR", n_chains=4, degree_of_polymerisation=20,
                 monomers=_enr(0.25), sequence_seed=1),
        GrowSpec(name="PBS", repeat_unit=PBS, n_chains=2,
                 degree_of_polymerisation=12),
    ]
    mass = 4 * 20 * (0.75 * 68.12 + 0.25 * 84.12) + 2 * 12 * 172.18
    res = grow_amorphous_cell(specs, _box(mass), temperature=413.0, seed=5)
    bm = backmap_cell(res, specs)

    # --- expected atoms, from the sequences actually grown
    per_unit = {s: molecular_formula(s) for s in (ISOPRENE, EPOXIDE, PBS)}
    want = Counter()
    n_chains = 0
    for c in res.chains:
        if c.sequence is not None:
            for u in c.sequence.order:
                want.update(per_unit[c.sequence.monomers[u].smiles])
        else:
            for _ in range(c.n_repeat_units):
                want.update(per_unit[PBS])
        n_chains += 1
    want["H"] += 2 * n_chains          # one end cap at each end of each chain

    got = Counter(a.element for a in bm.molecule.atoms)
    print(f"\n  {len(res.molecule.atoms)} beads -> {len(bm.molecule.atoms)} atoms")
    print(f"  got      {dict(sorted(got.items()))}")
    print(f"  expected {dict(sorted(want.items()))}")
    assert dict(got) == dict(want)


def test_the_enr_cell_has_sane_bond_lengths():
    specs = [
        GrowSpec(name="ENR", n_chains=4, degree_of_polymerisation=20,
                 monomers=_enr(0.25), sequence_seed=1),
        GrowSpec(name="PBS", repeat_unit=PBS, n_chains=2,
                 degree_of_polymerisation=12),
    ]
    mass = 4 * 20 * (0.75 * 68.12 + 0.25 * 84.12) + 2 * 12 * 172.18
    res = grow_amorphous_cell(specs, _box(mass), temperature=413.0, seed=5)
    bm = backmap_cell(res, specs)
    dims = np.array(res.box.bounding_box())
    xyz = np.array([a.xyz for a in bm.molecule.atoms])
    lengths = []
    for i, j, _o in bm.molecule.bonds:
        d = xyz[i] - xyz[j]
        d -= dims * np.round(d / dims)
        lengths.append(float(np.linalg.norm(d)))
    lengths = np.asarray(lengths)
    print(f"\n  {len(lengths)} bonds, {lengths.min():.3f}–{lengths.max():.3f} Å")
    print(f"  closest non-bonded contact {bm.worst_contact_a:.2f} Å")
    # The short end is a hydroxyl end cap on PBS (O-H is 0.97 Å).
    assert lengths.min() > 0.90
    assert lengths.max() < 2.05
    assert any("Bridged backbone ring" in n for n in bm.notes)


def test_each_copolymer_chain_gets_a_different_sequence():
    specs = [GrowSpec(name="ENR", n_chains=6, degree_of_polymerisation=20,
                      monomers=_enr(0.25), sequence_seed=1)]
    mass = 6 * 20 * (0.75 * 68.12 + 0.25 * 84.12)
    res = grow_amorphous_cell(specs, _box(mass, 0.90), temperature=413.0,
                              seed=5)
    orders = [tuple(c.sequence.order) for c in res.chains]
    fracs = [c.sequence.realised_fractions()[1] for c in res.chains]
    print(f"\n  epoxide fraction per chain: "
          f"{[f'{f:.2f}' for f in fracs]}")
    assert len(set(orders)) > 1, "every chain got the same sequence"


def test_a_single_monomer_list_is_treated_as_a_homopolymer():
    seq = build_sequence([Monomer(ISOPRENE, 1.0, "isoprene")], 5, "random")
    print(f"\n  {seq.arrangement}: {seq.order}")
    assert seq.arrangement == "homopolymer"
    assert seq.order == [0] * 5


# ============================================ the shared copolymer library
def test_a_saved_recipe_round_trips_through_the_shared_library(tmp_path):
    """The same CSV backs the simple Builder and the Amorphous cell tool.

    A definition saved from either must be readable by the other, which is
    the whole reason it lives in paaf.copolymer_library rather than in one
    tool's own store.
    """
    from paaf.copolymer_library import (
        CopolymerRecipe, load_copolymer_library, save_copolymer,
    )

    path = tmp_path / "copolymers.csv"
    save_copolymer(CopolymerRecipe(
        pid="SBR", smiles_a="[*]CC([*])c1ccccc1", smiles_b="[*]C/C=C\\C[*]",
        fraction_b=0.77, sequence_mode="random", random_seed=7), path)
    got = load_copolymer_library(path)
    print(f"\n  saved and reloaded: "
          f"{[(r.pid, r.fraction_b, r.sequence_mode) for r in got]}")
    assert len(got) == 1
    assert got[0].pid == "SBR"
    assert abs(got[0].fraction_b - 0.77) < 1e-9
    assert abs(got[0].fraction_a - 0.23) < 1e-9


def test_saving_the_same_name_twice_replaces_rather_than_duplicates(tmp_path):
    """A duplicate PID would silently shadow the original on read."""
    from paaf.copolymer_library import (
        CopolymerRecipe, load_copolymer_library, save_copolymer,
    )

    path = tmp_path / "copolymers.csv"
    for frac in (0.25, 0.50):
        save_copolymer(CopolymerRecipe(
            pid="ENR", smiles_a=ISOPRENE, smiles_b=EPOXIDE,
            fraction_b=frac, sequence_mode="random", random_seed=1), path)
    got = load_copolymer_library(path)
    print(f"\n  after two saves of 'ENR': {len(got)} row(s), "
          f"fraction_b = {got[0].fraction_b}")
    assert len(got) == 1
    assert abs(got[0].fraction_b - 0.50) < 1e-9


def test_a_library_recipe_becomes_a_growable_copolymer():
    """The bridge that matters: a saved recipe must reach the grower."""
    from paaf.copolymer_library import CopolymerRecipe
    from paaf.cell.sequence import Monomer, build_sequence

    rec = CopolymerRecipe(
        pid="ENR-25", smiles_a=ISOPRENE, smiles_b=EPOXIDE,
        fraction_b=0.25, sequence_mode="random", random_seed=1)
    mons = [Monomer(rec.smiles_a, rec.fraction_a, "A"),
            Monomer(rec.smiles_b, rec.fraction_b, "B")]
    seq = build_sequence(mons, 40, rec.sequence_mode, seed=rec.random_seed or 0)
    print(f"\n  {rec.description}")
    print(f"  -> {seq.describe()}")
    assert seq.n_units == 40
    assert abs(seq.realised_fractions()[1] - 0.25) < 0.20


def test_any_two_library_polymers_can_be_joined():
    """Not just ENR: any two repeat units with two attachment points.

    Styrene + butadiene is SBR; ethylene + propylene is EPR. Neither needs a
    preset — the sequence machinery only needs two polymerisation SMILES.
    """
    from paaf.cell.sequence import Monomer, build_sequence

    pairs = [
        ("SBR", "[*]CC([*])c1ccccc1", "[*]C/C=C\\C[*]", 0.77),
        ("EPR", "[*]CC[*]", "[*]CC([*])C", 0.50),
        ("PE/PBS", "[*]CC[*]", PBS, 0.30),
    ]
    print()
    for name, a, b, fb in pairs:
        seq = build_sequence(
            [Monomer(a, 1 - fb, "A"), Monomer(b, fb, "B")], 30, "random",
            seed=3)
        wt = seq.realised_weight_fractions()
        print(f"  {name:8s} {seq.n_units} units, {seq.n_beads:3d} beads, "
              f"{100 * wt[0]:5.1f}/{100 * wt[1]:5.1f} wt%")
        assert seq.n_beads > 0
        assert abs(sum(wt) - 1.0) < 1e-9


def test_block_copolymers_work_for_an_arbitrary_pair():
    """"Join any two as a block polymer" — the block arrangement, generally."""
    from paaf.cell.sequence import Monomer, build_sequence

    seq = build_sequence(
        [Monomer("[*]CC[*]", 0.5, "PE"), Monomer(PBS, 0.5, "PBS")],
        20, "block", seed=0)
    first_half = set(seq.order[:10])
    second_half = set(seq.order[10:])
    print(f"\n  block PE/PBS: {seq.order}")
    assert first_half == {0} and second_half == {1}, "the blocks interleaved"
