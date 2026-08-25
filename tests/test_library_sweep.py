"""Run the whole polymer library through the cell pipeline.

Testing two polymers proves two polymers work. This module runs every entry in
the 124-polymer library through chemistry, growth and back-mapping, and asserts
that each one either produces a chemically sound structure or is refused with a
specific reason. Silence is the thing being guarded against: the failure this
suite was written to catch produced bonds of 11.9 Å in 24 cells while raising
no error at all.

Marked slow; run with ``-m slow`` or unskip locally.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit", reason="library sweep needs RDKit")

from paaf.builder import list_library                        # noqa: E402
from paaf.cell.backmap import backmap_cell                   # noqa: E402
from paaf.cell.composition import Component, from_chain_counts  # noqa: E402
from paaf.cell.grow import (                                 # noqa: E402
    BackboneRingError, backbone_atoms_per_unit, backbone_ring_atoms,
    count_attachment_points, grow_amorphous_cell, repeat_unit_mass,
)

# Si-C is 1.87 Å, so a flat 1.8 Å ceiling would wrongly fail siloxanes.
MAX_BOND = 2.05
MIN_BOND = 0.90


def _library():
    return [r for r in list_library("all")
            if count_attachment_points(r.smiles) == 2]


def _ids(recs):
    return [r.pid or r.name for r in recs]


# ------------------------------------------------------------- chemistry
def test_every_library_entry_declares_two_attachment_points():
    """One entry does not, and it should be findable rather than mysterious."""
    all_recs = list_library("all")
    bad = [(r.pid or r.name, r.smiles) for r in all_recs
           if count_attachment_points(r.smiles) != 2]
    print(f"\n  {len(all_recs)} entries, {len(bad)} without a polymerisation "
          f"SMILES:")
    for pid, smi in bad:
        print(f"    {pid}: {smi}")
    # This is a data defect in the library, not a code defect. It is asserted
    # so that it cannot grow without someone noticing.
    assert len(bad) <= 1, f"more entries lost their [*] markers: {bad}"


@pytest.mark.parametrize("rec", _library(), ids=_ids(_library()))
def test_chemistry_resolves_for_every_polymer(rec):
    """Mass and granularity must come out of every library SMILES.

    This is what caught poly(methyl methacrylate): the branch-form expansion
    produced ``CC((C)C(=O)OC)`` — a doubled bracket that is not valid SMILES —
    so one of the four polymers the RIS library parameterises could not have
    its mass computed at all.
    """
    m = repeat_unit_mass(rec.smiles)
    b = backbone_atoms_per_unit(rec.smiles)
    assert m > 0, f"{rec.pid}: non-positive repeat-unit mass"
    assert b >= 1, f"{rec.pid}: no skeletal atoms found"


def test_backbone_ring_polymers_are_identified():
    """A ring IN the backbone must be distinguished from a pendant one."""
    recs = _library()
    ringed = [(r.pid or r.name, backbone_ring_atoms(r.smiles)) for r in recs]
    ringed = [(p, n) for p, n in ringed if n]
    print(f"\n  {len(ringed)} of {len(recs)} library polymers have a ring in "
          f"the backbone")
    for p, n in ringed[:8]:
        print(f"    {p}: {n} skeletal atoms in a ring")
    # Polystyrene's phenyl is pendant and must NOT be counted.
    assert backbone_ring_atoms("[*]CC([*])c1ccccc1") == 0
    assert backbone_ring_atoms("[*]CC[*]") == 0
    # PET's benzene ring IS in the backbone.
    assert backbone_ring_atoms("[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]") == 4
    assert 10 <= len(ringed) <= 40


# ------------------------------------------------------- the full pipeline
@pytest.mark.parametrize("rec", _library(), ids=_ids(_library()))
def test_every_polymer_either_builds_soundly_or_is_refused(rec):
    """Grow a small cell and back-map it. No silent structural damage.

    Either the polymer produces sane bond lengths, or PAAF raises
    :class:`BackboneRingError` explaining precisely why it cannot. What is not
    acceptable is a returned structure with a torn bond.
    """
    # The cell must be big enough for minimum image to hold: nothing may be
    # longer than half the box. A repeat unit with a long side chain needs a
    # bigger cell than polyethylene does, so the size is chosen per polymer
    # rather than fixed. Undersizing it is what produced the 12.29 Å "bonds"
    # that first looked like a back-mapping defect and were an artefact of
    # measuring across a boundary.
    comp = from_chain_counts(
        [Component(name=rec.pid or "P", repeat_unit=rec.smiles,
                   degree_of_polymerisation=20, n_chains=4)], 0.85)
    res = grow_amorphous_cell(comp.grow_specs(), comp.box(),
                              temperature=413.0, seed=5)
    # Growth works for every polymer, ring in the backbone or not, because a
    # bead is a skeletal atom either way.
    assert len(res.molecule.atoms) == comp.total_beads

    try:
        bm = backmap_cell(res, comp.grow_specs())
    except BackboneRingError as exc:
        assert backbone_ring_atoms(rec.smiles) > 0
        assert "runs THROUGH a ring" in str(exc)
        return

    dims = np.array(res.box.bounding_box())
    xyz = np.array([a.xyz for a in bm.molecule.atoms])
    lengths = []
    for i, j, _o in bm.molecule.bonds:
        d = xyz[i] - xyz[j]
        d -= dims * np.round(d / dims)
        lengths.append(float(np.linalg.norm(d)))
    lengths = np.asarray(lengths)
    worst = float(lengths.max())
    assert worst < MAX_BOND, (
        f"{rec.pid}: bond stretched to {worst:.2f} Å — the structure is "
        f"damaged, and it was returned without an error")
    assert float(lengths.min()) > MIN_BOND, (
        f"{rec.pid}: bond collapsed to {lengths.min():.2f} Å")
    assert len(bm.molecule.atoms) > len(res.molecule.atoms), (
        f"{rec.pid}: back-mapping added no atoms")


def test_a_backbone_ring_polymer_still_exports_coarse_grained(tmp_path):
    """Refusing the all-atom step must not throw away the valid bead cell."""
    from paaf.cell.cell_export import export_cell

    pet = "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]"
    comp = from_chain_counts(
        [Component(name="PET", repeat_unit=pet,
                   degree_of_polymerisation=6, n_chains=2)], 1.0)
    res = grow_amorphous_cell(comp.grow_specs(), comp.box(),
                              temperature=413.0, seed=5)
    exp = export_cell(res, comp.grow_specs(), tmp_path, name="pet",
                      run_typing=False)
    print(f"\n  {exp.summary()}")
    print(f"  reason given: {exp.messages[0].splitlines()[0]}")
    assert exp.bead_xyz.exists() and exp.bead_data.exists()
    assert exp.atomistic_xyz is None
    assert not exp.typed
    assert any("runs THROUGH a ring" in m for m in exp.messages)


def test_polymers_with_pendant_rings_do_back_map(tmp_path):
    """The counterpart: a pendant ring must NOT be refused."""
    ps = "[*]CC([*])c1ccccc1"
    comp = from_chain_counts(
        [Component(name="PS", repeat_unit=ps,
                   degree_of_polymerisation=8, n_chains=2)], 1.04)
    res = grow_amorphous_cell(comp.grow_specs(), comp.box(),
                              temperature=413.0, seed=5)
    bm = backmap_cell(res, comp.grow_specs())
    got = Counter(a.element for a in bm.molecule.atoms)
    print(f"\n  polystyrene back-mapped: {dict(got)}")
    assert dict(got) == {"C": 8 * 8 * 2, "H": (8 * 8 + 2) * 2}
