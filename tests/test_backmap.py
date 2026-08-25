"""Acceptance tests for back-mapping a bead cell to atoms, and exporting it.

The claim being tested is narrow and should stay narrow: back-mapping adds
atoms around a backbone **without moving the backbone**, and it adds the right
ones. Every measured value is printed.

What is deliberately NOT asserted: that the result is a physically relaxed
structure. It is not, and the module says so. Side-group rotamers come from
one template and neighbouring chains interpenetrate; that is what the
subsequent minimisation is for.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit", reason="back-mapping requires RDKit")

from paaf.cell.backmap import (                            # noqa: E402
    backmap_cell, backmap_chain, build_unit_template,
)
from paaf.cell.composition import Component, from_chain_counts   # noqa: E402
from paaf.cell.grow import grow_amorphous_cell               # noqa: E402

PE = "[*]CC[*]"
PS = "[*]CC([*])c1ccccc1"
PEO = "[*]OCC[*]"


def _cell(smiles, dp, n_chains, density, ris_key="", seed=5, name="P"):
    comp = from_chain_counts(
        [Component(name=name, repeat_unit=smiles,
                   degree_of_polymerisation=dp, n_chains=n_chains,
                   ris_key=ris_key)], density)
    specs = comp.grow_specs()
    res = grow_amorphous_cell(specs, comp.box(), temperature=413.0, seed=seed)
    return res, specs


# ------------------------------------------------------------- the template
@pytest.mark.parametrize("smiles,n_backbone,elements", [
    (PE, 2, {"C": 2, "H": 6}),          # incl. the 2 link hydrogens
    (PS, 2, {"C": 8, "H": 10}),
    (PEO, 3, {"O": 1, "C": 2, "H": 6}),
])
def test_template_finds_the_backbone_and_keeps_the_chemistry(
        smiles, n_backbone, elements):
    """The backbone is the path between attachments; side groups are branches."""
    t = build_unit_template(smiles)
    got = Counter(t.elements)
    print(f"\n  {smiles:26s} backbone {t.backbone} ({t.n_backbone} atoms)")
    print(f"  template formula: {dict(got)}   substituents per backbone atom: "
          f"{[len(a) for a in t.attached]}")
    assert t.n_backbone == n_backbone
    assert dict(got) == elements
    # Every atom is accounted for exactly once: backbone, substituent or link.
    covered = set(t.backbone) | {t.head_link, t.tail_link}
    for group in t.attached:
        covered |= set(group)
    assert len(covered) == len(t.elements)


def test_phenyl_is_a_branch_not_backbone():
    """Polystyrene's ring must not be mistaken for the chain."""
    t = build_unit_template(PS)
    sizes = [len(a) for a in t.attached]
    print(f"\n  PS substituent group sizes: {sizes} "
          f"(one CH2 hydrogen pair, one phenyl + H)")
    assert t.n_backbone == 2
    assert max(sizes) >= 11          # C6H5 = 11 atoms, plus the methine H


# ------------------------------------------------- the backbone does not move
def test_backbone_atoms_land_exactly_on_the_grown_beads():
    """Back-mapping must never relocate what growth validated."""
    res, specs = _cell(PE, 15, 2, 0.85, ris_key="PE")
    bm = backmap_cell(res, specs, push_off=False)
    flags = np.asarray(getattr(bm, "backbone_flags"), dtype=bool)

    beads = np.array([a.xyz for a in res.molecule.atoms])
    got = np.array([a.xyz for a in bm.molecule.atoms])[flags]
    dims = np.array(res.box.bounding_box())
    d = got - beads
    d -= dims * np.round(d / dims)
    worst = float(np.abs(d).max())
    print(f"\n  {flags.sum()} backbone atoms vs {len(beads)} beads")
    print(f"  worst coordinate difference (min image): {worst:.2e} Å")
    assert flags.sum() == len(beads)
    assert worst < 1e-9


def test_push_off_moves_side_atoms_only():
    res, specs = _cell(PE, 15, 2, 0.85, ris_key="PE")
    bm = backmap_cell(res, specs, push_off=True)
    flags = np.asarray(getattr(bm, "backbone_flags"), dtype=bool)
    beads = np.array([a.xyz for a in res.molecule.atoms])
    got = np.array([a.xyz for a in bm.molecule.atoms])[flags]
    dims = np.array(res.box.bounding_box())
    d = got - beads
    d -= dims * np.round(d / dims)
    print(f"\n  backbone displacement after push-off: {np.abs(d).max():.2e} Å")
    assert float(np.abs(d).max()) < 1e-9


# --------------------------------------------------------------- chemistry
def test_polyethylene_formula_is_exact():
    """A DP-n PE chain is C(2n)H(4n+2) — the caps are the +2."""
    dp, n_chains = 20, 3
    res, specs = _cell(PE, dp, n_chains, 0.85, ris_key="PE")
    bm = backmap_cell(res, specs)
    got = Counter(a.element for a in bm.molecule.atoms)
    want = {"C": 2 * dp * n_chains, "H": (4 * dp + 2) * n_chains}
    print(f"\n  {n_chains} x DP {dp} polyethylene")
    print(f"  got {dict(got)}   expected {want}")
    assert dict(got) == want


def test_polystyrene_formula_is_exact():
    """A DP-n PS chain is C(8n)H(8n+2)."""
    dp, n_chains = 10, 2
    res, specs = _cell(PS, dp, n_chains, 1.04, ris_key="PS")
    bm = backmap_cell(res, specs)
    got = Counter(a.element for a in bm.molecule.atoms)
    want = {"C": 8 * dp * n_chains, "H": (8 * dp + 2) * n_chains}
    print(f"\n  {n_chains} x DP {dp} polystyrene")
    print(f"  got {dict(got)}   expected {want}")
    assert dict(got) == want


def test_bond_lengths_are_chemically_sensible():
    """Every bond must be a real bond length, before and after the push-off."""
    res, specs = _cell(PE, 20, 3, 0.85, ris_key="PE")
    dims = np.array(res.box.bounding_box())
    print()
    for push in (False, True):
        bm = backmap_cell(res, specs, push_off=push)
        xyz = np.array([a.xyz for a in bm.molecule.atoms])
        lengths = []
        for i, j, _o in bm.molecule.bonds:
            d = xyz[i] - xyz[j]
            d -= dims * np.round(d / dims)
            lengths.append(float(np.linalg.norm(d)))
        lengths = np.asarray(lengths)
        print(f"  push_off={push!s:5s}  bonds {len(lengths)}  "
              f"min {lengths.min():.3f}  max {lengths.max():.3f} Å")
        assert lengths.min() > 0.9, "a bond collapsed"
        assert lengths.max() < 1.8, "a bond was stretched"


def test_ring_bonds_survive_the_push_off():
    """The case a spanning-tree constraint scheme silently destroys.

    An earlier push-off pinned each mobile atom to ONE parent. A tree cannot
    constrain a cycle, so every phenyl ring-closing bond was left free and was
    torn open to ~5 Å. Polyethylene could not reveal it — every mobile atom
    there is a hydrogen bonded straight to a pinned backbone carbon, for which
    a tree is sufficient. The test has to have the ring in it.
    """
    res, specs = _cell(PS, 10, 2, 1.04, ris_key="PS")
    dims = np.array(res.box.bounding_box())
    print()
    for push in (False, True):
        bm = backmap_cell(res, specs, push_off=push)
        xyz = np.array([a.xyz for a in bm.molecule.atoms])
        lengths = []
        for i, j, _o in bm.molecule.bonds:
            d = xyz[i] - xyz[j]
            d -= dims * np.round(d / dims)
            lengths.append(float(np.linalg.norm(d)))
        lengths = np.asarray(lengths)
        n_bad = int(np.sum(lengths > 1.8))
        print(f"  push_off={push!s:5s}  bonds {len(lengths)}  "
              f"min {lengths.min():.4f}  max {lengths.max():.4f}  "
              f"over 1.8 Å: {n_bad}")
        assert lengths.max() < 1.8, "a ring bond was torn open"
        assert lengths.min() > 0.9


def test_side_groups_are_rotated_about_their_own_bond():
    """A rigid ring can only be relieved by rotating it, not by pushing atoms."""
    from paaf.cell.backmap import rotate_side_groups

    res, specs = _cell(PS, 10, 2, 1.04, ris_key="PS")
    dims = np.array(res.box.bounding_box())
    bm = backmap_cell(res, specs, push_off=False)
    flags = getattr(bm, "backbone_flags")

    xyz0 = np.array([a.xyz for a in bm.molecule.atoms]).copy()
    n_rot = rotate_side_groups(bm.molecule, flags, dims)
    xyz1 = np.array([a.xyz for a in bm.molecule.atoms])

    lengths = []
    for i, j, _o in bm.molecule.bonds:
        d = xyz1[i] - xyz1[j]
        d -= dims * np.round(d / dims)
        lengths.append(float(np.linalg.norm(d)))
    lengths = np.asarray(lengths)
    moved = float(np.abs(xyz1 - xyz0).max())
    bb = np.asarray(flags, dtype=bool)
    print(f"\n  rotated {n_rot} side groups, max atom displacement {moved:.2f} Å")
    print(f"  bond lengths after rotation: {lengths.min():.4f} / "
          f"{lengths.max():.4f} Å  (rigid rotation preserves them exactly)")
    assert n_rot > 0, "no phenyl group was rotated"
    assert moved > 0.1
    assert lengths.max() < 1.8 and lengths.min() > 0.9
    # A rigid rotation of a side group must never move the backbone.
    assert float(np.abs(xyz1[bb] - xyz0[bb]).max()) < 1e-9


def test_ring_spearing_is_checked_on_the_all_atom_cell():
    """Growth cannot see rings; the finished cell can, and must be checked."""
    from paaf.cell.backmap import count_speared_rings, find_rings

    res, specs = _cell(PS, 10, 2, 1.04, ris_key="PS")
    bm = backmap_cell(res, specs)
    dims = np.array(res.box.bounding_box())
    rings = find_rings(bm.molecule)
    n_speared, offenders = count_speared_rings(bm.molecule, dims)
    print(f"\n  rings found in the all-atom cell : {len(rings)} "
          f"(expect 20 phenyls for 2 x DP 10)")
    print(f"  bonds threading a ring           : {n_speared}")
    assert len(rings) == 20
    assert n_speared == getattr(bm, "n_speared")
    assert any("Ring-spearing check" in n or "thread through a ring" in n
               for n in bm.notes)


def test_polyethylene_cell_reports_no_rings():
    from paaf.cell.backmap import find_rings

    res, specs = _cell(PE, 15, 2, 0.85, ris_key="PE")
    bm = backmap_cell(res, specs)
    print(f"\n  rings in a polyethylene cell: {len(find_rings(bm.molecule))}")
    assert find_rings(bm.molecule) == []
    assert getattr(bm, "n_speared") == 0


def test_push_off_relieves_the_worst_contact():
    """The whole point: make the cell safe to hand to a minimiser."""
    res, specs = _cell(PE, 20, 3, 0.85, ris_key="PE")
    raw = backmap_cell(res, specs, push_off=False)
    fixed = backmap_cell(res, specs, push_off=True)
    print(f"\n  closest non-bonded contact: {raw.worst_contact_a:.2f} Å "
          f"-> {fixed.worst_contact_a:.2f} Å after push-off")
    assert fixed.worst_contact_a > raw.worst_contact_a


# --------------------------------------------------------------- tacticity
def test_tacticity_changes_the_structure_and_is_labelled():
    res, specs = _cell(PS, 10, 2, 1.04, ris_key="PS")
    iso = backmap_cell(res, specs, tacticity="isotactic", push_off=False)
    syn = backmap_cell(res, specs, tacticity="syndiotactic", push_off=False)
    a = np.array([x.xyz for x in iso.molecule.atoms])
    b = np.array([x.xyz for x in syn.molecule.atoms])
    diff = float(np.abs(a - b).max())
    print(f"\n  isotactic vs syndiotactic: max atom displacement {diff:.2f} Å")
    print(f"  isotactic result labels itself: "
          f"{any('ISOTACTIC' in n for n in iso.notes)}")
    assert diff > 0.5, "tacticity had no effect on the coordinates"
    assert any("ISOTACTIC" in n for n in iso.notes)
    assert len(a) == len(b), "tacticity must not change the formula"


def test_unknown_tacticity_is_refused():
    res, specs = _cell(PE, 10, 1, 0.85, ris_key="PE")
    t = build_unit_template(PE)
    beads = np.array([a.xyz for a in res.molecule.atoms])
    with pytest.raises(ValueError):
        backmap_chain(beads, t, 10, tacticity="heterotactic")


def test_bead_count_mismatch_is_caught():
    t = build_unit_template(PE)
    with pytest.raises(ValueError):
        backmap_chain(np.zeros((7, 3)), t, 10)      # needs 20


# ------------------------------------------------------------------ export
def test_export_writes_the_bead_cell_and_a_valid_lammps_header(tmp_path):
    from paaf.cell.cell_export import export_cell

    res, specs = _cell(PE, 15, 3, 0.85, ris_key="PE")
    exp = export_cell(res, specs, tmp_path, name="pe", atomistic=False)
    text = exp.bead_data.read_text()
    head = text.splitlines()[:12]
    print("\n  " + "\n  ".join(head))
    assert exp.bead_xyz.exists() and exp.bead_data.exists()
    assert f"{len(res.molecule.atoms)} atoms" in text
    assert "Masses" in text and "Bonds" in text
    assert "Atoms # molecular" in text
    # Coarse-grained output must never claim to be typed.
    assert not exp.typed
    assert any("coarse-grained" in m for m in exp.messages)


def test_bead_masses_reproduce_the_target_density(tmp_path):
    """The masses written into the data file must give the cell's density."""
    from paaf.cell.cell_export import export_cell

    want = 0.88
    res, specs = _cell(PE, 20, 4, want, ris_key="PE")
    exp = export_cell(res, specs, tmp_path, name="pe", atomistic=False)
    lines = exp.bead_data.read_text().splitlines()
    i = lines.index("Masses")
    mass = float(lines[i + 2].split()[1])
    total = mass * len(res.molecule.atoms)
    vol = np.prod(res.box.bounding_box())
    got = (total / 6.02214076e23) / (vol * 1e-24)
    print(f"\n  bead mass in file : {mass:.4f} amu  (PE CH2 = 14.027)")
    print(f"  density from file : {got:.4f} g/cm³  (target {want})")
    assert abs(mass - 14.027) < 0.01
    assert abs(got - want) < 1e-6


def test_export_reports_when_typing_did_not_happen(tmp_path):
    """An untyped run must say so rather than leave a plausible .data file."""
    from paaf.cell.cell_export import export_cell

    res, specs = _cell(PE, 10, 2, 0.85, ris_key="PE")
    exp = export_cell(res, specs, tmp_path, name="pe", ff_key="")
    print(f"\n  {exp.summary()}")
    for m in exp.messages:
        print(f"    - {m}")
    assert exp.atomistic_xyz.exists()
    assert exp.typed_data is None
    assert not exp.typed
    assert any("No force field" in m for m in exp.messages)


def test_export_gromacs_format_delivers_gro_top_and_itps(tmp_path,
                                                         monkeypatch):
    """output_formats="both": the cell folder must end up with cell.data,
    cell.in, cell.gro, cell.top and the .itp includes — DL_FIELD asked once
    per format, same typing both times."""
    import paaf.cell.cell_export as ce
    from paaf.cell.cell_export import export_cell

    engines_asked = []

    def fake_run(structure, work_dir, ff_key, dl_dir, emit,
                 output_engine="lammps", box_ang=None):
        engines_asked.append(output_engine)
        assert box_ang is not None, "the box must be passed for BOTH engines"
        out = Path(work_dir) / "dlf_output1"
        out.mkdir(parents=True, exist_ok=True)
        if output_engine == "gromacs":
            (out / "gromacs.gro").write_text("cell\n0\n 9.2 9.2 9.2\n")
            (out / "gromacs.top").write_text('#include "gromacs1.itp"\n')
            (out / "gromacs1.itp").write_text("[ moleculetype ]\nXYZ 3\n")
            return out / "gromacs.gro"
        (out / "lammps1.data").write_text("2 atoms\n")
        (out / "lammps.in").write_text("pair_style lj/cut 12.0\n"
                                       "read_data lammps1.data\n")
        return out / "lammps1.data"

    monkeypatch.setattr(ce, "_run_dlfield", fake_run)
    res, specs = _cell(PE, 10, 2, 0.85, ris_key="PE")
    exp = export_cell(res, specs, tmp_path, name="pe",
                      ff_key="opls2005_dl", push_off=False,
                      output_formats="both")
    print(f"\n  engines asked: {engines_asked}")
    print(f"  files: {sorted(p.name for p in exp.folder.iterdir())}")
    assert engines_asked == ["lammps", "gromacs"]
    assert exp.typed
    assert exp.typed_data is not None and exp.typed_data.exists()
    assert exp.typed_gro is not None and exp.typed_gro.exists()
    assert exp.typed_top is not None and exp.typed_top.exists()
    assert (exp.folder / "gromacs1.itp").exists(), \
        "the .top #includes gromacs1.itp; it must travel with it"


def test_export_gromacs_only_needs_no_lammps_run(tmp_path, monkeypatch):
    import paaf.cell.cell_export as ce
    from paaf.cell.cell_export import export_cell

    engines_asked = []

    def fake_run(structure, work_dir, ff_key, dl_dir, emit,
                 output_engine="lammps", box_ang=None):
        engines_asked.append(output_engine)
        out = Path(work_dir) / "dlf_output1"
        out.mkdir(parents=True, exist_ok=True)
        (out / "gromacs.gro").write_text("cell\n0\n 9.2 9.2 9.2\n")
        (out / "gromacs.top").write_text("[ system ]\ncell\n")
        return out / "gromacs.gro"

    monkeypatch.setattr(ce, "_run_dlfield", fake_run)
    res, specs = _cell(PE, 10, 2, 0.85, ris_key="PE")
    exp = export_cell(res, specs, tmp_path, name="pe",
                      ff_key="opls2005_dl", push_off=False,
                      output_formats="gromacs")
    print(f"\n  engines asked: {engines_asked}")
    assert engines_asked == ["gromacs"]
    assert exp.typed
    assert exp.typed_data is None
    assert exp.typed_gro.exists() and exp.typed_top.exists()
    assert not any("did not produce a .data" in m for m in exp.messages), \
        "a gromacs-only export must not complain about the missing .data"
