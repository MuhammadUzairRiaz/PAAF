"""Relaxing each blend component before it is packed.

Packing only translates and rotates whole chains, so the geometry a chain
arrives with is the geometry every copy of it keeps. These tests cover the
step that fixes that: one minimisation per component, the relaxed
coordinates read back sorted by atom id, and honest reporting when it cannot
run.

The sorted read-back is the load-bearing part. Replication pairs template atom
*i* with coordinate *i*, so an unsorted dump would pair every atom with
someone else's position — producing a structure that looks plausible and is
scrambled.
"""
from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from paaf.blend_minimise import (                            # noqa: E402
    MinimiseSettings, find_component_input, minimise_component,
    read_last_dump_frame, write_component_min_input,
)

SINGLE_CHAIN = """LAMMPS test chain

4 atoms
3 bonds

2 atom types
1 bond types

0.0 40.0 xlo xhi
0.0 40.0 ylo yhi
0.0 40.0 zlo zhi

Masses

1 12.011
2 1.008

Atoms

1 1 1 -0.10 0.000 0.000 0.000
2 1 1 -0.10 1.530 0.000 0.000
3 1 2  0.05 2.100 0.900 0.000
4 1 2  0.05 9.000 0.000 0.000

Bonds

1 1 1 2
2 1 2 3
3 1 3 4
"""

TEMPLATE_IN = """units real
atom_style full
boundary p p p
pair_style lj/cut 10.0
bond_style harmonic
read_data original.data
thermo 50
thermo_style custom step pe
run 0
"""


def _component(tmp_path: Path) -> Path:
    d = tmp_path / "comp"
    d.mkdir()
    (d / "lammps1.data").write_text(SINGLE_CHAIN)
    (d / "lammps1.in").write_text(TEMPLATE_IN)
    return d / "lammps1.data"


# ------------------------------------------------------- finding the .in
def test_the_matching_input_file_is_found(tmp_path):
    data = _component(tmp_path)
    got = find_component_input(data)
    print(f"\n  {data.name} -> {got.name if got else None}")
    assert got == data.parent / "lammps1.in"


def test_a_lone_input_file_is_accepted(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    (d / "chain.data").write_text(SINGLE_CHAIN)
    (d / "whatever.in").write_text(TEMPLATE_IN)
    assert find_component_input(d / "chain.data") == d / "whatever.in"


def test_several_input_files_are_refused_rather_than_guessed(tmp_path):
    """Picking arbitrarily would minimise with a force field nobody chose."""
    d = tmp_path / "c"
    d.mkdir()
    (d / "chain.data").write_text(SINGLE_CHAIN)
    (d / "opls.in").write_text(TEMPLATE_IN)
    (d / "pcff.in").write_text(TEMPLATE_IN)
    print("\n  two .in files present -> refuses to choose")
    assert find_component_input(d / "chain.data") is None


# ------------------------------------------------------------- the deck
def test_the_minimisation_deck_keeps_the_real_force_field(tmp_path):
    """Styles come from the component's own .in, never from a guess."""
    data = _component(tmp_path)
    out = write_component_min_input(
        data.parent / "lammps1.in", tmp_path / "min.in",
        "component.data", "minimised.lammpstrj", MinimiseSettings())
    text = out.read_text()
    print("\n  " + "\n  ".join(
        l for l in text.splitlines() if l.strip() and not l.startswith("#")))
    assert "pair_style lj/cut 10.0" in text
    assert "bond_style harmonic" in text
    assert "read_data component.data" in text
    assert "minimize" in text


def test_the_dump_is_sorted_by_id(tmp_path):
    """Replication pairs template atom i with coordinate i."""
    data = _component(tmp_path)
    out = write_component_min_input(
        data.parent / "lammps1.in", tmp_path / "min.in",
        "component.data", "d.lammpstrj", MinimiseSettings())
    text = out.read_text()
    assert "dump_modify     dmin sort id" in text, \
        "an unsorted dump would scramble every chain"


def test_the_template_read_data_is_replaced_not_appended(tmp_path):
    data = _component(tmp_path)
    out = write_component_min_input(
        data.parent / "lammps1.in", tmp_path / "min.in",
        "component.data", "d.lammpstrj", MinimiseSettings())
    reads = [l for l in out.read_text().splitlines()
             if l.strip().startswith("read_data")]
    print(f"\n  read_data lines: {reads}")
    assert reads == ["read_data component.data"]


def test_conflicting_template_lines_are_dropped(tmp_path):
    """A template's own thermo/run 0 would fight the ones we add."""
    data = _component(tmp_path)
    out = write_component_min_input(
        data.parent / "lammps1.in", tmp_path / "min.in",
        "component.data", "d.lammpstrj", MinimiseSettings())
    text = out.read_text()
    assert text.count("thermo_style") == 1
    assert text.count("run             0") == 1


# -------------------------------------------------------------- the dump
def test_the_last_frame_wins_and_comes_back_sorted(tmp_path):
    dump = tmp_path / "d.lammpstrj"
    dump.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS id type x y z\n1 1 0.0 0.0 0.0\n2 1 1.0 0.0 0.0\n"
        "ITEM: TIMESTEP\n500\nITEM: NUMBER OF ATOMS\n2\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        # deliberately out of order
        "ITEM: ATOMS id type x y z\n2 1 8.0 9.0 1.0\n1 1 5.0 6.0 7.0\n")
    got = read_last_dump_frame(dump)
    print(f"\n  {got}")
    assert [a[0] for a in got] == [1, 2], "frame was not sorted by id"
    assert got[0][2:] == (5.0, 6.0, 7.0)


def test_a_dump_with_a_different_column_order_still_parses(tmp_path):
    """Columns are read from the header, not assumed."""
    dump = tmp_path / "d.lammpstrj"
    dump.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS type x y z id\n1 1.5 2.5 3.5 7\n")
    got = read_last_dump_frame(dump)
    print(f"\n  {got}")
    assert got == [(7, 1, 1.5, 2.5, 3.5)]


def test_a_dump_missing_coordinates_is_refused(tmp_path):
    dump = tmp_path / "d.lammpstrj"
    dump.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS id type\n1 1\n")
    with pytest.raises(RuntimeError):
        read_last_dump_frame(dump)


# ------------------------------------------------ graceful degradation
def test_no_input_template_means_no_minimisation(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    (d / "chain.data").write_text(SINGLE_CHAIN)          # no .in beside it
    rec, atoms = minimise_component("PE", d / "chain.data", tmp_path / "w")
    print(f"\n  {rec.summary()}")
    assert not rec.ran and atoms is None
    assert "no LAMMPS .in template" in rec.message


def test_no_lammps_means_no_minimisation(tmp_path, monkeypatch):
    import paaf.cell.relax as relax_mod
    monkeypatch.setattr(relax_mod, "find_lammps", lambda *_a, **_k: None)
    data = _component(tmp_path)
    rec, atoms = minimise_component("PE", data, tmp_path / "w")
    print(f"\n  {rec.summary()}")
    assert not rec.ran and atoms is None
    assert "no LAMMPS executable" in rec.message


def test_switching_it_off_is_honoured(tmp_path):
    data = _component(tmp_path)
    rec, atoms = minimise_component("PE", data, tmp_path / "w",
                                    settings=MinimiseSettings(enabled=False))
    assert not rec.ran and atoms is None
    assert "switched off" in rec.message


def test_a_failing_lammps_does_not_claim_a_relaxed_chain(tmp_path):
    """A failed run must fall back to the input geometry, and say so."""
    data = _component(tmp_path)
    exe = tmp_path / "lmp"
    exe.write_text("#!/bin/sh\necho 'ERROR: bad pair style' >&2\nexit 1\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    rec, atoms = minimise_component(
        "PE", data, tmp_path / "w",
        settings=MinimiseSettings(lammps_exe=str(exe)))
    print(f"\n  ran={rec.ran}; {rec.message.splitlines()[0]}")
    assert not rec.ran and atoms is None
    assert "unrelaxed geometry" in rec.message


def test_a_successful_run_reports_the_bond_it_fixed(tmp_path):
    """The chain has a deliberate 6.9 A bond; the stub 'relaxes' it."""
    data = _component(tmp_path)
    exe = tmp_path / "lmp"
    exe.write_text(f"""#!{sys.executable}
import pathlib
rows = [(1, 1, 0.0, 0.0, 0.0), (2, 1, 1.53, 0.0, 0.0),
        (3, 2, 2.10, 0.90, 0.0), (4, 2, 2.70, 1.80, 0.0)]
lines = ["ITEM: TIMESTEP", "0", "ITEM: NUMBER OF ATOMS", "4",
         "ITEM: BOX BOUNDS pp pp pp", "0 40", "0 40", "0 40",
         "ITEM: ATOMS id type x y z"]
for i, t, x, y, z in rows:
    lines.append(f"{{i}} {{t}} {{x}} {{y}} {{z}}")
pathlib.Path("minimised.lammpstrj").write_text("\\n".join(lines) + "\\n")
""")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)

    rec, atoms = minimise_component(
        "PE", data, tmp_path / "w",
        settings=MinimiseSettings(lammps_exe=str(exe)))
    print(f"\n  {rec.summary()}")
    assert rec.ran
    assert atoms is not None and len(atoms) == 4
    assert rec.max_bond_before > 6.0, "the strained input bond was not measured"
    assert rec.max_bond_after < 2.0, "the relaxed geometry was not read back"


def test_the_relaxed_geometry_is_what_gets_packed(tmp_path):
    """The point of the whole exercise: packmol must see the relaxed chain."""
    from paaf.lammps_replicator import _write_pdb_from_data

    data = _component(tmp_path)
    pdb_raw = tmp_path / "raw.pdb"
    pdb_relaxed = tmp_path / "relaxed.pdb"
    _write_pdb_from_data(data, pdb_raw)
    _write_pdb_from_data(data, pdb_relaxed,
                         coords={4: (2.70, 1.80, 0.0)})

    def _last_xyz(p):
        line = [l for l in p.read_text().splitlines()
                if l.startswith("HETATM")][-1]
        return line[30:54]

    print(f"\n  raw     atom 4 -> {_last_xyz(pdb_raw)}")
    print(f"  relaxed atom 4 -> {_last_xyz(pdb_relaxed)}")
    assert "9.000" in _last_xyz(pdb_raw)
    assert "2.700" in _last_xyz(pdb_relaxed)
    # Same atom count and order either way — only positions change.
    assert (len([l for l in pdb_raw.read_text().splitlines()
                 if l.startswith("HETATM")])
            == len([l for l in pdb_relaxed.read_text().splitlines()
                    if l.startswith("HETATM")]))
