"""The cell's box MUST reach DL_FIELD's control file.

For weeks the amorphous-cell typing failed on "unsaturated C" at atoms that
were geometrically perfect. Root cause: ``cell_export._run_dlfield`` never
passed ``box_ang``, so DL_FIELD fell back to its 250 A default cell. Under
that wrong box, every chain crossing the real 92.4 A boundary looked torn —
198 bonds measured >10 A — and the torn ends were untypeable.

These tests drive the REAL ``_run_dlfield`` code path with a fake dl_field
binary and assert the control file it writes carries the cell's box and a
periodic flag. If the forwarding ever breaks again, this fails loudly
instead of DL_FIELD failing cryptically.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from paaf.cell.cell_export import _run_dlfield


@pytest.fixture()
def fake_dlfield(tmp_path, monkeypatch):
    """A dl_field 'binary' that exits cleanly without producing output."""
    exe = tmp_path / "dl_field"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("DL_FIELD_EXE", str(exe))
    return exe


@pytest.fixture()
def tiny_xyz(tmp_path):
    xyz = tmp_path / "cell_atomistic.xyz"
    xyz.write_text("2\nmethane fragment\nC 0.0 0.0 0.0\nH 1.09 0.0 0.0\n")
    return xyz


def _control_after_run(tmp_path, tiny_xyz, box_ang):
    work = tmp_path / "_typing"
    try:
        _run_dlfield(tiny_xyz, work, "opls2005_dl", tmp_path / "no_lib",
                     lambda _m: None, box_ang=box_ang)
    except Exception:
        pass  # the fake binary makes no outputs; the control file is enough
    ctrls = list(work.glob("*.control"))
    assert ctrls, "control file was never written"
    return ctrls[0].read_text()


def test_the_cell_box_reaches_the_control_file(tmp_path, fake_dlfield,
                                               tiny_xyz):
    text = _control_after_run(tmp_path, tiny_xyz, (92.4, 87.1, 101.6))
    cell_lines = [l for l in text.splitlines() if "Cell vector" in l]
    print("\n".join(f"  {l.strip()}" for l in cell_lines))
    assert len(cell_lines) == 3
    assert "92.4" in cell_lines[0]
    assert "87.1" in cell_lines[1]
    assert "101.6" in cell_lines[2]
    for bad in ("250.0", "500.0"):
        assert all(bad not in l for l in cell_lines), \
            f"DL_FIELD default-box value {bad} leaked into the control file"


def test_a_box_implies_a_periodic_cell(tmp_path, fake_dlfield, tiny_xyz):
    text = _control_after_run(tmp_path, tiny_xyz, (92.4, 92.4, 92.4))
    periodic = [l for l in text.splitlines() if "Periodic" in l]
    assert periodic, "no periodic-condition line in the control file"
    flag = periodic[0].split()[0]
    print(f"  periodic flag = {flag!r}")
    assert flag != "0", "box passed but periodicity left OFF"


def test_both_export_call_sites_pass_the_real_box():
    """Grep-level guard: neither exporter may call _run_dlfield boxless."""
    src = Path(_run_dlfield.__code__.co_filename).read_text()
    calls = [l for l in src.splitlines()
             if "_run_dlfield(" in l and "def _run_dlfield" not in l]
    windows = []
    lines = src.splitlines()
    for n, l in enumerate(lines):
        if "_run_dlfield(" in l and "def " not in l:
            windows.append("\n".join(lines[n:n + 5]))
    assert windows, "no call sites found?"
    for w in windows:
        assert "box_ang=result.box.bounding_box()" in w, \
            f"a _run_dlfield call site does not pass the box:\n{w}"


def test_a_successful_run_is_not_declared_a_failure(tmp_path, monkeypatch):
    """DLFieldResult carries ``return_code``; the caller once probed
    ``returncode`` and its getattr default of 1 turned the first-ever
    successful dl_field run into "dl_field failed (rc=?)"."""
    import paaf.cell.cell_export as ce

    out = tmp_path / "w" / "dlf_output1"
    out.mkdir(parents=True)
    (out / "lammps1.data").write_text("10560 atoms\n")

    class R:  # what run_dlfield actually returns: return_code, no returncode
        return_code = 0

    monkeypatch.setattr("paaf.dlfield_runner.run_dlfield",
                        lambda **kw: R())
    said = []
    got = ce._run_dlfield(tmp_path / "cell.xyz", tmp_path / "w",
                          "opls2005_dl", None, said.append,
                          box_ang=(90.0, 90.0, 90.0))
    print(f"\n  returned: {got}\n  emitted: {said}")
    assert got is not None, "successful run reported as failed"
    assert got.name == "lammps1.data"
    assert not any("failed" in m for m in said)


def test_gromacs_engine_reaches_the_control_file(tmp_path, fake_dlfield,
                                                 tiny_xyz):
    """Same methodology, second format: asking for GROMACS output must set
    DL_FIELD's secondary-output slot to gromacs (and still carry the box)."""
    work = tmp_path / "_t"
    try:
        _run_dlfield(tiny_xyz, work, "opls2005_dl", tmp_path / "no_lib",
                     lambda _m: None, output_engine="gromacs",
                     box_ang=(92.4, 92.4, 92.4))
    except Exception:
        pass
    text = (list(work.glob("*.control"))[0]).read_text()
    sec = [l for l in text.splitlines() if "Secondary output" in l][0]
    print(f"  {sec.strip()}")
    assert sec.split()[0] == "gromacs"
    assert any("92.4" in l for l in text.splitlines() if "Cell vector" in l)


def test_a_gromacs_run_returns_the_gro_not_a_data_complaint(tmp_path,
                                                            monkeypatch):
    """With the secondary slot on gromacs there IS no lammps*.data; the
    artifact to find is the .gro. The old code emitted 'produced no
    lammps*.data' for a perfectly good GROMACS run."""
    import paaf.cell.cell_export as ce

    out = tmp_path / "w" / "dlf_output1"
    out.mkdir(parents=True)
    (out / "gromacs.gro").write_text("cell\n0\n 9.24 9.24 9.24\n")
    (out / "gromacs.top").write_text("[ system ]\ncell\n")

    class R:
        return_code = 0

    monkeypatch.setattr("paaf.dlfield_runner.run_dlfield", lambda **kw: R())
    said = []
    got = ce._run_dlfield(tmp_path / "cell.xyz", tmp_path / "w",
                          "opls2005_dl", None, said.append,
                          output_engine="gromacs", box_ang=(92.4,) * 3)
    print(f"\n  returned: {got}")
    assert got is not None and got.name == "gromacs.gro"
    assert not any("lammps" in m for m in said)
