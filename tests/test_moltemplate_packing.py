"""The moltemplate route must pack N chains INSIDE the box, not line them up.

Regression for: 103 PBS chains rendered as one rod outside a 400 Å box
because system.lt said ``new PBS_20[103].move(20.0, 0, 0)``.
"""
import math
import re
from pathlib import Path

import numpy as np
import pytest

from paaf import lt_writer
from paaf.lammps_replicator import (_clean, parse_lammps_data,
                                    replicate_single_chain)


def _single_chain_data(path: Path, natoms=12, box=60.0) -> Path:
    lines = ["single", "", f"{natoms} atoms", f"{natoms-1} bonds", "",
             "2 atom types", "1 bond types", "",
             f"0.0 {box} xlo xhi", f"0.0 {box} ylo yhi", f"0.0 {box} zlo zhi", "",
             "Masses", "", "1 12.011", "2 1.008", "", "Atoms  # full", ""]
    for i in range(natoms):
        lines.append(f"{i+1} 1 {1 if i % 2 == 0 else 2} 0.0 {1.5*i:.3f} 0.0 0.0")
    lines += ["", "Bonds", ""]
    for i in range(natoms - 1):
        lines.append(f"{i+1} 1 {i+1} {i+2}")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_system_lt_single_chain_has_no_linear_array(tmp_path):
    chain_lt = tmp_path / "PBS_20.lt"
    chain_lt.write_text("PBS_20 {}\n")
    p = lt_writer.write_system_lt(tmp_path, chain_lt, n_chains=1, box=[400, 400, 400])
    txt = p.read_text()
    assert "chains = new PBS_20" in txt
    assert ".move(20.0" not in txt
    assert "[" not in txt.split("write_once")[0]


def test_system_lt_multi_chain_grid_stays_inside_box(tmp_path):
    chain_lt = tmp_path / "X.lt"
    chain_lt.write_text("X {}\n")
    n, L = 103, 400.0
    p = lt_writer.write_system_lt(tmp_path, chain_lt, n_chains=n, box=[L, L, L])
    txt = p.read_text()
    moves = re.findall(r"new X\.move\(([-\d.]+), ([-\d.]+), ([-\d.]+)\)", txt)
    assert len(moves) == n
    xyz = np.array(moves, dtype=float)
    assert (xyz >= 0).all() and (xyz < L).all()
    assert len({tuple(m) for m in moves}) == n          # all distinct positions
    assert "[103]" not in txt                            # no 1-D array idiom


def test_replicate_puts_every_chain_inside_box(tmp_path):
    single = _single_chain_data(tmp_path / "single.data")
    n, L = 20, 60.0
    out = replicate_single_chain(single, n, (L, L, L), tmp_path / "packed_box.data")
    header, sec = parse_lammps_data(out)
    atoms = _clean(sec["Atoms"])
    assert len(atoms) == 12 * n
    xyz = np.array([[float(x) for x in l.split()[4:7]] for l in atoms])
    assert (xyz >= 0).all() and (xyz <= L).all()
    assert len({int(l.split()[1]) for l in atoms}) == n   # distinct molecule ids
    assert len(_clean(sec["Bonds"])) == 11 * n
    # bonds still 1.5 Å after packing — chains were moved rigidly
    pos = {int(l.split()[0]): np.array([float(x) for x in l.split()[4:7]]) for l in atoms}
    for b in _clean(sec["Bonds"]):
        _, _, i, j = (int(x) for x in b.split()[:4])
        assert math.isclose(np.linalg.norm(pos[i] - pos[j]), 1.5, abs_tol=1e-3)


def test_lammps_input_includes_init_and_charges(tmp_path):
    p = lt_writer.write_lammps_input(tmp_path, data_file="packed_box.data",
                                     init_file="system.in.init",
                                     charges_file="system.in.charges")
    lines = p.read_text().splitlines()
    i_init = next(i for i, l in enumerate(lines) if "include" in l and "system.in.init" in l)
    i_read = next(i for i, l in enumerate(lines) if l.startswith("read_data"))
    i_chg = next(i for i, l in enumerate(lines) if "system.in.charges" in l)
    assert i_init < i_read < i_chg              # styles before data, charges after
    assert "read_data       packed_box.data" in lines[i_read]


def test_lammps_input_default_unchanged(tmp_path):
    p = lt_writer.write_lammps_input(tmp_path)
    txt = p.read_text()
    assert "system.in.init" not in txt and "system.in.charges" not in txt


def test_pipeline_moltemplate_route_writes_single_chain_then_packs():
    """The pipeline must hand moltemplate ONE chain and pack copies after
    (mbuild is not installed here, so this checks the call graph in source)."""
    import ast
    import paaf.pipeline as pl
    tree = ast.parse(Path(pl.__file__).read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_pipeline")
    calls = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            calls.setdefault(name, []).append(node)
    (sys_lt,) = calls["write_system_lt"]
    kw = {k.arg: k.value for k in sys_lt.keywords}
    assert isinstance(kw["n_chains"], ast.Constant) and kw["n_chains"].value == 1
    # replicate_single_chain is called twice: DL_FIELD route and moltemplate route
    assert len(calls["replicate_single_chain"]) == 2
    assert "run_moltemplate" in calls
    inp = calls["write_lammps_input"][0]
    kws = {k.arg for k in inp.keywords}
    assert {"init_file", "charges_file", "data_file"} <= kws
