"""Every export writes benchmark.json / benchmark.txt and a history row."""
import csv
import json
import sys
import time
from pathlib import Path

import pytest

from paaf import benchmark as bench
from paaf.cell.packing import CancelToken, PackCancelled, run_cancellable


def _history(tmp_path):
    p = bench.history_path()
    assert str(p).startswith(str(tmp_path.parent)) or p.exists()
    with p.open() as fh:
        return list(csv.DictReader(fh))


def test_phases_external_time_and_files(tmp_path):
    out = tmp_path / "export"
    lines = []
    with bench.benchmark("unit_tool", out, workload={"n": 3},
                         emit=lines.append) as b:
        bench.phase("python work")
        time.sleep(0.05)
        bench.phase("child program")
        rc, _o, _e = run_cancellable(
            [sys.executable, "-c", "import time; time.sleep(0.2)"])
        assert rc == 0
        (out / "system.data").parent.mkdir(parents=True, exist_ok=True)
        (out / "system.data").write_text("LAMMPS\n\n1200 atoms\n3 bonds\n")
        b.size_from(out / "system.data")
        b.metric(chains=4)

    r = json.loads((out / bench.BENCH_JSON).read_text())
    assert r["status"] == "completed" and r["tool"] == "unit_tool"
    t = r["totals"]
    assert t["wall_s"] >= 0.25
    # The child's run time is credited to external programs, not to PAAF.
    assert 0.18 <= t["external_s"] <= t["wall_s"]
    assert abs(t["paaf_s"] + t["external_s"] - t["wall_s"]) < 1e-3
    (prog, stats), = r["external_programs"].items()
    assert stats["calls"] == 1 and Path(sys.executable).stem == prog
    names = [p["name"] for p in r["phases"]]
    assert names[:2] == ["python work", "child program"]
    child = r["phases"][1]
    assert child["external_s"] == pytest.approx(t["external_s"], abs=1e-3)
    assert r["size"] == {"chains": 4, "atoms": 1200}
    assert t["atoms_per_s"] == pytest.approx(1200 / t["wall_s"], rel=1e-2)
    assert t["files_written"] == 1
    assert r["environment"]["logical_cores"]
    assert "PAAF benchmark — unit_tool" in (out / bench.BENCH_TXT).read_text()
    assert any(ln.startswith("[benchmark] unit_tool") for ln in lines)

    rows = _history(tmp_path)
    assert rows[-1]["tool"] == "unit_tool" and rows[-1]["atoms"] == "1200"
    assert rows[-1]["workload_id"] == r["workload_id"]


def test_same_inputs_same_workload_id(tmp_path):
    ids = []
    for i in range(2):
        with bench.benchmark("wl", tmp_path / f"r{i}", workload={"a": 1}) as b:
            pass
        ids.append(b.report["workload_id"])
    with bench.benchmark("wl", tmp_path / "r2", workload={"a": 2}) as b:
        pass
    assert ids[0] == ids[1] != b.report["workload_id"]


def test_nested_export_becomes_a_phase(tmp_path):
    @bench.benchmarked("inner", folder=lambda a: a["out"])
    def inner(out, progress=None):
        bench.phase("step")
        return out

    with bench.benchmark("outer", tmp_path / "o") as b:
        bench.phase("grow")
        bench.end_phase()
        inner(tmp_path / "ignored")
    names = [p["name"] for p in b.report["phases"]
             if p["name"] != "(outside named phases)"]
    assert names == ["grow", "inner", "inner / step"]
    # Only the outermost run writes a report and a history row.
    assert (tmp_path / "o" / bench.BENCH_JSON).exists()
    assert not (tmp_path / "ignored").exists()
    assert [r["tool"] for r in _history(tmp_path)] == ["outer"]


def test_failed_and_cancelled_runs_are_still_recorded(tmp_path):
    with pytest.raises(RuntimeError):
        with bench.benchmark("t", tmp_path / "f"):
            raise RuntimeError("boom")
    r = json.loads((tmp_path / "f" / bench.BENCH_JSON).read_text())
    assert r["status"] == "failed" and "boom" in r["error"]

    tok = CancelToken()
    tok.cancel()
    shown = []
    with pytest.raises(PackCancelled):
        with bench.benchmark("t", tmp_path / "c", emit=shown.append):
            run_cancellable([sys.executable, "-c", "pass"], cancel=tok)
    r = json.loads((tmp_path / "c" / bench.BENCH_JSON).read_text())
    assert r["status"] == "cancelled"
    assert shown == []                 # nothing printed after a Cancel


def test_decorated_export_reports_even_when_the_callback_raises(tmp_path):
    def progress(_msg):
        raise PackCancelled("cancel check in the callback")

    @bench.benchmarked("deco", folder=lambda a: a["out"])
    def export(out, progress=None):
        Path(out).mkdir(parents=True, exist_ok=True)
        return 1

    assert export(tmp_path / "d", progress=progress) == 1
    assert (tmp_path / "d" / bench.BENCH_TXT).exists()


@pytest.mark.parametrize("name,text,n", [
    ("a.data", "title\n\n  37 atoms\n", 37),
    ("a.gro", "title\n  12\n", 12),
    ("a.xyz", "5\ncomment\n", 5),
    ("a.pdb", "ATOM  1\nHETATM 2\nEND\n", 2),
])
def test_count_atoms(tmp_path, name, text, n):
    (tmp_path / name).write_text(text)
    assert bench.count_atoms(tmp_path / name) == n


def test_amorphous_cell_export_writes_a_benchmark(tmp_path):
    pytest.importorskip("rdkit")
    from paaf.cell.cell_export import export_cell
    from paaf.cell.composition import Component, from_chain_counts
    from paaf.cell.grow import grow_amorphous_cell

    comp = from_chain_counts([Component(name="PE", repeat_unit="[*]CC[*]",
                                        degree_of_polymerisation=10,
                                        n_chains=2)], 0.5)
    res = grow_amorphous_cell(comp.grow_specs(), comp.box(), seed=3)
    exp = export_cell(res, comp.grow_specs(), tmp_path, name="pe",
                      run_typing=False)
    r = json.loads((exp.folder / bench.BENCH_JSON).read_text())
    assert r["tool"] == "amorphous_cell_export"
    assert r["size"]["atoms"] == exp.n_atoms > 0
    names = [p["name"] for p in r["phases"]]
    assert "back-map beads to atoms" in names
