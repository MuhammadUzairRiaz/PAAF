"""Cancel must kill external programs immediately, not wait them out.

The Cancel button used to "work" only between chains: once the build
reached the LAMMPS push-off or DL_FIELD, subprocess.run() blocked and the
GUI hung until the external program finished by itself.
"""
from __future__ import annotations

import threading
import time

import pytest

from paaf.cell.packing import CancelToken, PackCancelled, run_cancellable


def test_a_cancelled_token_kills_the_child_quickly():
    tok = CancelToken()
    threading.Timer(0.4, tok.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(PackCancelled):
        run_cancellable(["sleep", "30"], cancel=tok)
    took = time.monotonic() - t0
    print(f"\n  killed after {took:.2f}s (child wanted 30s)")
    assert took < 3.0, "cancel did not interrupt the child promptly"


def test_uncancelled_runs_complete_normally():
    rc, out, _err = run_cancellable(["echo", "hello"], cancel=CancelToken())
    assert rc == 0 and "hello" in out


def test_timeout_still_applies():
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        run_cancellable(["sleep", "30"], timeout_s=1.0)
    assert time.monotonic() - t0 < 5.0


def test_export_cell_accepts_the_cancel_token_the_gui_passes():
    """The GUI worker calls export_cell(..., cancel=token). A signature
    mismatch here broke EVERY build with "unexpected keyword argument
    'cancel'" — so the full chain of signatures is pinned."""
    import inspect

    from paaf.cell.cell_export import _run_dlfield, export_cell
    from paaf.cell.soft_pushoff import soft_pushoff
    from paaf.dlfield_runner import run_dlfield

    for fn in (export_cell, _run_dlfield, soft_pushoff, run_dlfield):
        assert "cancel" in inspect.signature(fn).parameters, fn.__name__


def test_export_cell_with_cancel_token_builds_untyped(tmp_path):
    """End-to-end: the exact GUI call shape, with a real token."""
    pytest.importorskip("rdkit")
    from paaf.cell.composition import Component, from_chain_counts
    from paaf.cell.grow import grow_amorphous_cell
    from paaf.cell.cell_export import export_cell

    comp = from_chain_counts(
        [Component(name="PE", repeat_unit="[*]CC[*]",
                   degree_of_polymerisation=10, n_chains=2, ris_key="PE")],
        0.85)
    res = grow_amorphous_cell(comp.grow_specs(), comp.box(),
                              temperature=413.0, seed=5)
    exp = export_cell(res, comp.grow_specs(), tmp_path, name="pe",
                      ff_key="", push_off=False, cancel=CancelToken())
    assert exp.bead_xyz is not None


def test_pipeline_cancel_token_stops_before_any_work(tmp_path):
    import inspect
    from paaf.cell.packing import CancelToken
    from paaf.config import Config
    from paaf.pipeline import PipelineCancelled, run_pipeline
    cfg = Config(); cfg.output_dir = str(tmp_path); cfg.project_name = "x"
    tok = CancelToken(); tok.cancel()
    seen = []
    with pytest.raises(PipelineCancelled):
        run_pipeline(cfg, progress=seen.append, cancel=tok)
    assert len(seen) == 1 and seen[0].startswith("[stage 1/")     # stopped at the first line


def test_cancel_reaches_every_external_program():
    """dl_field, moltemplate.sh, packmol and the optimiser all take `cancel`."""
    import inspect
    from paaf import optimizer, moltemplate_runner, lammps_replicator, dlfield_runner
    from paaf.geometry_repair import ensure_clean_geometry, relax_topology
    for fn in (optimizer.optimize, moltemplate_runner.run_moltemplate,
               lammps_replicator.replicate_single_chain, lammps_replicator._pack_with_packmol,
               dlfield_runner.run_dlfield, ensure_clean_geometry, relax_topology):
        assert "cancel" in inspect.signature(fn).parameters, fn.__name__


def test_run_cancellable_kills_child_reading_stdin(tmp_path):
    import sys
    from paaf.cell.packing import CancelToken, PackCancelled, run_cancellable
    tok = CancelToken()
    import threading
    threading.Timer(0.3, tok.cancel).start()
    inp = tmp_path / "in.txt"; inp.write_text("x\n")
    with inp.open() as fh, pytest.raises(PackCancelled):
        run_cancellable([sys.executable, "-c", "import time; time.sleep(30)"],
                        stdin=fh, cancel=tok, poll_s=0.1)


def test_geometry_repair_honours_cancel():
    import numpy as np
    from paaf.cell.packing import CancelToken, PackCancelled
    from paaf.geometry_repair import relax_topology
    from paaf.structure import Atom, Molecule
    mol = Molecule(atoms=[Atom(index=i, element="C", xyz=np.array([0.3 * i, 0, 0]))
                          for i in range(6)],
                   bonds=[(i, i + 1, 1.0) for i in range(5)])
    tok = CancelToken(); tok.cancel()
    with pytest.raises(PackCancelled):
        relax_topology(mol, cancel=tok)
