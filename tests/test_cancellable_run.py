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
