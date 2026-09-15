"""Per-run log file and energy ledger written next to the output files.

The console pane is transient; ``dl_field.log`` and ``moltemplate.log`` are
already left in the project folder, so the pipeline's own messages belong
there too. :func:`run_log` attaches a file handler to the ``paaf`` logger for
the duration of one run (``paaf_run.log``, overwritten each run) and collects
the optimiser's initial/final energies into ``energies.txt``.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import logging
import threading
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

RUN_LOG_NAME = "paaf_run.log"
ENERGY_FILE_NAME = "energies.txt"

_local = threading.local()


def record_energy(label: str, ff: str, e_initial: float, e_final: float,
                  n_atoms: int = 0) -> None:
    """Called by the optimiser after each minimisation (no-op outside a run)."""
    sink: Optional[List[Tuple[str, str, float, float, int]]] = getattr(_local, "energies", None)
    if sink is not None:
        sink.append((label, ff, float(e_initial), float(e_final), int(n_atoms)))


def write_energy_table(path: Path, rows) -> Optional[Path]:
    if not rows:
        return None
    lines = ["# PAAF optimisation energies (OpenBabel force field, kcal/mol)",
             "# label                  force_field   atoms   E_initial      E_final        dE",
             ]
    for label, ff, e0, e1, n in rows:
        lines.append(f"{label:<24s} {ff:<12s} {n:6d}   {e0:12.4f} {e1:12.4f} {e1 - e0:+12.4f}")
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


@contextlib.contextmanager
def run_log(out_dir: Path) -> Iterator[dict]:
    """Attach ``paaf_run.log`` to the ``paaf`` logger and collect energies.

    Yields a dict that is filled at exit with ``log_file`` and
    ``energy_file`` (``None`` when nothing was recorded).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("paaf")
    log_path = out_dir / RUN_LOG_NAME
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    root.info("PAAF run started %s — output folder %s",
              _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), out_dir)
    _local.energies = []
    info = {"log_file": str(log_path), "energy_file": None}
    try:
        yield info
    except BaseException as exc:
        root.error("Run ended with %s: %s", type(exc).__name__, exc)
        raise
    finally:
        rows = getattr(_local, "energies", []) or []
        _local.energies = None
        ep = write_energy_table(out_dir / ENERGY_FILE_NAME, rows)
        info["energy_file"] = str(ep) if ep else None
        if ep:
            root.info("Optimisation energies written to %s", ep)
        root.info("Run log: %s", log_path)
        root.removeHandler(handler)
        handler.close()
