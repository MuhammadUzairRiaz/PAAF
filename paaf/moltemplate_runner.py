"""Thin wrapper around ``moltemplate.sh``."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .logging_utils import get_logger

log = get_logger(__name__)


def find_moltemplate() -> Optional[str]:
    return shutil.which("moltemplate.sh")


def run_moltemplate(
    system_lt: str | Path,
    work_dir: str | Path | None = None,
    xyz: str | Path | None = None,
    extra_args: Optional[list[str]] = None,
) -> Path:
    """Run moltemplate.sh on `system_lt` and return path to the .data file."""
    exe = find_moltemplate()
    if not exe:
        raise RuntimeError(
            "moltemplate.sh not found on PATH. Install Moltemplate: "
            "`pip install moltemplate` or from https://moltemplate.org"
        )
    system_lt = Path(system_lt).resolve()
    wd = Path(work_dir) if work_dir else system_lt.parent
    cmd = [exe]
    if xyz is not None:
        cmd += ["-xyz", str(Path(xyz).resolve())]
    if extra_args:
        cmd += list(extra_args)
    cmd.append(system_lt.name)
    log.info("Running moltemplate: %s (cwd=%s)", " ".join(cmd), wd)
    proc = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True)

    # Always persist moltemplate's full stdout+stderr so the user can inspect
    # it in the output directory.
    log_path = wd / "moltemplate.log"
    try:
        log_path.write_text(
            "$ " + " ".join(cmd) + "\n"
            "(cwd=" + str(wd) + ")\n\n"
            "----- STDOUT -----\n" + (proc.stdout or "") + "\n"
            "----- STDERR -----\n" + (proc.stderr or "") + "\n"
        )
    except OSError:
        pass

    if proc.returncode != 0:
        log.error("moltemplate failed (exit %d):\n%s\n%s",
                  proc.returncode, proc.stdout, proc.stderr)
        # Take the last ~30 lines of stderr (or stdout if stderr is empty) so
        # the popup dialog shows the actual root cause, not a generic message.
        blob = (proc.stderr or "").strip() or (proc.stdout or "").strip()
        tail = "\n".join(blob.splitlines()[-30:]) or "(no output captured)"
        raise RuntimeError(
            f"moltemplate.sh returned {proc.returncode}.\n\n"
            f"Full log written to:\n  {log_path}\n\n"
            f"Last lines of moltemplate output:\n\n{tail}"
        )
    data = wd / f"{system_lt.stem}.data"
    if not data.exists():
        # moltemplate always writes system.data by default
        data = wd / "system.data"
    log.info("moltemplate wrote %s (log: %s)", data, log_path)
    return data
