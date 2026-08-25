"""Relax each blend component before it is packed.

Why this exists
---------------
Packing places whole rigid chains. Whatever geometry each chain has when it
goes in is the geometry it keeps: packmol only translates and rotates, it
never changes internal coordinates. So a chain that arrives strained — and a
freshly typed chain from DL_FIELD or moltemplate usually is, because its
coordinates came from a builder rather than from a force field — carries that
strain into every one of its copies, and into the blend.

Minimising once per component is cheap (one chain, a few hundred atoms) and
fixes it for every copy at a stroke. This is the ``prepare_component`` step of
the reference blend script.

What it does, per component
---------------------------
1. Copy the component's ``.data`` into a private working directory.
2. Build a minimisation input from the component's own ``.in`` file, so the
   real force-field styles and coefficients are used — the same reasoning as
   :mod:`paaf.cell.relax`: PAAF does not guess styles it can read.
3. Run ``minimize`` with conjugate gradient.
4. Dump the relaxed coordinates **sorted by atom id**, and read the last
   frame back.

The sort is not cosmetic. The replication step maps template atom *i* to
coordinate *i*; an unsorted dump would pair every atom with someone else's
position and produce a structure that looks plausible and is wrong.

When it cannot run
------------------
No LAMMPS, or no ``.in`` template for a component, means no minimisation. The
component is then packed with its input geometry, exactly as before, and the
result says so. It is never silently reported as relaxed.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["MinimiseSettings", "ComponentMinimisation", "minimise_component",
           "write_component_min_input", "read_last_dump_frame",
           "find_component_input"]


# =====================================================================
@dataclass
class MinimiseSettings:
    """How hard to minimise a single component."""
    enabled: bool = True
    etol: float = 1.0e-6
    ftol: float = 1.0e-8
    maxiter: int = 10_000
    maxeval: int = 100_000
    lammps_exe: str = ""            # blank -> auto-detect
    timeout_s: int = 1800


@dataclass
class ComponentMinimisation:
    """What happened to one component."""
    name: str
    ran: bool = False
    work_dir: Optional[Path] = None
    input_script: Optional[Path] = None
    dump_file: Optional[Path] = None
    n_atoms: int = 0
    max_bond_before: float = 0.0
    max_bond_after: float = 0.0
    message: str = ""

    def summary(self) -> str:
        if not self.ran:
            return f"{self.name}: not minimised — {self.message}"
        return (f"{self.name}: minimised, {self.n_atoms} atoms, "
                f"longest bond {self.max_bond_before:.3f} -> "
                f"{self.max_bond_after:.3f} Å")


# =====================================================================
def find_component_input(data_file: Path) -> Optional[Path]:
    """The LAMMPS ``.in`` that goes with a component's ``.data``.

    DL_FIELD writes ``lammps1.data`` beside ``lammps1.in``; moltemplate writes
    ``<name>.data`` beside ``<name>.in``. Both are checked, then any single
    ``.in`` in the same directory — but only if there is exactly one, because
    picking arbitrarily between several would mean minimising with a force
    field the user did not choose.
    """
    data_file = Path(data_file)
    folder = data_file.parent
    stem = data_file.stem

    for candidate in (folder / f"{stem}.in", folder / f"{stem}.in.init"):
        if candidate.is_file():
            return candidate

    ins = [p for p in sorted(folder.glob("*.in")) if p.is_file()]
    return ins[0] if len(ins) == 1 else None


def write_component_min_input(template_in: Path, out_path: Path,
                              data_name: str, dump_name: str,
                              settings: MinimiseSettings) -> Path:
    """Turn a component's own ``.in`` into a minimisation script.

    The template is reused rather than rewritten so the real styles and
    coefficients come from the file that was generated alongside the data.
    Three kinds of line are dropped: the original ``read_data`` (replaced with
    our working copy), and any ``thermo``/``thermo_style``/``run 0`` the
    template happens to carry, which would otherwise fight the ones added
    below.
    """
    lines: List[str] = []
    for raw in Path(template_in).read_text().splitlines(True):
        stripped = raw.strip()
        if stripped.startswith("read_data"):
            lines.append(f"read_data {data_name}\n")
        elif (stripped.startswith("thermo_style")
              or re.match(r"^thermo\s+\d+", stripped)
              or re.match(r"^run\s+0\b", stripped)
              or stripped.startswith("minimize")
              or stripped.startswith("write_data")
              or stripped.startswith("write_dump")):
            continue
        else:
            lines.append(raw)

    lines += [
        "\n# ---- added by PAAF: relax this component before packing ----\n",
        "thermo_style    custom step pe etotal press\n",
        "thermo          100\n",
        "min_style       cg\n",
        f"minimize        {settings.etol} {settings.ftol} "
        f"{settings.maxiter} {settings.maxeval}\n",
        # Sorted by id: replication pairs template atom i with coordinate i,
        # so an unsorted dump would silently scramble the whole chain.
        f"dump            dmin all custom 1 {dump_name} id type x y z\n",
        "dump_modify     dmin sort id\n",
        "run             0\n",
        "undump          dmin\n",
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines))
    return out_path


def read_last_dump_frame(path: Path) -> List[Tuple[int, int, float, float, float]]:
    """``(id, type, x, y, z)`` for the final frame, sorted by id.

    Column positions are read from the ``ITEM: ATOMS`` header rather than
    assumed, so a dump written with a different column order still parses.
    """
    lines = Path(path).read_text().splitlines()
    frames: List[List[Tuple[int, int, float, float, float]]] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        try:
            n_atoms = int(lines[i + 3])
        except (IndexError, ValueError):
            break
        head = i + 8
        if head >= len(lines) or not lines[head].startswith("ITEM: ATOMS"):
            i += 1
            continue
        cols = {name: k for k, name in enumerate(lines[head].split()[2:])}
        needed = ("id", "type", "x", "y", "z")
        if any(c not in cols for c in needed):
            raise RuntimeError(
                f"{Path(path).name}: dump is missing one of {needed}; "
                f"found {sorted(cols)}")
        frame = []
        for line in lines[head + 1: head + 1 + n_atoms]:
            parts = line.split()
            if len(parts) < len(cols):
                break
            frame.append((int(parts[cols["id"]]), int(parts[cols["type"]]),
                          float(parts[cols["x"]]), float(parts[cols["y"]]),
                          float(parts[cols["z"]])))
        if len(frame) == n_atoms:
            frame.sort(key=lambda a: a[0])
            frames.append(frame)
        i = head + 1 + n_atoms

    if not frames:
        raise RuntimeError(f"No complete frame found in {path}")
    return frames[-1]


# =====================================================================
def _longest_bond(data_file: Path,
                  coords: Optional[Dict[int, Tuple[float, float, float]]] = None
                  ) -> float:
    """Longest bond in a data file, optionally with replacement coordinates."""
    from .lammps_replicator import parse_lammps_data, _clean

    _header, sections = parse_lammps_data(Path(data_file))
    if coords is None:
        coords = {}
        for line in _clean(sections.get("Atoms")):
            p = line.split()
            if len(p) >= 7:
                coords[int(p[0])] = (float(p[4]), float(p[5]), float(p[6]))
    worst = 0.0
    for line in _clean(sections.get("Bonds")):
        p = line.split()
        if len(p) < 4:
            continue
        a, b = int(p[2]), int(p[3])
        if a not in coords or b not in coords:
            continue
        d = sum((coords[a][k] - coords[b][k]) ** 2 for k in range(3)) ** 0.5
        worst = max(worst, d)
    return worst


def minimise_component(name: str, data_file: Path, work_dir: Path,
                       settings: Optional[MinimiseSettings] = None,
                       input_file: Optional[Path] = None,
                       progress: Optional[Callable[[str], None]] = None
                       ) -> Tuple[ComponentMinimisation,
                                  Optional[List[Tuple[int, int, float, float, float]]]]:
    """Relax one component. Returns ``(record, atoms or None)``.

    ``atoms`` is ``None`` whenever minimisation did not run or did not
    produce a usable frame, and the caller should then pack the component's
    original geometry.
    """
    from .cell.relax import find_lammps

    emit = progress or (lambda _m: None)
    settings = settings or MinimiseSettings()
    data_file = Path(data_file)
    work = Path(work_dir) / name
    rec = ComponentMinimisation(name=name, work_dir=work)

    if not settings.enabled:
        rec.message = "minimisation switched off"
        return rec, None

    template = Path(input_file) if input_file else find_component_input(data_file)
    if template is None or not template.is_file():
        rec.message = (
            f"no LAMMPS .in template beside {data_file.name}, so the force "
            f"field is unknown and PAAF will not guess it")
        emit(f"  {name}: {rec.message}")
        return rec, None

    exe = find_lammps(settings.lammps_exe)
    if exe is None:
        rec.message = "no LAMMPS executable found"
        emit(f"  {name}: {rec.message}")
        return rec, None

    work.mkdir(parents=True, exist_ok=True)
    working_data = work / "component.data"
    shutil.copy2(data_file, working_data)
    try:
        shutil.copy2(template, work / template.name)
    except shutil.SameFileError:
        pass
    # Anything the template includes must come along too.
    for extra in template.parent.glob("*.in.*"):
        try:
            shutil.copy2(extra, work / extra.name)
        except (shutil.SameFileError, OSError):
            pass

    rec.max_bond_before = _longest_bond(working_data)
    dump_name = "minimised.lammpstrj"
    rec.input_script = write_component_min_input(
        template, work / "minimise_component.in", working_data.name,
        dump_name, settings)

    emit(f"  {name}: minimising the single chain …")
    try:
        proc = subprocess.run([str(exe), "-in", rec.input_script.name],
                              cwd=str(work), capture_output=True, text=True,
                              timeout=settings.timeout_s)
    except subprocess.TimeoutExpired:
        rec.message = f"LAMMPS exceeded {settings.timeout_s} s"
        return rec, None
    except Exception as exc:
        rec.message = f"LAMMPS could not be launched: {exc}"
        return rec, None

    (work / "minimise.log").write_text((proc.stdout or "") + "\n"
                                       + (proc.stderr or ""))
    if proc.returncode != 0:
        tail = "\n      ".join(
            ((proc.stderr or proc.stdout or "").strip().splitlines() or [""])[-6:])
        rec.message = (f"LAMMPS exited {proc.returncode}; packing the "
                       f"unrelaxed geometry instead. Last output:\n      {tail}")
        emit(f"  {name}: minimisation failed (rc={proc.returncode})")
        return rec, None

    dump = work / dump_name
    try:
        atoms = read_last_dump_frame(dump)
    except Exception as exc:
        rec.message = f"could not read {dump.name}: {exc}"
        return rec, None

    rec.dump_file = dump
    rec.n_atoms = len(atoms)
    rec.max_bond_after = _longest_bond(
        working_data, {a[0]: (a[2], a[3], a[4]) for a in atoms})
    rec.ran = True
    emit(f"  {rec.summary()}")
    log.info("%s", rec.summary())
    return rec, atoms
