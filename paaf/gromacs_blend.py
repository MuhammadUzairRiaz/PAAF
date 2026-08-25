"""Blend several single-chain GROMACS systems into one box.

The LAMMPS blend merges ``.data`` files; its GROMACS counterpart merges
``.top`` files and packs with ``gmx insert-molecules``. Each component is what
a single-chain PAAF GROMACS export produces::

    gromacs.gro     coordinates of one chain
    gromacs.top     [defaults] [atomtypes] #include "gromacs1.itp"
                    [system] [molecules]
    gromacs1.itp    the [moleculetype] itself

What merging has to get right
-----------------------------
* Every DL_FIELD topology names its molecule ``XYZ``. Two components both
  called XYZ cannot share a file, so each component's moleculetype is renamed
  to its component name, in a rewritten copy of its ``.itp``.
* ``[atomtypes]`` lines are pooled. The same name appearing twice is fine if
  the parameters agree (same force field — the usual case); the same name
  with DIFFERENT parameters is refused loudly, because GROMACS would silently
  use whichever came last and the blend would be typed inconsistently.
* ``[defaults]`` must agree exactly across components — mixing rules cannot
  be merged, only matched.
* ``[molecules]`` lists each component with its copy count, in insertion
  order, which must match the order the coordinates were packed in.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["GromacsBlendComponent", "parse_top", "merge_tops",
           "replicate_gromacs_blend", "BlendTopologyError"]


class BlendTopologyError(RuntimeError):
    """The component topologies cannot be combined."""


@dataclass
class GromacsBlendComponent:
    name: str
    gro_file: Path
    count: int
    #: Explicit topology; found beside the .gro when None.
    top_file: Optional[Path] = None


# ================================================================= parsing
def parse_top(path: Path) -> Dict[str, List[str]]:
    """``{section: [lines]}`` plus ``"#include"`` entries under that key."""
    out: Dict[str, List[str]] = {"#include": []}
    current: Optional[str] = None
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("#include"):
            out["#include"].append(line.split(None, 1)[1].strip().strip('"'))
            continue
        m = re.match(r"\[\s*(\S+)\s*\]", line)
        if m:
            current = m.group(1)
            out.setdefault(current, [])
            continue
        if current and line and not line.startswith(";"):
            out[current].append(line)
    return out


def _rename_moleculetype(itp_text: str, new_name: str) -> str:
    """A copy of an .itp with its [moleculetype] name replaced."""
    lines = itp_text.splitlines()
    out: List[str] = []
    in_moltype = False
    for line in lines:
        stripped = line.strip()
        m = re.match(r"\[\s*(\S+)\s*\]", stripped)
        if m:
            in_moltype = m.group(1) == "moleculetype"
            out.append(line)
            continue
        if in_moltype and stripped and not stripped.startswith(";"):
            parts = stripped.split()
            out.append(f"{new_name}     {parts[1] if len(parts) > 1 else 3}")
            in_moltype = False        # only the one name line
            continue
        out.append(line)
    return "\n".join(out) + "\n"


# ================================================================== merging
def merge_tops(components: List[GromacsBlendComponent], out_dir: Path,
               title: str = "PAAF polymer blend") -> Path:
    """Write ``packed_blend.top`` (+ per-component .itp copies) and return it.

    Raises :class:`BlendTopologyError` rather than writing anything that
    GROMACS would accept and simulate wrongly.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    defaults: Optional[List[str]] = None
    atomtypes: Dict[str, str] = {}          # name -> full line
    includes: List[str] = []

    for comp in components:
        top = comp.top_file or comp.gro_file.parent / "gromacs.top"
        if not Path(top).exists():
            raise BlendTopologyError(
                f"{comp.name}: no topology found at {top}. Each component "
                f"needs the .top from its own single-chain GROMACS export.")
        sections = parse_top(Path(top))

        d = sections.get("defaults", [])
        if defaults is None:
            defaults = d
        elif [l.split() for l in d] != [l.split() for l in defaults]:
            raise BlendTopologyError(
                f"{comp.name}: its [defaults] (comb-rule / fudge factors) "
                f"differ from the first component's. Mixing rules cannot be "
                f"merged — all components must come from the same force "
                f"field family.")

        for line in sections.get("atomtypes", []):
            name = line.split()[0]
            if name in atomtypes:
                if atomtypes[name].split() != line.split():
                    raise BlendTopologyError(
                        f"Atom type '{name}' has different parameters in "
                        f"{comp.name} than in an earlier component. GROMACS "
                        f"would silently keep one of the two, so the blend "
                        f"is refused instead.\n"
                        f"  earlier: {atomtypes[name]}\n"
                        f"  {comp.name}: {line}")
            else:
                atomtypes[name] = line

        # The component's own .itp, renamed after the component so two
        # DL_FIELD outputs (both "XYZ") can coexist in one file.
        wrote_itp = False
        for inc in sections.get("#include", []):
            src = Path(top).parent / inc
            if not src.exists():
                raise BlendTopologyError(
                    f"{comp.name}: {top} includes '{inc}' but it is not "
                    f"there. Copy the whole export folder, not the .top "
                    f"alone.")
            dst = out_dir / f"{comp.name}.itp"
            dst.write_text(_rename_moleculetype(
                src.read_text(errors="replace"), comp.name))
            includes.append(dst.name)
            wrote_itp = True
        if not wrote_itp:
            # Two different files answer to "the .top" in a PAAF project, and
            # only one of them can be blended. The project root holds a
            # SKELETON — [moleculetype] inline, atom type "C", charge 0.0,
            # mass 0.0, force field left as a commented-out include — written
            # as a starting point for hand editing. The typed topology that
            # DL_FIELD produced lives in dlf_output1/gromacs.top with its
            # gromacs1.itp. Naming the right file beats describing the shape.
            hint = Path(top).parent / "dlf_output1" / "gromacs.top"
            if "atomtypes" not in sections and "moleculetype" in sections:
                raise BlendTopologyError(
                    f"{comp.name}: {Path(top).name} is the skeleton topology "
                    f"(placeholder types, no force field) that PAAF writes "
                    f"for hand editing — it cannot be simulated or blended."
                    + (f"\n\nUse the typed one instead:\n    {hint}"
                       if hint.exists() else
                       "\n\nUse the gromacs.top from the component's "
                       "dlf_output1/ folder instead."))
            raise BlendTopologyError(
                f"{comp.name}: {top} has no #include for its moleculetype "
                f"— PAAF only knows how to merge DL_FIELD-shaped topologies "
                f"(.top + .itp)."
                + (f" Found one at:\n    {hint}" if hint.exists() else ""))

    lines = [f"; {title} — merged by PAAF from "
             f"{len(components)} component(s)", "",
             "[ defaults ]"]
    lines += defaults or []
    lines += ["", "[ atomtypes ]"]
    lines += [atomtypes[k] for k in atomtypes]
    lines += [""]
    lines += [f'#include "{inc}"' for inc in includes]
    lines += ["", "[ system ]", title, "", "[ molecules ]"]
    lines += [f"{c.name}    {int(c.count)}" for c in components]

    out_top = out_dir / "packed_blend.top"
    out_top.write_text("\n".join(lines) + "\n")
    log.info("Wrote merged GROMACS topology -> %s", out_top)
    return out_top


# ================================================================== packing
def _find_gmx() -> Optional[str]:
    from .gromacs_packer import _find_gmx as _f
    return _f()


def replicate_gromacs_blend(
    components: List[GromacsBlendComponent],
    box_edges_ang: Tuple[float, float, float],
    out_dir: Path,
    seed: int = -1,
    try_count: int = 100_000,
) -> Tuple[Path, Path]:
    """Pack all components into one box; return (packed .gro, merged .top).

    Coordinates via sequential ``gmx insert-molecules`` — the first call
    creates the box, each later call inserts into the previous result — so
    the [molecules] order in the merged topology matches insertion order by
    construction.
    """
    gmx = _find_gmx()
    if gmx is None:
        raise RuntimeError("gmx not found — the GROMACS blend needs GROMACS "
                           "installed (gmx or gmx_mpi on PATH).")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_top = merge_tops(components, out_dir)       # fail before any packing

    a, b, c = (e / 10.0 for e in box_edges_ang)     # Å -> nm
    current: Optional[Path] = None
    for i, comp in enumerate(components):
        target = out_dir / f"_blend_stage{i}.gro"
        cmd = [gmx, "insert-molecules",
               "-ci", str(comp.gro_file),
               "-nmol", str(int(comp.count)),
               "-try", str(try_count),
               "-o", str(target)]
        if current is None:
            cmd += ["-box", f"{a}", f"{b}", f"{c}"]
        else:
            cmd += ["-f", str(current)]
        if seed and seed > 0:
            cmd += ["-seed", str(seed + i)]
        r = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True)
        if r.returncode != 0 or not target.exists():
            raise RuntimeError(
                f"gmx insert-molecules failed on component "
                f"'{comp.name}' (rc={r.returncode}):\n{r.stderr[-800:]}")
        # Verify the count actually landed; a short insertion would desync
        # coordinates from the [molecules] section, which grompp reports as
        # a bare atom-count mismatch with no cause attached.
        added = re.search(r"Added\s+(\d+)\s+molecules", r.stderr or "")
        if added and int(added.group(1)) != int(comp.count):
            raise RuntimeError(
                f"Only {added.group(1)} of {comp.count} copies of "
                f"'{comp.name}' fitted in the box. Enlarge the box or lower "
                f"the counts; a partially packed blend is not written.")
        current = target

    out_gro = out_dir / "packed_blend.gro"
    shutil.copy2(current, out_gro)
    for i in range(len(components)):
        (out_dir / f"_blend_stage{i}.gro").unlink(missing_ok=True)
    log.info("Wrote packed GROMACS blend -> %s (+ %s)", out_gro, out_top.name)
    return out_gro, out_top
