"""Write a grown amorphous cell to disk, typed where that is possible.

Three levels of output, and the distinction between them matters
-----------------------------------------------------------------
1. **The bead cell** — ``cell_beads.xyz`` and ``cell_beads.data``. One site
   per skeletal atom, carrying that atom's share of the repeat-unit mass, with
   the bonds along each chain. The ``.data`` is a valid coarse-grained LAMMPS
   input: masses, bonds, a periodic box. It is exactly what was built and it
   claims nothing more.

2. **The atomistic cell** — ``cell_atomistic.xyz`` / ``.mol2``. Produced by
   :mod:`paaf.cell.backmap`, which dresses each backbone atom with its real
   substituents and hydrogens without moving anything growth decided.

3. **The typed cell** — ``cell.data``, with atom types, charges and the full
   bonded topology, routed by force field:

   ============================  =========================================
   force-field ``kind``          route
   ============================  =========================================
   ``dlfield``                   DL_FIELD -> LAMMPS ``.data`` directly
   ``moltemplate_native``        PAAF types the atoms, writes ``.lt``
                                 files, then runs ``moltemplate.sh``
   ``gaff`` / ``trappe``         whichever of the two the registry entry's
                                 ``atom_typer`` implies
   ============================  =========================================

The rule this module follows without exception: **if typing did not happen,
the result says so.** A ``.data`` file that silently contains default types
is worse than no file, because it runs.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..logging_utils import get_logger
from ..structure import Atom, Molecule

log = get_logger(__name__)

__all__ = ["CellExport", "export_cell", "write_bead_lammps_data",
           "write_cell_lt"]


# =====================================================================
@dataclass
class CellExport:
    """What was written, and what was not."""
    folder: Path
    bead_xyz: Optional[Path] = None
    bead_data: Optional[Path] = None
    atomistic_xyz: Optional[Path] = None
    atomistic_mol2: Optional[Path] = None
    typed_data: Optional[Path] = None
    #: The LAMMPS input beside the typed data — styles, coefficients, the
    #: commands to run it. A .data file alone cannot be simulated: the pair
    #: and bond styles live in the .in, and DL_FIELD writes it two folders
    #: down in _typing/dlf_output1/ where nobody would look for it.
    typed_input: Optional[Path] = None
    #: GROMACS pair: coordinates + topology, produced by asking DL_FIELD a
    #: second time with the gromacs output engine. Same typing methodology,
    #: same box, different file format.
    typed_gro: Optional[Path] = None
    typed_top: Optional[Path] = None
    lt_files: List[Path] = field(default_factory=list)
    n_beads: int = 0
    n_atoms: int = 0
    route: str = ""
    typed: bool = False
    relax: Optional[object] = None       # paaf.cell.relax.RelaxResult
    messages: List[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = [f"{self.n_beads} beads"]
        if self.n_atoms:
            bits.append(f"{self.n_atoms} atoms")
        state = (f"typed via {self.route}" if self.typed
                 else "NOT typed (coarse-grained output only)")
        return f"{', '.join(bits)}, {state}, in {self.folder}"


# =====================================================================
# Coarse-grained LAMMPS data
# =====================================================================
def write_bead_lammps_data(result, specs: Sequence, path: Path,
                           title: str = "PAAF amorphous cell (beads)") -> Path:
    """Write the bead cell as a coarse-grained LAMMPS ``data`` file.

    ``atom_style molecular`` — beads carry no charge, so ``full`` would only
    add a column of zeros and imply a charge model that does not exist here.

    One atom type per species, one bond type per species. Masses are the
    repeat-unit mass divided by the skeletal atoms per unit, which is what
    makes the cell's density come out at the requested value.

    No pair coefficients are written. Choosing them is a modelling decision
    (which coarse-grained potential, fitted to what) that this module has no
    basis to make for you, and writing a guess would be worse than leaving the
    line out.
    """
    from .grow import backbone_atoms_per_unit, repeat_unit_mass

    dims = np.asarray(result.box.bounding_box(), dtype=float)
    atoms = result.molecule.atoms
    bonds = result.molecule.bonds

    # Species index per atom, following the order chains were grown.
    type_of: List[int] = []
    mol_of: List[int] = []
    masses: List[float] = []
    labels: List[str] = []
    cursor = 0
    chain_id = 0
    for si, sp in enumerate(specs):
        bpu = (sp.backbone_atoms if sp.backbone_atoms
               else backbone_atoms_per_unit(sp.repeat_unit)) or 1
        unit_mass = sp.mass_amu if sp.mass_amu else repeat_unit_mass(sp.repeat_unit)
        masses.append(float(unit_mass) / bpu)
        labels.append(sp.name or sp.repeat_unit or f"species{si + 1}")
        per_chain = int(sp.degree_of_polymerisation) * bpu
        for _ in range(int(sp.n_chains)):
            chain_id += 1
            for _k in range(per_chain):
                type_of.append(si + 1)
                mol_of.append(chain_id)
            cursor += per_chain
    if cursor != len(atoms):
        raise ValueError(
            f"Spec list accounts for {cursor} beads but the cell has "
            f"{len(atoms)}. The specs must be the ones that were grown.")

    n_types = len(specs)
    lines: List[str] = [title, ""]
    lines.append(f"{len(atoms)} atoms")
    lines.append(f"{len(bonds)} bonds")
    lines.append("")
    lines.append(f"{n_types} atom types")
    lines.append(f"{max(n_types, 1)} bond types")
    lines.append("")
    lines.append(f"0.0 {dims[0]:.6f} xlo xhi")
    lines.append(f"0.0 {dims[1]:.6f} ylo yhi")
    lines.append(f"0.0 {dims[2]:.6f} zlo zhi")
    lines.append("")
    lines.append("Masses")
    lines.append("")
    for i, (m, lab) in enumerate(zip(masses, labels), start=1):
        lines.append(f"{i} {m:.5f}  # {lab} skeletal bead")
    lines.append("")
    lines.append("Atoms # molecular")
    lines.append("")
    for k, a in enumerate(atoms):
        x, y, z = a.xyz
        lines.append(f"{k + 1} {mol_of[k]} {type_of[k]} "
                     f"{x:.6f} {y:.6f} {z:.6f}")
    lines.append("")
    lines.append("Bonds")
    lines.append("")
    for b, (i, j, _o) in enumerate(bonds, start=1):
        lines.append(f"{b} {type_of[i]} {i + 1} {j + 1}")
    lines.append("")

    path = Path(path)
    path.write_text("\n".join(lines))
    log.info("Wrote coarse-grained LAMMPS data %s (%d beads, %d bonds)",
             path, len(atoms), len(bonds))
    return path


# =====================================================================
# Moltemplate route
# =====================================================================
def write_cell_lt(molecule: Molecule, atoms_per_chain: Sequence[int],
                  out_dir: Path, ff, box_dims: Sequence[float],
                  name: str = "cell") -> List[Path]:
    """Write one ``.lt`` object per chain plus a ``system.lt``.

    Every chain in a grown cell has its own conformation, so the usual
    Moltemplate idiom — define one molecule, then ``new Mol[N].move(...)`` —
    does not apply. Each chain is emitted as its own object with explicit
    coordinates and instantiated once, in place.
    """
    from ..lt_writer import _atom_line, _bond_line

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    # Split the flat molecule back into chains.
    starts: List[int] = []
    acc = 0
    for n in atoms_per_chain:
        starts.append(acc)
        acc += int(n)

    adj: Dict[int, List[int]] = {}
    for i, j, _o in molecule.bonds:
        adj.setdefault(i, []).append(j)

    imports = []
    for ci, (s, n) in enumerate(zip(starts, atoms_per_chain)):
        obj = f"{name}_chain{ci + 1}"
        sub = molecule.atoms[s:s + int(n)]
        lines = [f'import "{ff.lt_include}"'] if ff.lt_include else []
        inherit = f" inherits {ff.inherit}" if ff.inherit else ""
        lines.append(f"{obj}{inherit} {{")
        lines.append('  write("Data Atoms") {')
        for a in sub:
            lines.append("  " + _atom_line(a, obj))
        lines.append("  }")
        lines.append('  write("Data Bond List") {')
        bidx = 0
        for i, j, _o in molecule.bonds:
            if s <= i < s + int(n) and s <= j < s + int(n):
                lines.append("  " + _bond_line(bidx, i, j, molecule.atoms))
                bidx += 1
        lines.append("  }")
        lines.append("}")
        p = out_dir / f"{obj}.lt"
        p.write_text("\n".join(lines) + "\n")
        written.append(p)
        imports.append((obj, p.name))

    sys_lines: List[str] = []
    for obj, fname in imports:
        sys_lines.append(f'import "{fname}"')
    sys_lines.append("")
    for k, (obj, _f) in enumerate(imports, start=1):
        sys_lines.append(f"chain{k} = new {obj}")
    sys_lines.append("")
    sys_lines.append('write_once("Data Boundary") {')
    lx, ly, lz = box_dims
    sys_lines.append(f"  0.0 {lx:.6f} xlo xhi")
    sys_lines.append(f"  0.0 {ly:.6f} ylo yhi")
    sys_lines.append(f"  0.0 {lz:.6f} zlo zhi")
    sys_lines.append("}")
    sp = out_dir / f"{name}_system.lt"
    sp.write_text("\n".join(sys_lines) + "\n")
    written.append(sp)
    log.info("Wrote %d Moltemplate files under %s", len(written), out_dir)
    return written


def _run_moltemplate(system_lt: Path, work_dir: Path,
                     emit: Callable[[str], None]) -> Optional[Path]:
    exe = shutil.which("moltemplate.sh")
    if exe is None:
        emit("    moltemplate.sh not found on PATH — .lt files written, "
             "not compiled")
        return None
    emit(f"    moltemplate.sh {system_lt.name} …")
    try:
        proc = subprocess.run([exe, system_lt.name], cwd=str(work_dir),
                              capture_output=True, text=True, timeout=1800)
    except Exception as exc:
        emit(f"    moltemplate failed to launch: {exc}")
        return None
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        emit("    moltemplate failed:\n      " + "\n      ".join(tail))
        return None
    hits = sorted(work_dir.glob("*.data"))
    return hits[0] if hits else None


def _dlfield_root(dl_field_dir: Optional[Path]) -> Optional[Path]:
    """The folder CONTAINING dl_field, given whatever the user pointed at.

    Two different things get called "the DL_FIELD directory": the install
    root, which holds the ``dl_field`` executable, and its ``lib/``
    subfolder, which holds the parameter files. The GUI field is labelled
    "DL_FIELD lib dir" and the user correctly enters
    ``.../dl_f_4.13/lib`` — but ``find_dl_field`` looks for
    ``<dir>/dl_field``, so it searched inside lib/ and reported "dl_field
    executable not found" for a perfectly good installation. The pipeline
    passes ``dl_lib.parent`` and works; this path did not.

    Accepting either spelling is better than requiring the user to know
    which one a given page wants.
    """
    if dl_field_dir is None:
        return None
    d = Path(dl_field_dir)
    if (d / "dl_field").exists():
        return d
    if (d.parent / "dl_field").exists():
        return d.parent
    # Neither exists; hand back the parent, which is where the executable
    # belongs, so the error names the sensible location.
    return d.parent if d.name == "lib" else d


def _run_dlfield(structure: Path, work_dir: Path, ff_key: str,
                 dl_field_dir: Optional[Path],
                 emit: Callable[[str], None],
                 output_engine: str = "lammps",
                 box_ang=None) -> Optional[Path]:
    from ..dlfield_runner import run_dlfield

    dl_field_dir = _dlfield_root(dl_field_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    emit(f"    dl_field ({ff_key}, {output_engine}) on {structure.name} …")
    # box_ang MUST be the cell's real box. It was never passed here, so
    # dl_field fell back to its 250 A default — and a wrapped periodic cell
    # in the wrong box has every boundary-crossing bond torn apart: measured,
    # 198 bonds read as >10 A long, each torn end a 2- or 3-coordinate
    # carbon, and typing stopped on "unsaturated C atom" at a methylene that
    # was geometrically perfect under the true box. Every distance and angle
    # in the cell had been verified right; the box handed to the typer was
    # the one number nobody checked.
    result = run_dlfield(structure_path=structure, ff_key=ff_key,
                         work_dir=work_dir, dl_field_dir=dl_field_dir,
                         output_engine=output_engine,
                         box_ang=tuple(box_ang) if box_ang is not None
                         else None)
    # DLFieldResult's field is ``return_code`` — NOT subprocess's
    # ``returncode``. The old getattr(..., "returncode", 1) never found the
    # attribute, so its default of 1 declared every SUCCESSFUL run a failure
    # (rc=?). It could only bite on success, so it hid behind the earlier
    # box bug until the first clean run.
    rc = getattr(result, "return_code", getattr(result, "returncode", 1))
    if rc != 0:
        emit(f"    dl_field failed (rc={rc})")
        return None
    # What counts as "the output" depends on which engine was asked for:
    # the secondary-output slot in the control file writes lammps*.data OR
    # gromacs .gro/.top — never both in one run.
    if output_engine == "gromacs":
        hits = sorted((work_dir / "dlf_output1").glob("*.gro"))
        if not hits:
            hits = sorted(work_dir.rglob("*.gro"))
        if not hits:
            emit("    dl_field produced no .gro")
            return None
        return hits[0]
    hits = sorted((work_dir / "dlf_output1").glob("lammps*.data"))
    if not hits:
        hits = sorted(work_dir.glob("lammps*.data"))
    if not hits:
        emit("    dl_field produced no lammps*.data")
        return None
    return hits[0]




def data_file_charges(data_path: Path) -> Optional[Tuple[int, float, float]]:
    """``(n_atoms, total_charge, max_abs_charge)`` from a LAMMPS data file.

    ``atom_style full`` puts the charge in column 4 of the Atoms section. A
    cell whose charges are all zero runs perfectly happily and gives silently
    wrong electrostatics, so it is worth checking rather than assuming.

    Returns ``None`` when the file has no ``Atoms`` section to read.
    """
    try:
        lines = Path(data_path).read_text().splitlines()
    except OSError:
        return None
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Atoms"):
            start = i + 1
            break
    if start is None:
        return None
    total = 0.0
    biggest = 0.0
    n = 0
    for line in lines[start:]:
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            if n:
                break                      # blank line after the section
            continue
        parts = stripped.split()
        if len(parts) < 7:
            break                          # next section header
        try:
            q = float(parts[3])
        except ValueError:
            break
        total += q
        biggest = max(biggest, abs(q))
        n += 1
    return (n, total, biggest) if n else None


def _first_name(work_dir: Path, pattern: str) -> str:
    """Name of the first file matching ``pattern`` in ``work_dir``, or ""."""
    hits = sorted(Path(work_dir).glob(pattern))
    return hits[0].name if hits else ""

def _find_gromacs_inputs(work_dir: Path):
    """A ``.gro`` / ``.top`` pair produced by DL_FIELD, if there is one."""
    work_dir = Path(work_dir)
    if not work_dir.exists():
        return None, None
    gro = sorted(work_dir.rglob("*.gro"))
    top = sorted(work_dir.rglob("*.top"))
    return (gro[0] if gro else None), (top[0] if top else None)


def _find_dlfield_styles_file(work_dir: Path):
    """A LAMMPS input file DL_FIELD wrote alongside its data file.

    Preferred over PAAF's styles table whenever it exists, because it comes
    from the tool that assigned the parameters rather than from a lookup.
    """
    work_dir = Path(work_dir)
    if not work_dir.exists():
        return None
    for pattern in ("*.in.init", "lammps*.in", "*.in"):
        hits = sorted(work_dir.rglob(pattern))
        for h in hits:
            try:
                txt = h.read_text(errors="ignore")
            except OSError:
                continue
            if "pair_style" in txt:
                return h
    return None

# =====================================================================
def export_cell(
    result,
    specs: Sequence,
    out_dir: str | Path,
    *,
    name: str = "cell",
    ff_key: str = "",
    dl_field_dir: Optional[str | Path] = None,
    atomistic: bool = True,
    tacticity: str = "isotactic",
    run_typing: bool = True,
    push_off: bool = True,
    relax: bool = False,
    relax_settings: Optional[object] = None,
    output_formats: str = "lammps",   # "lammps" | "gromacs" | "both"
    progress: Optional[Callable[[str], None]] = None,
) -> CellExport:
    """Write everything the grown cell can honestly support.

    ``specs`` must be the same :class:`paaf.cell.grow.GrowSpec` list that was
    grown, in the same order — that is how each bead is traced back to its
    species and mass.
    """
    emit = progress or (lambda _m: None)
    folder = Path(out_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    exp = CellExport(folder=folder, n_beads=len(result.molecule.atoms))

    # ---- 1. the beads, always -------------------------------------
    emit("  writing the bead cell …")
    exp.bead_xyz = folder / "cell_beads.xyz"
    result.molecule.to_xyz(exp.bead_xyz)
    try:
        exp.bead_data = write_bead_lammps_data(
            result, specs, folder / "cell_beads.data")
    except Exception as exc:
        exp.messages.append(f"Coarse-grained .data not written: {exc}")

    if not atomistic:
        exp.messages.append(
            "Atomistic output was not requested. The .data file is "
            "coarse-grained: one site per skeletal atom, no pair "
            "coefficients — you must supply those.")
        emit(exp.summary())
        return exp

    # ---- 2. back-map to atoms --------------------------------------
    from . import backmap as _bm
    from . import grow as _grow
    if not _bm.available():
        exp.messages.append(
            "RDKit is not installed, so the cell could not be back-mapped to "
            "atoms and no force field could be applied. Coarse-grained output "
            "only. Install with: conda install -c conda-forge rdkit")
        emit(exp.summary())
        return exp

    emit("  back-mapping beads to atoms …")
    try:
        bm = _bm.backmap_cell(result, specs, tacticity=tacticity,
                              push_off=push_off, progress=emit)
    except _grow.BackboneRingError as exc:
        # Not a crash: the bead cell is valid and already written. Only the
        # all-atom reconstruction is impossible, and the reason is specific.
        exp.messages.append(str(exc))
        exp.messages.append(
            "Coarse-grained output only. To get an all-atom cell for this "
            "polymer you need a full-atomistic builder; PAAF's back-mapping "
            "assumes the substituents hanging off the backbone are trees.")
        emit("  back-mapping refused: backbone ring")
        emit(exp.summary())
        return exp
    exp.n_atoms = len(bm.molecule.atoms)
    exp.messages.extend(bm.notes)

    # Wrap into the periodic cell before writing.
    #
    # Back-mapping UNWRAPS each chain so it is continuous across the boundary
    # — a frame built from three points cannot be computed on a chain that
    # jumps a box length. But the exported file has to be wrapped again, and
    # it never was: measured on a 53.8 A cell, the coordinates spanned
    # 78.7 x 99.4 x 87.9 A. DL_FIELD applies the cell from the control file,
    # folds those coordinates back in, and atoms 99 A apart in the file land
    # on top of one another — carbons come out with 5 to 10 neighbours and it
    # stops with "atype = aliphatic". No change of density could fix that,
    # because the atoms were never inside the box to begin with.
    try:
        _dims = np.asarray(result.box.bounding_box(), dtype=float)
        if np.all(_dims > 0):
            _wrapped = 0
            for _a in bm.molecule.atoms:
                _before = _a.xyz.copy()
                _a.xyz = _a.xyz - np.floor(_a.xyz / _dims) * _dims
                if not np.allclose(_before, _a.xyz):
                    _wrapped += 1
            if _wrapped:
                emit(f"  wrapped {_wrapped} atoms into the "
                     f"{_dims[0]:.1f} A cell")
    except Exception as _exc:                        # pragma: no cover
        exp.messages.append(f"Coordinates were not wrapped ({_exc}); the "
                            f"cell may extend beyond its own box.")

    # Soft-core push-off in LAMMPS, before typing.
    #
    # The geometric push-off stalls: measured on this very cell, 1,167
    # non-bonded pairs were still closer than 1.65 A after 150 iterations —
    # hydrogens trapped between pinned backbone atoms have nowhere to go
    # without concerted motion. DL_FIELD bonds anything that close and
    # refuses the cell. A few thousand steps of pair_style soft with
    # nve/limit (the standard Kremer-Grest preparation push-off) resolves
    # them properly, and it needs NO atom types — which is what breaks the
    # circularity between "cannot type until contacts are gone" and
    # "cannot run a force field until typed". Skipped cleanly when LAMMPS
    # is not installed; the cell is then exported as before.
    if run_typing:
        try:
            from .soft_pushoff import soft_pushoff
            soft_pushoff(bm.molecule, _dims, folder / "_pushoff",
                         lammps_exe=getattr(relax_settings, "lammps_exe", "")
                         if relax_settings else "",
                         emit=emit)
        except Exception as _exc:                    # pragma: no cover
            emit(f"  soft push-off skipped ({_exc})")

    exp.atomistic_xyz = folder / "cell_atomistic.xyz"
    bm.molecule.to_xyz(exp.atomistic_xyz)
    try:
        from .. import structure as structure_mod
        exp.atomistic_mol2 = folder / "cell_atomistic.mol2"
        structure_mod.write(bm.molecule, exp.atomistic_mol2)
    except Exception as exc:
        exp.atomistic_mol2 = None
        exp.messages.append(
            f"mol2 not written ({exc}); DL_FIELD perceives bond orders better "
            f"from mol2 than from xyz.")

    if not run_typing:
        exp.messages.append("Force-field typing was switched off.")
        emit(exp.summary())
        return exp
    if not ff_key:
        exp.messages.append(
            "No force field was selected, so nothing was typed.")
        emit(exp.summary())
        return exp

    # ---- 3. type it, by the route the force field implies ----------
    from ..ff_registry import REGISTRY
    ff = REGISTRY.get(ff_key)
    if ff is None:
        exp.messages.append(f"Unknown force field key {ff_key!r}; not typed.")
        emit(exp.summary())
        return exp

    dims = list(np.asarray(result.box.bounding_box(), dtype=float))
    use_dlfield = (ff.kind == "dlfield" or ff.atom_typer == "dlfield_sf")

    if use_dlfield:
        exp.route = f"DL_FIELD ({ff.display_name})"
        # XYZ, not mol2.
        #
        # The mol2 carries bond orders, which is why it was preferred — a
        # typer that can read them does better. DL_FIELD cannot: handed the
        # cell's .mol2 it stopped with "Can't locate any sensible information
        # in config file" and wrote nothing. Every DL_FIELD run elsewhere in
        # PAAF feeds it an .xyz, and those all work; DL_FIELD perceives its
        # own bonds from geometry through DL_F Notation, so the bond orders
        # were never doing anything for this route anyway.
        src = exp.atomistic_xyz or exp.atomistic_mol2
        fmt = str(output_formats or "lammps").lower()
        want_lmp = fmt in ("lammps", "both")
        # GROMACS files are wanted either as a deliverable in their own
        # right, or because the relaxation stage will run gmx.
        want_gmx = (fmt in ("gromacs", "both")
                    or (relax and getattr(relax_settings, "engine", "auto")
                        in ("gromacs", "auto")))
        data = None
        if want_lmp:
            try:
                data = _run_dlfield(src, folder / "_typing", ff_key,
                                    Path(dl_field_dir) if dl_field_dir
                                    else None,
                                    emit, box_ang=result.box.bounding_box())
            except Exception as exc:
                exp.messages.append(f"DL_FIELD typing failed — {exc}")
        if data is not None:
            exp.typed_data = folder / "cell.data"
            shutil.copy2(data, exp.typed_data)
            exp.typed = True
            # ...and the input deck that makes it runnable.
            #
            # cell.data carries masses, charges and topology but NOT the
            # styles or the pair coefficients — those are in the lammps.in
            # DL_FIELD wrote alongside it, down in _typing/dlf_output1/.
            # Copying only the .data left the cell un-runnable and looked
            # like the export had simply forgotten a file.
            try:
                src_in = _find_dlfield_styles_file(data.parent)
                if src_in is not None:
                    exp.typed_input = folder / "cell.in"
                    shutil.copy2(src_in, exp.typed_input)
                    # DL_FIELD's script reads its own file name; ours is
                    # cell.data. Without this, `lmp -in cell.in` dies on
                    # a missing lammps1.data.
                    txt = exp.typed_input.read_text()
                    exp.typed_input.write_text(txt.replace(
                        f"read_data {data.name}", "read_data cell.data"))
                    emit(f"  wrote {exp.typed_input.name} "
                         f"(styles + pair coefficients)")
                    for extra in sorted(data.parent.glob("*.in.*")):
                        shutil.copy2(extra, folder / extra.name)
                else:
                    exp.messages.append(
                        "DL_FIELD wrote no lammps.in, so cell.data has no "
                        "styles or pair coefficients with it and cannot be "
                        "run as-is.")
            except Exception as exc:
                exp.messages.append(f"LAMMPS input not copied — {exc}")
        elif want_lmp:
            exp.messages.append(
                "DL_FIELD did not produce a .data file. Check that dl_field "
                "is installed and the DL_FIELD lib directory is set.")
        # GROMACS files come from DL_FIELD's *secondary*-output slot, which
        # holds one engine at a time — so the same typing run is asked a
        # second time with the gromacs engine. Identical methodology
        # (DL_F Notation, same force field, same box), different format.
        # This run goes AFTER the LAMMPS copy above so that whatever
        # dl_field does to _typing/dlf_output1 cannot cost us cell.data.
        if want_gmx:
            gro = None
            try:
                gro = _run_dlfield(src, folder / "_typing", ff_key,
                                   Path(dl_field_dir) if dl_field_dir
                                   else None,
                                   emit, output_engine="gromacs",
                                   box_ang=result.box.bounding_box())
            except Exception as exc:
                exp.messages.append(
                    f"DL_FIELD GROMACS output not produced ({exc})"
                    + ("; the LAMMPS route is still available."
                       if data is not None else "."))
            if gro is not None:
                exp.typed_gro = folder / "cell.gro"
                shutil.copy2(gro, exp.typed_gro)
                tops = sorted(gro.parent.glob("*.top"))
                if tops:
                    exp.typed_top = folder / "cell.top"
                    shutil.copy2(tops[0], exp.typed_top)
                # The .top #includes per-molecule .itp files by name; they
                # must travel with it or grompp dies on a missing include.
                for itp in sorted(gro.parent.glob("*.itp")):
                    shutil.copy2(itp, folder / itp.name)
                exp.typed = True
                emit(f"  wrote cell.gro"
                     + (" + cell.top (with itp includes)" if tops else "")
                     + " for GROMACS")
                if not tops:
                    exp.messages.append(
                        "DL_FIELD wrote a .gro but no .top; cell.gro has "
                        "coordinates only and cannot be simulated alone.")
    else:
        exp.route = f"Moltemplate ({ff.display_name})"
        try:
            from .. import ff_assigner
            emit("  assigning atom types …")
            ff_assigner.assign(
                bm.molecule, ff,
                dl_lib_dir=Path(dl_field_dir) if dl_field_dir else None)
        except Exception as exc:
            exp.messages.append(
                f"Atom typing failed — {exc}. The .lt files were not written, "
                f"because writing them with default types would produce a "
                f"LAMMPS file that runs and is wrong.")
            emit(exp.summary())
            return exp

        work = folder / "moltemplate"
        try:
            exp.lt_files = write_cell_lt(bm.molecule, bm.atoms_per_chain,
                                         work, ff, dims, name=name)
        except Exception as exc:
            exp.messages.append(f".lt files not written — {exc}")
            emit(exp.summary())
            return exp

        data = _run_moltemplate(exp.lt_files[-1], work, emit)
        if data is not None:
            exp.typed_data = folder / "cell.data"
            shutil.copy2(data, exp.typed_data)
            exp.typed = True
            # Moltemplate splits the deck across *.in.init / *.in.settings /
            # *.in.charges; all three are needed beside the .data.
            try:
                for pattern in ("*.in.init", "*.in.settings", "*.in.charges",
                                "*.in"):
                    for extra in sorted(work.glob(pattern)):
                        shutil.copy2(extra, folder / extra.name)
                        if extra.suffix == ".in" and exp.typed_input is None:
                            exp.typed_input = folder / extra.name
            except Exception as exc:
                exp.messages.append(f"LAMMPS input not copied — {exc}")
        else:
            exp.messages.append(
                "Moltemplate input was written but not compiled. Run it "
                f"yourself with:  cd {work} && moltemplate.sh "
                f"{exp.lt_files[-1].name}")

    if not exp.typed:
        emit(exp.summary())
        return exp

    # ---- 3b. is the typing actually complete? ----------------------
    # A data file with every charge at zero runs perfectly happily and gives
    # silently absent electrostatics. Checking costs nothing.
    if exp.typed_data is not None:
        q = data_file_charges(exp.typed_data)
        if q is not None:
            n_at, total, biggest = q
            if biggest < 1e-9:
                exp.messages.append(
                    f"WARNING: all {n_at} atoms in the typed data file carry "
                    f"zero charge. The cell will run and its electrostatics "
                    f"will be silently absent. On the Moltemplate route this "
                    f"means the .in.charges file was not applied.")
            elif abs(total) > 0.05:
                exp.messages.append(
                    f"WARNING: net charge {total:+.3f} e over {n_at} atoms. A "
                    f"neutral polymer should sum to zero; check the typing.")
            else:
                exp.messages.append(
                    f"Charges: {n_at} atoms, net {total:+.4f} e, largest "
                    f"|q| {biggest:.3f} e.")

    # ---- 4. relax, if asked ----------------------------------------
    if not relax:
        exp.messages.append(
            "The typed cell has NOT been energy-minimised. Constructed cells "
            "always carry strained contacts; minimise, then equilibrate, "
            "before any production run.")
        emit(exp.summary())
        return exp

    from .relax import (
        RelaxSettings, choose_engine, lammps_styles_for, relax_cell,
        relax_cell_gromacs,
    )

    rset = relax_settings or RelaxSettings()
    engine = choose_engine(ff_key, getattr(rset, "engine", "auto"))
    if engine == "gromacs" and not use_dlfield:
        exp.messages.append(
            f"{ff.display_name} is typed by Moltemplate, which produces LAMMPS "
            f"input only. Relaxing with LAMMPS instead of GROMACS.")
        engine = "lammps"

    emit(f"  relaxing with {engine.upper()} …")
    try:
        if engine == "gromacs":
            gro, top = _find_gromacs_inputs(folder / "_typing")
            if gro is None or top is None:
                exp.messages.append(
                    "GROMACS was requested but DL_FIELD produced no .gro/.top "
                    "pair. Re-run typing with the GROMACS output engine, or "
                    "relax with LAMMPS.")
                emit(exp.summary())
                return exp
            exp.relax = relax_cell_gromacs(
                bm.molecule, dims, gro, top, folder / "relax",
                settings=rset, progress=emit)
        else:
            init_f = settings_f = charges_f = ""
            styles = None
            src = ""
            if exp.lt_files:
                # moltemplate names its output after the INPUT .lt file, so a
                # cell built from cell_system.lt yields cell_system.in.init,
                # not system.in.init. Hardcoding the latter silently produced
                # a deck with no force field at all, which LAMMPS then ran and
                # PAAF reported as a successful relaxation.
                work = exp.lt_files[-1].parent
                init_f = _first_name(work, "*.in.init")
                settings_f = _first_name(work, "*.in.settings")
                charges_f = _first_name(work, "*.in.charges")
                for f in (init_f, settings_f, charges_f):
                    if f:
                        try:
                            (folder / "relax").mkdir(parents=True, exist_ok=True)
                            shutil.copy2(work / f, folder / "relax" / f)
                        except Exception:
                            pass
            if not init_f:
                # DL_FIELD route: hunt for a companion input first, and only
                # fall back to PAAF's table if there is none.
                found = _find_dlfield_styles_file(folder / "_typing")
                if found is not None:
                    init_f = found.name
                    src = f"DL_FIELD's own {found.name}"
                    try:
                        shutil.copy2(found, folder / "relax" / found.name)
                    except Exception:
                        pass
                else:
                    # Read the sub-styles out of the data file rather than
                    # look them up: DL_FIELD names each one in column 2 of
                    # every coefficient line, so this is the force field's
                    # own statement of its functional forms.
                    styles = lammps_styles_for(ff_key, exp.typed_data)
                    src = (f"read from {Path(exp.typed_data).name}"
                           if exp.typed_data else "PAAF's styles table")
            exp.relax = relax_cell(
                bm.molecule, dims, exp.typed_data, folder / "relax",
                settings=rset, init_file=init_f, settings_file=settings_f,
                charges_file=charges_f, styles=styles, styles_source=src,
                progress=emit)
        exp.messages.extend(exp.relax.messages)
        if exp.relax.ran:
            # The relaxed structure supersedes the constructed one.
            exp.atomistic_xyz = folder / "cell_atomistic_relaxed.xyz"
            bm.molecule.to_xyz(exp.atomistic_xyz)
            if exp.relax.relaxed_data is not None:
                dest = folder / "cell_relaxed.data"
                shutil.copy2(exp.relax.relaxed_data, dest)
                exp.typed_data = dest
    except Exception as exc:
        exp.messages.append(f"Relaxation failed — {exc}. The unrelaxed cell "
                            f"is still on disk and still valid.")

    emit(exp.summary())
    return exp
