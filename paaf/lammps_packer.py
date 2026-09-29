"""Pack N copies of a chain into a periodic box with LAMMPS instead of packmol.

Why not packmol
---------------
packmol places every copy at once and minimises one penalty function over all
of them. That is robust for small molecules, but for long polymer chains at
melt density it is slow, uses one core, and can spend a very long time without
converging. Its only way out is a time limit, and the Box page then waits with
nothing to show.

What this does instead
----------------------
The same job packmol does — rigid copies of one chain, placed so that no two
atoms of *different* chains are closer than a tolerance — done as a short
LAMMPS run that can use several cores:

1. **Placement (Python).** Each copy gets a uniformly random orientation and,
   by default, a uniformly random centre anywhere in the cell — packmol's
   random start. (``placement="lattice"`` spreads the centres on a shuffled,
   jittered lattice instead.)
2. **Rigid push-off (LAMMPS).** Every chain is one rigid body (``fix rigid
   molecule``), exactly packmol's degrees of freedom: translation and
   rotation only, so each copy keeps the builder's conformation. A purely
   repulsive ``pair_style soft`` reaching a little past the tolerance is
   ramped up, and ``fix viscous`` drains the energy so the chains drift apart
   instead of flying. Pairs within the same chain are excluded, so only
   contacts *between* chains count — packmol's definition of the tolerance.
   Enough on its own at modest density.
3. **Flexible finish (LAMMPS), only if contacts remain.** Rigid coils jam at
   melt density — this is also where packmol stalls. The finish lets atoms
   move a little: every bond and angle is held by a spring at *its own*
   built value (one type per bond, one per angle) and every torsion by a
   spring on its 1-4 distance; a harmonic repulsion pushes hardest on the
   deepest overlaps, ``fix nve/limit`` caps each step at 0.1 Å, and gentle
   300 K Langevin noise shakes jammed pairs loose. It runs in chunks and
   stops at the first one that meets the tolerance.

Unlike packmol this runs with periodic boundaries, so there is no need to
leave a gap at the box faces, and both stages have fixed step budgets: the
run always finishes, and the log reports the closest remaining contact.

Parallel runs
-------------
How LAMMPS can use N cores depends on how it was built, and getting it wrong
is worse than running on one core: ``mpirun -np 4`` on a build without MPI
("MPI STUBS", common for conda and pip installs) starts four *identical*
copies of the whole run, all writing the same files. So the build is probed
first (``lmp -h``): a real MPI build runs under ``mpirun -np N``, a build with
the OPENMP package runs N threads (``-sf omp -pk omp N``), and anything else
runs on one core and says so.
"""
from __future__ import annotations

import math
import os
import random
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["LammpsBuild", "LammpsPackResult", "probe_lammps",
           "launch_command", "lammps_pack", "place_chains"]

#: Soft-potential amplitude (kcal/mol) the rigid stage ramps up to.
_A_RIGID = 30.0
#: Flexible finish: harmonic repulsion k(rc - r)^2, kcal/mol/Å².
#:
#: Not the soft cosine of the rigid stage. Its force is zero when two atoms
#: coincide, so the deepest overlaps get the weakest push, and on 100 PIB
#: chains at 0.9 g/cm3 a few pairs stayed stuck near 0.1 Å through 20,000
#: steps. The harmonic form pushes hardest exactly there. With a 0.1 Å step
#: limit and gentle 300 K Langevin noise to shake jammed pairs loose, the
#: same box reached 2.0 Å in 8,000 steps (chain RMSD 2.9 Å vs. the
#: builder's conformation; without the noise a stiffer push stalled at
#: 1.9 Å and bent the chains twice as much).
_K_FLEX = 50.0
#: Fallback amplitude when the build lacks EXTRA-PAIR (no harmonic/cut).
_A_FLEX = 60.0
#: Step budgets. The rigid stage is short: it spreads chains that start
#: tangled at modest density, but at melt density rigid coils jam (measured
#: on 100 PIB chains at 0.9 g/cm3: closest contact still 0.14 Å after 6000
#: steps) and the flexible finish does the real work. That finish runs in
#: chunks and stops at the first chunk that meets the tolerance, up to a cap,
#: so an easy box ends early and a hard one still ends.
_STEPS_RIGID = 2000
_STEPS_FLEX_CHUNK = 2000
_FLEX_CHUNKS = 10
#: The soft repulsion reaches this far beyond the tolerance. Its force falls
#: to zero AT its cutoff, so with the cutoff equal to the tolerance the last
#: contacts stall a tenth of an Å short of it; reaching further pushes them
#: past it.
_REACH = 0.4
#: A contact this much closer than the tolerance still counts as resolved.
_SLACK = 0.01


# ================================================================ the build
@dataclass(frozen=True)
class LammpsBuild:
    """What a LAMMPS executable can do, from its ``-h`` output."""
    exe: str
    has_mpi: bool
    has_openmp: bool
    packages: frozenset = field(default_factory=frozenset)
    version: str = ""


@lru_cache(maxsize=8)
def probe_lammps(exe: str) -> LammpsBuild:
    """Ask a LAMMPS binary how it was built. Cached per path."""
    try:
        proc = subprocess.run([exe, "-h"], capture_output=True, text=True,
                              timeout=60)
        text = (proc.stdout or "") + (proc.stderr or "")
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("Could not ask %s how it was built (%s); assuming a "
                    "serial build.", exe, exc)
        return LammpsBuild(exe=exe, has_mpi=False, has_openmp=False)

    version = ""
    m = re.search(r"Large-scale Atomic/Molecular Massively Parallel "
                  r"Simulator - (.+)", text)
    if m:
        version = m.group(1).strip()
    # "MPI v1.0: LAMMPS MPI STUBS for LAMMPS version ..." is a serial build.
    # A real one names its library ("MPI v3.1: Open MPI v5.0.3 ...").
    mpi_line = next((l for l in text.splitlines() if l.startswith("MPI v")), "")
    has_mpi = bool(mpi_line) and "STUBS" not in mpi_line.upper()

    packages: set = set()
    grab = False
    for line in text.splitlines():
        if line.strip().startswith("Installed packages"):
            grab = True
            continue
        if grab:
            if not line.strip():
                if packages:
                    break
                continue
            packages.update(line.split())
    return LammpsBuild(exe=exe, has_mpi=has_mpi,
                       has_openmp="OPENMP" in packages,
                       packages=frozenset(packages), version=version)


def launch_command(build: LammpsBuild, nprocs: int
                   ) -> Tuple[List[str], str]:
    """``(command, description)`` for running ``build`` on ``nprocs`` cores.

    The input script is appended by the caller (``-in file``).
    """
    n = max(1, min(int(nprocs or 1), os.cpu_count() or 1))
    if n > 1 and build.has_mpi:
        mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
        if mpirun:
            return [mpirun, "-np", str(n), build.exe], f"mpirun -np {n}"
    if n > 1 and build.has_openmp:
        return ([build.exe, "-sf", "omp", "-pk", "omp", str(n)],
                f"{n} OpenMP threads")
    if n > 1:
        why = ("mpirun was not found" if build.has_mpi
               else "this LAMMPS build has neither MPI nor the OPENMP package")
        return [build.exe], f"1 core ({why})"
    return [build.exe], "1 core"


# ================================================================ placement
def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random rotation matrix (from a random unit quaternion)."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _h_matrix(box: Sequence[float]) -> np.ndarray:
    """Columns are the cell vectors, LAMMPS restricted-triclinic form."""
    lx, ly, lz = box[:3]
    xy, xz, yz = (tuple(box[3:6]) + (0.0, 0.0, 0.0))[:3]
    return np.array([[lx, xy, xz], [0.0, ly, yz], [0.0, 0.0, lz]])


def _lattice(n: int, box: Sequence[float]) -> Tuple[int, int, int]:
    """Sites per axis, in proportion to the edges, with at least ``n`` sites."""
    lx, ly, lz = box[:3]
    s = (lx * ly * lz / max(n, 1)) ** (1.0 / 3.0)
    m = [max(1, round(e / s)) for e in (lx, ly, lz)]
    while m[0] * m[1] * m[2] < n:
        # Split the axis whose cells are currently longest.
        k = max(range(3), key=lambda i: (lx, ly, lz)[i] / m[i])
        m[k] += 1
    return m[0], m[1], m[2]


#: How chain centres are chosen before the push-off.
PLACEMENTS = ("random", "lattice")


def place_chains(chain_xyz: np.ndarray, n: int, box: Sequence[float],
                 seed: int, placement: str = "random") -> np.ndarray:
    """``n`` randomly rotated copies of the chain. ``(n, natoms, 3)``.

    ``placement``:

    * ``"random"`` — every centre uniformly random anywhere in the cell,
      as packmol starts. Overlaps are left to the push-off to remove.
    * ``"lattice"`` — centres on a shuffled lattice with random jitter:
      still random, but evenly spread, so the push-off starts with fewer
      overlaps.

    Orientations are uniformly random either way. Centres are drawn in
    fractional coordinates, so a triclinic cell is covered as evenly as an
    orthogonal one.
    """
    rng = np.random.default_rng(seed)
    local = chain_xyz - chain_xyz.mean(axis=0)
    if placement == "random":
        frac = rng.uniform(0.0, 1.0, size=(n, 3))
    elif placement == "lattice":
        mx, my, mz = _lattice(n, box)
        sites = np.array([(i, j, k) for i in range(mx) for j in range(my)
                          for k in range(mz)], dtype=float)
        rng.shuffle(sites)
        sites = sites[:n]
        frac = (sites + 0.5 + rng.uniform(-0.25, 0.25, size=sites.shape)) \
            / np.array([mx, my, mz], dtype=float)
    else:
        raise ValueError(f"placement must be one of {PLACEMENTS}, "
                         f"not {placement!r}")
    centres = frac @ _h_matrix(box).T
    out = np.empty((n,) + local.shape)
    for c in range(n):
        out[c] = local @ _random_rotation(rng).T + centres[c]
    return out


# ================================================================ the input
@dataclass
class _Chain:
    """The pieces of the single-chain data file the packing run needs."""
    ids: List[int]                    # ascending
    types: List[int]
    xyz: np.ndarray                   # (natoms, 3), same order as ids
    masses: List[str]                 # "type mass" lines
    n_atom_types: int
    bonds: List[Tuple[int, int]]      # indices into ids
    angles: List[Tuple[int, int, int]]
    #: First and last atom of each dihedral, once per pair, excluding pairs
    #: that are already bonded.
    pairs14: List[Tuple[int, int]] = field(default_factory=list)


def _read_chain(data_file: Path) -> _Chain:
    from .lammps_replicator import _clean, _header_count, parse_lammps_data

    header, sections = parse_lammps_data(data_file)
    rows = []
    for line in _clean(sections.get("Atoms")):
        p = line.split("#")[0].split()
        # atom_style full: id mol type q x y z [ix iy iz]
        rows.append((int(p[0]), int(p[2]),
                     float(p[4]), float(p[5]), float(p[6])))
    if not rows:
        raise ValueError(f"{data_file} has no Atoms section to pack.")
    rows.sort()
    ids = [r[0] for r in rows]
    index = {aid: k for k, aid in enumerate(ids)}

    def _terms(section: str, width: int) -> List[tuple]:
        out = []
        for line in _clean(sections.get(section)):
            p = line.split("#")[0].split()
            try:
                out.append(tuple(index[int(a)] for a in p[2:2 + width]))
            except (KeyError, ValueError, IndexError):
                continue
        return out

    masses = [" ".join(l.split("#")[0].split()[:2])
              for l in _clean(sections.get("Masses"))]
    n_types = _header_count(header, "atom types") or max(r[1] for r in rows)
    bonds = _terms("Bonds", 2)
    bonded = {frozenset(b) for b in bonds}
    pairs14: List[Tuple[int, int]] = []
    for d in _terms("Dihedrals", 4):
        key = frozenset((d[0], d[3]))
        if len(key) == 2 and key not in bonded:
            bonded.add(key)
            pairs14.append((d[0], d[3]))
    return _Chain(ids=ids, types=[r[1] for r in rows],
                  xyz=np.array([r[2:] for r in rows], dtype=float),
                  masses=masses, n_atom_types=n_types,
                  bonds=bonds, angles=_terms("Angles", 3), pairs14=pairs14)


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a - b, c - b
    cosang = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosang))))


def _write_pack_data(path: Path, chain: _Chain, placed: np.ndarray,
                     box: Sequence[float]) -> None:
    """All copies, one bond type per chain bond and one angle type per angle.

    Per-term types are what let the flexible finish hold every bond and
    angle at the value the builder gave it, instead of a per-class average
    that would nudge every one of them.

    Torsions are held too, by a weaker spring on each dihedral's 1-4
    distance, written as extra bonds. Without it the chains keep their bond
    lengths and angles but rotate freely about their bonds and come out of
    the flexible finish with a different shape (measured: 5 Å RMSD on PIB).
    """
    n, natoms = placed.shape[0], placed.shape[1]
    springs = [(i, j, 300.0) for i, j in chain.bonds] \
        + [(i, j, 30.0) for i, j in chain.pairs14]
    nb, na = len(springs), len(chain.angles)
    lx, ly, lz = box[:3]
    xy, xz, yz = (tuple(box[3:6]) + (0.0, 0.0, 0.0))[:3]
    x = chain.xyz

    out = ["PAAF packing input (LAMMPS soft packer)", "",
           f"{n * natoms} atoms", f"{chain.n_atom_types} atom types"]
    if nb:
        out += [f"{n * nb} bonds", f"{nb} bond types"]
    if na:
        out += [f"{n * na} angles", f"{na} angle types"]
    out += ["", f"0.0 {lx:.6f} xlo xhi", f"0.0 {ly:.6f} ylo yhi",
            f"0.0 {lz:.6f} zlo zhi"]
    if any(abs(t) > 1e-9 for t in (xy, xz, yz)):
        out.append(f"{xy:.6f} {xz:.6f} {yz:.6f} xy xz yz")
    out += ["", "Masses", ""] + chain.masses
    if nb:
        out += ["", "Bond Coeffs", ""]
        out += [f"{k + 1} {kb} {np.linalg.norm(x[i] - x[j]):.6f}"
                for k, (i, j, kb) in enumerate(springs)]
    if na:
        out += ["", "Angle Coeffs", ""]
        out += [f"{k + 1} 60.0 {_angle_deg(x[i], x[j], x[l]):.6f}"
                for k, (i, j, l) in enumerate(chain.angles)]

    out += ["", "Atoms # full", ""]
    for c in range(n):
        base = c * natoms
        for k in range(natoms):
            px, py, pz = placed[c, k]
            out.append(f"{base + k + 1} {c + 1} {chain.types[k]} 0.0 "
                       f"{px:.6f} {py:.6f} {pz:.6f}")
    if nb:
        out += ["", "Bonds", ""]
        for c in range(n):
            base = c * natoms + 1
            for k, (i, j, _kb) in enumerate(springs):
                out.append(f"{c * nb + k + 1} {k + 1} {base + i} {base + j}")
    if na:
        out += ["", "Angles", ""]
        for c in range(n):
            base = c * natoms + 1
            for k, (i, j, l) in enumerate(chain.angles):
                out.append(f"{c * na + k + 1} {k + 1} "
                           f"{base + i} {base + j} {base + l}")
    path.write_text("\n".join(out) + "\n")


_SCRIPT = """\
# PAAF: pack chains with a soft repulsion instead of packmol.
units           real
atom_style      full
boundary        p p p
{bond_style}
{angle_style}
read_data       {data}

pair_style      soft {cutoff}
pair_coeff      * * 0.0
special_bonds   lj/coul 0.0 0.0 0.0
neighbor        1.0 bin
# Ghost atoms must reach the far end of the longest spring, or a spring
# that crosses a periodic face is attached to the wrong image of its atom.
comm_modify     cutoff {comm}
# Only contacts BETWEEN chains count, as in packmol.
neigh_modify    exclude molecule/intra all every 1 delay 0 check yes

compute         pl all pair/local dist
compute         mind all reduce min c_pl inputs local
variable        mind equal c_mind
thermo_style    custom step pe v_mind
thermo          500
thermo_modify   norm no lost warn

# ---- stage 1: rigid chains, packmol's degrees of freedom
variable        A1 equal ramp(0.5,{a_rigid})
fix             pushR all adapt 1 pair soft a * * v_A1
fix             rig all rigid molecule
fix             damp all viscous 2.0
timestep        1.0
run             {n_rigid}
print           "PAAF_RIGID_MINDIST ${{mind}}"
unfix           pushR
unfix           rig
unfix           damp
if "${{mind}} >= {tol_ok}" then "jump SELF done"

# ---- stage 2: flexible finish, bonds, angles and torsions held at their
#      own values. Runs in chunks and ends at the first chunk that meets
#      the tolerance.
{flex_pair}
fix             mv all nve/limit 0.1
fix             damp all langevin 300.0 300.0 50.0 {seed2}
timestep        0.5
run             {n_chunk}
variable        chunk loop {n_chunks}
label           flexloop
if "${{mind}} >= {tol_ok}" then "jump SELF done"
run             {n_chunk}
next            chunk
jump            SELF flexloop

label           done
run             0
print           "PAAF_FINAL_MINDIST ${{mind}}"
write_dump      all custom {dump} id mol xu yu zu modify sort id
"""


# ================================================================ the result
@dataclass
class LammpsPackResult:
    #: ``(n_chains * natoms, 3)``: chain by chain, atoms in ascending id order
    #: within each chain — the order ``lammps_replicator`` pairs them with.
    coords: np.ndarray
    #: Closest pair of atoms on different chains after packing, Å. ``inf``
    #: when no pair is within the tolerance.
    min_distance: float
    rigid_only: bool
    launch: str
    work_dir: Path
    log_path: Path


def _whole_in_cell(xyz: np.ndarray, natoms: int, box: Sequence[float]
                   ) -> np.ndarray:
    """Shift each chain by whole cell vectors so its centre lies in the cell.

    LAMMPS reports unwrapped positions, so a chain that drifted through a
    face comes back whole but outside. Chains stay whole, as packmol and
    ``gmx insert-molecules`` leave them.
    """
    h = _h_matrix(box)
    hinv = np.linalg.inv(h)
    out = xyz.copy()
    for c in range(len(xyz) // natoms):
        s = slice(c * natoms, (c + 1) * natoms)
        frac = hinv @ out[s].mean(axis=0)
        out[s] -= h @ np.floor(frac)
    return out


def _read_dump(path: Path) -> Dict[int, Tuple[float, float, float]]:
    lines = path.read_text().splitlines()
    try:
        start = next(i for i, l in enumerate(lines)
                     if l.startswith("ITEM: ATOMS")) + 1
    except StopIteration:
        raise RuntimeError(f"{path} holds no atoms") from None
    out: Dict[int, Tuple[float, float, float]] = {}
    for line in lines[start:]:
        p = line.split()
        if len(p) >= 5:
            out[int(p[0])] = (float(p[2]), float(p[3]), float(p[4]))
    return out


def _last_value(log_text: str, tag: str) -> Optional[float]:
    vals = re.findall(rf"{tag}\s+(\S+)", log_text)
    if not vals:
        return None
    try:
        return float(vals[-1])
    except ValueError:
        return None


def lammps_pack(single_data_file: Path, n_chains: int,
                box: Sequence[float], *, tolerance: float = 2.0,
                seed: int = -1, nprocs: int = 4, lammps_exe: str = "",
                placement: str = "random",
                work_dir: Optional[Path] = None, cancel=None,
                progress=None) -> LammpsPackResult:
    """Pack ``n_chains`` copies of the chain in ``single_data_file``.

    ``box`` is ``(lx, ly, lz)`` or ``(lx, ly, lz, xy, xz, yz)`` in Å with
    the origin at 0, as ``BoxShape.lammps_params()`` gives it.
    """
    from .cell.packing import run_cancellable
    from .cell.relax import find_lammps

    say = progress or (lambda _m: None)
    exe = find_lammps(lammps_exe)
    if exe is None:
        raise RuntimeError(
            "The LAMMPS packer needs a LAMMPS executable (lmp, lmp_mpi or "
            "lmp_serial) and none was found. Install LAMMPS, give its path "
            "on the Box page, or choose Packmol as the packer.")
    build = probe_lammps(str(exe))
    missing = {"MOLECULE", "RIGID"} - set(build.packages)
    if build.packages and missing:
        raise RuntimeError(
            f"{exe} was built without the {', '.join(sorted(missing))} "
            f"package(s), which the LAMMPS packer needs (bonds and rigid "
            f"chains). Use a fuller LAMMPS build, or choose Packmol.")

    if seed is None or int(seed) < 0:
        seed = random.randint(1, 2_000_000_000)
    seed = int(seed)
    tolerance = float(tolerance)

    chain = _read_chain(Path(single_data_file))
    natoms = len(chain.ids)
    work = Path(work_dir) if work_dir else Path(single_data_file).parent \
        / "lammps_pack"
    work.mkdir(parents=True, exist_ok=True)

    x = chain.xyz
    longest = max([float(np.linalg.norm(x[i] - x[j]))
                   for i, j in chain.bonds + chain.pairs14]
                  + [float(np.linalg.norm(x[i] - x[l]))
                     for i, _j, l in chain.angles] + [0.0])
    comm = max(tolerance + _REACH + 1.0, 2.0 * longest)

    if not build.packages or "EXTRA-PAIR" in build.packages:
        flex_pair = (f"pair_style      harmonic/cut\n"
                     f"pair_coeff      * * {_K_FLEX} {tolerance + _REACH:.4f}")
    else:
        flex_pair = (f"pair_style      soft {tolerance + _REACH:.4f}\n"
                     f"pair_coeff      * * {_A_FLEX}")

    placed = place_chains(chain.xyz, int(n_chains), box, seed, placement)
    _write_pack_data(work / "pack_in.data", chain, placed, box)
    script = _SCRIPT.format(
        data="pack_in.data", dump="packed.dump",
        cutoff=f"{tolerance + _REACH:.4f}", comm=f"{comm:.2f}",
        tol_ok=f"{tolerance - _SLACK:.4f}",
        bond_style=("bond_style      harmonic"
                    if chain.bonds or chain.pairs14 else ""),
        angle_style="angle_style     harmonic" if chain.angles else "",
        a_rigid=_A_RIGID, flex_pair=flex_pair,
        seed2=seed % 900_000_000 + 1,
        n_rigid=_STEPS_RIGID, n_chunk=_STEPS_FLEX_CHUNK,
        n_chunks=_FLEX_CHUNKS - 1)
    (work / "pack.in").write_text(script)

    cmd, how = launch_command(build, nprocs)
    cmd = cmd + ["-in", "pack.in", "-log", "pack.log", "-screen", "none"]
    say(f"[lammps-pack] {n_chains} chains × {natoms} atoms, {placement} "
        f"placement, seed {seed}, "
        f"tolerance {tolerance:.2f} Å, LAMMPS on {how}")
    log.info("LAMMPS packer: %s (cwd=%s)", " ".join(cmd), work)
    rc, out, err = run_cancellable(cmd, cwd=str(work), cancel=cancel)

    log_path = work / "pack.log"
    log_text = log_path.read_text(errors="replace") if log_path.exists() \
        else (out or "")
    dump = work / "packed.dump"
    if rc != 0 or not dump.exists():
        tail = "\n".join((log_text + "\n" + (err or "")).strip()
                         .splitlines()[-25:])
        raise RuntimeError(
            f"LAMMPS packing failed (exit code {rc}). Last lines of "
            f"{log_path}:\n\n{tail}")

    by_id = _read_dump(dump)
    try:
        xyz = np.array([by_id[i + 1] for i in range(int(n_chains) * natoms)])
    except KeyError as exc:
        raise RuntimeError(f"LAMMPS lost atom {exc} during packing; see "
                           f"{log_path}") from None
    xyz = _whole_in_cell(xyz, natoms, box)

    rigid_md = _last_value(log_text, "PAAF_RIGID_MINDIST")
    final_md = _last_value(log_text, "PAAF_FINAL_MINDIST")
    # compute reduce min over an empty set is LAMMPS's BIG (1e20): no pair
    # within the cutoff at all.
    md = math.inf if final_md is None or final_md > 1e10 else final_md
    # The script skips the flexible finish exactly when this holds.
    rigid_only = rigid_md is not None and rigid_md >= tolerance - _SLACK
    if md >= tolerance - _SLACK:
        say(f"[lammps-pack] done: no two chains closer than "
            f"{tolerance:.2f} Å"
            + (" (rigid stage was enough)" if rigid_only
               else " (after the flexible finish)"))
    else:
        say(f"[lammps-pack] done, but the closest contact between chains is "
            f"{md:.2f} Å (< {tolerance:.2f} Å). The box is very tight for "
            f"these chains; minimise before dynamics, or enlarge the box.")
    return LammpsPackResult(coords=xyz, min_distance=md,
                            rigid_only=rigid_only, launch=how,
                            work_dir=work, log_path=log_path)
