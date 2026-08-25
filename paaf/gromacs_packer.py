"""GROMACS-native box packing via ``gmx insert-molecules``.

Ported from the user's ``create_box.py`` — same pattern:

1. (optional) run a quick GROMACS energy minimisation on the single chain
   so the chain has a sensible geometry before insertion.
2. Read the atom count from the ``.gro`` file.
3. Compute how many copies to insert:
     * if ``n_chains`` is given → use that count.
     * else if ``atom_limit`` is given → ``atom_limit // atoms_per_chain``.
4. Run ``gmx insert-molecules -ci chain.gro -nmol N -box a b c -o filled.gro``.
5. Rewrite the topology file so the last ``moleculetype`` count is N
   instead of 1.

The result is a packed ``.gro`` in the same directory as the single-chain
input, plus an updated ``.top``.
"""
from __future__ import annotations

import math
import os
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .logging_utils import get_logger

log = get_logger(__name__)


_ENMIN_MDP = """; PAAF pre-insertion energy minimisation (steepest descent)
integrator      = steep
emtol           = 500
emstep          = 0.01
nsteps          = 500000
nstenergy       = 500
nstlog          = 500

nstlist         = 1
cutoff-scheme   = Verlet
ns_type         = grid
coulombtype     = PME
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = xyz
"""


@dataclass
class GmxPackResult:
    filled_gro: Path
    updated_top: Optional[Path]
    n_inserted: int
    energy_minimised: bool


def _find_gmx() -> Optional[str]:
    """Locate gmx or gmx_mpi. Returns the executable path or None.

    Also probes the conda-forge GROMACS layout on Apple Silicon, which puts
    the binary in a SIMD-suffixed subfolder (``bin.ARM_NEON_ASIMD/gmx``)
    instead of plain ``bin/gmx``. Without this override PATH may miss it.
    """
    for name in ("gmx", "gmx_mpi"):
        p = shutil.which(name)
        if p:
            return p
    # Explicit fallbacks for conda-forge GROMACS 2025.x on macOS ARM.
    home = Path.home()
    _CANDIDATES = [
        "/usr/local/gromacs/bin/gmx",
        "/opt/homebrew/bin/gmx",
        "/opt/local/bin/gmx",
        "/opt/homebrew/Caskroom/miniconda/base/envs/mta/bin.ARM_NEON_ASIMD/gmx",
        "/opt/homebrew/Caskroom/miniconda/base/envs/mta/bin/gmx",
        str(home / "miniconda3/envs/mta/bin.ARM_NEON_ASIMD/gmx"),
        str(home / "miniconda3/envs/mta/bin/gmx"),
        str(home / "opt/miniconda3/envs/mta/bin.ARM_NEON_ASIMD/gmx"),
    ]
    # If CONDA_PREFIX is set, also try its SIMD-suffixed bin dirs.
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        _CANDIDATES.extend([
            str(Path(conda_prefix) / "bin.ARM_NEON_ASIMD/gmx"),
            str(Path(conda_prefix) / "bin/gmx"),
            str(Path(conda_prefix) / "bin.AVX_512/gmx"),
            str(Path(conda_prefix) / "bin.AVX2_256/gmx"),
        ])
    for cand in _CANDIDATES:
        if Path(cand).exists():
            return cand
    return None


def _read_natoms(gro_path: Path) -> int:
    """The 2nd line of a .gro file is the atom count."""
    with gro_path.open() as fh:
        _ = fh.readline()
        return int(fh.readline().strip())


def _rewrite_top_count(top_path: Path, new_count: int) -> None:
    """Update the last ``moleculename count`` line in [ molecules ] section
    from 1 → new_count. Mirrors what create_box.py does."""
    if not top_path.exists():
        return
    lines = top_path.read_text().splitlines(keepends=True)
    out: list[str] = []
    replaced = False
    for line in reversed(lines):
        # Match a line like "MyMolName   1"
        m = re.match(r"^\s*(\S+)\s+(\d+)\s*$", line)
        if not replaced and m and int(m.group(2)) == 1:
            out.append(f"{m.group(1)}\t{new_count}\n")
            replaced = True
        else:
            out.append(line)
    out.reverse()
    top_path.write_text("".join(out))
    log.info("Rewrote %s: molecule count → %d", top_path, new_count)


def _run(cmd: list, cwd: Path) -> subprocess.CompletedProcess:
    """Run a subprocess with combined stdout/stderr captured, log it."""
    log.info("$ %s   (cwd=%s)", " ".join(cmd), cwd)
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        log.warning("Command failed (rc=%d). Last stderr:\n%s",
                    p.returncode, "\n".join((p.stderr or "").splitlines()[-30:]))
    return p


def energy_minimise_single_chain(gro_path: Path, top_path: Path,
                                 gmx: Optional[str] = None) -> Path:
    """Run one steepest-descent minimisation on ``gro_path`` using its
    matching ``top_path``. Returns the path to the minimised .gro."""
    gmx = gmx or _find_gmx()
    if gmx is None:
        raise RuntimeError("gmx not found on PATH; can't run energy minimisation.")
    workdir = gro_path.parent
    mdp = workdir / "enmin.mdp"
    mdp.write_text(_ENMIN_MDP)
    tpr = workdir / "enmin.tpr"
    r = _run([gmx, "grompp",
              "-f",   str(mdp.name),
              "-c",   str(gro_path.name),
              "-p",   str(top_path.name),
              "-o",   str(tpr.name),
              "-maxwarn", "3"],
             cwd=workdir)
    if r.returncode != 0:
        raise RuntimeError(f"gmx grompp failed: rc={r.returncode}\n{r.stderr[-600:]}")
    r = _run([gmx, "mdrun", "-deffnm", "enmin"], cwd=workdir)
    if r.returncode != 0:
        raise RuntimeError(f"gmx mdrun failed: rc={r.returncode}\n{r.stderr[-600:]}")
    return workdir / "enmin.gro"


# ================================================================ pure-Python fallback
def _parse_gro(gro_path: Path) -> Tuple[str, List[dict], Tuple[float, float, float]]:
    """Return (title, atoms, box) where atoms is a list of dicts:
    {resnum, resname, atomname, atomid, x, y, z}. Positions in nm.
    Box is a 3-tuple of edge lengths in nm (from last line; ignores tilt).
    """
    lines = gro_path.read_text().splitlines()
    title = lines[0]
    natoms = int(lines[1].strip())
    atoms: List[dict] = []
    for k in range(natoms):
        line = lines[2 + k]
        # Fixed-width GRO format:
        #   residue number (i5) + residue name (a5) + atom name (a5)
        #   + atom number (i5) + x,y,z (3 f8.3)
        atoms.append({
            "resnum":   int(line[0:5]),
            "resname":  line[5:10].strip(),
            "atomname": line[10:15].strip(),
            "atomid":   int(line[15:20]),
            "x":        float(line[20:28]),
            "y":        float(line[28:36]),
            "z":        float(line[36:44]),
        })
    box_parts = lines[2 + natoms].split()
    box = (float(box_parts[0]), float(box_parts[1]), float(box_parts[2])) \
          if len(box_parts) >= 3 else (0.0, 0.0, 0.0)
    return title, atoms, box


def _write_gro(path: Path, title: str, atoms: List[dict],
              box: Tuple[float, float, float]) -> None:
    """Write a GROMACS .gro file with strict column widths."""
    with path.open("w") as fh:
        fh.write(f"{title}\n")
        fh.write(f"{len(atoms):5d}\n")
        for a in atoms:
            fh.write(
                f"{a['resnum'] % 100000:5d}"
                f"{a['resname'][:5]:>5s}"
                f"{a['atomname'][:5]:>5s}"
                f"{a['atomid'] % 100000:5d}"
                f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}\n"
            )
        fh.write(f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}\n")


def _random_rotation_matrix(rng: random.Random) -> Tuple[Tuple[float, ...], ...]:
    """Uniform-random 3D rotation matrix via three Euler angles."""
    a = rng.uniform(0, 2 * math.pi)
    b = rng.uniform(0, 2 * math.pi)
    c = rng.uniform(0, 2 * math.pi)
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cc, sc = math.cos(c), math.sin(c)
    return (
        (cb * cc,               -cb * sc,               sb),
        (sa * sb * cc + ca * sc, -sa * sb * sc + ca * cc, -sa * cb),
        (-ca * sb * cc + sa * sc, ca * sb * sc + sa * cc,  ca * cb),
    )


def _apply_rot_and_translate(coords: List[Tuple[float, float, float]],
                             R, t: Tuple[float, float, float]
                             ) -> List[Tuple[float, float, float]]:
    out = []
    for x, y, z in coords:
        rx = R[0][0] * x + R[0][1] * y + R[0][2] * z + t[0]
        ry = R[1][0] * x + R[1][1] * y + R[1][2] * z + t[1]
        rz = R[2][0] * x + R[2][1] * y + R[2][2] * z + t[2]
        out.append((rx, ry, rz))
    return out


def _pack_pure_python(single_gro: Path, box_edges_nm: Tuple[float, float, float],
                     n_copies: int, out_gro: Path, seed: int = -1,
                     max_tries_per_chain: int = 400,
                     min_dist_nm: float = 0.20) -> Tuple[int, List[dict]]:
    """Place ``n_copies`` of ``single_gro`` at random positions + rotations
    inside a box of ``box_edges_nm``, avoiding overlaps closer than
    ``min_dist_nm`` (0.20 nm ≈ ~2.0 Å tolerance, similar to packmol).

    Returns (actually_inserted, packed_atoms). If packing runs out of retries
    before all N copies land, the caller can inform the user; we don't raise.
    """
    if seed is None or seed < 0:
        seed = random.randint(1, 2_000_000_000)
    rng = random.Random(seed)
    title, atoms, _ = _parse_gro(single_gro)
    n_per_chain = len(atoms)
    # Zero-mean the source coords so rotation is around chain centroid.
    cx = sum(a["x"] for a in atoms) / n_per_chain
    cy = sum(a["y"] for a in atoms) / n_per_chain
    cz = sum(a["z"] for a in atoms) / n_per_chain
    src_coords = [(a["x"] - cx, a["y"] - cy, a["z"] - cz) for a in atoms]
    span = max(math.sqrt((a[0] * a[0]) + (a[1] * a[1]) + (a[2] * a[2]))
               for a in src_coords)
    # Reserve a margin so a rotated chain still fits in the box.
    margin = min(span, min(box_edges_nm) / 2.0 - 0.05)
    if margin <= 0:
        raise RuntimeError(
            f"Single chain (radius {span:.2f} nm) does not fit into a box of "
            f"{box_edges_nm} nm. Increase the cell edges.")
    a_nm, b_nm, c_nm = box_edges_nm

    # Fast neighbour test: cell grid keyed on int coords, cell size = min_dist_nm.
    cell = min_dist_nm
    grid: dict = {}
    def _key(x, y, z):
        return (int(x / cell), int(y / cell), int(z / cell))
    def _add(x, y, z):
        grid.setdefault(_key(x, y, z), []).append((x, y, z))
    def _clash(x, y, z) -> bool:
        kx, ky, kz = _key(x, y, z)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for (px, py, pz) in grid.get((kx + dx, ky + dy, kz + dz), ()):
                        if (px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2 < cell * cell:
                            return True
        return False

    packed: List[dict] = []
    resnum_offset = 0
    atomid_offset = 0
    inserted = 0
    for chain_i in range(n_copies):
        placed = False
        for _ in range(max_tries_per_chain):
            tx = rng.uniform(margin, a_nm - margin)
            ty = rng.uniform(margin, b_nm - margin)
            tz = rng.uniform(margin, c_nm - margin)
            R = _random_rotation_matrix(rng)
            new_coords = _apply_rot_and_translate(src_coords, R, (tx, ty, tz))
            # Check first-atom + centroid distance against grid; if any point
            # collides with existing placement, retry.
            if any(_clash(x, y, z) for (x, y, z) in new_coords):
                continue
            # Commit
            for (x, y, z) in new_coords: _add(x, y, z)
            for (src_atom, (x, y, z)) in zip(atoms, new_coords):
                packed.append({
                    "resnum":   (src_atom["resnum"] + resnum_offset),
                    "resname":  src_atom["resname"],
                    "atomname": src_atom["atomname"],
                    "atomid":   (src_atom["atomid"] + atomid_offset),
                    "x": x, "y": y, "z": z,
                })
            resnum_offset += max(a["resnum"] for a in atoms)
            atomid_offset += n_per_chain
            inserted += 1
            placed = True
            break
        if not placed:
            break
    _write_gro(out_gro, title + f" — {inserted} copies (paaf packed)", packed,
               box_edges_nm)
    return inserted, packed


def pack_with_gmx_insert(
    single_gro: Path,
    top_file: Path,
    box_edges: Tuple[float, float, float],
    n_chains: Optional[int] = None,
    atom_limit: Optional[int] = None,
    try_count: int = 100000,
    pre_minimise: bool = True,
    gmx: Optional[str] = None,
    out_name: str = "packed_box.gro",
) -> GmxPackResult:
    """Fill a box with N copies of ``single_gro`` via ``gmx insert-molecules``.

    Exactly one of ``n_chains`` or ``atom_limit`` must be given.
    Box edges are in **nanometres** (GROMACS convention).
    """
    if (n_chains is None) == (atom_limit is None):
        raise ValueError("Give exactly ONE of n_chains, atom_limit.")
    if not single_gro.exists():
        raise FileNotFoundError(single_gro)
    if not top_file.exists():
        log.warning("Topology file not found next to single chain: %s", top_file)

    # ---------- pure-Python fallback when gmx isn't installed --------------
    gmx = gmx or _find_gmx()
    if gmx is None:
        log.warning("gmx not found — falling back to PAAF's pure-Python "
                    "insert-molecules replacement.")
        natoms = _read_natoms(single_gro)
        nmol = int(n_chains) if n_chains is not None else \
               max(1, int(atom_limit) // max(1, natoms))
        out_gro = single_gro.parent / out_name
        inserted, _ = _pack_pure_python(
            single_gro, box_edges, n_copies=nmol, out_gro=out_gro)
        if inserted < nmol:
            log.warning("Pure-Python packer only fit %d / %d chains — "
                        "increase the box edges or reduce chain count.",
                        inserted, nmol)
        updated_top = None
        if top_file.exists():
            _rewrite_top_count(top_file, inserted)
            updated_top = top_file
        return GmxPackResult(filled_gro=out_gro, updated_top=updated_top,
                             n_inserted=inserted, energy_minimised=False)

    workdir = single_gro.parent
    minimised = False
    if pre_minimise and top_file.exists():
        try:
            single_gro = energy_minimise_single_chain(single_gro, top_file, gmx=gmx)
            minimised = True
            log.info("Single chain minimised → %s", single_gro)
        except Exception as e:
            log.warning("Pre-minimisation failed (%s). Continuing with unminimised chain.", e)

    natoms = _read_natoms(single_gro)
    if n_chains is not None:
        nmol = int(n_chains)
        log.info("Chain has %d atoms; inserting %d copies (explicit chain "
                 "count) = %d atoms.", natoms, nmol, nmol * natoms)
    else:
        # NEAREST whole chain, not the floor.
        #
        # A target of 50000 atoms with a 242-atom chain is 206.6 chains.
        # Truncating gave 206 and 49852 atoms — 148 short — and said nothing,
        # so the box quietly differed from what was asked for. 207 chains is
        # 50094, which is 94 out: closer to the target in the direction the
        # user cares about, which is "about this many atoms".
        #
        # A whole number of chains can rarely hit the target exactly, so the
        # shortfall is REPORTED rather than hidden. That is the part that was
        # missing; either rounding is defensible, silence is not.
        target = max(1, int(atom_limit))
        per = max(1, natoms)
        nmol = max(1, int(round(target / per)))
        delivered = nmol * per
        log.info("Chain has %d atoms; target %d atoms → %d copies = %d atoms "
                 "(%+d, %.1f%% off target; a whole number of chains cannot "
                 "always hit it exactly).",
                 natoms, target, nmol, delivered, delivered - target,
                 100.0 * abs(delivered - target) / target)

    a, b, c = box_edges
    out_gro = workdir / out_name
    r = _run([gmx, "insert-molecules",
              "-ci",  str(single_gro.name),
              "-nmol", str(nmol),
              "-box",  f"{a}", f"{b}", f"{c}",
              "-try",  str(try_count),
              "-o",    str(out_gro.name)],
             cwd=workdir)
    if r.returncode != 0 or not out_gro.exists():
        raise RuntimeError(
            f"gmx insert-molecules failed (rc={r.returncode}):\n{r.stderr[-800:]}")

    # How many actually went in, counted from the file gmx wrote.
    #
    # ``gmx insert-molecules`` does not promise to place what it was asked
    # for: it makes up to ``-try`` attempts per molecule and inserts as many
    # as fit. Writing ``nmol`` into the topology regardless is how a .gro and
    # a .top come to disagree, and grompp's complaint about that names atom
    # counts rather than the cause. The pure-Python fallback in this module
    # has always counted the real number; the gmx path did not.
    placed = nmol
    try:
        actual_atoms = _read_natoms(out_gro)
        if natoms > 0 and actual_atoms % natoms == 0:
            placed = actual_atoms // natoms
    except Exception as exc:                          # pragma: no cover
        log.warning("Could not read back %s to confirm how many chains were "
                    "inserted (%s); assuming %d.", out_gro, exc, nmol)

    if placed < nmol:
        log.warning("gmx insert-molecules placed %d of %d chains — the box is "
                    "too small for the rest, or -try (%d) was too low. The "
                    "topology is written for %d so it MATCHES the "
                    "coordinates; the box is less dense than requested.",
                    placed, nmol, try_count, placed)

    # Update the topology molecule count from 1 → however many are really there.
    updated_top = None
    if top_file.exists():
        _rewrite_top_count(top_file, placed)
        updated_top = top_file

    return GmxPackResult(
        filled_gro=out_gro,
        updated_top=updated_top,
        n_inserted=placed,
        energy_minimised=minimised,
    )
