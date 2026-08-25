"""Relax a constructed amorphous cell with LAMMPS.

The gap this closes
-------------------
BIOVIA describe Amorphous Cell as two steps: an initial guess structure,
*followed by relaxation to a state of minimum potential energy*. PAAF has only
ever done the first. This module does the second, with whatever force field
the cell was typed in.

Note what this is and is not. Minimisation finds a nearby local minimum ---
effectively a 0 K glass. It removes strained contacts and makes the cell safe
to run, and it is exactly what Materials Studio's relaxation step does. It
does **not** equilibrate the melt: chain dimensions relax on a diffusive
timescale that only molecular dynamics reaches. The over-extension that
sequential construction produces will still be there after minimisation.

Why a plain ``minimize`` is not enough
--------------------------------------
A constructed cell has residual close contacts --- PAAF measures 1.0--1.7 Å
after its geometric push-off. Handing that to a Lennard-Jones potential is
catastrophic. At :math:`r = 1.0` Å with :math:`\\sigma = 3.5` Å the repulsive
term alone is

.. math::  (\\sigma/r)^{12} = (3.5/1.0)^{12} \\approx 3.4 \\times 10^{6}\\,\\epsilon

so the force is astronomically large. LAMMPS responds by aborting with "Out of
range atoms" or by launching atoms across the box, and either way the run is
lost.

The established remedy is a **soft push-off**: replace the real pair style
with a bounded, purely repulsive one whose prefactor is ramped up from zero,
run a short capped-displacement dynamics so overlapping atoms slide apart
without ever seeing a singular force, and only then switch to the real
potential and minimise::

    pair_style  soft 3.0
    variable    pre equal ramp(0, 60)
    fix         push all adapt 1 pair soft a * * v_pre
    fix         lim  all nve/limit 0.05
    run         <n>

``soft`` has the form :math:`A[1 + \\cos(\\pi r / r_c)]`, which is finite at
:math:`r = 0` --- that finiteness is the entire point. ``nve/limit`` caps how
far any atom may move in one step, so even a badly placed atom cannot be
ejected.

Force-field styles
------------------
The pair, bond, angle and dihedral styles come from the cell's own typing.
Moltemplate writes them into ``system.in.init`` and the coefficients into
``system.in.settings``, so that route needs no guessing. For a bare
``.data`` file the styles must be supplied by the caller.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..logging_utils import get_logger
from ..structure import Molecule

log = get_logger(__name__)

__all__ = ["RelaxSettings", "RelaxResult", "CellMetrics", "find_lammps",
           "find_gromacs", "write_relax_input", "relax_cell", "measure_cell",
           "choose_engine", "lammps_styles_for", "write_em_mdp",
           "relax_cell_gromacs"]


# =====================================================================
# Which engine minimises which force field
# =====================================================================
def choose_engine(ff_key: str, preference: str = "auto") -> str:
    """``"lammps"`` or ``"gromacs"`` for a given force field.

    The typing tool decides what the engine *can* be, which is why this is not
    a free choice:

    ``moltemplate_native`` (OPLS-AA, L-OPLS, TraPPE, DREIDING, COMPASS
        published, MARTINI, SDK)
        Moltemplate emits a LAMMPS data file plus ``system.in.init`` and
        ``system.in.settings``. **LAMMPS only** --- there is no GROMACS output
        on this path.

    ``dlfield`` (PCFF, COMPASS, CVFF, OPLS-2005, CHARMM36, AMBER/GAFF,
        GROMOS-54A7, TraPPE-EH, OPLS-UA)
        DL_FIELD writes either LAMMPS or GROMACS output, selected by its
        control file. **Either engine.** GROMACS is the better-conditioned
        route here: DL_FIELD's ``.top`` states every functional form
        explicitly, so nothing has to be inferred, whereas its LAMMPS data
        file leaves the pair/bond/angle/dihedral styles to be supplied.

    ``gaff`` (antechamber)
        LAMMPS, via the same data-file route.

    ``preference`` of ``"lammps"`` or ``"gromacs"`` is honoured where the
    force field allows it, and otherwise falls back with the caller free to
    report the substitution.
    """
    from ..ff_registry import REGISTRY

    pref = (preference or "auto").lower()
    ff = REGISTRY.get(ff_key)
    if ff is None:
        return "lammps" if pref != "gromacs" else "gromacs"

    dl = (ff.kind == "dlfield")
    if pref == "gromacs":
        return "gromacs" if dl else "lammps"
    if pref == "lammps":
        return "lammps"

    # auto: prefer the better-conditioned route where one exists AND is
    # installed. For a DL_FIELD force field that is GROMACS, because its .top
    # states every functional form and nothing has to be inferred; the LAMMPS
    # route there falls back to a displacement-capped minimisation and a
    # styles table PAAF supplies. Fall back to LAMMPS when gmx is absent
    # rather than choosing an engine the user does not have.
    if dl and find_gromacs() is not None:
        return "gromacs"
    return "lammps"


# DEPRECATED — kept only as a last resort when no data file is available to
# read from. Prefer :func:`paaf.dlfield_styles.styles_for_data`, which reads
# each sub-style out of the data file's own coefficient lines.
#
# This table was a guess dressed as knowledge, and it was wrong. It claimed
# PCFF needs angle/dihedral/improper ``class2``, on the reasoning that PCFF is
# a class-II force field. What DL_FIELD actually writes is ``class2`` bonds
# with ``quartic`` angles, ``fourier`` dihedrals and ``inversion/harmonic``
# impropers — because ``angle_style class2`` requires BondBond and BondAngle
# cross-term sections that DL_FIELD does not emit. Declaring it over a file
# without them makes LAMMPS demand sections that do not exist.
#
# The lesson generalises: the sub-style is written in column 2 of every
# DL_FIELD coefficient line, so it can be read instead of guessed.
_DLFIELD_STYLES: Dict[str, Dict[str, str]] = {
    "pcff": {
        "pair_style": "lj/class2/coul/long 10.0",
        "bond_style": "class2", "angle_style": "class2",
        "dihedral_style": "class2", "improper_style": "class2",
        "kspace_style": "pppm 1.0e-4",
        "special_bonds": "lj/coul 0.0 0.0 1.0",
    },
    "compass": {
        "pair_style": "lj/class2/coul/long 10.0",
        "bond_style": "class2", "angle_style": "class2",
        "dihedral_style": "class2", "improper_style": "class2",
        "kspace_style": "pppm 1.0e-4",
        "special_bonds": "lj/coul 0.0 0.0 1.0",
    },
    "cvff": {
        "pair_style": "lj/cut/coul/long 10.0",
        "bond_style": "harmonic", "angle_style": "harmonic",
        "dihedral_style": "harmonic", "improper_style": "cvff",
        "kspace_style": "pppm 1.0e-4",
        "special_bonds": "lj/coul 0.0 0.0 0.5",
    },
    "opls2005_dl": {
        "pair_style": "lj/cut/coul/long 10.0",
        "bond_style": "harmonic", "angle_style": "harmonic",
        "dihedral_style": "opls", "improper_style": "harmonic",
        "kspace_style": "pppm 1.0e-4",
        "special_bonds": "lj/coul 0.0 0.0 0.5",
    },
    "opls_ua": {
        "pair_style": "lj/cut/coul/long 10.0",
        "bond_style": "harmonic", "angle_style": "harmonic",
        "dihedral_style": "opls", "improper_style": "harmonic",
        "kspace_style": "pppm 1.0e-4",
        "special_bonds": "lj/coul 0.0 0.0 0.5",
    },
    "charmm36": {
        "pair_style": "lj/charmm/coul/long 8.0 10.0",
        "bond_style": "harmonic", "angle_style": "charmm",
        "dihedral_style": "charmm", "improper_style": "harmonic",
        "kspace_style": "pppm 1.0e-4",
        "special_bonds": "charmm",
    },
    "amber_gaff": {
        "pair_style": "lj/cut/coul/long 10.0",
        "bond_style": "harmonic", "angle_style": "harmonic",
        "dihedral_style": "harmonic", "improper_style": "cvff",
        "kspace_style": "pppm 1.0e-4",
        "special_bonds": "amber",
    },
    "trappe_eh": {
        "pair_style": "lj/cut 14.0",
        "bond_style": "harmonic", "angle_style": "harmonic",
        "dihedral_style": "opls",
        "special_bonds": "lj 0.0 0.0 0.0",
    },
}


def lammps_styles_for(ff_key: str, data_file: Optional[Path] = None
                      ) -> Optional[Dict[str, str]]:
    """LAMMPS styles for a DL_FIELD force field, or ``None``.

    When ``data_file`` is given the bonded styles are **read** from its
    coefficient sections, where DL_FIELD names each sub-style in column 2.
    That is what the force field actually wrote, so it needs no table and
    stays right for force fields nobody has added an entry for.

    Without a data file the deprecated table is consulted, and ``None`` means
    PAAF has no entry and will not guess. Running a minimisation with the
    wrong functional form produces numbers that look fine and are wrong,
    which is worse than refusing.
    """
    if data_file is not None and Path(data_file).is_file():
        from ..dlfield_styles import styles_for_data
        styles = styles_for_data(data_file, ff_key)
        if styles:
            return styles
    return _DLFIELD_STYLES.get((ff_key or "").lower())

_N_AVOGADRO = 6.02214076e23

# Names a LAMMPS binary is commonly installed under, most specific first.
_LAMMPS_NAMES = ("lmp", "lmp_serial", "lmp_mpi", "lammps", "lmp_ubuntu",
                 "lmp_openmpi", "lmp_daily")


# =====================================================================
@dataclass
class RelaxSettings:
    """How hard to push, and how far to minimise."""
    # --- soft push-off
    push_off: bool = True
    soft_cutoff: float = 3.0          # Å, where the soft potential dies
    soft_prefactor: float = 60.0      # kcal/mol at full ramp
    push_steps: int = 10_000
    nve_limit: float = 0.05           # Å, max displacement per step
    timestep_fs: float = 1.0

    # --- minimisation
    minimize: bool = True
    etol: float = 1e-4
    ftol: float = 1e-6
    maxiter: int = 10_000
    maxeval: int = 100_000

    # --- environment
    engine: str = "auto"              # auto | lammps | gromacs
    lammps_exe: str = ""              # blank -> auto-detect
    gmx_exe: str = ""                 # blank -> auto-detect
    mpi_ranks: int = 1
    timeout_s: int = 3600

    # --- GROMACS steepest descent
    gmx_emtol: float = 100.0          # kJ/mol/nm
    gmx_emstep: float = 0.001         # nm, small: the cell starts strained
    gmx_steps: int = 50_000


@dataclass
class CellMetrics:
    """The numbers worth comparing before and against after."""
    n_atoms: int = 0
    density_g_cm3: float = 0.0
    closest_contact_a: float = 0.0
    r_gyration_a: float = 0.0
    bond_min_a: float = 0.0
    bond_max_a: float = 0.0

    def row(self, label: str) -> str:
        return (f"  {label:9s} {self.density_g_cm3:8.4f} "
                f"{self.closest_contact_a:9.2f} {self.r_gyration_a:8.2f} "
                f"{self.bond_min_a:7.3f} {self.bond_max_a:7.3f}")


@dataclass
class RelaxResult:
    folder: Path
    engine: str = ""
    input_script: Optional[Path] = None
    log_file: Optional[Path] = None
    dump_file: Optional[Path] = None
    relaxed_data: Optional[Path] = None
    ran: bool = False
    before: Optional[CellMetrics] = None
    after: Optional[CellMetrics] = None
    potential_energy: Optional[float] = None
    messages: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.ran:
            return (f"relaxation NOT run; input written to "
                    f"{self.input_script}" if self.input_script
                    else "relaxation not run")
        lines = [f"relaxed with {self.engine.upper() or 'LAMMPS'}",
                 "            density  contact      Rg   bond min    max"]
        if self.before:
            lines.append(self.before.row("before"))
        if self.after:
            lines.append(self.after.row("after"))
        if self.potential_energy is not None:
            lines.append(f"  final potential energy: "
                         f"{self.potential_energy:,.1f} kcal/mol")
        return "\n".join(lines)


# =====================================================================
def _conda_env_bin_dirs() -> List[Path]:
    """Every ``bin`` directory of every sibling conda environment.

    LAMMPS is very often installed in a *different* environment from the one
    running PAAF --- ``conda create -n lammps -c conda-forge lammps`` is the
    documented way to get it. PATH then contains only the active environment,
    so ``shutil.which`` finds nothing and the user is told LAMMPS is missing
    when it is sitting one directory away.

    So the sibling environments are searched too. ``$CONDA_PREFIX`` for an
    activated env is ``<root>/envs/<name>``, which makes ``<root>/envs`` the
    parent to scan; the usual installation roots are checked as well.
    """
    import os

    roots: List[Path] = []
    prefix = os.environ.get("CONDA_PREFIX", "")
    if prefix:
        p = Path(prefix)
        # Activated non-base env: .../envs/<name>  -> scan .../envs
        if p.parent.name == "envs":
            roots.append(p.parent)
        # Activated base: <root> -> scan <root>/envs
        roots.append(p / "envs")
    for guess in ("~/miniconda3/envs", "~/anaconda3/envs", "~/miniforge3/envs",
                  "~/mambaforge/envs", "~/opt/miniconda3/envs",
                  "/opt/homebrew/Caskroom/miniconda/base/envs",
                  "/opt/conda/envs", "/usr/local/Caskroom/miniconda/base/envs"):
        roots.append(Path(guess).expanduser())

    out: List[Path] = []
    seen: set = set()
    for root in roots:
        try:
            if not root.is_dir():
                continue
            for env in sorted(root.iterdir()):
                b = env / "bin"
                key = str(b)
                if b.is_dir() and key not in seen:
                    seen.add(key)
                    out.append(b)
        except OSError:                            # pragma: no cover
            continue
    return out


def find_lammps(explicit: str = "", search_conda_envs: bool = True
                ) -> Optional[Path]:
    """Locate a LAMMPS executable, or ``None``.

    Order: an explicit path or command name, then ``PATH``, then the ``bin``
    directory of every sibling conda environment. Distributions disagree about
    which binary name they install, so several are tried.
    """
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        found = shutil.which(explicit)
        if found:
            return Path(found)
        # An environment NAME rather than a path, e.g. "polyrapid".
        for b in _conda_env_bin_dirs():
            if b.parent.name == explicit:
                for name in _LAMMPS_NAMES:
                    cand = b / name
                    if cand.is_file():
                        return cand
        return None

    for name in _LAMMPS_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)

    if search_conda_envs:
        for b in _conda_env_bin_dirs():
            for name in _LAMMPS_NAMES:
                cand = b / name
                if cand.is_file():
                    log.info("Found LAMMPS in a sibling conda environment: %s",
                             cand)
                    return cand
    return None


# =====================================================================
_STYLE_KEYWORDS = ("pair_style", "bond_style", "angle_style", "dihedral_style",
                   "improper_style", "kspace_style", "pair_modify",
                   "special_bonds")


def extract_style_lines(init_path: Path) -> List[str]:
    """The force-field style commands from a moltemplate ``.in.init``.

    The whole file cannot simply be re-included after the soft push-off,
    because it begins with ``units real``, ``atom_style full`` and
    ``boundary p p p`` — and LAMMPS refuses those once a simulation box
    exists: *"Units command after simulation box is defined"*. Only the style
    commands may be reissued, so only those are taken.

    Returned in file order, which matters: ``pair_modify`` and
    ``special_bonds`` must follow the ``pair_style`` they modify.
    """
    try:
        text = Path(init_path).read_text()
    except OSError:
        return []
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        head = line.split()[0]
        if head in _STYLE_KEYWORDS:
            out.append(line)
    return out


def find_gromacs(explicit: str = "") -> Optional[Path]:
    """Locate ``gmx``, reusing the search the GROMACS packer already has.

    That search matters on Apple Silicon, where conda-forge installs the
    binary into a SIMD-suffixed directory (``bin.ARM_NEON_ASIMD/gmx``) that
    is not on ``PATH``.
    """
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        found = shutil.which(explicit)
        if found:
            return Path(found)
    from ..gromacs_packer import _find_gmx
    found = _find_gmx()
    return Path(found) if found else None


# =====================================================================
# GROMACS route
# =====================================================================
def write_em_mdp(out_dir: Path, settings: RelaxSettings,
                 name: str = "em.mdp") -> Path:
    """Steepest-descent minimisation parameters for a strained cell.

    Steepest descent rather than conjugate gradient, and a deliberately small
    ``emstep``. A constructed cell begins with contacts around 1 Å; conjugate
    gradient assumes a locally quadratic surface and a large first step there
    is enough to lose the structure. Steepest descent with a capped step is
    the GROMACS counterpart of the soft push-off used on the LAMMPS route.

    Constraints are off. Constraining bonds while atoms are still resolving
    an overlap is how LINCS failures appear.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join([
        "; Energy minimisation of a PAAF-constructed amorphous cell",
        "; steep, not cg: the starting structure is strained and a quadratic",
        "; assumption does not hold near a 1 A contact.",
        "integrator      = steep",
        f"emtol           = {settings.gmx_emtol}",
        f"emstep          = {settings.gmx_emstep}",
        f"nsteps          = {settings.gmx_steps}",
        "",
        "nstlist         = 10",
        "cutoff-scheme   = Verlet",
        "ns_type         = grid",
        "pbc             = xyz",
        "coulombtype     = PME",
        "rcoulomb        = 1.0",
        "rvdw            = 1.0",
        "rlist           = 1.0",
        "",
        "; No constraints while overlaps are still being resolved.",
        "constraints     = none",
        "",
    ])
    path = out_dir / name
    path.write_text(text)
    log.info("Wrote GROMACS minimisation parameters %s", path)
    return path


def read_gro_coordinates(gro_path: Path, n_atoms: int) -> Optional[np.ndarray]:
    """Coordinates from a ``.gro`` file, converted nm -> Å.

    Fixed-width fields, not whitespace: a ``.gro`` with large coordinates runs
    its columns together and splitting on spaces silently mis-parses it.
    """
    try:
        lines = Path(gro_path).read_text().splitlines()
    except OSError:
        return None
    if len(lines) < 3:
        return None
    try:
        declared = int(lines[1].strip())
    except ValueError:
        return None
    if declared != n_atoms:
        return None
    coords = np.zeros((n_atoms, 3), dtype=float)
    for k in range(n_atoms):
        line = lines[2 + k]
        try:
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])
        except (ValueError, IndexError):
            return None
        coords[k] = (x * 10.0, y * 10.0, z * 10.0)
    return coords


def relax_cell_gromacs(molecule: Molecule, box_dims: Sequence[float],
                       gro_file: Path, top_file: Path, work_dir: str | Path,
                       settings: Optional[RelaxSettings] = None,
                       total_mass_amu: Optional[float] = None,
                       progress: Optional[Callable[[str], None]] = None
                       ) -> RelaxResult:
    """Minimise with ``gmx grompp`` then ``gmx mdrun``.

    Preferred for DL_FIELD force fields: the ``.top`` DL_FIELD writes states
    every functional form explicitly, so unlike the LAMMPS route nothing has
    to be inferred about pair, bond, angle or dihedral styles.
    """
    emit = progress or (lambda _m: None)
    settings = settings or RelaxSettings()
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    res = RelaxResult(folder=work, engine="gromacs")
    res.before = measure_cell(molecule, box_dims, total_mass_amu)

    for src in (gro_file, top_file):
        src = Path(src)
        if src.is_file() and src.parent != work:
            try:
                shutil.copy2(src, work / src.name)
            except shutil.SameFileError:
                pass
    res.input_script = write_em_mdp(work, settings)

    gmx = find_gromacs(settings.gmx_exe)
    if gmx is None:
        res.messages.append(
            "No GROMACS executable found. The minimisation parameters were "
            "written but NOT run, so the cell is still the unrelaxed "
            f"construction. Run it yourself with:  cd {work} && "
            f"gmx grompp -f em.mdp -c {Path(gro_file).name} "
            f"-p {Path(top_file).name} -o em.tpr -maxwarn 3 && "
            f"gmx mdrun -deffnm em")
        emit("  GROMACS not found — parameters written, not run")
        return res

    grompp = [str(gmx), "grompp", "-f", "em.mdp",
              "-c", Path(gro_file).name, "-p", Path(top_file).name,
              "-o", "em.tpr", "-maxwarn", "3"]
    emit(f"  {' '.join(grompp[1:3])} …")
    try:
        p1 = subprocess.run(grompp, cwd=str(work), capture_output=True,
                            text=True, timeout=settings.timeout_s)
    except Exception as exc:
        res.messages.append(f"gmx grompp could not be launched: {exc}")
        return res
    if p1.returncode != 0:
        tail = "\n      ".join((p1.stderr or "").strip().splitlines()[-8:])
        res.messages.append(
            f"gmx grompp failed (rc={p1.returncode}). The cell was NOT "
            f"relaxed. Last output:\n      {tail}")
        return res

    emit("  gmx mdrun …")
    try:
        p2 = subprocess.run([str(gmx), "mdrun", "-deffnm", "em"],
                            cwd=str(work), capture_output=True, text=True,
                            timeout=settings.timeout_s)
    except Exception as exc:
        res.messages.append(f"gmx mdrun could not be launched: {exc}")
        return res

    log_text = (p1.stdout or "") + (p1.stderr or "") + \
               (p2.stdout or "") + (p2.stderr or "")
    res.log_file = work / "paaf_relax.log"
    res.log_file.write_text(log_text)
    if p2.returncode != 0:
        tail = "\n      ".join((p2.stderr or "").strip().splitlines()[-8:])
        res.messages.append(
            f"gmx mdrun failed (rc={p2.returncode}). The cell was NOT "
            f"relaxed. Last output:\n      {tail}")
        return res

    out_gro = work / "em.gro"
    coords = read_gro_coordinates(out_gro, len(molecule.atoms))
    if coords is None:
        res.messages.append(
            f"GROMACS ran but {out_gro.name} could not be read, so the "
            f"coordinates in memory are still the unrelaxed ones.")
        return res
    for k, atom in enumerate(molecule.atoms):
        atom.xyz = coords[k]
    res.dump_file = out_gro
    res.relaxed_data = out_gro
    res.ran = True

    m = re.search(r"Potential Energy\s*=\s*(-?[\d.eE+]+)", log_text)
    if m:
        try:
            res.potential_energy = float(m.group(1))
        except ValueError:
            pass
    res.after = measure_cell(molecule, box_dims, total_mass_amu)
    res.messages.append(
        "Minimisation finds a nearby local minimum — effectively a 0 K "
        "glass. It does NOT equilibrate the melt; chain dimensions relax on "
        "a diffusive timescale that only dynamics reaches. Run NVT then NPT "
        "before quoting any property.")
    emit("  relaxed")
    return res


def measure_cell(molecule: Molecule, box_dims: Sequence[float],
                 total_mass_amu: Optional[float] = None) -> CellMetrics:
    """Density, closest non-bonded contact, R_g and bond-length range."""
    from .backmap import _closest_nonbonded

    xyz = np.array([a.xyz for a in molecule.atoms], dtype=float)
    dims = np.asarray(box_dims, dtype=float)
    m = CellMetrics(n_atoms=len(xyz))
    if len(xyz) == 0:
        return m

    if total_mass_amu is None:
        from .packing import atomic_mass
        total_mass_amu = sum(atomic_mass(a.element) for a in molecule.atoms)
    volume_a3 = float(np.prod(dims))
    m.density_g_cm3 = (total_mass_amu / _N_AVOGADRO) / (volume_a3 * 1e-24)

    m.closest_contact_a = _closest_nonbonded(xyz, molecule.bonds, dims)

    com = xyz.mean(axis=0)
    d = xyz - com
    m.r_gyration_a = float(np.sqrt((d * d).sum() / len(xyz)))

    if molecule.bonds:
        lengths = []
        for i, j, _o in molecule.bonds:
            v = xyz[i] - xyz[j]
            v -= dims * np.round(v / dims)
            lengths.append(float(np.linalg.norm(v)))
        m.bond_min_a = float(min(lengths))
        m.bond_max_a = float(max(lengths))
    return m


# =====================================================================
def write_relax_input(out_dir: Path, data_file: str,
                      settings: RelaxSettings,
                      init_file: str = "",
                      settings_file: str = "",
                      charges_file: str = "",
                      restore_lines: Optional[Sequence[str]] = None,
                      styles: Optional[Dict[str, str]] = None,
                      styles_source: str = "",
                      name: str = "relax.in",
                      dump_name: str = "relaxed.lammpstrj",
                      final_data: str = "relaxed.data") -> Path:
    """Write the two-stage relaxation deck.

    ``init_file`` and ``settings_file`` are Moltemplate's ``system.in.init``
    and ``system.in.settings``. When they are given, the real force-field
    styles are taken from them and restored after the push-off. When they are
    not, the deck still runs the push-off and minimises with whatever styles
    the data file's own coefficients imply, and says so in a comment.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L: List[str] = []

    _push = settings.push_off and bool(init_file)
    L.append("# ------------------------------------------------------------")
    L.append("# Relaxation of a PAAF-constructed amorphous cell")
    L.append("#")
    if _push:
        L.append("# Stage 1  soft push-off   bounded repulsion, ramped from")
        L.append("#                          zero, capped displacement.")
        L.append("#                          Constructed cells contain")
        L.append("#                          contacts near 1 A; a 12-6")
        L.append("#                          potential there is ~3e6 x")
        L.append("#                          epsilon and blows the run up.")
        L.append("# Stage 2  minimisation    real force field, conjugate")
        L.append("#                          gradient.")
    else:
        L.append("# Displacement-capped minimisation. The soft push-off needs")
        L.append("# the pair coefficients to live outside the data file, which")
        L.append("# is the moltemplate layout; they are not available here.")
    L.append("# ------------------------------------------------------------")
    L.append("units           real")
    L.append("atom_style      full")
    L.append("boundary        p p p")
    L.append("")
    if not init_file and not styles:
        L.append("# WARNING: no force-field styles are available for this")
        L.append("# cell. LAMMPS cannot read a data file without styles for")
        L.append("# its sections, so this script will NOT run as written.")
        L.append("# Supply moltemplate's system.in.init / system.in.settings,")
        L.append("# or use the real force field via the GROMACS route.")
        L.append("")
    if init_file:
        L.append(f"include         {init_file}       # real styles")
    elif styles:
        L.append(f"# Styles supplied by PAAF ({styles_source or 'table'}).")
        L.append("# DL_FIELD's LAMMPS data file does not carry them, and a")
        L.append("# data file cannot be read without styles for its sections.")
        L.append("# CHECK THESE against your force field before trusting the")
        L.append("# energies: a wrong functional form runs and is wrong.")
        for key in ("pair_style", "bond_style", "angle_style",
                    "dihedral_style", "improper_style", "special_bonds"):
            if styles.get(key):
                L.append(f"{key:<15s} {styles[key]}")
    L.append(f"read_data       {data_file}")
    if styles and not init_file and styles.get("kspace_style"):
        # kspace must come after read_data, when charges exist.
        L.append(f"kspace_style    {styles['kspace_style']}")
    if settings_file:
        L.append(f"include         {settings_file}   # real coefficients")
    if charges_file:
        # Moltemplate keeps partial charges in their own file. Omitting it
        # leaves every charge at zero, which runs and is wrong.
        L.append(f"include         {charges_file}   # partial charges")
    if not settings_file:
        L.append("# No .in.settings supplied: coefficients must already be in")
        L.append("# the data file.")
    L.append("")
    L.append("neighbor        2.0 bin")
    L.append("neigh_modify    every 1 delay 0 check yes one 10000 page 1000000")
    L.append(f"timestep        {settings.timestep_fs}")
    L.append("thermo          500")
    L.append("thermo_style    custom step temp press density pe evdwl ebond")
    L.append("")

    # The soft push-off requires switching pair_style and switching back, and
    # switching back means re-applying every pair coefficient. That is only
    # possible when the coefficients live OUTSIDE the data file, which is the
    # moltemplate layout (system.in.settings). DL_FIELD writes them inside the
    # data file, where they cannot be recovered after a style change — so on
    # that route the push-off is replaced by a displacement-capped
    # minimisation, which is weaker but correct.
    can_push = settings.push_off and bool(init_file)
    capped_only = settings.push_off and not init_file

    if capped_only:
        L.append("# ---- Displacement-capped minimisation (no style switch) ----")
        L.append("# The pair coefficients are inside the data file, so the")
        L.append("# soft push-off cannot be used: switching pair_style would")
        L.append("# discard them with no way to restore them. Instead the")
        L.append("# minimiser's per-iteration step is capped, which survives")
        L.append("# close contacts without changing the potential.")
        L.append("# For DL_FIELD force fields the GROMACS route avoids this")
        L.append("# entirely — its .top states every functional form.")
        L.append("min_modify      dmax 0.01")
        L.append("")

    if can_push:
        L.append("# ---------------- Stage 1: soft push-off ----------------")
        L.append("# soft:  E = A [1 + cos(pi r / rc)]  -- finite at r = 0,")
        L.append("# which is the whole point. A is ramped 0 -> A_max so atoms")
        L.append("# that start on top of one another separate gently.")
        L.append(f"pair_style      soft {settings.soft_cutoff}")
        # The prefactor MUST be given explicitly and start at zero; fix adapt
        # then ramps it. Omitting it leaves the coefficient unset.
        L.append("pair_coeff      * * 0.0")
        L.append(f"variable        prefactor equal ramp(0,{settings.soft_prefactor})")
        L.append("fix             push all adapt 1 pair soft a * * v_prefactor")
        L.append(f"fix             lim all nve/limit {settings.nve_limit}")
        L.append("velocity        all create 300.0 4928459 mom yes rot yes dist gaussian")
        L.append(f"run             {settings.push_steps}")
        L.append("unfix           push")
        L.append("unfix           lim")
        L.append("reset_timestep  0")
        L.append("")
        if init_file:
            L.append("# Restore the real force field before minimising.")
            L.append("# Changing pair_style clears every pair coefficient, so")
            L.append("# the coefficients have to be re-applied. Moltemplate")
            L.append("# keeps them in their own file, which is what makes")
            L.append("# this possible.")
            L.append("#")
            L.append("# Only the STYLE commands are reissued, not the whole")
            L.append("# .in.init: that file also carries units, atom_style")
            L.append("# and boundary, and LAMMPS rejects those once a box")
            L.append("# exists (\"Units command after simulation box is")
            L.append("# defined\").")
            if restore_lines:
                for line in restore_lines:
                    L.append(line)
            else:
                L.append("# NOTE: the init file could not be read, so the")
                L.append("# styles below are unknown. Check this deck.")
            if settings_file:
                L.append(f"include         {settings_file}")
            if charges_file:
                L.append(f"include         {charges_file}")
            L.append("")
        elif not styles:
            L.append("# WARNING: no force-field styles are available, so the")
            L.append("# soft pair style would still be active at minimisation")
            L.append("# and the result would be meaningless. Supply the")
            L.append("# moltemplate init/settings files, or use GROMACS.")
            L.append("")

    if settings.minimize:
        L.append("# ---------------- Stage 2: minimisation ----------------")
        L.append("min_style       cg")
        L.append(f"minimize        {settings.etol} {settings.ftol} "
                 f"{settings.maxiter} {settings.maxeval}")
        L.append("")

    L.append("# Final state, written both as a trajectory frame and as data.")
    L.append(f"write_dump      all custom {dump_name} id type x y z modify sort id")
    L.append(f"write_data      {final_data} nocoeff")
    L.append("")
    L.append("# The final energy is read from the thermo output rather than")
    L.append("# printed with $(pe). An immediate variable forces an energy")
    L.append("# evaluation, which fails outright if no run or minimisation")
    L.append("# actually happened — as under -skiprun, where it turns a")
    L.append("# successful syntax check into a spurious error.")

    path = out_dir / name
    path.write_text("\n".join(L) + "\n")
    log.info("Wrote LAMMPS relaxation deck %s", path)
    return path


# =====================================================================
def read_dump_coordinates(dump_path: Path, n_atoms: int) -> Optional[np.ndarray]:
    """Read the last frame of a LAMMPS custom dump as an (N, 3) array.

    The dump is written with ``modify sort id`` so the row order matches the
    data file's atom ordering, which is what makes the coordinates droppable
    straight back into the PAAF molecule.
    """
    try:
        text = Path(dump_path).read_text()
    except OSError:
        return None
    blocks = text.split("ITEM: TIMESTEP")
    if len(blocks) < 2:
        return None
    last = blocks[-1]
    m = re.search(r"ITEM: ATOMS ([^\n]*)\n", last)
    if not m:
        return None
    columns = m.group(1).split()
    try:
        ix, iy, iz = columns.index("x"), columns.index("y"), columns.index("z")
    except ValueError:
        return None
    body = last[m.end():].strip().splitlines()
    coords = np.zeros((n_atoms, 3), dtype=float)
    seen = 0
    for line in body[:n_atoms]:
        parts = line.split()
        if len(parts) <= max(ix, iy, iz):
            continue
        coords[seen] = (float(parts[ix]), float(parts[iy]), float(parts[iz]))
        seen += 1
    if seen != n_atoms:
        return None
    return coords


def _parse_log(log_text: str) -> Tuple[Optional[float], Optional[float]]:
    """Final potential energy and density, read from the thermo output.

    Not from ``print "$(pe)"``. An immediate variable forces LAMMPS to
    evaluate the energy there and then, which fails outright if no run or
    minimisation actually took place — including under ``-skiprun``, where it
    turned a clean syntax check into ``ERROR: Energy was not tallied on
    needed timestep``. Reading the thermo table has no such precondition.

    The table is located by its header, so the column order in
    ``thermo_style`` can change without breaking this.
    """
    pe = dens = None
    header_re = re.compile(r"^\s*Step\s+.*PotEng.*$", re.MULTILINE)
    m = None
    for m in header_re.finditer(log_text):
        pass                                  # keep the LAST thermo block
    if m is None:
        return None, None
    columns = m.group(0).split()
    try:
        i_pe = columns.index("PotEng")
    except ValueError:
        return None, None
    i_dens = columns.index("Density") if "Density" in columns else None

    for line in log_text[m.end():].splitlines():
        parts = line.split()
        if len(parts) != len(columns):
            if pe is not None:
                break                         # table finished
            continue
        try:
            values = [float(x) for x in parts]
        except ValueError:
            if pe is not None:
                break
            continue
        pe = values[i_pe]
        if i_dens is not None:
            dens = values[i_dens]
    return pe, dens


# =====================================================================
def relax_cell(molecule: Molecule, box_dims: Sequence[float],
               data_file: Path, work_dir: str | Path,
               settings: Optional[RelaxSettings] = None,
               init_file: str = "", settings_file: str = "",
               charges_file: str = "",
               styles: Optional[Dict[str, str]] = None,
               styles_source: str = "",
               total_mass_amu: Optional[float] = None,
               progress: Optional[Callable[[str], None]] = None) -> RelaxResult:
    """Push off, minimise, and read the relaxed coordinates back.

    ``molecule`` is updated **in place** with the relaxed coordinates when the
    run succeeds. When LAMMPS cannot be found the deck is still written and
    the result says clearly that nothing was run --- a cell that was not
    relaxed must never be mistaken for one that was.
    """
    emit = progress or (lambda _m: None)
    settings = settings or RelaxSettings()
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    res = RelaxResult(folder=work, engine="lammps")

    res.before = measure_cell(molecule, box_dims, total_mass_amu)

    data_file = Path(data_file)
    if data_file.parent != work:
        try:
            shutil.copy2(data_file, work / data_file.name)
        except shutil.SameFileError:
            pass
    restore = extract_style_lines(work / init_file) if init_file else []
    if init_file and not restore:
        res.messages.append(
            f"Could not read any style commands out of {init_file}. The "
            f"push-off would leave the soft potential in place, so it has "
            f"been disabled and a displacement-capped minimisation used "
            f"instead.")
        settings = replace(settings, push_off=False)

    res.input_script = write_relax_input(
        work, data_file.name, settings, init_file=init_file,
        settings_file=settings_file, charges_file=charges_file,
        restore_lines=restore, styles=styles, styles_source=styles_source)

    # Without styles LAMMPS will read the data file, warn that no bond,
    # angle or dihedral style is set, and minimise with NO FORCE FIELD AT
    # ALL. It exits zero, so the run looks successful. Refuse instead: a
    # meaningless minimisation reported as a real one is the worst outcome
    # available here.
    if not init_file and not styles:
        res.messages.append(
            "Refusing to run: no force-field styles are available for this "
            "cell, so LAMMPS would minimise it with no bonded or non-bonded "
            "interactions and still exit successfully. The input was written "
            "for inspection. This normally means moltemplate did not produce "
            "its .in.init / .in.settings files, or a DL_FIELD force field was "
            "used for which PAAF has no styles table — in which case relax "
            "with GROMACS instead.")
        emit("  refusing to run: no force field in the deck")
        return res
    if styles and not init_file:
        res.messages.append(
            f"LAMMPS styles were supplied by PAAF ({styles_source or 'table'}) "
            f"because DL_FIELD's data file does not carry them. Check them "
            f"against your force field before quoting energies.")

    exe = find_lammps(settings.lammps_exe)
    if exe is None:
        res.messages.append(
            "No LAMMPS executable found (looked for: "
            + ", ".join(_LAMMPS_NAMES) +
            "). The relaxation input was written but NOT run, so the cell is "
            "still the unrelaxed construction. Run it yourself with:  "
            f"cd {work} && lmp -in {res.input_script.name}")
        emit("  LAMMPS not found — relaxation deck written, not run")
        return res

    cmd = [str(exe), "-in", res.input_script.name]
    if settings.mpi_ranks > 1 and shutil.which("mpirun"):
        cmd = ["mpirun", "-np", str(settings.mpi_ranks)] + cmd
    emit(f"  running {' '.join(cmd)} …")
    try:
        proc = subprocess.run(cmd, cwd=str(work), capture_output=True,
                              text=True, timeout=settings.timeout_s)
    except subprocess.TimeoutExpired:
        res.messages.append(
            f"LAMMPS exceeded the {settings.timeout_s} s timeout. The cell "
            f"was not relaxed.")
        return res
    except Exception as exc:
        res.messages.append(f"LAMMPS could not be launched: {exc}")
        return res

    log_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    res.log_file = work / "paaf_relax.log"
    res.log_file.write_text(log_text)

    if proc.returncode != 0:
        tail = "\n      ".join(log_text.strip().splitlines()[-8:])
        res.messages.append(
            f"LAMMPS exited with code {proc.returncode}. The cell was NOT "
            f"relaxed. Last output:\n      {tail}")
        emit(f"  LAMMPS failed (rc={proc.returncode})")
        return res

    dump = work / "relaxed.lammpstrj"
    coords = read_dump_coordinates(dump, len(molecule.atoms))
    if coords is None:
        res.messages.append(
            "LAMMPS ran but its dump could not be read, so the coordinates "
            "in memory are still the unrelaxed ones. The relaxed structure is "
            f"on disk at {dump}.")
        emit("  ran, but the dump could not be parsed")
        return res

    for k, atom in enumerate(molecule.atoms):
        atom.xyz = coords[k]
    res.dump_file = dump
    final_data = work / "relaxed.data"
    res.relaxed_data = final_data if final_data.exists() else None
    res.ran = True

    pe, _dens = _parse_log(log_text)
    res.potential_energy = pe
    res.after = measure_cell(molecule, box_dims, total_mass_amu)

    res.messages.append(
        "Minimisation finds a nearby local minimum — effectively a 0 K glass. "
        "It removes strained contacts and makes the cell safe to run. It does "
        "NOT equilibrate the melt: chain dimensions relax on a diffusive "
        "timescale that only molecular dynamics reaches, so the construction's "
        "chain over-extension will still be present. Run NVT then NPT before "
        "quoting any property.")
    emit("  relaxed")
    return res
