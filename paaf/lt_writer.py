"""Emit Moltemplate .lt files for monomers, polymer chains, and simulation boxes."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .ff_registry import ForceField, get_ff
from .logging_utils import get_logger
from .monomer import Monomer
from .structure import Atom, Molecule

log = get_logger(__name__)


# ---------------------------------------------------------------- helpers
def _atom_line(a: Atom, mol_prefix: str) -> str:
    """Emit one Moltemplate Data-Atoms line for atom `a`.

    Defensive: ``ff_type`` MUST already be a proper FF type (numeric or
    symbolic) coming from the typer. If it's None, empty, or degenerates
    to just an element symbol like "H" or "C", we log a warning and
    substitute a safe generic OPLS numeric so moltemplate's sorter never
    hits a mixed int/str atom-type column (it crashes with a TypeError).
    """
    ft = a.ff_type
    if not ft or (isinstance(ft, str) and ft.strip() == ""):
        ft = "135"
    ft = str(ft)
    # An element-symbol type would mix strings and ints in the LAMMPS data
    # column that moltemplate's renumber_DATA_first_column.py sorts,
    # crashing it with a TypeError. Coerce to a safe numeric fallback.
    if ft in {"H", "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si"}:
        fallback = {
            "H": "140", "C": "135", "N": "739", "O": "154", "S": "202",
            "F": "719", "Cl": "264", "Br": "722", "I": "725",
            "P": "440", "Si": "500",
        }[ft]
        log.warning("atom %s had element-only type %r; substituting numeric %s",
                    a.name, ft, fallback)
        ft = fallback
    return (
        f"    $atom:{a.name} $mol:. @atom:{ft} "
        f"{a.charge:.6f} {a.xyz[0]:.6f} {a.xyz[1]:.6f} {a.xyz[2]:.6f}"
    )


def _bond_line(idx: int, i: int, j: int, atoms: Sequence[Atom]) -> str:
    return f"    $bond:b{idx} $atom:{atoms[i].name} $atom:{atoms[j].name}"


# --------------------------------------------------------- monomer -> .lt
def write_monomer_lt(
    monomer: Monomer,
    out_dir: str | Path,
    ff: ForceField,
    include_head_tail: bool = True,
) -> Path:
    """Write ``<monomer.name>.lt`` importing the requested FF.

    The monomer inherits from ``ff.inherit`` so Moltemplate can look up
    bonded parameters automatically.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{monomer.name}.lt"
    m = monomer.molecule

    lines: List[str] = []
    if ff.lt_include:
        lines.append(f'import "{ff.lt_include}"')
    parent = f" inherits {ff.inherit}" if ff.inherit else ""
    lines.append("")
    lines.append(f"{monomer.name}{parent} {{")
    lines.append("")
    lines.append("    # atom-id  mol-id  atom-type  charge  x  y  z")
    lines.append("    write('Data Atoms') {")
    for a in m.atoms:
        lines.append(_atom_line(a, monomer.name))
    lines.append("    }")
    lines.append("")
    lines.append("    write('Data Bond List') {")
    for k, (i, j, _) in enumerate(m.bonds):
        lines.append(_bond_line(k, i, j, m.atoms))
    lines.append("    }")
    if include_head_tail:
        head_name = m.atoms[monomer.head_index].name
        tail_name = m.atoms[monomer.tail_index].name
        lines.append("")
        lines.append(f"    # exposed connection atoms")
        lines.append(f"    write('In Settings') {{}}   # placeholder")
        lines.append(f"    # $atom:{head_name} = polymer head")
        lines.append(f"    # $atom:{tail_name} = polymer tail")
    lines.append(f"}} # {monomer.name}")
    fname.write_text("\n".join(lines) + "\n")
    log.info("Wrote monomer .lt %s", fname)
    return fname


# --------------------------------------------------------- chain -> .lt
def write_chain_lt(
    chain: Molecule,
    out_dir: str | Path,
    ff: ForceField,
    name: str = "polymer",
) -> Path:
    """Write a full-chain .lt (single self-contained object).

    This route is simplest when we have already assembled the chain in
    Python and just need Moltemplate to look up bonded parameters + write
    the LAMMPS data file. Alternative route (compose from monomer .lt using
    Moltemplate's polymerize) is left for a future release.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{name}.lt"

    lines: List[str] = []
    if ff.lt_include:
        lines.append(f'import "{ff.lt_include}"')
    parent = f" inherits {ff.inherit}" if ff.inherit else ""
    lines.append("")
    lines.append(f"{name}{parent} {{")
    lines.append("")
    lines.append("    write('Data Atoms') {")
    for a in chain.atoms:
        lines.append(_atom_line(a, name))
    lines.append("    }")
    lines.append("")
    lines.append("    write('Data Bond List') {")
    for k, (i, j, _) in enumerate(chain.bonds):
        lines.append(_bond_line(k, i, j, chain.atoms))
    lines.append("    }")
    lines.append(f"}} # {name}")
    fname.write_text("\n".join(lines) + "\n")
    # Log the distinct atom types actually written (small sample for large chains)
    unique = sorted({(a.ff_type or "?") for a in chain.atoms})
    log.info("Wrote chain .lt %s  (%d atoms, %d distinct FF types: %s)",
             fname, len(chain.atoms), len(unique), unique[:20])
    return fname


# --------------------------------------------------------- system .lt (box)
def write_system_lt(
    out_dir: str | Path,
    chain_lt: str | Path,
    n_chains: int = 1,
    box: Sequence[float] = (50.0, 50.0, 50.0),
    name: str = "system",
    ff: Optional[ForceField] = None,
) -> Path:
    out_dir = Path(out_dir)
    fname = out_dir / f"{name}.lt"
    chain_stem = Path(chain_lt).stem

    lines: List[str] = []
    lines.append(f'import "{Path(chain_lt).name}"')
    lines.append("")
    lines.append(f"chains = new {chain_stem}[{n_chains}].move(20.0, 0, 0)")
    lines.append("")
    lines.append("write_once('Data Boundary') {")
    lx, ly, lz = box
    lines.append(f"    0.0 {lx:.4f} xlo xhi")
    lines.append(f"    0.0 {ly:.4f} ylo yhi")
    lines.append(f"    0.0 {lz:.4f} zlo zhi")
    lines.append("}")
    fname.write_text("\n".join(lines) + "\n")
    log.info("Wrote system .lt %s", fname)
    return fname


# --------------------------------------------------------- LAMMPS input
def _fix_line(ensemble: str, tag: str, T: float, tdamp: float, P: float, pdamp: float,
              barostat: str, coupling: str, thermostat: str, seed: int) -> str:
    """Compose a LAMMPS ``fix`` line for the requested ensemble."""
    e = ensemble.lower()
    if e == "nve":
        return f"fix {tag} all nve"
    if e == "nvt":
        if thermostat == "langevin":
            return (f"fix {tag}_nve all nve\n"
                    f"fix {tag} all langevin {T} {T} {tdamp} {seed}")
        if thermostat == "berendsen":
            return (f"fix {tag}_nve all nve\n"
                    f"fix {tag} all temp/berendsen {T} {T} {tdamp}")
        if thermostat == "csvr":
            return (f"fix {tag}_nve all nve\n"
                    f"fix {tag} all temp/csvr {T} {T} {tdamp} {seed}")
        # default: Nose-Hoover
        return f"fix {tag} all nvt temp {T} {T} {tdamp}"
    if e == "npt":
        # Nose-Hoover NPT is the standard; barostat coupling is set by iso/aniso/tri
        return (f"fix {tag} all npt temp {T} {T} {tdamp} "
                f"{coupling} {P} {P} {pdamp}")
    raise ValueError(f"Unknown ensemble {ensemble!r}")


def write_lammps_input(
    out_dir: str | Path,
    data_file: str = "system.data",
    settings_file: str = "system.in.settings",
    name: str = "run.in",
    ensemble: str = "npt",
    temperature: float = 300.0,
    pressure: float = 1.0,
    steps: int = 500000,
    timestep: float = 1.0,
    tdamp_fs: float = 100.0,
    pdamp_fs: float = 1000.0,
    thermostat: str = "nose-hoover",
    barostat: str = "nose-hoover",
    pressure_coupling: str = "iso",
    minimize: bool = True,
    min_etol: float = 1e-4,
    min_ftol: float = 1e-6,
    min_maxiter: int = 10000,
    min_maxeval: int = 100000,
    multistage: bool = False,
    nvt_steps: int = 100000,
    nvt_temperature: float | None = None,
    npt_steps: int = 500000,
    thermo_every: int = 1000,
    dump_every: int = 0,
    seed: int = 4928459,
) -> Path:
    """Write a LAMMPS input script.

    Two modes:
      * Single-stage (default): minimize -> velocity init -> chosen `ensemble`.
      * Multi-stage (``multistage=True``): minimize -> NVT eq -> NPT prod.
    """
    out_dir = Path(out_dir)
    fname = out_dir / name
    Tnvt = nvt_temperature if nvt_temperature is not None else temperature

    lines: list[str] = []
    lines.append("# LAMMPS input generated by paaf")
    lines.append("# damping: tdamp={} fs, pdamp={} fs".format(tdamp_fs, pdamp_fs))
    lines.append("# thermostat={}, barostat={}, coupling={}".format(
        thermostat, barostat, pressure_coupling))
    lines.append("units           real")
    lines.append("atom_style      full")
    lines.append("boundary        p p p")
    lines.append("")
    lines.append(f"read_data       {data_file}")
    lines.append(f"include         {settings_file}")
    lines.append("")
    lines.append("neighbor        2.0 bin")
    lines.append("neigh_modify    every 1 delay 0 check yes one 10000 page 1000000")
    lines.append("")
    lines.append(f"timestep        {timestep}")
    lines.append(f"thermo          {thermo_every}")
    lines.append("thermo_style    custom step temp press density pe ke etotal ebond eangle edihed eimp evdwl ecoul elong")
    lines.append("")

    # Optional trajectory dump
    if dump_every > 0:
        lines.append(f"dump            traj all atom {dump_every} traj.lammpstrj")
        lines.append("")

    # 1) Minimization
    if minimize:
        lines.append("# ---- Stage 0: energy minimization ----")
        lines.append("min_style       cg")
        lines.append(f"minimize        {min_etol} {min_ftol} {min_maxiter} {min_maxeval}")
        lines.append("reset_timestep  0")
        lines.append("")

    lines.append(f"velocity        all create {temperature} {seed} mom yes rot yes dist gaussian")
    lines.append("")

    if multistage:
        # 2) NVT equilibration
        lines.append(f"# ---- Stage 1: NVT equilibration @ {Tnvt} K ({nvt_steps} steps) ----")
        fix_nvt = _fix_line("nvt", "EQ", Tnvt, tdamp_fs, pressure, pdamp_fs,
                            barostat, pressure_coupling, thermostat, seed)
        lines.append(fix_nvt)
        lines.append(f"run             {nvt_steps}")
        # Unfix all NVT-related tags
        for tag in ("EQ", "EQ_nve"):
            lines.append(f"unfix           {tag}   # (harmless if not defined)")
        lines.append("reset_timestep  0")
        lines.append("")

        # 3) NPT production
        lines.append(f"# ---- Stage 2: NPT production @ {temperature} K / {pressure} atm ({npt_steps} steps) ----")
        fix_npt = _fix_line("npt", "PROD", temperature, tdamp_fs, pressure, pdamp_fs,
                            barostat, pressure_coupling, thermostat, seed)
        lines.append(fix_npt)
        lines.append(f"run             {npt_steps}")
        lines.append("unfix           PROD")
    else:
        lines.append(f"# ---- {ensemble.upper()} run @ {temperature} K ({steps} steps) ----")
        fix_line = _fix_line(ensemble, "RELAX", temperature, tdamp_fs, pressure, pdamp_fs,
                             barostat, pressure_coupling, thermostat, seed)
        lines.append(fix_line)
        lines.append(f"run             {steps}")
        # Unfix everything defensively
        for tag in ("RELAX", "RELAX_nve"):
            lines.append(f"unfix           {tag}   # (harmless if not defined)")

    lines.append("")
    lines.append("write_data      final.data pair ij")

    fname.write_text("\n".join(lines) + "\n")
    log.info("Wrote LAMMPS input %s (%s, tdamp=%g, pdamp=%g)",
             fname, "multistage" if multistage else ensemble, tdamp_fs, pdamp_fs)
    return fname
