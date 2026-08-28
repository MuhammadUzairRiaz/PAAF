"""Kremer-Grest coarse-grained model of a grown bead cell.

The Amorphous grower already produces a bead-per-skeletal-atom cell; this
module turns it into a RUNNABLE Kremer-Grest (KG) bead-spring system:

* **Bonds** — FENE springs, k = 30 eps/sigma^2, R0 = 1.5 sigma (the
  canonical parameters; together with the repulsion they make bonds
  uncrossable, so entanglements are real).
* **Pairs** — WCA: Lennard-Jones cut & shifted at 2^(1/6) sigma
  (purely repulsive, melt regime).
* **Angles (optional)** — LAMMPS ``angle_style cosine``, E = K (1 + cos
  theta), minimised at the straight chain. K can be FITTED so the CG
  chain reproduces a target characteristic ratio C_inf: for independent
  angles the bond-vector correlation is the Langevin function
  L(K/eps) = coth(K/eps) - eps/K, and C_inf = (1 + <b.b>)/(1 - <b.b>),
  which inverts numerically in a few bisection steps.

Chemistry enters through three derived numbers per species:

* bead mass  — repeat-unit mass / beads per unit (known from the SMILES),
* sigma      — from the KG melt condition: bead number density 0.85/sigma^3
               at the cell's own density,
* eps        — k_B T at the build temperature (real units) or 1 (reduced).

Two unit systems: ``"real"`` (A, kcal/mol, fs — runs alongside the
all-atom outputs) and ``"lj"`` (classic reduced units, sigma = eps =
m = 1 for the FIRST species; the literature's convention).

Two mappings: ``"backbone"`` (one bead per skeletal atom — the grower's
native resolution) and ``"monomer"`` (one bead per repeat unit, centre of
mass of its skeletal atoms — the classic CG picture).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["CGSettings", "build_cg_cell", "fit_angle_k", "langevin"]

_KB_KCAL = 0.0019872041          # kcal/mol/K
_NA = 6.02214076e23


# ================================================================ settings
@dataclass
class CGSettings:
    units: str = "real"            # "real" | "lj"
    mapping: str = "backbone"      # "backbone" | "monomer"
    angle_mode: str = "auto"       # "auto" (fit to C_inf) | "manual" | "off"
    angle_k: float = 1.5           # manual K, in eps units
    target_cn: float = 0.0         # C_inf for the auto fit (0 = use measured)
    temperature_k: float = 413.0


# ================================================================ physics
def langevin(a: float) -> float:
    """L(a) = coth(a) - 1/a, numerically safe near 0."""
    if a < 1e-6:
        return a / 3.0
    return 1.0 / math.tanh(a) - 1.0 / a


def fit_angle_k(target_cn: float) -> float:
    """K/eps for ``angle_style cosine`` that reproduces ``target_cn``.

    C_inf = (1 + L)/(1 - L) with L = <b_i . b_i+1> = L(K/eps) — invert the
    Langevin function by bisection. C_inf <= 1 means a freely jointed
    chain or floppier: no angle term can help, returns 0.
    """
    if target_cn <= 1.0:
        return 0.0
    target_l = (target_cn - 1.0) / (target_cn + 1.0)
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if langevin(mid) < target_l:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ================================================================ mapping
def _bead_groups(specs: Sequence, mapping: str
                 ) -> Tuple[List[int], List[int], List[float], List[str],
                            List[List[int]]]:
    """Per-OUTPUT-bead species type, molecule id, mass, and source indices.

    The grower lays beads out chain after chain, species after species,
    each chain contiguous — reproduced here from the specs, exactly as
    ``write_bead_lammps_data`` does.
    """
    from .grow import backbone_atoms_per_unit, repeat_unit_mass

    type_of: List[int] = []
    mol_of: List[int] = []
    mass_of_type: List[float] = []
    labels: List[str] = []
    groups: List[List[int]] = []      # grower-bead indices per output bead
    cursor = 0
    chain_id = 0
    for si, sp in enumerate(specs):
        bpu = (sp.backbone_atoms if getattr(sp, "backbone_atoms", 0)
               else backbone_atoms_per_unit(sp.repeat_unit)) or 1
        unit_mass = (sp.mass_amu if getattr(sp, "mass_amu", 0.0)
                     else repeat_unit_mass(sp.repeat_unit))
        labels.append(sp.name or sp.repeat_unit or f"species{si + 1}")
        if mapping == "monomer":
            mass_of_type.append(float(unit_mass))
        else:
            mass_of_type.append(float(unit_mass) / bpu)
        dp = int(sp.degree_of_polymerisation)
        for _c in range(int(sp.n_chains)):
            chain_id += 1
            if mapping == "monomer":
                for u in range(dp):
                    start = cursor + u * bpu
                    groups.append(list(range(start, start + bpu)))
                    type_of.append(si + 1)
                    mol_of.append(chain_id)
            else:
                for k in range(dp * bpu):
                    groups.append([cursor + k])
                    type_of.append(si + 1)
                    mol_of.append(chain_id)
            cursor += dp * bpu
    return type_of, mol_of, mass_of_type, labels, groups


def _group_positions(xyz: np.ndarray, groups: List[List[int]],
                     dims: np.ndarray) -> np.ndarray:
    """Centre of mass of each group, minimum-imaged within the group.

    Chains may cross the periodic boundary; averaging raw coordinates of a
    split unit would put its bead in the middle of the box. Each member is
    imaged next to the group's first bead before averaging.
    """
    out = np.empty((len(groups), 3), dtype=float)
    for g, idx in enumerate(groups):
        base = xyz[idx[0]]
        acc = np.zeros(3)
        for i in idx:
            d = xyz[i] - base
            d -= dims * np.round(d / dims)
            acc += base + d
        out[g] = acc / len(idx)
    return out


# ================================================================ builder
def build_cg_cell(
    result,
    specs: Sequence,
    out_dir: str | Path,
    *,
    name: str = "cell",
    settings: Optional[CGSettings] = None,
    measured_cn: float = 0.0,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[Path, Path]:
    """Write ``cell_cg.data`` + ``cell_cg.in`` for the grown cell.

    Returns the two paths. Everything is derived — no lookup tables.
    """
    st = settings or CGSettings()
    emit = progress or (lambda _m: None)
    folder = Path(out_dir) / name
    folder.mkdir(parents=True, exist_ok=True)

    dims = np.asarray(result.box.bounding_box(), dtype=float)
    xyz = np.array([a.xyz for a in result.molecule.atoms], dtype=float)

    type_of, mol_of, mass_of_type, labels, groups = _bead_groups(
        specs, st.mapping)
    pos = _group_positions(xyz, groups, dims)
    pos -= np.floor(pos / dims) * dims                 # wrap into the box
    n_types = len(mass_of_type)

    # ---- sigma per species: KG melt condition, bead density 0.85/sigma^3
    volume = float(np.prod(dims))                      # A^3
    n_beads = len(groups)
    number_density = n_beads / volume                  # beads / A^3
    sigma_global = (0.85 / number_density) ** (1.0 / 3.0)
    # Species share the packing but scale with (bead mass)^(1/3), so a
    # styrene-heavy bead is fatter than a CH2 bead.
    mean_mass = float(np.mean([mass_of_type[t - 1] for t in type_of]))
    sigma = [sigma_global * (m / mean_mass) ** (1.0 / 3.0)
             for m in mass_of_type]
    eps_real = _KB_KCAL * float(st.temperature_k)      # kcal/mol

    # ---- unit system
    if st.units == "lj":
        s0, m0, e0 = sigma[0], mass_of_type[0], 1.0
        length = lambda x: x / s0
        eps_of = [1.0] * n_types
        masses = [m / m0 for m in mass_of_type]
        sig_out = [s / s0 for s in sigma]
        units_line, ts = "lj", 0.005
    else:
        length = lambda x: x
        eps_of = [eps_real] * n_types
        masses = list(mass_of_type)
        sig_out = list(sigma)
        units_line, ts = "real", 4.0    # fs; conservative for soft beads

    # ---- angle stiffness
    cn = float(st.target_cn) or float(measured_cn) or 0.0
    if st.angle_mode == "off":
        k_over_eps = 0.0
    elif st.angle_mode == "manual":
        k_over_eps = max(0.0, float(st.angle_k))
    else:
        k_over_eps = fit_angle_k(cn) if cn > 1.0 else 0.0
        if cn > 1.0:
            emit(f"  angle stiffness fitted: K = {k_over_eps:.3f} eps "
                 f"for C_inf = {cn:.2f}")
        else:
            emit("  no usable C_inf; angle term off "
                 "(pass one, or set it manually)")
    use_angles = k_over_eps > 0.0

    # ---- topology: consecutive beads within each molecule
    bonds: List[Tuple[int, int, int]] = []             # type, i, j (1-based)
    angles: List[Tuple[int, int, int, int]] = []
    for i in range(1, n_beads):
        if mol_of[i] == mol_of[i - 1]:
            bt = type_of[i] if type_of[i] == type_of[i - 1] else 1
            bonds.append((bt, i, i + 1))
    if use_angles:
        for i in range(2, n_beads):
            if mol_of[i] == mol_of[i - 1] == mol_of[i - 2]:
                angles.append((type_of[i - 1], i - 1, i, i + 1))

    # ---- data file ---------------------------------------------------
    L = [f"PAAF Kremer-Grest CG cell ({st.mapping} mapping, "
         f"{units_line} units)", "",
         f"{n_beads} atoms", f"{len(bonds)} bonds"]
    if use_angles:
        L.append(f"{len(angles)} angles")
    L += ["", f"{n_types} atom types", f"{n_types} bond types"]
    if use_angles:
        L.append(f"{n_types} angle types")
    L += ["",
          f"0.0 {length(dims[0]):.6f} xlo xhi",
          f"0.0 {length(dims[1]):.6f} ylo yhi",
          f"0.0 {length(dims[2]):.6f} zlo zhi",
          "", "Masses", ""]
    for t in range(n_types):
        # Always state the real mass: in reduced units the number is
        # scaled (reference bead = 1) and users rightly ask where the
        # chemistry went.
        L.append(f"{t + 1} {masses[t]:.4f}  # {labels[t]} "
                 f"({mass_of_type[t]:.2f} amu/bead, "
                 f"{'1 repeat unit' if st.mapping == 'monomer' else '1 skeletal atom'}/bead)")
    L += ["", "Atoms # molecular", ""]
    for i in range(n_beads):
        p = pos[i]
        L.append(f"{i + 1} {mol_of[i]} {type_of[i]} "
                 f"{length(p[0]):.6f} {length(p[1]):.6f} {length(p[2]):.6f}")
    L += ["", "Bonds", ""]
    for b, (bt, i, j) in enumerate(bonds):
        L.append(f"{b + 1} {bt} {i} {j}")
    if use_angles:
        L += ["", "Angles", ""]
        for a, (at, i, j, k) in enumerate(angles):
            L.append(f"{a + 1} {at} {i} {j} {k}")
    data = folder / "cell_cg.data"
    data.write_text("\n".join(L) + "\n")

    # ---- input script ------------------------------------------------
    wca = [2.0 ** (1.0 / 6.0) * s for s in sig_out]
    S = [f"# PAAF Kremer-Grest CG cell — {st.mapping} mapping, "
         f"{units_line} units",
         f"# sigma/species: " + ", ".join(
             f"{labels[t]}={sig_out[t]:.3f}" for t in range(n_types)),
         f"units           {units_line}",
         "atom_style      molecular",
         "boundary        p p p",
         f"read_data       {data.name}",
         "",
         "# ---- stage 1: soft push-off (constructed cells have overlaps)",
         "pair_style      soft " + f"{max(wca):.4f}",
         "pair_coeff      * * 0.0",
         "variable        pf equal ramp(1.0,"
         + ("60.0" if st.units == "lj" else f"{60.0 * eps_real:.2f}") + ")",
         "fix             PUSH all adapt 1 pair soft a * * v_pf",
         "special_bonds   fene",
         "bond_style      fene",]
    for t in range(n_types):
        k_fene = 30.0 * eps_of[t] / (sig_out[t] ** 2)
        S.append(f"bond_coeff      {t + 1} {k_fene:.4f} "
                 f"{1.5 * sig_out[t]:.4f} {eps_of[t]:.4f} "
                 f"{sig_out[t]:.4f}")
    if use_angles:
        S.append("angle_style     cosine")
        for t in range(n_types):
            S.append(f"angle_coeff     {t + 1} "
                     f"{k_over_eps * eps_of[t]:.4f}")
    S += [f"velocity        all create "
          + ("1.0" if st.units == "lj" else f"{st.temperature_k:.1f}")
          + " 4928459",
          f"fix             NVE all nve/limit "
          + ("0.05" if st.units == "lj" else f"{0.05 * sig_out[0]:.3f}"),
          f"fix             LNG all langevin "
          + ("1.0 1.0" if st.units == "lj"
             else f"{st.temperature_k:.1f} {st.temperature_k:.1f}")
          + " " + ("10.0" if st.units == "lj" else "100.0") + " 904297",
          f"timestep        {ts}",
          "thermo          500",
          "run             10000",
          "unfix           PUSH",
          "",
          "# ---- stage 2: the real Kremer-Grest interactions (WCA)",
          "pair_style      lj/cut " + f"{max(wca):.4f}",
          "pair_modify     shift yes",]
    for t in range(n_types):
        for u in range(t, n_types):
            s_mix = 0.5 * (sig_out[t] + sig_out[u])
            e_mix = math.sqrt(eps_of[t] * eps_of[u])
            S.append(f"pair_coeff      {t + 1} {u + 1} {e_mix:.4f} "
                     f"{s_mix:.4f} {2.0 ** (1 / 6) * s_mix:.4f}")
    S += ["unfix           NVE",
          "fix             NVE2 all nve",
          "run             50000",
          "",
          "write_data      cell_cg_equilibrated.data",
          "# Production: extend the run, or switch fix LNG to a thermostat",
          "# of your choice. Densify with fix npt if a melt density is the",
          "# goal (reduced-units KG melts sit near 0.85 beads/sigma^3,",
          "# which this cell was sized to)."]
    inp = folder / "cell_cg.in"
    inp.write_text("\n".join(S) + "\n")

    emit(f"  CG cell: {n_beads} beads ({st.mapping} mapping), "
         f"{len(bonds)} FENE bonds"
         + (f", {len(angles)} angles" if use_angles else "")
         + f", {units_line} units -> {data.name} + {inp.name}")
    return data, inp
