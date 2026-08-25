"""Config schema for the CLI / GUI.

A single Config object drives the whole pipeline. The GUI serializes into
the same object so headless and GUI runs are perfectly interchangeable.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False


@dataclass
class MonomerSpec:
    file: str
    name: Optional[str] = None
    head: Optional[int] = None       # 1-based Avogadro index
    tail: Optional[int] = None
    head_h: Optional[int] = None     # H atom to remove when linking head
    tail_h: Optional[int] = None
    cap_mode: str = "terminal_h"


@dataclass
class OptimizerCfg:
    enabled: bool = True
    ff: str = "MMFF94"       # OpenBabel FF
    steps: int = 10000
    tol: float = 1e-6
    algorithm: str = "cg"


@dataclass
class ChainCfg:
    n_monomers: int = 20
    mode: str = "homopolymer"           # homopolymer|alternating|block|random
    fractions: Optional[List[float]] = None
    block_sizes: Optional[List[int]] = None
    seed: Optional[int] = None
    backend: str = "auto"
    # When True, if the monomer's tail heavy atom is a carbonyl carbon
    # (e.g. PBS: [*]OCCCCOC(=O)CCC(=O)[*]) the chain's terminal
    # -C(=O)H is auto-capped as -C(=O)OH (carboxylic acid). Set to False
    # to keep the raw aldehyde end.
    cap_carboxyl_end: bool = True


@dataclass
class BoxCfg:
    """Packing / periodic-cell settings.

    Two ways to define the cell:
      * fixed dimensions:   ``shape`` + ``a`` (+ ``b/c/alpha/beta/gamma`` as needed)
      * target density:     ``use_density=True`` + ``density_g_cm3`` — the tool
        derives the cell edge length while preserving the requested shape's
        aspect ratio and angles.

    Density is stored in g/cm³ (the unit users think in for polymers). Everything
    downstream (``pack_cell``) uses kg/m³ internally; conversion is done at
    call-site (× 1000).
    """
    n_chains: int = 1
    shape: str = "cubic"                  # cubic | orthorhombic | triclinic
    a: float = 50.0
    b: float = 50.0
    c: float = 50.0
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    use_density: bool = False
    density_g_cm3: float = 1.0            # 1 g/cm³ = 1000 kg/m³ (water baseline)
    packmol: bool = True
    # Explicit path to the packmol binary. When set, overrides PATH lookup
    # and env-var detection. Leave empty for auto-detection.
    packmol_path: str = ""
    # Random-number seed for packmol. -1 (default) = fresh time-based seed
    # every run → different packing each time. Non-negative = fixed layout.
    packmol_seed: int = -1
    # packmol 'tolerance' in Å (min distance between placed atoms).
    # Same knob as the Blend page. 2.0 is safe for polymers; drop to 1.5 for
    # tighter packing at higher density.
    packmol_tolerance: float = 2.0
    # ------------------------------------------------------------------
    # GROMACS-native packing (via ``gmx insert-molecules``). Only used
    # when engine is 'gromacs'. Two ways to specify chain count:
    #   * n_chains > 0  → insert exactly that many copies
    #   * target_atoms > 0 → insert atom_limit // atoms_per_chain copies
    # If both are set, target_atoms takes precedence.
    # ------------------------------------------------------------------
    gmx_target_atoms: int = 0
    gmx_pre_minimise: bool = True
    gmx_try_count: int = 100000

    # Deprecated but kept for backwards compat with old configs.
    # Old code path stored a 3-tuple in `size`.
    size: List[float] = field(default_factory=lambda: [50.0, 50.0, 50.0])


@dataclass
class ForceFieldCfg:
    # Default to OPLS 2005 via DL_FIELD — the DL_FIELD path is the primary
    # supported flow and produces LAMMPS/GROMACS files directly.
    key: str = "opls2005_dl"
    dl_lib_dir: Optional[str] = None    # path to dl_f_4.13/lib
    manual_types: Dict[int, str] = field(default_factory=dict)
    convert_par_to: Optional[str] = None  # emit converted .lt to this path


@dataclass
class BlendComponentCfg:
    """One species in a multi-component blend.

    ``data_file`` is a pre-built single-chain LAMMPS data file — typically the
    ``dlf_output1/lammps1.data`` a previous PAAF run produced for one polymer.
    ``count`` is how many copies of it to pack.
    """
    name: str = ""
    data_file: str = ""
    count: int = 1


@dataclass
class BlendCfg:
    """Multi-component blend packing settings.

    When ``enabled`` and ``components`` has at least two entries, the pipeline
    invokes the blend replicator instead of (or after) the single-chain flow,
    packing every copy of every component into one box and merging their
    Masses / Pair / Bond / ... coefficient sections with per-component type
    offsets so type IDs never collide.
    """
    enabled: bool = False
    components: List["BlendComponentCfg"] = field(default_factory=list)
    seed: int = -1                # <0 → time-based, different each run
    tolerance: float = 2.0
    out_data: str = "packed_blend.data"
    out_input: str = "packed_blend.in"


@dataclass
class LammpsCfg:
    """LAMMPS relaxation / production settings.

    Damping constants follow LAMMPS conventions:
      - ``tdamp_fs`` — thermostat relaxation time (fs). Standard rule of
        thumb: ~100 * timestep for Nose-Hoover.
      - ``pdamp_fs`` — barostat relaxation time (fs). Rule of thumb:
        ~1000 * timestep.

    The pipeline can emit either a single-stage run or a full multi-stage
    protocol: minimize -> NVT equilibration -> NPT production.
    """
    ensemble: str = "npt"                    # single-stage ensemble: nve / nvt / npt
    temperature: float = 300.0               # K
    pressure: float = 1.0                    # atm
    steps: int = 500_000                     # production steps
    timestep: float = 1.0                    # fs

    # Damping constants
    tdamp_fs: float = 100.0                  # thermostat coupling time (fs)
    pdamp_fs: float = 1000.0                 # barostat coupling time (fs)

    # Thermostat / barostat styles
    thermostat: str = "nose-hoover"          # nose-hoover | langevin | berendsen | csvr
    barostat: str = "nose-hoover"            # nose-hoover | berendsen | parrinello
    pressure_coupling: str = "iso"           # iso | aniso | tri | x | y | z

    # Multi-stage protocol
    multistage: bool = False
    minimize: bool = True                    # run CG minimization first
    min_etol: float = 1.0e-4
    min_ftol: float = 1.0e-6
    min_maxiter: int = 10_000
    min_maxeval: int = 100_000

    # NVT equilibration stage (only used when multistage=True)
    nvt_steps: int = 100_000
    nvt_temperature: Optional[float] = None  # defaults to `temperature` if None

    # NPT production stage (only used when multistage=True; overrides steps)
    npt_steps: int = 500_000

    # Miscellaneous
    thermo_every: int = 1000
    dump_every: int = 0                      # 0 disables trajectory dump
    seed: int = 4928459


@dataclass
class Config:
    project_name: str = "polymer"
    output_dir: str = "output"
    monomers: List[MonomerSpec] = field(default_factory=list)
    optimizer: OptimizerCfg = field(default_factory=OptimizerCfg)
    chain: ChainCfg = field(default_factory=ChainCfg)
    box: BoxCfg = field(default_factory=BoxCfg)
    force_field: ForceFieldCfg = field(default_factory=ForceFieldCfg)
    lammps: LammpsCfg = field(default_factory=LammpsCfg)
    blend: BlendCfg = field(default_factory=BlendCfg)
    # Auto-execute the topology builder (moltemplate.sh / dl_field) after
    # writing .lt files. Default True: without this, PAAF only emits the
    # scaffold .lt (Moltemplate) or writes control file (DL_FIELD); nothing
    # converts them into a LAMMPS system.data / .in.settings / .in.init or
    # a GROMACS .top / .gro / .mdp. Turn OFF only if you want to hand-edit
    # the .lt files before running moltemplate yourself.
    run_moltemplate: bool = True

    # Which topology builder to invoke when run_moltemplate is True:
    #   "auto"        - pick based on ff.kind (dl_field for DL_FIELD FFs, moltemplate.sh otherwise)
    #   "moltemplate" - always use moltemplate.sh
    #   "dl_field"    - always use dl_field
    topology_builder: str = "auto"

    # MD engine outputs to emit. Choose "lammps" (default) to only write
    # LAMMPS files, "gromacs" to only write GROMACS files (.gro/.top/.mdp),
    # or "both" to write both. Damping constants and multi-stage settings on
    # LammpsCfg are also used to configure the GROMACS MDPs.
    engine: str = "lammps"
    gromacs_include_itp: Optional[str] = None   # e.g. "oplsaa.ff/forcefield.itp"

    # ----------------------------------------------------- IO
    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        text = path.read_text()
        if path.suffix.lower() in {".yaml", ".yml"}:
            if not _HAVE_YAML:
                raise RuntimeError("PyYAML is not installed; use JSON or `pip install pyyaml`.")
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        payload = asdict(self)
        if path.suffix.lower() in {".yaml", ".yml"}:
            if not _HAVE_YAML:
                raise RuntimeError("PyYAML not installed")
            path.write_text(yaml.safe_dump(payload, sort_keys=False))
        else:
            path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        d = dict(d)
        monos = [MonomerSpec(**m) for m in d.pop("monomers", [])]
        opt = OptimizerCfg(**d.pop("optimizer", {}))
        ch = ChainCfg(**d.pop("chain", {}))
        bx = BoxCfg(**d.pop("box", {}))
        ff = ForceFieldCfg(**d.pop("force_field", {}))
        lm = LammpsCfg(**d.pop("lammps", {}))
        blend_d = d.pop("blend", {}) or {}
        blend_components = [BlendComponentCfg(**c) for c in blend_d.pop("components", [])]
        bl = BlendCfg(components=blend_components, **blend_d)
        return cls(
            monomers=monos,
            optimizer=opt,
            chain=ch,
            box=bx,
            force_field=ff,
            lammps=lm,
            blend=bl,
            **d,
        )
