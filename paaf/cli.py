"""Command-line entry point.

Usage::

    moltemplate-auto run config.yaml
    moltemplate-auto list-ffs
    moltemplate-auto convert-par --par dl_f_4.13/lib/PCFF.par --out PCFF.lt
    moltemplate-auto build --monomer PBS.pdb --n 20 --ff oplsaa --output out/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .__version__ import __version__
from .config import ChainCfg, Config, ForceFieldCfg, MonomerSpec, OptimizerCfg, BoxCfg, LammpsCfg
from .dlfield_converter import convert_par
from .ff_registry import get_ff, list_ffs
from .logging_utils import get_logger
from .pipeline import run_pipeline

log = get_logger("paaf.cli")


def _list_ffs(_: argparse.Namespace) -> int:
    for ff in list_ffs():
        tag = ", ".join(ff.tags)
        print(f"  {ff.key:<15s} {ff.display_name:<45s} [{tag}]")
    return 0


def _convert_par(ns: argparse.Namespace) -> int:
    out = convert_par(ns.par, ns.out, ff_object_name=ns.name)
    print(f"Wrote {out}")
    return 0


def _run(ns: argparse.Namespace) -> int:
    cfg = Config.load(ns.config)
    res = run_pipeline(cfg)
    print(json.dumps(res, indent=2))
    return 0


def _build(ns: argparse.Namespace) -> int:
    cfg = Config(
        project_name=ns.name,
        output_dir=ns.output,
        monomers=[MonomerSpec(file=m) for m in ns.monomer],
        optimizer=OptimizerCfg(enabled=not ns.no_opt, ff=ns.opt_ff, steps=ns.opt_steps),
        chain=ChainCfg(n_monomers=ns.n, mode=ns.mode),
        box=BoxCfg(n_chains=ns.n_chains, size=ns.box),
        force_field=ForceFieldCfg(key=ns.ff, dl_lib_dir=ns.dl_lib),
        lammps=LammpsCfg(ensemble=ns.ensemble, temperature=ns.temperature),
        run_moltemplate=not ns.no_moltemplate,
        engine=ns.engine,
        gromacs_include_itp=ns.gromacs_itp,
    )
    res = run_pipeline(cfg)
    print(json.dumps(res, indent=2))
    return 0


def _gui(_: argparse.Namespace) -> int:
    from .gui.app import launch
    return launch()


def _pack_cell(ns: argparse.Namespace) -> int:
    from .cell import pack_cell, PackSpec
    specs = []
    for s in ns.species:
        parts = s.split(":")
        file = parts[0]; count = int(parts[1]) if len(parts) > 1 else 1
        name = parts[2] if len(parts) > 2 else Path(file).stem
        specs.append(PackSpec(file=file, count=count, name=name))
    mol, box = pack_cell(specs, density_kg_m3=ns.density, box_ang=ns.box,
                         out_path=ns.out, seed=ns.seed)
    print(f"Wrote {ns.out}  ({len(mol.atoms)} atoms, box={box:.2f} Å)")
    return 0


def _build_crystal(ns: argparse.Namespace) -> int:
    from .cell import build_preset, list_presets
    if ns.list:
        for p in list_presets():
            print(f"  {p.name:<15s} {p.description}")
        return 0
    if not ns.preset or not ns.out:
        raise SystemExit("--preset and --out are required (use --list to see presets)")
    from .structure import write
    mol = build_preset(ns.preset, nx=ns.nx, ny=ns.ny, nz=ns.nz)
    write(mol, ns.out)
    print(f"Wrote {ns.out}  ({len(mol.atoms)} atoms)")
    return 0


def _build_slab(ns: argparse.Namespace) -> int:
    from .cell import cleave_surface
    from .structure import write
    mol = cleave_surface(ns.preset, tuple(ns.hkl), supercell=tuple(ns.supercell),
                        thickness_ang=ns.thickness, vacuum_ang=ns.vacuum)
    write(mol, ns.out)
    print(f"Wrote {ns.out}  ({len(mol.atoms)} atoms)")
    return 0


def _build_cnt(ns: argparse.Namespace) -> int:
    from .cell import build_nanotube
    from .structure import write
    mol = build_nanotube(ns.n, ns.m, length_ang=ns.length, element=ns.element)
    write(mol, ns.out)
    print(f"Wrote {ns.out}  ({len(mol.atoms)} atoms)")
    return 0


def _extract_oplsua(ns: argparse.Namespace) -> int:
    from .oplsua_extractor import extract, default_source, default_dest
    src = ns.source or default_source()
    dst = ns.out or default_dest()
    p = extract(src, dst, object_name=ns.object_name)
    print(f"Wrote {p}")
    return 0


def _build_layers(ns: argparse.Namespace) -> int:
    from .cell import build_layers, LayerSpec
    specs = []
    for raw in ns.layer:
        parts = raw.split(":")
        kind = parts[0].lower()
        if kind == "file":
            path = parts[1]
            name = parts[2] if len(parts) > 2 else Path(path).stem
            off = [0.0, 0.0]; gap = None
            if len(parts) > 3 and parts[3]:
                off = [float(x) for x in parts[3].split(",")]
            if len(parts) > 4 and parts[4]:
                gap = float(parts[4])
            specs.append(LayerSpec(file=path, name=name,
                                   lateral_offset=tuple(off), gap_before=gap))
        elif kind == "preset":
            preset = parts[1]; sc = (1, 1, 1)
            if len(parts) > 2 and parts[2]:
                sc = tuple(int(x) for x in parts[2].split(","))
            off = [0.0, 0.0]; gap = None
            if len(parts) > 3 and parts[3]:
                off = [float(x) for x in parts[3].split(",")]
            if len(parts) > 4 and parts[4]:
                gap = float(parts[4])
            specs.append(LayerSpec(preset=preset, supercell=sc, name=preset,
                                   lateral_offset=tuple(off), gap_before=gap))
        else:
            raise SystemExit(f"--layer must start with 'file:' or 'preset:' (got {raw!r})")
    mol, box = build_layers(specs, gap_ang=ns.gap, center_xy=not ns.no_center,
                           out_path=ns.out)
    print(f"Wrote {ns.out}  ({len(mol.atoms)} atoms, box={box[0]:.1f} x {box[1]:.1f} x {box[2]:.1f} Å)")
    return 0


def _solvate_cli(ns: argparse.Namespace) -> int:
    from .cell import solvate
    mol, box = solvate(ns.solute, solvent=ns.solvent, n_solvent=ns.n_solvent,
                      density_kg_m3=ns.density, out_path=ns.out)
    print(f"Wrote {ns.out}  ({len(mol.atoms)} atoms, box={box:.2f} Å)")
    return 0


def _reactions_learn(ns: argparse.Namespace) -> int:
    from .reaction import ReactionLibrary
    if ns.folder:
        lib = ReactionLibrary.from_folder(ns.folder)
    else:
        specs = []
        if len(ns.reactant) != len(ns.product):
            raise SystemExit("--reactant and --product must be provided in pairs")
        for i, (r, p) in enumerate(zip(ns.reactant, ns.product)):
            name = ns.name[i] if ns.name and i < len(ns.name) else f"reaction_{i + 1}"
            specs.append({"name": name, "reactant": r, "product": p})
        lib = ReactionLibrary.learn_many(specs)
    lib.save(ns.out)
    print(lib.summary())
    print(f"Saved -> {ns.out}")
    return 0


def _reactions_apply(ns: argparse.Namespace) -> int:
    from .reaction import ReactionLibrary
    from .xlink_engine import apply_library
    lib = ReactionLibrary.load(ns.library)
    stats = apply_library(ns.system, lib, ns.out, max_events=ns.max_events, cutoff=ns.cutoff)
    print(json.dumps(stats.as_dict(), indent=2))
    print(f"Wrote crosslinked system -> {ns.out}")
    return 0


def _build_molecule(ns: argparse.Namespace) -> int:
    from .builder import build_from_smiles, build_recipe, list_library, search_library
    if ns.list:
        rows = list_library(ns.source)
        print(f"# {len(rows)} polymer(s) in library (source={ns.source})")
        print(f"# {'KEY':<14s} {'DESCRIPTION':<45s} {'Tg(K)':>8s} {'ρ(kg/m³)':>10s}  SMILES")
        for r in rows:
            tg = f"{r.tg_k:.1f}" if r.tg_k is not None else "-"
            rho = f"{r.density_kg_m3:.1f}" if r.density_kg_m3 is not None else "-"
            key = r.pid or r.name
            print(f"  {key:<14s} {r.description:<45s} {tg:>8s} {rho:>10s}  {r.monomer_smiles}")
        return 0
    if ns.search:
        for r in search_library(ns.search):
            print(f"  {r.pid or r.name:<14s} {r.description}")
        return 0
    if ns.recipe:
        out = build_recipe(ns.recipe, ns.out, optimize=not ns.no_opt)
    elif ns.smiles:
        out = build_from_smiles(ns.smiles, ns.out, optimize=not ns.no_opt,
                                opt_ff=ns.opt_ff, opt_steps=ns.opt_steps, name=ns.name)
    else:
        raise SystemExit("Provide --smiles OR --recipe OR --list OR --search.")
    print(f"Wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="paaf",
        description="PAAF — Polymer Auto-Assembly Framework. "
                    "Automated polymer builder, reactive-MD engine and "
                    "Moltemplate / DL_FIELD / LAMMPS / GROMACS system generator.",
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list-ffs", help="Show all supported force fields")
    sp.set_defaults(func=_list_ffs)

    sp = sub.add_parser("convert-par", help="Convert DL_FIELD .par -> Moltemplate .lt")
    sp.add_argument("--par", required=True, type=Path)
    sp.add_argument("--out", required=True, type=Path)
    sp.add_argument("--name", default=None, help="Moltemplate object name (default: from .par)")
    sp.set_defaults(func=_convert_par)

    sp = sub.add_parser("run", help="Run pipeline from a YAML/JSON config file")
    sp.add_argument("config", type=Path)
    sp.set_defaults(func=_run)

    sp = sub.add_parser("build", help="One-shot build without a config file")
    sp.add_argument("--monomer", action="append", required=True,
                    help="Path to monomer file; repeat for copolymers")
    sp.add_argument("--n", type=int, default=20, help="Monomers per chain")
    sp.add_argument("--n-chains", type=int, default=1)
    sp.add_argument("--box", type=float, nargs=3, default=[50.0, 50.0, 50.0])
    sp.add_argument("--mode", default="homopolymer",
                    choices=["homopolymer", "alternating", "block", "random"])
    sp.add_argument("--ff", default="oplsaa", help="Force field key (see list-ffs)")
    sp.add_argument("--dl-lib", type=Path, default=None,
                    help="Path to dl_f_4.13/lib (needed for PCFF/COMPASS/CVFF/...)")
    sp.add_argument("--output", type=Path, default=Path("./output"))
    sp.add_argument("--name", default="polymer")
    sp.add_argument("--no-opt", action="store_true", help="Skip OpenBabel optimization")
    sp.add_argument("--opt-ff", default="MMFF94")
    sp.add_argument("--opt-steps", type=int, default=10000)
    sp.add_argument("--ensemble", default="npt")
    sp.add_argument("--temperature", type=float, default=300.0)
    sp.add_argument("--no-moltemplate", action="store_true",
                    help="Skip running moltemplate.sh at the end")
    sp.add_argument("--engine", choices=["lammps", "gromacs", "both"], default="lammps",
                    help="Which MD engine files to emit")
    sp.add_argument("--gromacs-itp", default=None,
                    help="Optional GROMACS FF .itp path to #include in the .top")
    sp.set_defaults(func=_build)

    sp = sub.add_parser("gui", help="Launch the PyQt5 GUI")
    sp.set_defaults(func=_gui)

    # ---------- reactions-learn
    sp = sub.add_parser("reactions-learn",
                        help="Learn a reaction library from reactant/product structures")
    sp.add_argument("--reactant", action="append", default=[],
                    help="Reactant structure file (xyz/pdb/mol2/...); repeat")
    sp.add_argument("--product", action="append", default=[],
                    help="Product structure file; repeat")
    sp.add_argument("--name", action="append", default=[],
                    help="Optional reaction name; repeat")
    sp.add_argument("--folder", type=Path, default=None,
                    help="Root folder with reaction_XXX/{reactant,product}.<ext>")
    sp.add_argument("--out", type=Path, required=True, help="Output JSON library")
    sp.set_defaults(func=_reactions_learn)

    # ---------- reactions-apply
    sp = sub.add_parser("reactions-apply",
                        help="Apply a learned reaction library to a packed system")
    sp.add_argument("--library", type=Path, required=True)
    sp.add_argument("--system", type=Path, required=True,
                    help="Packed system (xyz/pdb/mol2)")
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--max-events", type=int, default=100_000)
    sp.add_argument("--cutoff", type=float, default=5.0,
                    help="Reactive-pair distance cutoff, Å")
    sp.set_defaults(func=_reactions_apply)

    # ---------- pack-cell (amorphous cell / blend)
    sp = sub.add_parser("pack-cell", help="Pack molecules into an amorphous periodic box")
    sp.add_argument("--species", action="append", required=True,
                    help="file:count[:name] triplet; repeat for blends")
    sp.add_argument("--density", type=float, default=None, help="target density kg/m³")
    sp.add_argument("--box", type=float, default=None, help="cubic box side (Å)")
    sp.add_argument("--seed", type=int, default=12345)
    sp.add_argument("--out", type=Path, required=True)
    sp.set_defaults(func=_pack_cell)

    # ---------- build-crystal
    sp = sub.add_parser("build-crystal", help="Build a preset crystal supercell")
    sp.add_argument("--preset", default=None,
                    help="Cu, Al, Au, Fe, Si_diamond, NaCl, graphene, quartz_alpha, ...")
    sp.add_argument("--nx", type=int, default=4)
    sp.add_argument("--ny", type=int, default=4)
    sp.add_argument("--nz", type=int, default=4)
    sp.add_argument("--out", type=Path, default=None)
    sp.add_argument("--list", action="store_true")
    sp.set_defaults(func=_build_crystal)

    # ---------- build-slab
    sp = sub.add_parser("build-slab", help="Cleave a preset crystal along (hkl)")
    sp.add_argument("--preset", required=True)
    sp.add_argument("--hkl", nargs=3, type=int, default=[1, 0, 0])
    sp.add_argument("--supercell", nargs=3, type=int, default=[5, 5, 5])
    sp.add_argument("--thickness", type=float, default=12.0)
    sp.add_argument("--vacuum", type=float, default=15.0)
    sp.add_argument("--out", type=Path, required=True)
    sp.set_defaults(func=_build_slab)

    # ---------- build-cnt
    sp = sub.add_parser("build-cnt", help="Build a (n,m) single-wall nanotube")
    sp.add_argument("--n", type=int, required=True)
    sp.add_argument("--m", type=int, required=True)
    sp.add_argument("--length", type=float, default=20.0)
    sp.add_argument("--element", default="C")
    sp.add_argument("--out", type=Path, required=True)
    sp.set_defaults(func=_build_cnt)

    # ---------- extract-oplsua
    sp = sub.add_parser("extract-oplsua",
                        help="Extract UA subset of oplsaa2024.lt into oplsua_2024.lt")
    sp.add_argument("--source", type=Path, default=None,
                    help="Path to oplsaa2024.lt (default: bundled copy)")
    sp.add_argument("--out", type=Path, default=None,
                    help="Output .lt path (default: ff_libraries/moltemplate/oplsua_2024.lt)")
    sp.add_argument("--object-name", default="OPLSUA_2024")
    sp.set_defaults(func=_extract_oplsua)

    # ---------- build-layers
    sp = sub.add_parser("build-layers",
                        help="Stack N structures/presets along Z (bilayer/trilayer/interface)")
    sp.add_argument("--layer", action="append", required=True,
                    help="'file:path[:name[:offx,offy[:gap]]]' or "
                         "'preset:name:nx,ny,nz[:offx,offy[:gap]]'; repeat")
    sp.add_argument("--gap", type=float, default=3.0, help="default Z gap between layers (Å)")
    sp.add_argument("--no-center", action="store_true",
                    help="Don't center layers on the largest xy footprint")
    sp.add_argument("--out", type=Path, required=True)
    sp.set_defaults(func=_build_layers)

    # ---------- solvate
    sp = sub.add_parser("solvate", help="Solvate a molecule with a chosen solvent")
    sp.add_argument("--solute", type=Path, required=True)
    sp.add_argument("--solvent", default="water")
    sp.add_argument("--n-solvent", type=int, default=500)
    sp.add_argument("--density", type=float, default=None)
    sp.add_argument("--out", type=Path, required=True)
    sp.set_defaults(func=_solvate_cli)

    # ---------- build-molecule
    sp = sub.add_parser("build-molecule",
                        help="Build a molecule from SMILES or the built-in library")
    sp.add_argument("--smiles", default=None)
    sp.add_argument("--recipe", default=None, help="Name from the polymer library")
    sp.add_argument("--out", type=Path, default=Path("molecule.pdb"))
    sp.add_argument("--name", default="MOL")
    sp.add_argument("--no-opt", action="store_true")
    sp.add_argument("--opt-ff", default="MMFF94")
    sp.add_argument("--opt-steps", type=int, default=2000)
    sp.add_argument("--list", action="store_true", help="List the built-in polymer library")
    sp.add_argument("--source", choices=["all", "curated", "database"], default="all",
                    help="Filter --list output by source")
    sp.add_argument("--search", default=None,
                    help="Case-insensitive search across name, PID, description")
    sp.set_defaults(func=_build_molecule)

    ns = p.parse_args(argv)
    return int(ns.func(ns) or 0)


if __name__ == "__main__":
    sys.exit(main())
