"""Top-level pipeline that ties every module together.

Consumed by both the CLI and the GUI.
"""
from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Optional

from . import chain_builder, ff_assigner, lt_writer, monomer as mono_mod, optimizer, structure
from .config import Config
from .dlfield_converter import convert_par
from .ff_registry import get_ff
from .logging_utils import get_logger
from .moltemplate_runner import find_moltemplate, run_moltemplate

log = get_logger(__name__)


def _hoist_dlfield_outputs(out_dir: Path, engine: str) -> list[Path]:
    """Copy dl_field's output files from dlf_output1/ up to `out_dir`.

    DL_FIELD 4.x writes into ``<out_dir>/dlf_output1/`` under these names:
      * ``dl_poly.CONFIG``, ``dl_poly.FIELD`` (DL_POLY topology + coords)
      * ``lammps<N>.data``, ``lammps<N>.in`` (LAMMPS files, N is 1..K)
      * ``system.gro``, ``system.top``, ``*.mdp`` (GROMACS)
    We copy them to the parent folder and also rename ``lammps1.data`` and
    ``lammps1.in`` to plain ``lammps.data`` / ``lammps.in`` for convenience.
    """
    import shutil
    dlf = out_dir / "dlf_output1"
    if not dlf.exists():
        return []
    hoisted: list[Path] = []

    # Exact-name matches
    for name in ("dl_poly.CONFIG", "dl_poly.FIELD", "REVIVE", "OUTPUT",
                 "STATIS", "system.gro", "system.top"):
        src = dlf / name
        if src.exists() and src.is_file():
            dst = out_dir / name
            try:
                shutil.copy2(src, dst); hoisted.append(dst)
            except Exception as exc:
                log.warning("Could not hoist %s -> %s: %s", src, dst, exc)

    # Glob matches: lammpsN.data / lammpsN.in / *.mdp
    lammps_data_files = sorted(dlf.glob("lammps*.data"))
    lammps_in_files   = sorted(dlf.glob("lammps*.in"))
    for src in lammps_data_files + lammps_in_files:
        dst = out_dir / src.name
        try:
            shutil.copy2(src, dst); hoisted.append(dst)
        except Exception:
            pass
    # Convenience aliases for the single-copy common case
    if len(lammps_data_files) == 1:
        try:
            shutil.copy2(lammps_data_files[0], out_dir / "lammps.data")
            hoisted.append(out_dir / "lammps.data")
        except Exception:
            pass
    if len(lammps_in_files) == 1:
        try:
            shutil.copy2(lammps_in_files[0], out_dir / "lammps.in")
            hoisted.append(out_dir / "lammps.in")
        except Exception:
            pass

    for src in sorted(dlf.glob("*.mdp")):
        dst = out_dir / src.name
        try:
            shutil.copy2(src, dst); hoisted.append(dst)
        except Exception:
            pass
    return hoisted


def run_pipeline(cfg: Config, progress: Optional[Callable[[str], None]] = None) -> dict:
    """Execute the whole pipeline described by `cfg` and return summary."""
    def _p(msg: str) -> None:
        log.info(msg)
        if progress:
            progress(msg)

    out_dir = Path(cfg.output_dir) / cfg.project_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and optimize monomers
    _p("Loading monomers...")
    monomers = []
    _DUMMY = {"*", "Xx", "XX", "Du", "DU", ""}
    for spec in cfg.monomers:
        m = mono_mod.monomer_from_file(
            path=spec.file,
            head=spec.head, tail=spec.tail,
            head_h=spec.head_h, tail_h=spec.tail_h,
            cap_mode=spec.cap_mode,
            name=spec.name,
        )
        # Defensive: even if the loader didn't strip them, remove any dummy
        # element atoms (*, Xx, Du) here. Otherwise they poison every
        # downstream step and crash moltemplate's data-sort with int/str.
        bad = {a.index for a in m.molecule.atoms if a.element in _DUMMY}
        if bad:
            _p(f"[pipeline] stripped {len(bad)} dummy atom(s) from {m.name}")
            keep = [a for a in m.molecule.atoms if a.index not in bad]
            remap = {a.index: k for k, a in enumerate(keep)}
            for k, a in enumerate(keep):
                a.index = k; a.name = f"{a.element}{k + 1}"
            m.molecule.atoms = keep
            m.molecule.bonds = [(remap[i], remap[j], o)
                                for (i, j, o) in m.molecule.bonds
                                if i in remap and j in remap]
        if cfg.optimizer.enabled:
            _p(f"Optimizing monomer {m.name}")
            optimizer.optimize(
                m.molecule,
                ff=cfg.optimizer.ff,
                steps=cfg.optimizer.steps,
                tol=cfg.optimizer.tol,
                algorithm=cfg.optimizer.algorithm,
            )
        structure.write(m.molecule, out_dir / f"{m.name}_opt.xyz")
        monomers.append(m)

    # 2. Build chain
    _p(f"Building chain ({cfg.chain.n_monomers} units, mode={cfg.chain.mode})")
    _p("  (NOTE: mbuild produces a fully-extended initial geometry — that's "
       "expected. The chain will collapse to a realistic conformation during "
       "MD equilibration in LAMMPS/GROMACS.)")
    chain = chain_builder.build_chain(
        monomers=monomers,
        n=cfg.chain.n_monomers,
        mode=cfg.chain.mode,
        fractions=cfg.chain.fractions,
        block_sizes=cfg.chain.block_sizes,
        seed=cfg.chain.seed,
        backend=cfg.chain.backend,
        cap_carboxyl_end=getattr(cfg.chain, "cap_carboxyl_end", True),
    )
    chain.name = cfg.project_name
    structure.write(chain, out_dir / f"{cfg.project_name}.mol2")

    # 3. Optimize chain
    if cfg.optimizer.enabled:
        _p("Optimizing full chain")
        optimizer.optimize(chain, ff=cfg.optimizer.ff, steps=cfg.optimizer.steps,
                           tol=cfg.optimizer.tol, algorithm=cfg.optimizer.algorithm)
        structure.write(chain, out_dir / f"{cfg.project_name}_opt.xyz")

    # 4. Force field: convert if requested, then type atoms
    ff = get_ff(cfg.force_field.key)
    _p(f"Force field: {ff.display_name}")

    dl_lib = Path(cfg.force_field.dl_lib_dir) if cfg.force_field.dl_lib_dir else None
    ff_lt_path: Optional[Path] = None
    # DL_FIELD-based FFs use the dl_field binary directly — no Moltemplate .lt
    # scaffolding is needed. We still validate that the lib dir is set.
    if ff.kind == "dlfield":
        if dl_lib is None:
            raise RuntimeError(
                f"{ff.display_name} requires 'DL_FIELD lib dir' on the "
                f"Force-field page (e.g. /Users/uzair/project/dl_f_4.13/lib)."
            )
    else:
        # For Moltemplate-native FFs, drop the bundled .lt next to system.lt
        # so moltemplate.sh finds it locally and users don't need it installed
        # in their Moltemplate distribution.
        bundled = ff.bundled_path()
        if bundled and bundled.exists():
            ff_lt_path = out_dir / bundled.name
            shutil.copy2(bundled, ff_lt_path)
            _p(f"Copied bundled FF library {bundled.name} -> output dir")

    _p("Assigning atom types")
    # The manual types the user assigned refer to ONE monomer; the chain is n
    # of them with a cap hydrogen deleted at every junction. Hand the monomer
    # and the unit count over so the two numberings can be reconciled exactly
    # instead of assumed equal — assuming equal is what put the user's types
    # on the first fifteen atoms and nowhere else.
    #
    # DL_FIELD types atoms itself and never sees manual_types, so this affects
    # the Moltemplate-native route only.
    _first_monomer = monomers[0] if monomers else None
    ff_assigner.assign(chain, ff, dl_lib_dir=dl_lib,
                       manual_types=cfg.force_field.manual_types,
                       monomer=_first_monomer,
                       n_units=int(cfg.chain.n_monomers))

    # Nothing may leave here with an atom type belonging to another element.
    # The guard used to live only in the typing dialog, so it caught a bad
    # MANUAL assignment and missed the automatic typer entirely — which then
    # wrote two backbone carbons with hydrogen's mass. Check the final types,
    # whatever produced them.
    if ff.kind == "moltemplate_native":
        from .type_guard import check_all
        _problems = check_all({a.index: a.element for a in chain.atoms},
                              {a.index: a.ff_type for a in chain.atoms
                               if a.ff_type})
        if _problems:
            _p("!" * 60)
            for _idx, _msg in _problems[:10]:
                _p(f"!! atom {_idx}: {_msg}")
            raise RuntimeError(
                f"{len(_problems)} atoms carry a force-field type belonging to "
                f"a different element. Writing this would give them the wrong "
                f"mass and charge, and nothing downstream would object. "
                f"First: atom {_problems[0][0]} — {_problems[0][1]}")

    # Sanity check: for Moltemplate-native FFs the atom types produced by
    # OpenBabel/Ghemical (e.g. "C.3", "H") do NOT match the numeric or
    # symbolic types used inside oplsaa2024.lt / gaff.lt / dreiding.lt etc.
    # If we hand those to moltemplate.sh it will error with "unknown atom
    # type". Warn loudly so the user knows to either provide manual_types
    # or pick a DL_FIELD-based FF that types atoms itself.
    #
    # For OPLS-AA specifically, our SMARTS typer already emits valid numeric
    # types (135, 140, 145, ...), so the "looks_bad" heuristic below only
    # trips for other Moltemplate-native FFs (GAFF, TraPPE, DREIDING) that
    # still route through the raw OpenBabel Ghemical path.
    if ff.kind == "moltemplate_native" and ff.key not in (
            "oplsaa", "oplsaa2008", "loplsaa", "loplsaa2008"):
        sample = {a.ff_type for a in chain.atoms[:20] if a.ff_type}
        looks_bad = any("." in t or t.isalpha() and len(t) <= 2 for t in sample)
        if looks_bad and not cfg.force_field.manual_types:
            _p("!" * 60)
            _p(f"!! WARNING: atom typing for {ff.display_name} is only a "
               f"rough OpenBabel guess ({sorted(sample)[:5]} ...).")
            _p("!! moltemplate.sh will almost certainly reject these types "
               "because they don't match the numeric IDs in the shipped")
            _p(f"!! {ff.lt_include}. Options:")
            _p("!!   (a) Provide manual atom types on the Force-field page")
            _p("!!       (one line per atom index: '3:@atom:80').")
            _p("!!   (b) Switch to a DL_FIELD-based FF (PCFF, COMPASS, CVFF, "
               "OPLS 2005, TraPPE-EH, CHARMM36, GROMOS 54A7, AMBER GAFF,")
            _p("!!       OPLS-UA) — those type atoms automatically via dl_field.")
            _p("!" * 60)

    # 5. Compute the requested periodic cell (cubic / orthorhombic / triclinic)
    #    from the Box config, resolving target density when requested.
    from .cell.amorphous import BoxShape, _shape_for_density, PackSpec
    if getattr(cfg.box, "shape", "cubic") == "cubic":
        box_shape = BoxShape(shape="cubic", a=cfg.box.a, b=cfg.box.a, c=cfg.box.a)
    elif cfg.box.shape == "orthorhombic":
        box_shape = BoxShape(shape="orthorhombic", a=cfg.box.a, b=cfg.box.b, c=cfg.box.c)
    else:
        box_shape = BoxShape(shape="triclinic",
                             a=cfg.box.a, b=cfg.box.b, c=cfg.box.c,
                             alpha=cfg.box.alpha, beta=cfg.box.beta, gamma=cfg.box.gamma)
    if getattr(cfg.box, "use_density", False):
        density_kg_m3 = float(cfg.box.density_g_cm3) * 1000.0
        # Scale the requested shape to the target density (n_chains copies).
        template = box_shape
        pseudo_specs = [PackSpec(molecule=chain, count=cfg.box.n_chains, name=chain.name)]
        box_shape = _shape_for_density(pseudo_specs, [chain], density_kg_m3,
                                       template=template)
        _p(f"Box scaled to target density {cfg.box.density_g_cm3} g/cm³: {box_shape}")

    # The box, stated in the log as ONE unambiguous line — and sanity-checked.
    #
    # A 206-chain PIB cell once went to GROMACS in a 500 Å box when the Box
    # page said 50 Å: density 0.003 g/cm³ where the real polymer is ~0.92, a
    # near-vacuum that would have simulated without a single error message.
    # The ×10 could not be reproduced afterwards, which is exactly why the
    # value the pipeline is ACTUALLY using has to be printed at the moment it
    # is used, not reconstructed later from output files.
    _p(f"Box (as configured): {box_shape.a:.1f} × {box_shape.b:.1f} × "
       f"{box_shape.c:.1f} Å ({box_shape.shape})")
    try:
        _n_for_density = max(int(cfg.box.n_chains), 1)
        _tgt = int(getattr(cfg.box, "gmx_target_atoms", 0))
        if _tgt > 0 and len(chain.atoms) > 0:
            _n_for_density = max(_n_for_density,
                                 round(_tgt / len(chain.atoms)))
        _mass_g = _n_for_density * sum(
            {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
             "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45,
             "Br": 79.904}.get(a.element, 12.0) for a in chain.atoms)
        _rho = (_mass_g / 6.02214076e23) / (box_shape.volume_ang3() * 1e-24)
        _p(f"Box density check: {_n_for_density} chain(s) in this box "
           f"≈ {_rho:.3f} g/cm³")
        if _rho < 0.05:
            _p("WARNING " + "=" * 52)
            _p(f"WARNING: {_rho:.4f} g/cm³ is far below any polymer melt "
               f"(~0.9-1.4). The box is much too large for its contents — "
               f"check the edge lengths on the Box page (they are in "
               f"ANGSTROM) before trusting this cell.")
            _p("WARNING " + "=" * 52)
        elif _rho > 3.0:
            _p("WARNING " + "=" * 52)
            _p(f"WARNING: {_rho:.2f} g/cm³ is denser than any polymer — the "
               f"box is too small for its contents and packing will fail or "
               f"overlap. Check the Box page edge lengths (ANGSTROM).")
            _p("WARNING " + "=" * 52)
    except Exception:
        pass

    # 5b. Fit the box to the chain and wrap coordinates so no atom falls
    # outside the cell (fixes chains protruding through the periodic box).
    import numpy as _np
    coords = chain.coords()
    span = coords.max(axis=0) - coords.min(axis=0)
    bx, by, bz = box_shape.bounding_box()
    # If the chain is bigger than the box on any axis, grow the box with 10%
    # padding — otherwise packing / visualization will show it sticking out.
    if any(s > b for s, b in zip(span, (bx, by, bz))):
        pad = 1.10
        new_a = max(box_shape.a, span[0] * pad)
        new_b = max(box_shape.b, span[1] * pad)
        new_c = max(box_shape.c, span[2] * pad)
        _p(f"Chain span {span.round(1).tolist()} Å exceeds box "
           f"({bx:.1f}x{by:.1f}x{bz:.1f} Å). Growing box to "
           f"{new_a:.1f}x{new_b:.1f}x{new_c:.1f} Å (+10% pad).")
        box_shape.a, box_shape.b, box_shape.c = new_a, new_b, new_c
        if box_shape.shape == "cubic":
            # keep cubic invariant: use the largest dim on all three edges
            side = max(new_a, new_b, new_c)
            box_shape.a = box_shape.b = box_shape.c = side
    # Wrap: shift the whole chain so its min corner sits at the origin, then
    # apply periodic-image translation for anything still outside [0, edge].
    coords = chain.coords()
    lo = coords.min(axis=0)
    shift = -lo
    for a in chain.atoms:
        a.xyz = a.xyz + shift
    bx, by, bz = box_shape.bounding_box()
    for a in chain.atoms:
        a.xyz[0] = a.xyz[0] % bx if bx > 0 else a.xyz[0]
        a.xyz[1] = a.xyz[1] % by if by > 0 else a.xyz[1]
        a.xyz[2] = a.xyz[2] % bz if bz > 0 else a.xyz[2]

    # 6. Write Moltemplate .lt files — ONLY for Moltemplate-native FFs.
    # DL_FIELD-based FFs don't need any .lt files (they use dl_field directly).
    chain_lt: Optional[Path] = None
    system_lt: Optional[Path] = None
    # moltemplate is run on ONE chain; the N-chain box is then packed with
    # packmol from that single-chain data file (same replicator as the
    # DL_FIELD route). `new X[N].move(...)` would only line the copies up.
    _n_chains_mt = int(cfg.box.n_chains)
    _pack_after_mt = (ff.kind != "dlfield" and _n_chains_mt > 1)
    if ff.kind != "dlfield":
        _p("Writing Moltemplate files")
        chain_lt = lt_writer.write_chain_lt(chain, out_dir, ff, name=cfg.project_name)
        system_lt = lt_writer.write_system_lt(
            out_dir, chain_lt, n_chains=1,
            box=list(box_shape.bounding_box()),
            name="system", ff=ff,
        )
        if _pack_after_mt:
            _p(f"system.lt holds ONE chain; {_n_chains_mt} copies will be "
               f"packed into the box with packmol after moltemplate runs.")
    else:
        _p("DL_FIELD path — skipping Moltemplate .lt scaffolding "
           "(dl_field will build the topology directly).")

    want_lammps = cfg.engine in ("lammps", "both")
    want_gromacs = cfg.engine in ("gromacs", "both")

    input_file = None
    gromacs_files: dict = {}

    # Only emit a LAMMPS run script for the MOLTEMPLATE path. The DL_FIELD
    # path produces its own lammps.in via dl_field, so writing one here is
    # both wasteful and confusing (users end up with two conflicting run.in
    # / lammps.in files in the same folder).
    _tb_choice_pre = getattr(cfg, "topology_builder", "auto")
    _use_dl_pre = (_tb_choice_pre == "dl_field"
                   or (_tb_choice_pre == "auto" and ff.kind == "dlfield"))
    if want_lammps and cfg.run_moltemplate and not _use_dl_pre:
        input_file = lt_writer.write_lammps_input(
            out_dir,
            data_file=("packed_box.data" if _pack_after_mt else "system.data"),
            settings_file="system.in.settings",
            init_file="system.in.init",
            charges_file="system.in.charges",
            ensemble=cfg.lammps.ensemble,
            temperature=cfg.lammps.temperature,
            pressure=cfg.lammps.pressure,
            steps=cfg.lammps.steps,
            timestep=cfg.lammps.timestep,
            tdamp_fs=cfg.lammps.tdamp_fs,
            pdamp_fs=cfg.lammps.pdamp_fs,
            thermostat=cfg.lammps.thermostat,
            barostat=cfg.lammps.barostat,
            pressure_coupling=cfg.lammps.pressure_coupling,
            minimize=cfg.lammps.minimize,
            min_etol=cfg.lammps.min_etol,
            min_ftol=cfg.lammps.min_ftol,
            min_maxiter=cfg.lammps.min_maxiter,
            min_maxeval=cfg.lammps.min_maxeval,
            multistage=cfg.lammps.multistage,
            nvt_steps=cfg.lammps.nvt_steps,
            nvt_temperature=cfg.lammps.nvt_temperature,
            npt_steps=cfg.lammps.npt_steps,
            thermo_every=cfg.lammps.thermo_every,
            dump_every=cfg.lammps.dump_every,
            seed=cfg.lammps.seed,
        )

    if want_gromacs:
        _p("Writing GROMACS files (.gro / .top / .mdp)")
        from . import gromacs_writer
        gromacs_files = gromacs_writer.write_all(
            chain, out_dir,
            project_name=cfg.project_name,
            box_ang=tuple(box_shape.bounding_box()),
            include_itp=cfg.gromacs_include_itp,
            ff_name=ff.display_name,
            ensemble=cfg.lammps.ensemble,
            temperature=cfg.lammps.temperature,
            pressure=cfg.lammps.pressure,
            steps=cfg.lammps.steps,
            timestep_fs=cfg.lammps.timestep,
            tdamp_fs=cfg.lammps.tdamp_fs,
            pdamp_fs=cfg.lammps.pdamp_fs,
            thermostat=cfg.lammps.thermostat,
            barostat=cfg.lammps.barostat,
            pressure_coupling=cfg.lammps.pressure_coupling,
            minimize=cfg.lammps.minimize,
            multistage=cfg.lammps.multistage,
            nvt_steps=cfg.lammps.nvt_steps,
            nvt_temperature=cfg.lammps.nvt_temperature,
            npt_steps=cfg.lammps.npt_steps,
            thermo_every=cfg.lammps.thermo_every,
            dump_every=cfg.lammps.dump_every,
            seed=cfg.lammps.seed,
        )

    # 6. OPTIONAL topology-builder execution.
    # The default behaviour is generation-only — the tool builds structures
    # and writes .lt / .top / .mdp / .in files so users can inspect them.
    # Set cfg.run_moltemplate=True (checkbox in GUI Run tab) to also invoke
    # the appropriate binary for the chosen FF:
    #   moltemplate_native / gaff  -> moltemplate.sh
    #   dlfield                    -> dl_field
    data_file = None
    dlfield_result = None

    # Choose which topology builder to run. Config.topology_builder is
    # "auto" (based on ff.kind), "moltemplate", or "dl_field".
    _tb_choice = getattr(cfg, "topology_builder", "auto")
    if _tb_choice == "auto":
        use_dlfield = (ff.kind == "dlfield")
    elif _tb_choice == "dl_field":
        use_dlfield = True
    else:
        use_dlfield = False
    _p(f"Topology builder (config={_tb_choice!r}) -> "
       f"{'dl_field' if use_dlfield else 'moltemplate.sh'}")

    # If this is the DL_FIELD path, ALWAYS write the polymer.control file so
    # the user has a scaffold they can inspect / run by hand — regardless of
    # whether Auto-run is on. The dl_field binary invocation itself is still
    # gated on cfg.run_moltemplate below.
    control_file: Optional[Path] = None
    if use_dlfield:
        from .dlfield_runner import write_dlfield_control
        _struct = out_dir / (cfg.project_name + "_opt.xyz")
        if not _struct.exists():
            _struct = out_dir / (cfg.project_name + ".mol2")
        control_file = write_dlfield_control(
            out_dir / "polymer.control",
            _struct,
            ff_key=ff.key,
            output_engine=("gromacs" if cfg.engine == "gromacs"
                           else "lammps" if cfg.engine in ("lammps", "both") else "none"),
            box_ang=tuple(box_shape.bounding_box()),
        )
        _p(f"Wrote DL_FIELD control file -> {control_file}")

    # DL_FIELD-based FFs ALWAYS run dl_field (the checkbox is ignored for this
    # path — the whole point of picking a DL_FIELD FF is to get its topology
    # output). Only the Moltemplate-native path is gated by run_moltemplate,
    # because there users may want to hand-edit .lt files before invoking
    # moltemplate.sh.
    effective_run = cfg.run_moltemplate or use_dlfield

    if not effective_run:
        _p("!" * 60)
        _p("!! GENERATION-ONLY MODE: moltemplate.sh NOT executed.")
        _p("!! Only the scaffold .lt files were written.")
        _p("!! Tick 'Auto-run' on the Export tab to also run moltemplate.sh.")
        _p("!" * 60)
    elif use_dlfield:
        _p("=" * 60)
        _p("Auto-running dl_field for " + ff.display_name)
        _p("=" * 60)
        from .dlfield_runner import run_dlfield, find_dl_field
        _dl_exe = find_dl_field(dl_lib.parent if dl_lib else None)
        _p(f"  DL_FIELD lib dir: {dl_lib}")
        _p(f"  dl_field binary : {_dl_exe}")
        if _dl_exe is None:
            raise RuntimeError(
                "dl_field executable not found. Set $DL_FIELD_EXE, put "
                "'dl_field' next to the lib/ folder, or add it to PATH."
            )
        if True:
            # NOTE: name this `structure_path`, not `structure` — the module
            # `structure` is imported at the top of this file and used above
            # (structure.write(...)); shadowing it here would turn every
            # `structure` reference in the function into a not-yet-assigned
            # local (UnboundLocalError).
            structure_path = out_dir / f"{cfg.project_name}_opt.xyz"
            if not structure_path.exists():
                structure_path = out_dir / f"{cfg.project_name}.mol2"
            _p(f"  Input structure : {structure_path}")
            _p(f"  Control file    : {control_file}")
            _p(f"  Command         : cd {out_dir} && {_dl_exe} polymer.control")
            dl_engine = ("gromacs" if cfg.engine == "gromacs"
                         else "lammps" if cfg.engine in ("lammps", "both") else "none")
            dlfield_result = run_dlfield(
                structure_path, ff.key, work_dir=out_dir,
                dl_field_dir=dl_lib.parent if dl_lib else None,
                output_engine=dl_engine,
                box_ang=tuple(box_shape.bounding_box()),
            )
            _p(f"  Exit code       : {dlfield_result.return_code}")
            _p(f"  Output files    : {len(dlfield_result.outputs)}")
            for pf in dlfield_result.outputs:
                _p(f"    -> {pf}")
            hoisted_paths = _hoist_dlfield_outputs(out_dir, cfg.engine)
            if hoisted_paths:
                _p("  Hoisted to top of output dir:")
                for h in hoisted_paths:
                    _p("    -> " + str(h))
                dlfield_result.outputs = sorted(set(list(dlfield_result.outputs) + hoisted_paths), key=str)

            # If the user asked for more than 1 chain, replicate the single-
            # chain lammps data into an N-chain packed box (packmol-based,
            # non-hybrid, ported from create_box_lammps_nonhybrid_chains.py).
            n_chains_wanted = int(cfg.box.n_chains)
            if n_chains_wanted > 1 and cfg.engine in ("lammps", "both"):
                single_data = None
                for cand in ("lammps1.data", "lammps.data"):
                    p = out_dir / cand
                    if p.exists():
                        single_data = p; break
                if single_data is None:
                    _p(f"[replicator] no single-chain data file found — skipping packing")
                else:
                    _p(f"[replicator] Packing {n_chains_wanted} chains into "
                       f"box ({box_shape.a:.1f} × {box_shape.b:.1f} × "
                       f"{box_shape.c:.1f} Å) via packmol (falls back to grid)")
                    from .lammps_replicator import replicate_single_chain
                    packed_data = replicate_single_chain(
                        single_data, n_chains_wanted,
                        (box_shape.a, box_shape.b, box_shape.c),
                        out_dir / "packed_box.data",
                        packmol_path=getattr(cfg.box, "packmol_path", "") or None,
                        seed=int(getattr(cfg.box, "packmol_seed", -1)),
                        tolerance=float(getattr(cfg.box, "packmol_tolerance", 2.0)),
                    )
                    _p(f"[replicator] Wrote {packed_data}")
                    dlfield_result.outputs = sorted(
                        set(list(dlfield_result.outputs) + [packed_data]), key=str)

            # -------- GROMACS-native packing via gmx insert-molecules -------
            # When the user picks engine=gromacs we can't use packmol/LAMMPS
            # replicator (they operate on lammps.data). Instead, run
            # `gmx insert-molecules` on the dl_field-produced .gro, using
            # either an explicit chain count or a target atom count.
            gmx_target_atoms = int(getattr(cfg.box, "gmx_target_atoms", 0))
            _need_gmx_pack = (cfg.engine in ("gromacs", "both") and
                              (n_chains_wanted > 1 or gmx_target_atoms > 0))
            if _need_gmx_pack:
                single_gro = None
                for cand in ("gromacs.gro", "gromacs1.gro"):
                    p = out_dir / cand
                    if p.exists():
                        single_gro = p; break
                if single_gro is None:
                    dlf = out_dir / "dlf_output1"
                    for cand in ("gromacs.gro", "gromacs1.gro"):
                        p = dlf / cand
                        if p.exists():
                            single_gro = p; break
                if single_gro is None:
                    _p("[gmx-pack] no single-chain gromacs.gro found — skipping packing")
                else:
                    top_file = single_gro.parent / "gromacs.top"
                    try:
                        from .gromacs_packer import pack_with_gmx_insert
                        # GROMACS wants nm, not Å. Divide by 10.
                        box_nm = (box_shape.a / 10.0, box_shape.b / 10.0, box_shape.c / 10.0)
                        _p(f"[gmx-pack] Packing via gmx insert-molecules into "
                           f"{box_nm[0]:.2f}×{box_nm[1]:.2f}×{box_nm[2]:.2f} nm "
                           f"({'target atoms='+str(gmx_target_atoms) if gmx_target_atoms>0 else 'n_chains='+str(n_chains_wanted)})")
                        gmx_result = pack_with_gmx_insert(
                            single_gro, top_file, box_nm,
                            n_chains=(None if gmx_target_atoms > 0 else n_chains_wanted),
                            atom_limit=(gmx_target_atoms if gmx_target_atoms > 0 else None),
                            try_count=int(getattr(cfg.box, "gmx_try_count", 100000)),
                            pre_minimise=bool(getattr(cfg.box, "gmx_pre_minimise", True)),
                        )
                        _p(f"[gmx-pack] Inserted {gmx_result.n_inserted} copies → "
                           f"{gmx_result.filled_gro}")
                        if gmx_result.updated_top:
                            _p(f"[gmx-pack] Updated topology: {gmx_result.updated_top}")
                        # Hoist the packed .gro to the project root so it's easy to find.
                        packed_top = out_dir / "packed_box.gro"
                        try:
                            shutil.copy2(gmx_result.filled_gro, packed_top)
                            dlfield_result.outputs = sorted(
                                set(list(dlfield_result.outputs) + [packed_top]), key=str)
                        except Exception:
                            pass
                        # ...and its TOPOLOGY with it. Hoisting the coordinates
                        # alone is what made a correctly packed box look broken.
                        #
                        # The packer does update the topology — measured on a
                        # 206-chain PIB box, dlf_output1/gromacs.top correctly
                        # read "XYZ 206". But it stays down in dlf_output1/
                        # while the .gro is lifted to the project root, next to
                        # a DIFFERENT topology (<project>.top) that still says
                        # 1 molecule. Pairing those two — the obvious pair, and
                        # the only one visible — gives grompp a coordinate file
                        # of 49852 atoms and a topology expecting 242.
                        #
                        # The matching pair now sits together, named alike.
                        if gmx_result.updated_top:
                            packed_topol = out_dir / "packed_box.top"
                            try:
                                shutil.copy2(gmx_result.updated_top, packed_topol)
                                # ...and everything the .top #includes.
                                #
                                # A GROMACS topology is not one file. This one
                                # carries `#include "gromacs1.itp"`, resolved
                                # relative to the .top's own directory — so
                                # copying the .top alone moved it away from its
                                # includes and grompp stopped with
                                # 'Topology include file "gromacs1.itp" not
                                # found'. It had already accepted the atom
                                # counts by then, which is the part this whole
                                # change was about; the includes were simply
                                # the next thing to trip over.
                                src_dir = Path(gmx_result.updated_top).parent
                                for inc in sorted(src_dir.glob("*.itp")):
                                    try:
                                        shutil.copy2(inc, out_dir / inc.name)
                                    except Exception:
                                        pass
                                dlfield_result.outputs = sorted(
                                    set(list(dlfield_result.outputs)
                                        + [packed_topol]), key=str)
                                _p(f"[gmx-pack] Use packed_box.gro WITH "
                                   f"packed_box.top ({gmx_result.n_inserted} "
                                   f"chains). The single-chain .top in this "
                                   f"folder describes one chain only.")
                            except Exception as exc:
                                _p(f"[gmx-pack] WARNING: could not copy the "
                                   f"packed topology ({exc}). Use "
                                   f"{gmx_result.updated_top} with "
                                   f"packed_box.gro — NOT the single-chain "
                                   f".top, which would not match.")
                    except Exception as e:
                        _p(f"[gmx-pack] FAILED: {type(e).__name__}: {e}")

            if cfg.engine in ("lammps", "both"):
                names = [p.name for p in dlfield_result.outputs]
                # dl_field writes lammps<N>.data (numbered) and dl_poly.FIELD
                _has_lammps = (
                    "lammps.data" in names
                    or "dl_poly.FIELD" in names
                    or "FIELD" in names
                    or any(n.startswith("lammps") and n.endswith(".data") for n in names)
                )
                if not _has_lammps:
                    # dl_field returned OK but produced no topology — usually an
                    # atom-typing failure. Show the last 40 lines of DL_FIELD's
                    # own output so the user sees exactly which atom / bond /
                    # residue DL_FIELD couldn't handle.
                    dl_log_path = out_dir / "dl_field.log"
                    tail = ""
                    if dl_log_path.exists():
                        tail = "\n".join(dl_log_path.read_text(errors="replace").splitlines()[-40:])
                    if not tail:
                        tail = "\n".join(dlfield_result.log.splitlines()[-40:])
                    raise RuntimeError(
                        "dl_field ran (exit 0) but produced NO lammps.data / FIELD.\n\n"
                        "This usually means DL_FIELD couldn't type one or more atoms\n"
                        "against its force-field library (e.g. bad element labels in\n"
                        "the xyz, or a group DL_FIELD doesn't recognise).\n\n"
                        "Full log: " + str(dl_log_path) + "\n"
                        "Working dir: " + str(out_dir) + "\n\n"
                        "Last lines of dl_field output:\n\n" + tail
                    )
            if dlfield_result.return_code != 0:
                dl_log_path = out_dir / "dl_field.log"
                tail = ""
                if dl_log_path.exists():
                    tail = "\n".join(dl_log_path.read_text(errors="replace").splitlines()[-40:])
                if not tail:
                    tail = "\n".join(dlfield_result.log.splitlines()[-40:])
                raise RuntimeError(
                    "dl_field returned " + str(dlfield_result.return_code) + ".\n"
                    "Working dir: " + str(out_dir) + "\n"
                    "Full log: " + str(dl_log_path) + "\n"
                    "Control file: " + str(dlfield_result.control_file) + "\n\n"
                    "Last lines of dl_field output:\n\n" + tail
                )
    else:
        # Moltemplate-native pipeline
        if want_lammps:
            if find_moltemplate():
                data_file = run_moltemplate(system_lt, work_dir=out_dir)
                if _pack_after_mt and data_file and Path(data_file).exists():
                    _p(f"[replicator] Packing {_n_chains_mt} chains into box "
                       f"({box_shape.a:.1f} × {box_shape.b:.1f} × "
                       f"{box_shape.c:.1f} Å) via packmol (falls back to grid)")
                    from .lammps_replicator import replicate_single_chain
                    packed_data = replicate_single_chain(
                        Path(data_file), _n_chains_mt,
                        (box_shape.a, box_shape.b, box_shape.c),
                        out_dir / "packed_box.data",
                        packmol_path=getattr(cfg.box, "packmol_path", "") or None,
                        seed=int(getattr(cfg.box, "packmol_seed", -1)),
                        tolerance=float(getattr(cfg.box, "packmol_tolerance", 2.0)),
                    )
                    _p(f"[replicator] Wrote {packed_data} "
                       f"(run.in reads this file; system.data is the single chain)")
                    data_file = packed_data
            else:
                log.warning(
                    "moltemplate.sh not found; skipping. LAMMPS data will not be generated. "
                    "Install moltemplate to enable this step."
                )

    # ---- Optional: multi-component blend packing --------------------------
    blend_out = None
    try:
        from .blend_replicator import run_blend_from_config
        blend_out = run_blend_from_config(cfg, out_dir)
        if blend_out is not None:
            _p(f"[blend] Wrote merged multi-component data -> {blend_out}")
    except Exception as e:  # pragma: no cover - defensive
        _p(f"[blend] Skipped ({e.__class__.__name__}: {e})")

    return {
        "output_dir": str(out_dir),
        "engine": cfg.engine,
        "runner": ("dl_field" if ff.kind == "dlfield" else "moltemplate"),
        "monomer_files": [str(out_dir / f"{m.name}_opt.xyz") for m in monomers],
        "chain_lt": str(chain_lt),
        "system_lt": str(system_lt),
        "input_file": str(input_file) if input_file else None,
        "ff_lt": str(ff_lt_path) if ff_lt_path else None,
        "data_file": str(data_file) if data_file else None,
        "gromacs_files": gromacs_files,
        "dlfield_outputs": ([str(p) for p in dlfield_result.outputs]
                            if dlfield_result else []),
        "blend_data": str(blend_out) if blend_out else None,
        "config": asdict(cfg),
    }
