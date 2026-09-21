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
    def _copy(src: Path, dst: Path) -> None:
        # Remove the old file first: a failed copy must not leave a stale
        # lammps.data from a previous run as the one users open.
        try:
            if dst.exists():
                dst.unlink()
            shutil.copy2(src, dst)
            hoisted.append(dst)
        except OSError as exc:
            log.warning("Could not copy %s -> %s: %s", src, dst, exc)

    for src in lammps_data_files + lammps_in_files:
        _copy(src, out_dir / src.name)
    # Convenience aliases for the single-copy common case
    if len(lammps_data_files) == 1:
        _copy(lammps_data_files[0], out_dir / "lammps.data")
    if len(lammps_in_files) == 1:
        _copy(lammps_in_files[0], out_dir / "lammps.in")

    for src in sorted(dlf.glob("*.mdp")):
        _copy(src, out_dir / src.name)
    return hoisted


_N_STAGES = 7


class PipelineCancelled(RuntimeError):
    """The user pressed Cancel; nothing is wrong with the inputs."""


def run_pipeline(cfg: Config, progress: Optional[Callable[[str], None]] = None,
                 cancel=None) -> dict:
    """Execute the whole pipeline described by `cfg` and return summary.

    ``cancel`` is a :class:`paaf.cell.packing.CancelToken`. It is polled at
    every log line and inside the long steps (optimiser chunks, geometry
    push-off stages), and external programs (dl_field, moltemplate.sh,
    packmol) are killed the moment it is set. Raises PipelineCancelled.
    """
    from .cell.packing import PackCancelled

    def _check() -> None:
        if cancel is not None and cancel.is_cancelled():
            raise PipelineCancelled("cancelled by user")

    def _p(msg: str) -> None:
        log.info(msg)
        if progress:
            progress(msg)
        _check()

    # Stage markers. The GUI parses "[stage k/N] label" to drive its
    # progress bar; everything else is plain log text.
    def _stage(k: int, label: str) -> None:
        bench.phase(f"{k}. {label}")
        _p(f"[stage {k}/{_N_STAGES}] {label}")

    from . import benchmark as bench
    from .run_log import run_log
    out_dir = Path(cfg.output_dir) / cfg.project_name
    with run_log(out_dir) as rl:
        with bench.benchmark("main_pipeline", out_dir,
                             workload=_pipeline_workload(cfg),
                             emit=progress) as b:
            try:
                result = _run_pipeline_body(cfg, _p, _stage, _check, cancel)
            except PackCancelled as exc:
                raise PipelineCancelled(str(exc)) from exc
            b.metric(chains=int(cfg.box.n_chains),
                     repeat_units=int(cfg.chain.n_monomers))
            for f in (out_dir / "packed_box.data", result.get("data_file"),
                      out_dir / "lammps.data", out_dir / "lammps1.data",
                      out_dir / "system.data", out_dir / "packed_box.gro",
                      out_dir / "system.gro"):
                b.size_from(f)
    result["benchmark_file"] = str(b.txt_file) if b.txt_file else None
    result["log_file"] = rl.get("log_file")
    result["energy_file"] = rl.get("energy_file")
    if rl.get("energy_file"):
        _p(f"Optimisation energies: {rl['energy_file']}")
    _p(f"Run log: {rl['log_file']}")
    return result


def _pipeline_workload(cfg: Config) -> dict:
    """The inputs that set a pipeline run's cost (hashed into workload_id)."""
    box = cfg.box
    return {
        "monomers": [Path(m.file).name for m in cfg.monomers],
        "repeat_units": cfg.chain.n_monomers,
        "chain_mode": cfg.chain.mode,
        "n_chains": box.n_chains,
        "box": (f"{box.shape} {box.a:g}x{box.b:g}x{box.c:g}"
                + (f" @ {box.density_g_cm3} g/cm3" if box.use_density else "")),
        "force_field": cfg.force_field.key,
        "engine": cfg.engine,
        "lammps_styles": getattr(cfg, "lammps_styles", "hybrid"),
        "optimizer": (f"{cfg.optimizer.ff} {cfg.optimizer.steps} steps"
                      if cfg.optimizer.enabled else "off"),
        "packmol": bool(box.packmol),
        "blend_components": (len(cfg.blend.components)
                             if cfg.blend.enabled else 0),
    }


def _run_pipeline_body(cfg: Config, _p, _stage, _check, cancel) -> dict:
    from .cell.packing import PackCancelled  # noqa: F401  (re-raised by caller)
    out_dir = Path(cfg.output_dir) / cfg.project_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and optimize monomers
    _stage(1, "Loading monomers")
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
                report_energy=getattr(cfg.optimizer, "report_energy", True),
                fallback=getattr(cfg.optimizer, "fallback", True),
                strict=not getattr(cfg.optimizer, "fallback", True),
                cancel=cancel,
            )
        structure.write(m.molecule, out_dir / f"{m.name}_opt.xyz")
        monomers.append(m)

    # 2. Build chain
    _stage(2, f"Building chain ({cfg.chain.n_monomers} units, mode={cfg.chain.mode})")
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
        # Step 3 below minimises with the user's Optimize-page settings; no
        # hidden extra pass inside the builder.
        optimize=False,
        cancel=cancel,
    )
    chain.name = cfg.project_name
    structure.write(chain, out_dir / f"{cfg.project_name}.mol2")

    # 3. Optimize chain
    if cfg.optimizer.enabled:
        _stage(3, "Optimizing full chain")
    else:
        _stage(3, "Chain optimisation skipped (disabled)")
    if cfg.optimizer.enabled:
        optimizer.optimize(chain, ff=cfg.optimizer.ff, steps=cfg.optimizer.steps,
                           tol=cfg.optimizer.tol, algorithm=cfg.optimizer.algorithm,
                report_energy=getattr(cfg.optimizer, "report_energy", True),
                fallback=getattr(cfg.optimizer, "fallback", True),
                strict=not getattr(cfg.optimizer, "fallback", True),
                           cancel=cancel)
        structure.write(chain, out_dir / f"{cfg.project_name}_opt.xyz")

    # 3b. Geometry sanity: mbuild's straight-line join overlaps bulky side
    # groups and a local minimiser cannot untangle that. dl_field infers
    # bonds from distances, so a clashed chain reads as the wrong molecule
    # ("Fail to decide type of unsaturated C atom"). Rebuild coordinates
    # from the bond topology when needed — general for any polymer.
    from .geometry_repair import ensure_clean_geometry, is_clean
    _geom_ok = True
    if not is_clean(chain):
        _geom_ok = ensure_clean_geometry(chain, _p, seed=int(cfg.chain.seed or 7),
                                         cancel=cancel)
        if _geom_ok and cfg.optimizer.enabled:
            # Push-off geometry is approximate (covalent-radius bond lengths,
            # ideal angles): polish it with the real force field, then make
            # sure the minimiser did not fold it back into a clash.
            _p("Re-optimising the repaired chain")
            optimizer.optimize(chain, ff=cfg.optimizer.ff, steps=cfg.optimizer.steps,
                               tol=cfg.optimizer.tol, algorithm=cfg.optimizer.algorithm,
                report_energy=getattr(cfg.optimizer, "report_energy", True),
                fallback=getattr(cfg.optimizer, "fallback", True),
                strict=not getattr(cfg.optimizer, "fallback", True),
                               cancel=cancel)
            if not is_clean(chain):
                _geom_ok = ensure_clean_geometry(chain, _p, seed=int(cfg.chain.seed or 7) + 1,
                                                 cancel=cancel)
    if not _geom_ok:
        _p("WARNING: chain geometry still has overlaps; force-field typing "
           "from coordinates may fail.")
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

    _stage(4, "Assigning atom types")
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
        from .type_guard import check_all, elements_for_ff
        _problems = check_all({a.index: a.element for a in chain.atoms},
                              {a.index: a.ff_type for a in chain.atoms
                               if a.ff_type},
                              elements_for_ff(ff.key))
        if _problems:
            _p("!" * 60)
            for _idx, _msg in _problems[:10]:
                _p(f"!! atom {_idx}: {_msg}")
            raise RuntimeError(
                f"{len(_problems)} atoms carry a force-field type belonging to "
                f"a different element. Writing this would give them the wrong "
                f"mass and charge, and nothing downstream would object. "
                f"First: atom {_problems[0][0]} — {_problems[0][1]}")

    # 4b. United-atom beads: absorb hydrogens (UA-only force field, or the
    # Advanced typing UA/AA mix). New chain, fewer atoms, bead masses noted.
    _hyb = None
    _ua_key = getattr(cfg.force_field, "ua_secondary_key", None)
    if ff.kind == "moltemplate_native" and not ff.united_atom and not _ua_key:
        # UA-block types picked straight from the OPLS-AA table count too.
        from .ua_hybrid import implicit_ua_library
        _ua_key = implicit_ua_library(ff, chain)
        if _ua_key:
            _p(f"United-atom bead types found on the chain — absorbing their "
               f"hydrogens ({get_ff(_ua_key).display_name}).")
    if ff.kind == "moltemplate_native" and (ff.united_atom or _ua_key):
        from .ua_hybrid import apply_hybrid
        _ua_ff = ff if ff.united_atom else get_ff(_ua_key)
        _hyb = apply_hybrid(chain, ff, _ua_ff, out_dir, _p)
        chain = _hyb.chain
        chain.name = cfg.project_name
        structure.write(chain, out_dir / f"{cfg.project_name}_ua.mol2")

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
        if _hyb is not None and _hyb.removed_h:
            # UA beads carry their absorbed hydrogens' mass; the element-mass
            # sum above does not, so grow the volume to match.
            from .cell.amorphous import _mass_of
            _bare = max(_mass_of(chain), 1e-9)
            box_shape = box_shape.scaled(
                ((_bare + _hyb.removed_h * 1.008) / _bare) ** (1.0 / 3.0))
        _p(f"Box scaled to target density {cfg.box.density_g_cm3} g/cm³: {box_shape}")

    _stage(5, "Sizing the box")
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
        if _hyb is not None:
            _mass_g += _n_for_density * _hyb.removed_h * 1.008   # absorbed H
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
    except Exception as exc:
        _p(f"WARNING: box density check could not be computed ({exc}).")

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
        chain_lt = lt_writer.write_chain_lt(
            chain, out_dir, ff, name=cfg.project_name,
            inherit=(_hyb.inherit if _hyb else None),
            lt_include=(_hyb.lt_include if _hyb else None),
            bond_type=(_hyb.bond_type if _hyb else None))
        system_lt = lt_writer.write_system_lt(
            out_dir, chain_lt, n_chains=1,
            box=list(box_shape.lammps_params()),
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
        _stage(6, "Topology: generation-only (moltemplate.sh not run)")
        _p("!" * 60)
        _p("!! GENERATION-ONLY MODE: moltemplate.sh NOT executed.")
        _p("!! Only the scaffold .lt files were written.")
        _p("!! Tick 'Auto-run' on the Export tab to also run moltemplate.sh.")
        _p("!" * 60)
    elif use_dlfield:
        _stage(6, "Typing with dl_field (" + ff.display_name + ")")
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
                cancel=cancel,
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
                _stage(7, f"Packing {n_chains_wanted} chains into the box")
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
                        box_shape.lammps_params(),
                        out_dir / "packed_box.data",
                        packmol_path=getattr(cfg.box, "packmol_path", "") or None,
                        seed=int(getattr(cfg.box, "packmol_seed", -1)),
                        tolerance=float(getattr(cfg.box, "packmol_tolerance", 2.0)),
                        use_packmol=bool(getattr(cfg.box, "packmol", True)),
                        cancel=cancel,
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
                            cancel=cancel,
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
                        except OSError as _cp_err:
                            _p(f"WARNING: could not copy {gmx_result.filled_gro} to "
                               f"{packed_top} ({_cp_err}); use the file in "
                               f"{Path(gmx_result.filled_gro).parent}")
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
                                    except OSError as _cp_err:
                                        _p(f"WARNING: could not copy {inc.name} into "
                                           f"{out_dir} ({_cp_err}); grompp there will "
                                           f"not find this #include")
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
                    hint = ""
                    if "unsaturated" in tail or "Fail to decide type" in tail:
                        from .geometry_repair import find_clashes as _fc
                        _cl, _bb = _fc(chain)
                        hint = ("\n'Fail to decide type of unsaturated C atom' means dl_field\n"
                                "perceived a bond count from the coordinates that does not\n"
                                "match the element (it infers bonds from distances). "
                                + (f"The chain\nstill has {len(_cl)} overlapping pair(s) and "
                                   f"{len(_bb)} distorted bond(s).\n"
                                   if (_cl or _bb) else
                                   "PAAF's own\ngeometry check found no clashes, so the atom is "
                                   "probably one\nthis force field has no type for.\n"))
                    raise RuntimeError(
                        "dl_field ran (exit 0) but produced NO lammps.data / FIELD.\n\n"
                        "This usually means DL_FIELD couldn't type one or more atoms\n"
                        "against its force-field library (e.g. bad element labels in\n"
                        "the xyz, or a group DL_FIELD doesn't recognise)." + hint + "\n\n"
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
        _stage(6, "Running moltemplate.sh")
        if want_lammps:
            if find_moltemplate():
                try:
                    data_file = run_moltemplate(system_lt, work_dir=out_dir, cancel=cancel)
                except RuntimeError as _mt_err:
                    _msg = str(_mt_err)
                    _missing = ("No bond types" in _msg or "No angle types" in _msg
                                or "No dihedral types" in _msg or "No improper types" in _msg)
                    if not (_hyb is not None and _hyb.native_beads and _missing):
                        raise
                    from .ua_hybrid import bridge_native_beads
                    bridge_native_beads(_hyb, ff, out_dir, _p)
                    chain = _hyb.chain
                    chain_lt = lt_writer.write_chain_lt(
                        chain, out_dir, ff, name=cfg.project_name,
                        inherit=_hyb.inherit, lt_include=_hyb.lt_include,
                        bond_type=_hyb.bond_type)
                    system_lt = lt_writer.write_system_lt(
                        out_dir, chain_lt, n_chains=1,
                        box=list(box_shape.lammps_params()), name="system", ff=ff)
                    data_file = run_moltemplate(system_lt, work_dir=out_dir, cancel=cancel)
                if _hyb is not None and _hyb.bead_masses:
                    from .ua_hybrid import patch_data_masses
                    _nm = patch_data_masses(Path(data_file), _hyb.bead_masses)
                    _p(f"[united-atom] bead masses written for {_nm} type(s) in {Path(data_file).name}")
                    if not ff.united_atom:
                        # UA beads next to all-atom groups: the two libraries'
                        # charges do not add up to a neutral molecule.
                        from .ua_hybrid import rebalance_hybrid_charges
                        rebalance_hybrid_charges(
                            Path(data_file),
                            Path(data_file).with_name("system.in.charges"),
                            list(_hyb.bead_masses), _p)
                if _pack_after_mt and data_file and Path(data_file).exists():
                    _stage(7, f"Packing {_n_chains_mt} chains into the box")
                    _p(f"[replicator] Packing {_n_chains_mt} chains into box "
                       f"({box_shape.a:.1f} × {box_shape.b:.1f} × "
                       f"{box_shape.c:.1f} Å) via packmol (falls back to grid)")
                    from .lammps_replicator import replicate_single_chain
                    packed_data = replicate_single_chain(
                        Path(data_file), _n_chains_mt,
                        box_shape.lammps_params(),
                        out_dir / "packed_box.data",
                        packmol_path=getattr(cfg.box, "packmol_path", "") or None,
                        seed=int(getattr(cfg.box, "packmol_seed", -1)),
                        tolerance=float(getattr(cfg.box, "packmol_tolerance", 2.0)),
                        use_packmol=bool(getattr(cfg.box, "packmol", True)),
                        cancel=cancel,
                    )
                    _p(f"[replicator] Wrote {packed_data} "
                       f"(run.in reads this file; system.data is the single chain)")
                    data_file = packed_data
            else:
                log.warning(
                    "moltemplate.sh not found; skipping. LAMMPS data will not be generated. "
                    "Install moltemplate to enable this step."
                )

    # ---- Hybrid -> non-hybrid LAMMPS styles (DL_FIELD route) --------------
    nonhybrid_files: list = []
    _style_choice = str(getattr(cfg, "lammps_styles", "hybrid") or "hybrid").lower()
    if (dlfield_result is not None and want_lammps
            and _style_choice in ("non_hybrid", "nonhybrid", "both")):
        from .lammps_hybrid import convert_to_nonhybrid, is_hybrid_input
        _in = out_dir / "lammps.in"
        # Only the file lammps.in will read: the packed N-chain box when
        # there is one, else the single chain (lammps.data == lammps1.data).
        _packed = out_dir / "packed_box.data"
        if _packed.exists():
            _datas = [_packed]
        else:
            _datas = [out_dir / "lammps.data"] if (out_dir / "lammps.data").exists() \
                     else [out_dir / "lammps1.data"]
        if _in.exists() and is_hybrid_input(_in):
            _rd = "packed_box.data" if _packed.exists() else _datas[0].name
            _nh_in, _nh_data = convert_to_nonhybrid(
                _in, _datas, out_dir / "non_hybrid", read_data=_rd)
            nonhybrid_files = [_nh_in] + list(_nh_data)
            _p("[styles] Non-hybrid LAMMPS files written to " + str(out_dir / "non_hybrid"))
            for f in nonhybrid_files:
                _p("    -> " + str(f))
            if _rd:
                _p("[styles] non_hybrid/lammps.in reads packed_box.data "
                   "(the N-chain box).")
            dlfield_result.outputs = sorted(
                set(list(dlfield_result.outputs) + nonhybrid_files), key=str)
        elif _in.exists():
            _p("[styles] lammps.in already uses plain styles — nothing to convert")

    _p(f"[stage done/{_N_STAGES}] Finished — files in {out_dir}")
    from .benchmark import phase as _bench_phase
    _bench_phase("Blend packing and charge checks")

    # ---- Optional: multi-component blend packing --------------------------
    blend_out = None
    try:
        from .blend_replicator import run_blend_from_config
        blend_out = run_blend_from_config(cfg, out_dir)
        if blend_out is not None:
            _p(f"[blend] Wrote merged multi-component data -> {blend_out}")
    except Exception as e:  # pragma: no cover - defensive
        _p(f"[blend] Skipped ({e.__class__.__name__}: {e})")

    # Net-charge check on the data file users will actually run.
    net_charge = None
    _cands = [out_dir / "packed_box.data"]
    if data_file:
        _cands.insert(0, Path(data_file))
    _cands += [out_dir / "lammps1.data", out_dir / "lammps.data",
               out_dir / "system.data"]
    for _cand in _cands:
        if not _cand.exists():
            continue
        try:
            from .cell.cell_export import data_file_charges
            # Moltemplate sets OPLS charges by type in system.in.charges and
            # leaves the data column at zero; the run script includes it.
            _chg = out_dir / "system.in.charges"
            _q = data_file_charges(_cand, _chg if _chg.exists() else None)
        except Exception as exc:
            _p(f"WARNING: could not read charges from {_cand.name} ({exc})")
            break
        if _q is None:
            continue
        _n_at, net_charge, _qmax = _q
        _p(f"Charges: {_n_at} atoms in {_cand.name}, net {net_charge:+.4f} e")
        if abs(net_charge) > 0.05:
            _p(f"WARNING: {_cand.name} carries a net charge of {net_charge:+.3f} e; "
               f"with PPPM LAMMPS adds a neutralising background. Check manual "
               f"type overrides / united-atom beads.")
        elif _qmax == 0.0:
            _p(f"WARNING: every charge in {_cand.name} is zero — electrostatics "
               f"will be missing if this force field is meant to be charged.")
        break

    # ...and the same check on the GROMACS topology, which grompp reads.
    # A .top carries its charges per moleculetype and multiplies them by the
    # [ molecules ] counts, so a rounding error worth 0.001 e on one chain
    # becomes 0.2 e on a 200-chain box — exactly the case worth catching
    # before grompp does.
    net_charge_gromacs = None
    if cfg.engine in ("gromacs", "both"):
        for _tcand in (out_dir / "packed_box.top", out_dir / "gromacs.top",
                       out_dir / "gromacs1.top", out_dir / "system.top"):
            if not _tcand.exists():
                continue
            try:
                from .cell.cell_export import top_file_charges
                _tq = top_file_charges(_tcand)
            except Exception as exc:
                _p(f"WARNING: could not read charges from {_tcand.name} ({exc})")
                break
            if _tq is None:
                break
            _n_at, net_charge_gromacs, _qmax = _tq
            _p(f"Charges: {_n_at} atoms in {_tcand.name}, "
               f"net {net_charge_gromacs:+.4f} e")
            if abs(net_charge_gromacs) > 0.05:
                _p(f"WARNING: {_tcand.name} carries a net charge of "
                   f"{net_charge_gromacs:+.3f} e; grompp will report a "
                   f"non-zero total charge. Check manual type overrides / "
                   f"united-atom beads.")
            elif _qmax == 0.0:
                _p(f"WARNING: every charge in {_tcand.name} is zero — "
                   f"electrostatics will be missing if this force field is "
                   f"meant to be charged.")
            break

    return {
        "output_dir": str(out_dir),
        "engine": cfg.engine,
        "net_charge": net_charge,
        "net_charge_gromacs": net_charge_gromacs,
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
        "nonhybrid_files": [str(p) for p in nonhybrid_files],
        "config": asdict(cfg),
    }
