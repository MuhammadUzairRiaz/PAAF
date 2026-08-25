"""Multi-component blend replicator.

Port of `create_blend_box_lammps_with_mah_nonhybrid.py`. Given a list of
components, each described by a **single-chain LAMMPS data file** produced
by `dl_field` and a **copy count**, this module:

1. Parses each component's LAMMPS data (header type counts + coefficient
   sections + atoms / bonds / angles / dihedrals / impropers).
2. Computes cumulative type-id offsets per component so that atom / bond /
   angle / dihedral / improper type numbers do not collide when the data
   files are merged.
3. Uses packmol to pack every copy of every component into one box.
4. Reads back the packed coordinates and assigns molecule ids consecutively
   (component 0 chain 0, 1, 2, ...; then component 1 chain 0, 1, 2, ...).
5. Writes a single consolidated ``packed_blend.data`` with merged Masses,
   Pair/Bond/Angle/Dihedral/Improper Coeffs and combined Atoms/Bonds/... .
6. Optionally writes a ``packed_blend.in`` that reads the packed data and
   emits merged pair_coeff commands with the correct atom-type offsets.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .lammps_replicator import (
    parse_lammps_data, _clean, _header_count,
    _write_pdb_from_data, _read_xyz_or_pdb_coords,
    _find_packmol, TOPOLOGY_SECTIONS, COEFFICIENT_SECTIONS,
)
from .blend_styles import (
    StyleMismatch,
    StyleBlock, dehybridise_coeff_lines, dehybridise_pair_coeff,
    strip_substyle_tokens,
    find_forcefield_input, merge_style_blocks, read_style_block,
)
from .logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class BlendComponent:
    """One species in the blend.

    Attributes
    ----------
    name : str
        Human label used in log messages and molecule-id comments.
    data_file : Path
        Path to the single-chain LAMMPS data file (from ``dlf_output1/lammps1.data``).
    count : int
        Number of copies to pack.
    """
    name: str
    data_file: Path
    count: int
    header: List[str] = field(default_factory=list)
    sections: Dict[str, List[str]] = field(default_factory=dict)
    natoms_chain: int = 0
    ntypes: Dict[str, int] = field(default_factory=dict)
    #: The LAMMPS input carrying this component's force-field styles. Resolved
    #: at load time; the blend cannot be written without at least one.
    forcefield_input: Optional[Path] = None


# ================================================================ parsing
def _load_components(components: List[BlendComponent]) -> None:
    for comp in components:
        header, sections = parse_lammps_data(comp.data_file)
        comp.header = header
        comp.sections = sections
        comp.natoms_chain = len(_clean(sections.get("Atoms")))
        comp.ntypes = {
            "atom":     _header_count(header, "atom types"),
            "bond":     _header_count(header, "bond types"),
            "angle":    _header_count(header, "angle types"),
            "dihedral": _header_count(header, "dihedral types"),
            "improper": _header_count(header, "improper types"),
        }
        if comp.forcefield_input is None:
            comp.forcefield_input = find_forcefield_input(comp.data_file)
        log.info("Loaded %s from %s: %d atoms/chain, ntypes=%s, styles from %s",
                 comp.name, comp.data_file, comp.natoms_chain, comp.ntypes,
                 comp.forcefield_input or "(none found)")


def _cumulative_offsets(components: List[BlendComponent]) -> List[Dict[str, int]]:
    """Return per-component offsets so type IDs don't collide across components."""
    running = {"atom": 0, "bond": 0, "angle": 0, "dihedral": 0, "improper": 0}
    out: List[Dict[str, int]] = []
    for comp in components:
        out.append(dict(running))
        for k in running:
            running[k] += comp.ntypes.get(k, 0)
    return out


# ================================================================ packmol
def _build_blend_packmol(pdb_files: List[Path], counts: List[int],
                        out_pdb: Path, box_edges: Tuple[float, float, float],
                        tolerance: float, seed: int,
                        packmol_path: Optional[str] = None,
                        regions: Optional[List[Tuple[float, float, float,
                                                     float, float, float]]]
                        = None) -> bool:
    import random
    exe = _find_packmol(explicit=packmol_path)
    if exe is None:
        log.warning("packmol not found; blend needs packmol on PATH or in Box tab.")
        return False
    if seed is None or int(seed) < 0:
        seed = random.randint(1, 2_000_000_000)
        log.info("blend packmol: random seed %d", seed)
    a, b, c = box_edges
    inp = out_pdb.parent / "pack_blend.inp"
    lines = [
        f"tolerance {tolerance}",
        f"seed {seed}",
        "filetype pdb",
        f"output {out_pdb}",
    ]
    # A blend shares one region (the whole box); a LAYERED system gives each
    # component its own sub-box, so packmol confines every species to its
    # slab and the interfaces land exactly where the gaps put them.
    if regions is None:
        regions = [(0.0, 0.0, 0.0, a, b, c)] * len(pdb_files)
    for pdb, n, reg in zip(pdb_files, counts, regions):
        x0, y0, z0, x1, y1, z1 = reg
        lines += [
            f"structure {pdb}",
            f"  number {n}",
            f"  inside box {x0:.4f} {y0:.4f} {z0:.4f} "
            f"{x1:.4f} {y1:.4f} {z1:.4f}",
            "end structure",
        ]
    inp.write_text("\n".join(lines) + "\n")
    log_path = out_pdb.parent / "packmol.log"
    log.info("Packing %d components (total %d copies) into %.1fx%.1fx%.1f Å (log: %s)",
             len(pdb_files), sum(counts), a, b, c, log_path)
    with inp.open() as fh:
        proc = subprocess.run([exe], stdin=fh, capture_output=True, text=True)
    try:
        log_path.write_text(
            "$ " + exe + " < " + str(inp) + "\n\n"
            + "----- STDOUT -----\n" + (proc.stdout or "")
            + "\n----- STDERR -----\n" + (proc.stderr or "") + "\n"
        )
    except OSError:
        pass
    if proc.returncode != 0 or not out_pdb.exists():
        tail = "\n".join((proc.stdout or "").splitlines()[-30:])
        log.warning("packmol (blend) failed rc=%d:\n%s", proc.returncode, tail)
        return False
    return True


# ================================================================ merge
def _offset_coeff_lines(lines: List[str], offset: int) -> List[str]:
    """Add `offset` to the first token (type id) of every coefficient line."""
    out = []
    for line in lines:
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        parts[0] = str(int(parts[0]) + offset)
        out.append(" ".join(parts))
    return out


def _replicate_component_atoms(comp: BlendComponent, coords: List[Tuple[float, float, float]],
                              atom_id_offset: int, atom_type_offset: int,
                              molid_offset: int) -> List[str]:
    """Replicated ``Atoms`` lines for one component, ids already offset.

    ``atom_id_offset`` is the number of atoms every earlier component
    contributed. Without it the second component restarts numbering at 1 and
    collides with the first, while its bonds — which *are* offset — then point
    at atoms that do not exist.
    """
    atoms_orig = _clean(comp.sections.get("Atoms"))
    parsed = []
    for line in atoms_orig:
        parts = line.split()
        parsed.append((int(parts[0]), int(parts[2]), float(parts[3])))
    parsed.sort()
    n = comp.natoms_chain
    if len(coords) != n * comp.count:
        raise RuntimeError(
            f"Component {comp.name}: coord count {len(coords)} != "
            f"{comp.count} chains * {n} atoms/chain = {n * comp.count}")
    out: List[str] = []
    ci = 0
    for chain_i in range(comp.count):
        aid_offset = atom_id_offset + chain_i * n
        molid = molid_offset + chain_i + 1
        for oid, atype, charge in parsed:
            x, y, z = coords[ci]; ci += 1
            out.append(f"{oid + aid_offset} {molid} {atype + atom_type_offset} "
                       f"{charge:.8f} {x:.8f} {y:.8f} {z:.8f}")
    return out


def _component_styles(comp: BlendComponent) -> StyleBlock:
    """The force-field styles for one component.

    The component's own ``.in`` wins: DL_FIELD wrote it for this exact system,
    so it states the cutoffs and exclusions that were actually used. Failing
    that, the styles are read out of the data file's coefficient sections,
    which name their own sub-style — that covers a data file that arrived
    without its input script, and is still a reading rather than a guess.
    """
    if comp.forcefield_input:
        block = read_style_block(comp.forcefield_input)
        if not block.is_empty():
            return block

    from .dlfield_styles import force_field_scheme_in, styles_for_data

    inferred = styles_for_data(comp.data_file)
    if not inferred:
        return StyleBlock()
    scheme = force_field_scheme_in(comp.data_file)
    log.info("%s: no .in found; styles inferred from the data file%s",
             comp.name, f" ({scheme})" if scheme else "")
    block = StyleBlock(source=Path(comp.data_file), directives=dict(inferred))
    block.directives.setdefault("units", "real")
    block.directives.setdefault("atom_style", "full")
    block.directives.setdefault("boundary", "p p p")
    return block


def component_mass_amu(comp: BlendComponent) -> float:
    """Mass of one chain of this component, from the data file's own Masses."""
    masses: Dict[int, float] = {}
    for line in _clean(comp.sections.get("Masses")):
        p = line.split()
        if len(p) >= 2:
            masses[int(p[0])] = float(p[1])
    total = 0.0
    for line in _clean(comp.sections.get("Atoms")):
        p = line.split()
        if len(p) >= 3:
            total += masses.get(int(p[2]), 0.0)
    return total


def box_edge_for_density(components: List[BlendComponent],
                         density_g_cm3: float) -> float:
    """Cubic edge in Å that puts the blend at a target density.

    .. math::

        L = \\left( \\frac{\\sum_i N_i M_i}{N_A \\rho} \\right)^{1/3}

    with the usual 10^24 Å³/cm³ conversion. Packing straight to the bulk
    density is normally too hard for packmol — chains cannot be threaded past
    one another once there is no room — so callers typically ask for a
    fraction of the target here and compress the rest with NPT.
    """
    if density_g_cm3 <= 0:
        raise ValueError("Density must be positive")
    total_amu = sum(component_mass_amu(c) * c.count for c in components)
    if total_amu <= 0:
        raise ValueError("Total mass is zero; are the Masses sections present?")
    volume_cm3 = total_amu / (6.02214076e23 * density_g_cm3)
    return (volume_cm3 * 1.0e24) ** (1.0 / 3.0)


def validate_bond_lengths(atom_lines: List[str], bond_lines: List[str],
                          limit: float = 3.0) -> float:
    """Longest bond in the merged cell; raises if anything exceeds ``limit``.

    A chemical bond is 1–2 Å. Anything much longer means the two atoms are no
    longer neighbours: a coordinate was paired with the wrong atom, or a
    molecule straddles the periodic boundary and was written unwrapped. Both
    produce a file LAMMPS will happily read and then tear apart on the first
    timestep, which is a far more expensive way to find out.
    """
    coords = {}
    for line in atom_lines:
        p = line.split()
        coords[int(p[0])] = (float(p[4]), float(p[5]), float(p[6]))

    worst = 0.0
    offenders: List[Tuple[int, int, int, float]] = []
    for line in bond_lines:
        p = line.split()
        if len(p) < 4:
            continue
        a, b = int(p[2]), int(p[3])
        if a not in coords or b not in coords:
            continue
        d = sum((coords[a][k] - coords[b][k]) ** 2 for k in range(3)) ** 0.5
        worst = max(worst, d)
        if d > limit:
            offenders.append((int(p[0]), a, b, d))

    if offenders:
        shown = "\n".join(f"    bond {i}: atoms {a}–{b} at {d:.2f} Å"
                          for i, a, b, d in sorted(
                              offenders, key=lambda t: -t[3])[:10])
        raise RuntimeError(
            f"{len(offenders)} bonds in the merged blend are longer than "
            f"{limit} Å (longest {worst:.2f} Å). Either a chain's coordinates "
            f"were paired with the wrong atoms, or a molecule crosses the box "
            f"edge and was written unwrapped.\n{shown}\n"
            f"The file has not been written.")
    return worst


def validate_merged_topology(atom_lines: List[str],
                             topology: Dict[str, List[str]]) -> None:
    """Refuse to write a data file whose bonds point at atoms that don't exist.

    This exists because that failure is silent. A merged file with duplicated
    atom ids parses, has the right atom count, has the right molecule count,
    and *looks* correct in every summary — then LAMMPS or OVITO rejects it
    with a line number, thousands of lines from the actual mistake.

    Two things are checked:

    * every atom id appears exactly once — a duplicate means one component's
      numbering was not offset past the previous one's;
    * every id referenced by a bond, angle, dihedral or improper exists.

    Raising here costs a re-run. Not raising costs a simulation that either
    refuses to start or, worse, starts with the wrong connectivity.
    """
    seen: set = set()
    duplicates: set = set()
    for line in atom_lines:
        aid = int(line.split()[0])
        (duplicates if aid in seen else seen).add(aid)

    if duplicates:
        example = sorted(duplicates)[:5]
        raise RuntimeError(
            f"Merged blend has {len(duplicates)} duplicated atom ids "
            f"(e.g. {example}) among {len(atom_lines)} atoms. A component's "
            f"atom numbering was not offset past the components before it, so "
            f"two different atoms share an id and every bond touching them is "
            f"ambiguous. The file has not been written.")

    for section, lines in topology.items():
        missing = {}
        for line in lines:
            parts = line.split()
            for token in parts[2:]:
                aid = int(token)
                if aid not in seen:
                    missing.setdefault(aid, line)
        if missing:
            worst = sorted(missing)[:5]
            raise RuntimeError(
                f"Merged blend: {len(missing)} atom ids referenced by "
                f"{section} do not exist in the Atoms section "
                f"(e.g. {worst}; highest real id is {max(seen)}).\n"
                f"  first offending line: {missing[worst[0]]}\n"
                f"This is the corruption LAMMPS reports as 'Nonexistent atom "
                f"ID'. The file has not been written.")


def _replicate_component_topology(comp: BlendComponent, section: str,
                                  atom_id_offset: int, type_offset: int,
                                  entry_id_offset: int) -> List[str]:
    """Return replicated topology lines for one component. IDs already offset."""
    orig = _clean(comp.sections.get(section))
    if not orig:
        return []
    parsed = []
    for line in orig:
        parts = line.split()
        if len(parts) < 4: continue
        parsed.append((int(parts[0]), int(parts[1]), [int(x) for x in parts[2:]]))
    parsed.sort()
    n = comp.natoms_chain
    nterms = len(parsed)
    out: List[str] = []
    for chain_i in range(comp.count):
        atom_offset = atom_id_offset + chain_i * n
        term_offset = entry_id_offset + chain_i * nterms
        for oid, ttype, aids in parsed:
            new_aids = " ".join(str(a + atom_offset) for a in aids)
            out.append(f"{oid + term_offset} {ttype + type_offset} {new_aids}")
    return out


# ================================================================ top-level
def replicate_blend(
    components: List[BlendComponent],
    box_edges: Tuple[float, float, float],
    out_data_file: Path,
    out_input_file: Optional[Path] = None,
    seed: int = 12345,
    tolerance: float = 2.0,
    packmol_path: Optional[str] = None,
    minimise: Optional[object] = None,
    max_bond_length: float = 3.0,
    progress: Optional[Callable[[str], None]] = None,
    regions: Optional[List[Tuple[float, float, float,
                                 float, float, float]]] = None,
) -> Path:
    """Pack a multi-component blend and write a consolidated LAMMPS data file.

    ``minimise`` takes a :class:`paaf.blend_minimise.MinimiseSettings`. When
    enabled, each component's single chain is relaxed in LAMMPS *before* any
    copy of it is placed. Packing only translates and rotates whole chains, so
    whatever strain a chain arrives with is inherited by every copy — relaxing
    once per component fixes it for all of them.
    """
    if not components:
        raise ValueError("At least one component required")
    _load_components(components)
    offsets = _cumulative_offsets(components)
    emit = progress or (lambda _m: None)

    work = Path(tempfile.mkdtemp(prefix="paaf_blend_"))

    # ---- relax each component once, before it is copied ----------------
    from .blend_minimise import MinimiseSettings, minimise_component
    settings = minimise if minimise is not None else MinimiseSettings()
    relaxed: Dict[int, list] = {}
    reports: List[str] = []
    if getattr(settings, "enabled", False):
        emit("Relaxing each component before packing …")
        for i, comp in enumerate(components):
            rec, atoms = minimise_component(
                comp.name, comp.data_file, work / "minimise",
                settings=settings, input_file=comp.forcefield_input,
                progress=emit)
            reports.append(rec.summary())
            if atoms is not None:
                relaxed[i] = atoms
    else:
        reports.append("Per-component minimisation was switched off; each "
                       "chain is packed with its input geometry.")
    for line in reports:
        log.info("blend: %s", line)

    # Build a PDB for each component and hand them all to packmol together.
    pdb_files: List[Path] = []
    for i, comp in enumerate(components):
        pdb = work / f"comp_{i}_{comp.name}.pdb"
        _write_pdb_from_data(comp.data_file, pdb,
                             coords={a[0]: (a[2], a[3], a[4])
                                     for a in relaxed[i]} if i in relaxed
                             else None)
        pdb_files.append(pdb)
    packed_pdb = work / "packed_blend.pdb"
    if not _build_blend_packmol(pdb_files, [c.count for c in components],
                                packed_pdb, box_edges, tolerance, seed,
                                packmol_path=packmol_path, regions=regions):
        raise RuntimeError(
            "packmol failed to produce packed_blend.pdb. "
            "Check packmol.log in the output directory.")
    coords = _read_xyz_or_pdb_coords(packed_pdb)

    # Split coords into per-component slices, in the same order packmol
    # placed them: component 0's atoms first (count * natoms_chain of them),
    # then component 1, ...
    slices: List[List[Tuple[float, float, float]]] = []
    idx = 0
    for comp in components:
        n = comp.count * comp.natoms_chain
        slices.append(coords[idx:idx + n]); idx += n
    if idx != len(coords):
        raise RuntimeError(
            f"Blend coord count mismatch: consumed {idx} of {len(coords)}")

    # Build merged atom / topology / coefficient sections.
    merged_atoms: List[str] = []
    merged_bonds: List[str] = []
    merged_angles: List[str] = []
    merged_dihedrals: List[str] = []
    merged_impropers: List[str] = []
    merged_masses: List[str] = []
    merged_pair: List[str] = []
    merged_bond_c: List[str] = []
    merged_angle_c: List[str] = []
    merged_dih_c: List[str] = []
    merged_imp_c: List[str] = []

    # running offsets for atom-id and topology-entry-id across components
    running_atomid = 0
    running_bondid = 0
    running_angleid = 0
    running_dihid = 0
    running_impid = 0
    running_molid = 0

    for comp, off, coord_slice in zip(components, offsets, slices):
        # atoms
        merged_atoms.extend(_replicate_component_atoms(
            comp, coord_slice, running_atomid, off["atom"], running_molid))
        # topology
        n_bonds = len(_clean(comp.sections.get("Bonds")))
        n_ang   = len(_clean(comp.sections.get("Angles")))
        n_dih   = len(_clean(comp.sections.get("Dihedrals")))
        n_imp   = len(_clean(comp.sections.get("Impropers")))
        merged_bonds.extend(_replicate_component_topology(
            comp, "Bonds", running_atomid, off["bond"], running_bondid))
        merged_angles.extend(_replicate_component_topology(
            comp, "Angles", running_atomid, off["angle"], running_angleid))
        merged_dihedrals.extend(_replicate_component_topology(
            comp, "Dihedrals", running_atomid, off["dihedral"], running_dihid))
        merged_impropers.extend(_replicate_component_topology(
            comp, "Impropers", running_atomid, off["improper"], running_impid))
        # coefficient sections with the SAME per-component type offsets
        merged_masses.extend(_offset_coeff_lines(_clean(comp.sections.get("Masses")), off["atom"]))
        merged_pair.extend(_offset_coeff_lines(_clean(comp.sections.get("Pair Coeffs")), off["atom"]))
        merged_bond_c.extend(_offset_coeff_lines(_clean(comp.sections.get("Bond Coeffs")), off["bond"]))
        merged_angle_c.extend(_offset_coeff_lines(_clean(comp.sections.get("Angle Coeffs")), off["angle"]))
        merged_dih_c.extend(_offset_coeff_lines(_clean(comp.sections.get("Dihedral Coeffs")), off["dihedral"]))
        merged_imp_c.extend(_offset_coeff_lines(_clean(comp.sections.get("Improper Coeffs")), off["improper"]))
        # advance running counters
        running_atomid += comp.count * comp.natoms_chain
        running_bondid += comp.count * n_bonds
        running_angleid += comp.count * n_ang
        running_dihid += comp.count * n_dih
        running_impid += comp.count * n_imp
        running_molid += comp.count

    validate_merged_topology(merged_atoms, {
        "Bonds": merged_bonds, "Angles": merged_angles,
        "Dihedrals": merged_dihedrals, "Impropers": merged_impropers})

    # The blend's own force-field styles, read from the components' inputs and
    # converted out of DL_FIELD's single-sub-style hybrid form. The same
    # sub-style token has to come off the coefficient tables below, or the
    # data file and the input file will disagree about how many columns a
    # coefficient line has.
    style_blocks = [_component_styles(comp) for comp in components]
    try:
        merged_styles, substyles = merge_style_blocks(
            style_blocks, [c.name for c in components])
    except StyleMismatch:
        # Without an input file the styles are unknowable. That is fatal only
        # if we were asked to write one; a caller who wants the data file
        # alone presumably has its force field elsewhere. Nothing is
        # de-hybridised in that case, because the sub-style names are exactly
        # what we failed to learn.
        if out_input_file is not None:
            raise
        merged_styles, substyles = StyleBlock(), {}
        emit("  no component input file found — writing the data file only, "
             "with its coefficient tables exactly as they came in")
    # The token is found in the lines themselves rather than taken from the
    # declaration. Both routes need it stripped: a hybrid declaration puts it
    # there, and so does DL_FIELD even when the style is read back as a plain
    # name. Leaving it makes LAMMPS parse "class2" as a bond length.
    for _section, _directive, _key in (
            (merged_pair, "pair_style", "pair"),
            (merged_bond_c, "bond_style", "bond"),
            (merged_angle_c, "angle_style", "angle"),
            (merged_dih_c, "dihedral_style", "dihedral"),
            (merged_imp_c, "improper_style", "improper")):
        stripped, token = strip_substyle_tokens(
            _section, substyles.get(_directive, ""))
        _section[:] = stripped
        if token:
            substyles.setdefault(_directive, token)

    max_bond = validate_bond_lengths(merged_atoms, merged_bonds,
                                     max_bond_length)
    emit(f"  longest bond in the merged cell: {max_bond:.3f} Å")

    # Header type counts = sum across components
    total_atom_types    = sum(c.ntypes["atom"]     for c in components)
    total_bond_types    = sum(c.ntypes["bond"]     for c in components)
    total_angle_types   = sum(c.ntypes["angle"]    for c in components)
    total_dih_types     = sum(c.ntypes["dihedral"] for c in components)
    total_imp_types     = sum(c.ntypes["improper"] for c in components)

    a, b, c_ = box_edges
    out: List[str] = []
    out.append(f"LAMMPS blend of {len(components)} components — PAAF")
    out.append("")
    out.append(f"{len(merged_atoms)} atoms")
    if merged_bonds: out.append(f"{len(merged_bonds)} bonds")
    if merged_angles: out.append(f"{len(merged_angles)} angles")
    if merged_dihedrals: out.append(f"{len(merged_dihedrals)} dihedrals")
    if merged_impropers: out.append(f"{len(merged_impropers)} impropers")
    out.append("")
    out.append(f"{total_atom_types} atom types")
    if total_bond_types:  out.append(f"{total_bond_types} bond types")
    if total_angle_types: out.append(f"{total_angle_types} angle types")
    if total_dih_types:   out.append(f"{total_dih_types} dihedral types")
    if total_imp_types:   out.append(f"{total_imp_types} improper types")
    out.append("")
    out.append(f"0.0 {a:.4f} xlo xhi")
    out.append(f"0.0 {b:.4f} ylo yhi")
    out.append(f"0.0 {c_:.4f} zlo zhi")
    out.append("")

    for name, block in (
        ("Masses",           merged_masses),
        ("Pair Coeffs",      merged_pair),
        ("Bond Coeffs",      merged_bond_c),
        ("Angle Coeffs",     merged_angle_c),
        ("Dihedral Coeffs",  merged_dih_c),
        ("Improper Coeffs",  merged_imp_c),
    ):
        if block:
            out.append(name); out.append(""); out.extend(block); out.append("")

    out.append("Atoms  # full"); out.append(""); out.extend(merged_atoms); out.append("")
    for name, block in (
        ("Bonds",     merged_bonds),
        ("Angles",    merged_angles),
        ("Dihedrals", merged_dihedrals),
        ("Impropers", merged_impropers),
    ):
        if block:
            out.append(name); out.append(""); out.extend(block); out.append("")

    out_data_file.parent.mkdir(parents=True, exist_ok=True)
    out_data_file.write_text("\n".join(out) + "\n")
    log.info("Wrote packed blend LAMMPS data (%d components, %d atoms) -> %s",
             len(components), len(merged_atoms), out_data_file)

    # Optional: also stitch together a minimal packed_blend.in that reads the
    # data and mixes pair_coeffs from each component's own lammps.in file.
    if out_input_file is not None:
        _write_blend_input(components, offsets, merged_styles, substyles,
                           out_data_file, out_input_file)

    return out_data_file


def _write_blend_input(components: List[BlendComponent],
                       offsets: List[Dict[str, int]],
                       styles: StyleBlock, substyles: Dict[str, str],
                       data_file: Path, in_file: Path) -> None:
    """Emit a merged LAMMPS input that can actually be read back.

    The order is not decorative. Every style must be declared *before*
    ``read_data``, because LAMMPS needs to know how many coefficients a line
    holds while it is parsing the coefficient tables. ``pair_coeff`` lines
    come after, since they refer to types the data file has just defined.

    Earlier versions wrote only ``units``/``atom_style``/``boundary`` and then
    a block of ``pair_coeff`` lines. That file cannot run: LAMMPS aborts at
    ``read_data`` with no bond style defined.
    """
    lines: List[str] = [
        "# LAMMPS input generated by PAAF blend replicator",
        f"# {len(components)} components merged from:",
    ]
    for comp in components:
        lines.append(f"#   - {comp.name} x{comp.count}: {comp.data_file}")
    if styles.source:
        lines.append(f"# Force-field styles taken from {styles.source}")
    if substyles:
        lines.append("# DL_FIELD single-sub-style hybrids converted to "
                     "plain styles: "
                     + ", ".join(f"{k.split('_')[0]}={v}"
                                 for k, v in sorted(substyles.items())))

    # ---- 1. global settings and styles, before read_data ----------------
    lines.append("")
    lines.extend(styles.lines())

    # ---- 2. the data file ------------------------------------------------
    lines += ["", f"read_data {data_file.name}", ""]

    # ---- 3. one group per component, over its own atom-type range -------
    # DL_FIELD writes a group per molecular group. For a blend the useful
    # split is per component, so per-species thermo output and fixes can be
    # written without working the offsets out by hand.
    lines.append("# Define Group-ID for atoms from the respective components.")
    for comp, off in zip(components, offsets):
        n_types = comp.ntypes.get("atom", 0)
        if n_types <= 0:
            continue
        span = " ".join(str(off["atom"] + t) for t in range(1, n_types + 1))
        lines.append(f"group {_group_name(comp.name)} type {span}")
    lines.append("")

    # ---- 4. pair coefficients, after the types exist --------------------
    pair_sub = substyles.get("pair_style", "")
    for comp, off in zip(components, offsets):
        source = comp.forcefield_input
        if source is None or not Path(source).is_file():
            continue
        block: List[str] = []
        for raw in Path(source).read_text(errors="replace").splitlines():
            stripped = raw.strip()
            if not stripped.startswith("pair_coeff"):
                continue
            stripped = dehybridise_pair_coeff(stripped, pair_sub)
            parts = stripped.split()
            try:
                parts[1] = str(int(parts[1]) + off["atom"])
                parts[2] = str(int(parts[2]) + off["atom"])
            except (ValueError, IndexError):
                # A wildcard like "pair_coeff * *" cannot be offset, and
                # applying it to the blend would silently overwrite the other
                # component's types. Skip it and say so.
                log.warning("%s: skipping un-offsettable pair_coeff: %s",
                            comp.name, stripped)
                continue
            block.append(" ".join(parts))
        if block:
            lines.append(f"# {comp.name}: atom types {off['atom'] + 1}"
                         f"–{off['atom'] + comp.ntypes['atom']}")
            lines.extend(block)
            lines.append("")

    if "pair_modify" not in styles.directives:
        lines += [
            "# Every pair WITHIN a component is given explicitly above. Pairs",
            "# ACROSS components are not, so LAMMPS mixes them. OPLS-AA mixes",
            "# geometrically in both epsilon and sigma; stating it beats",
            "# relying on the default for the pair style in use.",
            "pair_modify     mix geometric",
            "",
        ]

    # ---- 5. run settings -------------------------------------------------
    lines += [
        "neighbor        2.0 bin",
        "neigh_modify    every 1 delay 0 check yes",
        "",
        "thermo          100",
        "thermo_style    custom step temp press density pe ke etotal "
        "ebond eangle edihed eimp evdwl ecoul elong",
        "",
        "# A packed cell is not an equilibrated one. Relax the contacts here,",
        "# then compress to a physical density before measuring anything.",
        "minimize        1.0e-4 1.0e-6 10000 100000",
        "",
        "# velocity        all create 300.0 4928459 dist gaussian",
        "# fix             eq all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0",
        "# run             500000",
        "run             0",
    ]
    in_file.write_text("\n".join(lines) + "\n")
    log.info("Wrote merged LAMMPS input for blend -> %s", in_file)


def _group_name(name: str) -> str:
    """A LAMMPS-safe group id: letters, digits and underscores only."""
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_"
                      for ch in name).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = "g" + cleaned
    return cleaned


# ================================================================ config bridge
def run_blend_from_config(cfg, out_dir: Path) -> Optional[Path]:
    """Invoke the blend replicator from a top-level Config.

    Returns the path to the packed data file, or ``None`` if the blend is
    disabled or has fewer than two components.
    """
    blend = getattr(cfg, "blend", None)
    if blend is None or not getattr(blend, "enabled", False):
        return None
    comps_cfg = list(blend.components or [])
    if len(comps_cfg) < 2:
        log.info("Blend disabled or has <2 components; skipping.")
        return None
    components = []
    for c in comps_cfg:
        p = Path(c.data_file)
        if not p.is_absolute():
            p = (out_dir / p).resolve() if not p.exists() else p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"Blend component data file missing: {p}")
        components.append(BlendComponent(
            name=c.name or p.stem, data_file=p, count=int(c.count)))
    box = cfg.box
    out_data = out_dir / (blend.out_data or "packed_blend.data")
    out_in   = out_dir / (blend.out_input or "packed_blend.in")
    return replicate_blend(
        components,
        box_edges=(box.a, box.b, box.c),
        out_data_file=out_data,
        out_input_file=out_in,
        seed=int(blend.seed),
        tolerance=float(blend.tolerance),
        packmol_path=getattr(box, "packmol_path", "") or None,
    )
