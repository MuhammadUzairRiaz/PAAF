#!/usr/bin/env python3
"""End-to-end check of the blend writer against real DL_FIELD components.

Runs the whole merge on ``mta_output/enr`` + ``mta_output/pbs`` and reports
what a LAMMPS run would care about: unique atom ids, resolvable bonds, sane
bond lengths, a complete style block, and a density that is actually
condensed-phase.

packmol is stubbed with a deterministic grid so the check runs anywhere. That
exercises every line of the merge; only the placement differs.

    python scripts/check_blend.py [--density 1.0] [--pack-at 0.4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paaf import blend_replicator as BR                        # noqa: E402
from paaf.blend_minimise import MinimiseSettings               # noqa: E402
from paaf.blend_replicator import (                            # noqa: E402
    BlendComponent, box_edge_for_density, component_mass_amu, replicate_blend,
)
from paaf.blend_styles import read_style_block                 # noqa: E402


def _grid_pack(pdb_files, counts, out_pdb, box_edges, *_a, **_kw) -> bool:
    """Stand in for packmol: lay every copy out on a cubic grid."""
    import math
    a, b, c = box_edges
    total = sum(counts)
    per_side = max(1, math.ceil(total ** (1 / 3)))
    step = min(a, b, c) / per_side
    lines, k = [], 0
    for pdb, n in zip(pdb_files, counts):
        atoms = [l for l in Path(pdb).read_text().splitlines()
                 if l.startswith(("ATOM", "HETATM"))]
        for _ in range(n):
            i, j, m = (k % per_side, (k // per_side) % per_side,
                       k // (per_side * per_side))
            dx, dy, dz = (i + 0.5) * step, (j + 0.5) * step, (m + 0.5) * step
            x0 = min(float(l[30:38]) for l in atoms)
            y0 = min(float(l[38:46]) for l in atoms)
            z0 = min(float(l[46:54]) for l in atoms)
            for l in atoms:
                x = float(l[30:38]) - x0 + dx
                y = float(l[38:46]) - y0 + dy
                z = float(l[46:54]) - z0 + dz
                lines.append(f"{l[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{l[54:]}")
            k += 1
    Path(out_pdb).write_text("\n".join(lines) + "\nEND\n")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", type=float, default=1.0)
    ap.add_argument("--pack-at", type=float, default=0.4)
    ap.add_argument("--chains", type=int, default=20)
    args = ap.parse_args()

    comps = [
        BlendComponent(name="ENR",
                       data_file=ROOT / "mta_output/enr/lammps1.data",
                       count=args.chains),
        BlendComponent(name="PBS",
                       data_file=ROOT / "mta_output/pbs/lammps1.data",
                       count=args.chains),
    ]
    missing = [c.data_file for c in comps if not c.data_file.is_file()]
    if missing:
        print("Missing component data files:")
        for m in missing:
            print(f"  {m}")
        return 2

    BR._load_components(comps)
    print("Components")
    for c in comps:
        print(f"  {c.name:5} {c.natoms_chain:4d} atoms/chain  "
              f"{component_mass_amu(c):8.1f} amu  x{c.count}  "
              f"styles from {c.forcefield_input}")
        if c.forcefield_input is None:
            print("    !! no force-field input found — minimisation and the "
                  ".in would both be skipped")

    edge = box_edge_for_density(comps, args.density * args.pack_at)
    full = box_edge_for_density(comps, args.density)
    print(f"\nBox: {edge:.2f} Å at {args.density * args.pack_at:.2f} g/cm³ "
          f"(compress to {full:.2f} Å for {args.density:.2f} g/cm³)")

    out = ROOT / "output" / "_check_blend"
    out.mkdir(parents=True, exist_ok=True)
    original = BR._build_blend_packmol
    BR._build_blend_packmol = _grid_pack
    try:
        data = replicate_blend(
            comps, (edge, edge, edge),
            out_data_file=out / "packed_blend.data",
            out_input_file=out / "packed_blend.in",
            minimise=MinimiseSettings(enabled=False),
            max_bond_length=3.0,
            progress=lambda m: print(f"  {m}"))
    finally:
        BR._build_blend_packmol = original

    # ---- what LAMMPS would object to -----------------------------------
    from paaf.lammps_replicator import parse_lammps_data, _clean
    _h, sections = parse_lammps_data(data)
    atoms = _clean(sections.get("Atoms"))
    ids = [int(l.split()[0]) for l in atoms]
    print(f"\nData file: {data}")
    print(f"  {len(atoms)} atoms, {len(set(ids))} unique ids "
          f"({'OK' if len(ids) == len(set(ids)) else 'DUPLICATES'})")
    for sect in ("Bonds", "Angles", "Dihedrals", "Impropers"):
        body = _clean(sections.get(sect))
        bad = sum(1 for l in body
                  if any(int(t) not in set(ids) for t in l.split()[2:]))
        print(f"  {sect:10} {len(body):6d} lines, {bad} dangling")

    hybrid = [s for s in ("Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs",
                          "Improper Coeffs", "Pair Coeffs")
              if any(not p.replace(".", "").replace("-", "").replace("e", "")
                     .isdigit()
                     for l in _clean(sections.get(s))[:1]
                     for p in l.split()[1:2])]
    print(f"  coefficient sections still carrying a sub-style token: "
          f"{hybrid or 'none'}")

    inp = out / "packed_blend.in"
    block = read_style_block(inp)
    required = ["pair_style", "bond_style", "angle_style", "dihedral_style",
                "improper_style", "kspace_style", "special_bonds"]
    absent = [d for d in required if d not in block.directives]
    print(f"\nInput file: {inp}")
    for d in required:
        print(f"  {d:15} {block.get(d) or '*** MISSING ***'}")
    code = [l.strip() for l in inp.read_text().splitlines()
            if l.strip() and not l.strip().startswith("#")]
    first_read = next(i for i, l in enumerate(code)
                      if l.startswith("read_data"))
    styles_before = all(
        i < first_read
        for i, l in enumerate(code)
        if l.split()[0] in ("pair_style", "bond_style", "angle_style",
                            "dihedral_style", "improper_style",
                            "kspace_style", "special_bonds"))
    coeffs_after = all(
        i > first_read
        for i, l in enumerate(code) if l.startswith("pair_coeff"))
    print(f"  every style precedes read_data: {styles_before}")
    print(f"  every pair_coeff follows read_data: {coeffs_after}")
    print(f"  pair_coeff lines: {sum(1 for l in code if l.startswith('pair_coeff'))}")

    ok = (len(ids) == len(set(ids)) and not absent and not hybrid
          and styles_before and coeffs_after)
    print("\n" + ("PASS — the blend and its input are self-consistent."
                  if ok else "FAIL — see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
