#!/usr/bin/env python3
"""Build a small cell, write the LAMMPS relaxation deck, and validate it.

    conda activate mta
    cd /Users/uzair/project/PAAF
    python scripts/check_relax.py

The point of this script is the one thing PAAF's test suite cannot check: the
tests drive a *stub* LAMMPS, so they prove PAAF calls the binary correctly and
parses what comes back, but not that real LAMMPS accepts the script PAAF
writes. This builds a deliberately tiny cell so the whole loop takes seconds,
then runs ``lmp -skiprun``, which reads and validates every command without
computing anything.

It reports honestly at each stage. A missing moltemplate or an untyped cell is
not a failure of the relaxation code, and the output says which stage stopped.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paaf.cell.cell_export import export_cell                 # noqa: E402
from paaf.cell.composition import Component, from_chain_counts  # noqa: E402
from paaf.cell.grow import grow_amorphous_cell                # noqa: E402
from paaf.cell.relax import RelaxSettings, find_lammps        # noqa: E402

OUT = ROOT / "output" / "relax_check"
FF = "oplsaa"          # moltemplate route: styles come from system.in.init


def main() -> int:
    print("=" * 68)
    print("PAAF relaxation deck check")
    print("=" * 68)

    lmp = find_lammps()
    print(f"  LAMMPS       : {lmp or 'NOT FOUND'}")
    print(f"  moltemplate  : {shutil.which('moltemplate.sh') or 'NOT FOUND'}")
    print(f"  force field  : {FF}")
    print()

    # --- a deliberately tiny cell: 2 chains of DP 10 polyethylene
    comp = from_chain_counts(
        [Component(name="PE", repeat_unit="[*]CC[*]",
                   degree_of_polymerisation=10, n_chains=2, ris_key="PE")],
        0.85)
    print(f"  {comp.total_chains} chains, {comp.total_beads} skeletal beads, "
          f"box {comp.box_edge_a:.2f} A")

    specs = comp.grow_specs()
    res = grow_amorphous_cell(specs, comp.box(), temperature=413.0, seed=5)
    print(f"  grown: density {res.density_kg_m3 / 1000:.4f} g/cm3")

    if OUT.exists():
        shutil.rmtree(OUT, ignore_errors=True)
    exp = export_cell(res, specs, OUT, name="cell", ff_key=FF,
                      relax=True, relax_settings=RelaxSettings(
                          push_steps=200, maxiter=200),
                      progress=lambda m: print(f"  {m}"))

    print()
    print(f"  {exp.summary()}")
    for m in exp.messages:
        print(f"    - {m.splitlines()[0]}")

    deck = OUT / "cell" / "relax" / "relax.in"
    if not deck.exists():
        print()
        print("  No relaxation deck was written. That happens when typing did")
        print("  not succeed, which is a moltemplate/OpenBabel problem rather")
        print("  than a relaxation one — see the messages above.")
        return 1

    print()
    print("=" * 68)
    print(f"deck: {deck}")
    print("=" * 68)
    print(deck.read_text())

    if lmp is None:
        print("LAMMPS not found, so the deck was not validated.")
        return 0

    print("=" * 68)
    print(f"validating with: {lmp} -skiprun -in relax.in")
    print("=" * 68)
    proc = subprocess.run([str(lmp), "-skiprun", "-in", "relax.in"],
                          cwd=str(deck.parent), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out[-4000:])
    if proc.returncode == 0:
        print("\n  OK — LAMMPS read and accepted every command.")
        return 0
    print(f"\n  LAMMPS rejected the deck (rc={proc.returncode}).")
    print("  If it complains about -skiprun the flag is too new for this")
    print("  build; drop it and let the tiny cell actually run instead.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
