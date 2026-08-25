#!/usr/bin/env python3
"""Build an amorphous cell by chain growth and write it out for viewing.

Run::

    conda activate mta
    cd /Users/uzair/project/PAAF
    python scripts/demo_amorphous.py

Writes ``output/<name>_cell.xyz`` which you can open in VMD, OVITO, Avogadro
or PyMOL. One bead per repeat unit — see the note the builder prints.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paaf.cell.amorphous import BoxShape                       # noqa: E402
from paaf.cell.grow import (                                   # noqa: E402
    GrowSpec, grow_amorphous_cell, repeat_unit_mass,
)
from paaf.cell.packing import PackFailed                       # noqa: E402

OUT = ROOT / "output"


def box_for_density(total_mass_amu: float, density_g_cm3: float) -> BoxShape:
    """Cubic cell whose volume gives the requested density."""
    volume_cm3 = total_mass_amu / (6.02214076e23 * density_g_cm3)
    edge = (volume_cm3 * 1.0e24) ** (1.0 / 3.0)
    return BoxShape(shape="cubic", a=edge)


def build(name: str, smiles: str, ris_key: str, n_chains: int, dp: int,
          density: float, temperature: float = 413.0, seed: int = 12345):
    from paaf.cell.grow import backbone_atoms_per_unit            # noqa: E402

    mass = repeat_unit_mass(smiles)
    bpu = backbone_atoms_per_unit(smiles)
    total = n_chains * dp * mass
    box = box_for_density(total, density)

    print(f"\n{'=' * 66}\n{name}\n{'=' * 66}")
    print(f"  repeat unit    : {smiles}")
    print(f"  unit mass      : {mass:.2f} g/mol   (implicit H included)")
    print(f"  granularity    : {bpu} skeletal atoms per repeat unit "
          f"-> one growth step is one backbone bond")
    print(f"  chains         : {n_chains} x DP {dp} "
          f"= {n_chains * dp * bpu} beads")
    print(f"  chain mass     : {dp * mass:,.0f} g/mol")
    print(f"  target density : {density:.3f} g/cm³")
    print(f"  cubic box      : {box.a:.2f} Å   (V = {box.volume_ang3():,.0f} Å³)")

    def show(p):
        print(f"\r  {p.message:<52}", end="", flush=True)

    t0 = time.time()
    try:
        res = grow_amorphous_cell(
            [GrowSpec(repeat_unit=smiles, n_chains=n_chains,
                      degree_of_polymerisation=dp, name=name,
                      ris_key=ris_key)],
            box, temperature=temperature, seed=seed, progress=show)
    except PackFailed as exc:
        print(f"\n  FAILED: {exc}")
        return None
    print(f"\r  {'grown':<52}")

    print(f"\n  density        : {res.density_kg_m3 / 1000:.4f} g/cm³")
    print(f"  mean C_n       : {res.mean_c_n:.2f}")
    print(f"  torsion t/g+/g-: " +
          " / ".join(f"{p:.3f}" for p in res.torsion_populations))
    rg = [c.radius_of_gyration for c in res.chains]
    r2 = [c.r_end_to_end for c in res.chains]
    print(f"  radius of gyr. : {sum(rg) / len(rg):.2f} Å")
    print(f"  end-to-end     : {sum(r2) / len(r2):.2f} Å")
    print(f"  rejected       : {res.n_rejected_overlap} overlap, "
          f"{res.n_rejected_spearing} spearing, {res.n_restarts} restarts")
    print(f"  wall time      : {time.time() - t0:.1f} s")
    for n in res.notes:
        print(f"  note           : {n}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}_cell.xyz"
    res.molecule.to_xyz(path)
    print(f"\n  wrote {path.relative_to(ROOT)}  "
          f"({len(res.molecule.atoms)} beads) — open it in VMD / OVITO")
    return res


def main() -> int:
    # Two homopolymer melts, then a blend solved from weight percentages.
    build("PE", "[*]CC[*]", "PE", n_chains=8, dp=30, density=0.85)
    build("PS", "[*]CC([*])c1ccccc1", "PS", n_chains=6, dp=25, density=1.04)

    print(f"\n{'=' * 66}\nblend: PE + PS solved from weight %\n{'=' * 66}")
    from paaf.cell.composition import (                          # noqa: E402
        Component, solve_from_weight_fractions,
    )
    comp = solve_from_weight_fractions(
        [Component(name="PE", repeat_unit="[*]CC[*]",
                   degree_of_polymerisation=30, weight_percent=70.0,
                   ris_key="PE"),
         Component(name="PS", repeat_unit="[*]CC([*])c1ccccc1",
                   degree_of_polymerisation=30, weight_percent=30.0,
                   ris_key="PS")],
        0.95, target_beads=900)
    print(f"  {comp.summary()}")
    print(f"  requested      : " +
          " / ".join(f"{w:.1f} wt%" for w in comp.requested_weight_percent))
    print("  (chain masses come from the SMILES, hydrogens included; the "
          "counts are integers, so the realised split is what you get)")

    res = grow_amorphous_cell(comp.grow_specs(), comp.box(),
                              temperature=413.0, seed=99)
    print(f"  density        : {res.density_kg_m3 / 1000:.4f} g/cm³")
    print(f"  chains         : " +
          ", ".join(f"{c.species}(Rg={c.radius_of_gyration:.1f})"
                    for c in res.chains))
    OUT.mkdir(parents=True, exist_ok=True)
    res.molecule.to_xyz(OUT / "blend_cell.xyz")
    print("  wrote output/blend_cell.xyz")

    # ---- and the same cell as atoms, ready for a force field
    print(f"\n{'=' * 66}\nback-mapping the blend to all-atom\n{'=' * 66}")
    try:
        from paaf.cell.cell_export import export_cell             # noqa: E402
        exp = export_cell(res, comp.grow_specs(), OUT, name="blend_atomistic",
                          run_typing=False, progress=lambda m: print(f"  {m}"))
        print(f"\n  {exp.summary()}")
        for m in exp.messages:
            print(f"  note: {m}")
    except Exception as exc:
        print(f"  back-mapping unavailable: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
