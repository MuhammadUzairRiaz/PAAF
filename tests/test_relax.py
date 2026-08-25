"""Acceptance tests for LAMMPS relaxation of a constructed cell.

No LAMMPS installation is required. The deck is checked by reading it, the
executable search is checked against a temporary directory, and the *run* path
is exercised with a stub executable that writes a dump in LAMMPS' own format —
so the subprocess call, the dump parser and the coordinate write-back are all
covered without depending on a real binary being present.

The property being protected throughout: a cell that was not relaxed must
never be reported as one that was.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

from paaf.cell.relax import (                                # noqa: E402
    CellMetrics, RelaxSettings, find_lammps, measure_cell,
    read_dump_coordinates, relax_cell, write_relax_input,
)
from paaf.structure import Atom, Molecule                    # noqa: E402


def _tiny_molecule(n: int = 6) -> Molecule:
    atoms = [Atom(index=i, element="C",
                  xyz=np.array([1.6 * i, 0.0, 0.0]), name=f"C{i + 1}")
             for i in range(n)]
    bonds = [(i, i + 1, 1.0) for i in range(n - 1)]
    return Molecule(atoms=atoms, bonds=bonds, name="chain")


# ------------------------------------------------------------- the deck
def test_deck_ramps_a_soft_potential_before_touching_the_real_one(tmp_path):
    """The ordering IS the algorithm: soft first, real second.

    At a 1 Å contact a 12-6 potential is ~3e6 x epsilon. Minimising straight
    into that is how a constructed cell explodes, so the soft stage is not
    optional decoration.
    """
    p = write_relax_input(
        tmp_path, "cell.data", RelaxSettings(),
        init_file="system.in.init", settings_file="system.in.settings",
        restore_lines=["pair_style lj/cut/coul/long 10.0 10.0",
                       "dihedral_style opls"])
    text = p.read_text()
    print("\n  " + "\n  ".join(
        l for l in text.splitlines()
        if l.strip() and not l.startswith("#"))[:900])

    assert "pair_style      soft" in text
    # The prefactor must be explicit and start at zero, or fix adapt has
    # nothing to ramp.
    assert "pair_coeff      * * 0.0" in text
    assert "ramp(0," in text
    assert "fix             push all adapt" in text
    assert "nve/limit" in text
    assert "minimize" in text

    # Ordering: soft push-off, then the real styles restored, then minimise.
    # The restore is the reissued style line, not a re-include of .in.init —
    # that file carries units/atom_style and cannot be read twice.
    i_soft = text.index("pair_style      soft")
    i_restore = text.index("pair_style lj/cut/coul/long")
    i_min = text.index("minimize")
    assert i_soft < i_restore < i_min, "the real force field must be restored " \
                                       "before minimisation, not after"


def test_deck_writes_back_a_sorted_dump_and_a_data_file(tmp_path):
    """Sorted by id, or the coordinates cannot be mapped back to our atoms."""
    p = write_relax_input(tmp_path, "cell.data", RelaxSettings())
    text = p.read_text()
    assert "write_dump" in text and "modify sort id" in text
    assert "write_data" in text
    # The final energy is READ from thermo, never printed with $(pe): an
    # immediate variable forces an energy evaluation that fails when no run
    # happened, which broke `lmp -skiprun` validation.
    commands = [l for l in text.splitlines()
                if l.strip() and not l.lstrip().startswith("#")]
    assert not any("$(pe)" in l for l in commands), \
        "an immediate energy variable is back in the deck"


def test_deck_says_so_when_no_force_field_styles_were_supplied(tmp_path):
    """Without .in.init the soft style would still be active at minimisation.

    Silently minimising a cell under a soft potential and calling it relaxed
    would be worse than refusing, so the deck carries the warning.
    """
    p = write_relax_input(tmp_path, "cell.data", RelaxSettings())
    text = p.read_text()
    print(f"\n  warning present: {'WARNING' in text}")
    assert "WARNING" in text
    assert "will NOT run as written" in text
    # And with a styles table it must NOT warn, because it will run.
    from paaf.cell.relax import lammps_styles_for
    p2 = write_relax_input(tmp_path, "cell.data", RelaxSettings(),
                           styles=lammps_styles_for("pcff"),
                           styles_source="PAAF table", name="relax2.in")
    t2 = p2.read_text()
    assert "WARNING" not in t2
    assert "lj/class2/coul/long" in t2
    assert "CHECK THESE" in t2


def test_push_off_can_be_disabled(tmp_path):
    p = write_relax_input(tmp_path, "cell.data",
                          RelaxSettings(push_off=False))
    text = p.read_text()
    assert "pair_style      soft" not in text
    assert "minimize" in text


# --------------------------------------------------------- finding LAMMPS
def test_missing_lammps_is_reported_as_missing():
    assert find_lammps("definitely-not-a-real-binary-xyz") is None


def test_an_explicit_path_is_honoured(tmp_path):
    fake = tmp_path / "lmp"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    assert find_lammps(str(fake)) == fake


def test_sibling_conda_environments_are_searched(tmp_path, monkeypatch):
    """LAMMPS is usually installed in a DIFFERENT env from the one running us.

    ``conda create -n lammps -c conda-forge lammps`` is the documented way to
    install it, and PATH then contains only the active environment. Telling
    the user LAMMPS is missing when it sits one directory away is a bad
    failure, so sibling environments are searched.
    """
    root = tmp_path / "envs"
    (root / "mta" / "bin").mkdir(parents=True)
    other = root / "polyrapid" / "bin"
    other.mkdir(parents=True)
    exe = other / "lmp"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)

    monkeypatch.setenv("CONDA_PREFIX", str(root / "mta"))
    monkeypatch.setenv("PATH", str(root / "mta" / "bin"))
    found = find_lammps()
    print(f"\n  active env: {root / 'mta'}")
    print(f"  found     : {found}")
    assert found == exe

    # And by environment name.
    assert find_lammps("polyrapid") == exe


def test_conda_search_can_be_switched_off(tmp_path, monkeypatch):
    root = tmp_path / "envs"
    other = root / "polyrapid" / "bin"
    other.mkdir(parents=True)
    exe = other / "lmp"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("CONDA_PREFIX", str(root / "mta"))
    monkeypatch.setenv("PATH", "/nonexistent")
    assert find_lammps(search_conda_envs=False) is None


# ------------------------------------------------------------- the dump
def test_dump_is_parsed_from_the_last_frame(tmp_path):
    """Two frames; the relaxed structure is the second one."""
    dump = tmp_path / "d.lammpstrj"
    dump.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS id type x y z\n1 1 0.0 0.0 0.0\n2 1 1.0 0.0 0.0\n"
        "ITEM: TIMESTEP\n100\nITEM: NUMBER OF ATOMS\n2\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS id type x y z\n1 1 5.0 6.0 7.0\n2 1 8.0 9.0 1.0\n")
    got = read_dump_coordinates(dump, 2)
    print(f"\n  last frame read: {got.tolist()}")
    assert got is not None
    assert np.allclose(got[0], [5.0, 6.0, 7.0])
    assert np.allclose(got[1], [8.0, 9.0, 1.0])


def test_a_truncated_dump_is_refused_not_half_read(tmp_path):
    dump = tmp_path / "d.lammpstrj"
    dump.write_text("ITEM: TIMESTEP\n0\nITEM: ATOMS id type x y z\n"
                    "1 1 0.0 0.0 0.0\n")
    assert read_dump_coordinates(dump, 5) is None


# ------------------------------------------------------------- metrics
def test_metrics_report_density_contact_and_bonds():
    mol = _tiny_molecule()
    m = measure_cell(mol, (20.0, 20.0, 20.0))
    print(f"\n  atoms {m.n_atoms}  density {m.density_g_cm3:.5f} g/cm3  "
          f"bonds {m.bond_min_a:.2f}-{m.bond_max_a:.2f} A  "
          f"Rg {m.r_gyration_a:.2f} A")
    assert m.n_atoms == 6
    assert m.density_g_cm3 > 0
    assert abs(m.bond_min_a - 1.6) < 1e-9
    assert m.r_gyration_a > 0


# ------------------------------------------------------ the run, stubbed
def _stub_lammps(tmp_path: Path, n_atoms: int, shift: float = 3.0) -> Path:
    """A fake LAMMPS that writes a plausible dump and exits cleanly."""
    exe = tmp_path / "lmp"
    body = f"""#!{sys.executable}
import sys, pathlib
n = {n_atoms}
lines = ["ITEM: TIMESTEP", "0", "ITEM: NUMBER OF ATOMS", str(n),
         "ITEM: BOX BOUNDS pp pp pp", "0 20", "0 20", "0 20",
         "ITEM: ATOMS id type x y z"]
for i in range(n):
    lines.append(f"{{i+1}} 1 {{i * 1.6 + {shift}}} 0.0 0.0")
pathlib.Path("relaxed.lammpstrj").write_text("\\n".join(lines) + "\\n")
pathlib.Path("relaxed.data").write_text("stub\\n")
print("Step Temp Press Density PotEng E_vdwl E_bond")
print("       0    0   -50.0   0.900   4321.0   10.0   5.0")
print("      99    0   -12.0   0.912  -1234.5   -8.0   2.0")
print("Loop time of 0.01")
"""
    exe.write_text(body)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def test_a_successful_run_updates_the_coordinates_and_reports_both_states(tmp_path):
    mol = _tiny_molecule()
    before = np.array([a.xyz for a in mol.atoms]).copy()
    data = tmp_path / "cell.data"
    data.write_text("dummy\n")
    exe = _stub_lammps(tmp_path, len(mol.atoms), shift=3.0)

    res = relax_cell(mol, (20.0, 20.0, 20.0), data, tmp_path / "work",
                     settings=RelaxSettings(lammps_exe=str(exe)),
                     init_file="cell_system.in.init", settings_file="cell_system.in.settings")
    after = np.array([a.xyz for a in mol.atoms])
    print(f"\n  {res.summary()}")
    assert res.ran
    assert res.before is not None and res.after is not None
    assert res.potential_energy == pytest.approx(-1234.5)
    assert not np.allclose(before, after), "coordinates were not updated"
    assert np.allclose(after[0], [3.0, 0.0, 0.0])


def test_a_failing_lammps_does_not_claim_success(tmp_path):
    """The cell must not be reported as relaxed when the run died."""
    mol = _tiny_molecule()
    before = np.array([a.xyz for a in mol.atoms]).copy()
    data = tmp_path / "cell.data"
    data.write_text("dummy\n")
    exe = tmp_path / "lmp"
    exe.write_text("#!/bin/sh\necho 'ERROR: Out of range atoms' >&2\nexit 1\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)

    res = relax_cell(mol, (20.0, 20.0, 20.0), data, tmp_path / "work",
                     settings=RelaxSettings(lammps_exe=str(exe)),
                     init_file="cell_system.in.init", settings_file="cell_system.in.settings")
    after = np.array([a.xyz for a in mol.atoms])
    print(f"\n  ran={res.ran}")
    print(f"  message: {res.messages[0].splitlines()[0]}")
    assert not res.ran
    assert np.allclose(before, after), "coordinates changed despite failure"
    assert any("NOT relaxed" in m for m in res.messages)


def test_without_lammps_the_deck_is_written_and_the_fact_is_stated(tmp_path,
                                                                   monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "nope"))
    mol = _tiny_molecule()
    data = tmp_path / "cell.data"
    data.write_text("dummy\n")
    res = relax_cell(mol, (20.0, 20.0, 20.0), data, tmp_path / "work",
                     init_file="cell_system.in.init", settings_file="cell_system.in.settings")
    print(f"\n  {res.summary()}")
    print(f"  {res.messages[0][:120]}")
    assert not res.ran
    assert res.input_script is not None and res.input_script.exists()
    assert any("NOT run" in m for m in res.messages)


def test_result_summary_never_implies_a_run_that_did_not_happen(tmp_path):
    mol = _tiny_molecule()
    data = tmp_path / "cell.data"
    data.write_text("dummy\n")
    exe = tmp_path / "lmp"
    exe.write_text("#!/bin/sh\nexit 1\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    res = relax_cell(mol, (20.0, 20.0, 20.0), data, tmp_path / "work",
                     settings=RelaxSettings(lammps_exe=str(exe)),
                     init_file="cell_system.in.init", settings_file="cell_system.in.settings")
    assert "relaxed with LAMMPS" not in res.summary()


def test_relaxation_warns_that_it_is_not_equilibration(tmp_path):
    """Minimisation gives a 0 K glass, not an equilibrated melt."""
    mol = _tiny_molecule()
    data = tmp_path / "cell.data"
    data.write_text("dummy\n")
    exe = _stub_lammps(tmp_path, len(mol.atoms))
    res = relax_cell(mol, (20.0, 20.0, 20.0), data, tmp_path / "work",
                     settings=RelaxSettings(lammps_exe=str(exe)),
                     init_file="cell_system.in.init", settings_file="cell_system.in.settings")
    joined = " ".join(res.messages)
    print(f"\n  {[m[:90] for m in res.messages]}")
    assert "does not" in joined.lower() or "NOT equilibrate" in joined
    assert "NPT" in joined or "dynamics" in joined.lower()


# =========================================================== engine routing
def test_moltemplate_force_fields_are_lammps_only(monkeypatch):
    """Moltemplate writes LAMMPS input and nothing else.

    Offering GROMACS for OPLS-AA and then silently substituting would be a
    trap, so the choice is constrained rather than pretended.
    """
    from paaf.cell.relax import choose_engine
    for key in ("oplsaa", "loplsaa", "trappe_ua", "dreiding"):
        assert choose_engine(key) == "lammps"
        assert choose_engine(key, "gromacs") == "lammps", \
            f"{key} must not be routed to GROMACS"


def test_dlfield_force_fields_can_use_either_engine():
    from paaf.cell.relax import choose_engine
    for key in ("pcff", "compass", "charmm36", "opls2005_dl"):
        assert choose_engine(key, "gromacs") == "gromacs"
        assert choose_engine(key, "lammps") == "lammps"


def test_auto_prefers_gromacs_for_dlfield_only_when_it_is_installed(
        tmp_path, monkeypatch):
    """Auto should pick the better-conditioned route, not an absent one."""
    import paaf.cell.relax as relax_mod
    from paaf.cell.relax import choose_engine

    monkeypatch.setattr(relax_mod, "find_gromacs", lambda *_a, **_k: None)
    print(f"\n  gmx absent : pcff -> {choose_engine('pcff')}")
    assert choose_engine("pcff") == "lammps"

    fake = tmp_path / "gmx"
    fake.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(relax_mod, "find_gromacs", lambda *_a, **_k: fake)
    print(f"  gmx present: pcff -> {choose_engine('pcff')}")
    assert choose_engine("pcff") == "gromacs"
    # A moltemplate force field is still LAMMPS even with gmx installed.
    assert choose_engine("oplsaa") == "lammps"


def test_every_dlfield_force_field_can_be_given_styles(tmp_path):
    """Every registered DL_FIELD force field must reach a usable style block.

    This used to demand a hand-written table entry per force field, which put
    a ceiling on how many could be supported and was wrong for the one that
    mattered most. Styles now come from the generated data file, so the
    guarantee is per-*file* rather than per-table-entry — and it holds for
    force fields nobody has written an entry for.
    """
    from paaf.ff_registry import REGISTRY
    from paaf.cell.relax import lammps_styles_for

    data = tmp_path / "lammps1.data"
    data.write_text(
        "# Force field scheme: x\n\n1 atoms\n\n1 atom types\n1 bond types\n\n"
        "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n"
        "Masses\n\n1 12.011\n\n"
        "Bond Coeffs\n\n1 class2 1.52 253.7 -423.0 396.9\n\n"
        "Atoms\n\n1 1 1 0.0 0.0 0.0 0.0\n")

    missing = []
    print()
    for key, ff in REGISTRY.items():
        if ff.kind != "dlfield":
            continue
        st = lammps_styles_for(key, data)
        ok = bool(st) and "bond_style" in st and "pair_style" in st
        print(f"  {key:26s} {'ok' if ok else 'NO STYLES'}")
        if not ok:
            missing.append(key)
    assert not missing, f"no styles reachable for: {missing}"


def test_the_bonded_styles_come_from_the_file_not_the_force_field_name(tmp_path):
    """Naming a force field must not override what the file plainly says.

    PCFF is the case that burned us: "PCFF is class-II" led to declaring
    class2 angles, dihedrals and impropers, while DL_FIELD writes quartic,
    fourier and inversion/harmonic.
    """
    from paaf.cell.relax import lammps_styles_for

    data = tmp_path / "lammps1.data"
    data.write_text(
        "# Force field scheme: pcff\n\n1 atoms\n\n1 atom types\n"
        "1 bond types\n1 angle types\n\n"
        "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n"
        "Masses\n\n1 12.011\n\n"
        "Bond Coeffs\n\n1 class2 1.52 253.7 -423.0 396.9\n\n"
        "Angle Coeffs\n\n1 quartic 100.3 38.9 -3.8 -8.0\n\n"
        "Atoms\n\n1 1 1 0.0 0.0 0.0 0.0\n")

    st = lammps_styles_for("pcff", data)
    print(f"\n  bond {st['bond_style']} / angle {st['angle_style']}")
    assert st["bond_style"] == "class2"
    assert st["angle_style"] == "quartic", \
        "class2 angles would demand BondBond/BondAngle sections DL_FIELD omits"
    # The non-bonded half still comes from the scheme, since the data file
    # carries no Pair Coeffs section.
    assert "class2" in st["pair_style"]
    assert st["pair_modify"] == "mix sixthpower"


# ============================================================== GROMACS deck
def test_em_mdp_uses_steepest_descent_with_a_capped_step(tmp_path):
    """cg assumes a locally quadratic surface; a 1 A contact is not that."""
    from paaf.cell.relax import write_em_mdp
    p = write_em_mdp(tmp_path, RelaxSettings())
    text = p.read_text()
    print("\n  " + "\n  ".join(
        l for l in text.splitlines() if l and not l.startswith(";")))
    assert "integrator      = steep" in text
    assert "emstep" in text
    assert "constraints     = none" in text, \
        "constraining bonds during overlap resolution causes LINCS failures"
    assert "pbc             = xyz" in text


def test_gro_is_parsed_by_column_and_converted_to_angstrom(tmp_path):
    """.gro is fixed-width and in nm; splitting on spaces mis-parses it."""
    from paaf.cell.relax import read_gro_coordinates
    gro = tmp_path / "em.gro"
    gro.write_text(
        "cell\n 2\n"
        "    1PE      C1    1   1.000   2.000   3.000\n"
        "    1PE      C2    2  11.111  22.222  33.333\n"
        "   4.00000   4.00000   4.00000\n")
    got = read_gro_coordinates(gro, 2)
    print(f"\n  parsed (A): {got.tolist()}")
    assert np.allclose(got[0], [10.0, 20.0, 30.0])
    assert np.allclose(got[1], [111.11, 222.22, 333.33], atol=1e-2)


def test_gro_with_the_wrong_atom_count_is_refused(tmp_path):
    from paaf.cell.relax import read_gro_coordinates
    gro = tmp_path / "em.gro"
    gro.write_text("cell\n 2\n"
                   "    1PE      C1    1   1.000   2.000   3.000\n"
                   "   4.0   4.0   4.0\n")
    assert read_gro_coordinates(gro, 5) is None


def test_gromacs_route_reports_when_gmx_is_absent(tmp_path, monkeypatch):
    import paaf.cell.relax as relax_mod
    from paaf.cell.relax import relax_cell_gromacs
    monkeypatch.setattr(relax_mod, "find_gromacs", lambda *_a, **_k: None)
    mol = _tiny_molecule()
    gro = tmp_path / "cell.gro"; gro.write_text("x\n 0\n 1 1 1\n")
    top = tmp_path / "cell.top"; top.write_text("; top\n")
    res = relax_cell_gromacs(mol, (20.0, 20.0, 20.0), gro, top,
                             tmp_path / "work")
    print(f"\n  {res.messages[0][:110]}")
    assert not res.ran
    assert res.engine == "gromacs"
    assert res.input_script is not None and res.input_script.exists()
    assert any("NOT run" in m for m in res.messages)


# ============================================ styles must actually be found
def test_moltemplate_output_names_are_not_assumed(tmp_path):
    """moltemplate names its output after the INPUT .lt file.

    A cell built from ``cell_system.lt`` yields ``cell_system.in.init``, not
    ``system.in.init``. Hardcoding the latter produced a deck with no force
    field at all — which LAMMPS ran, warning "Bonds are defined but no bond
    style is set", and which PAAF then reported as a successful relaxation.
    """
    from paaf.cell.cell_export import _first_name

    work = tmp_path / "moltemplate"
    work.mkdir()
    for n in ("cell_system.in.init", "cell_system.in.settings",
              "cell_system.in.charges", "cell_system.data", "cell_chain1.lt"):
        (work / n).write_text("x\n")
    got = (_first_name(work, "*.in.init"),
           _first_name(work, "*.in.settings"),
           _first_name(work, "*.in.charges"))
    print(f"\n  found: {got}")
    assert got == ("cell_system.in.init", "cell_system.in.settings",
                   "cell_system.in.charges")
    assert _first_name(work, "*.in.nothing") == ""


def test_charges_file_is_included_when_present(tmp_path):
    """Moltemplate keeps partial charges separate; omitting them zeroes them."""
    p = write_relax_input(tmp_path, "cell.data", RelaxSettings(),
                          init_file="cell_system.in.init",
                          settings_file="cell_system.in.settings",
                          charges_file="cell_system.in.charges")
    text = p.read_text()
    print(f"\n  charges included: "
          f"{text.count('cell_system.in.charges')} time(s)")
    # Once when the data file is read, once when the force field is restored
    # after the soft push-off.
    assert text.count("cell_system.in.charges") == 2


def test_a_deck_with_no_force_field_is_refused_not_run(tmp_path):
    """LAMMPS exits ZERO after minimising with no styles at all.

    It warns, does nothing physical, and returns success — so "the run
    worked" is not evidence that anything was relaxed. PAAF must refuse
    rather than rely on the return code.
    """
    mol = _tiny_molecule()
    before = np.array([a.xyz for a in mol.atoms]).copy()
    data = tmp_path / "cell.data"
    data.write_text("dummy\n")
    exe = _stub_lammps(tmp_path, len(mol.atoms), shift=3.0)

    res = relax_cell(mol, (20.0, 20.0, 20.0), data, tmp_path / "work",
                     settings=RelaxSettings(lammps_exe=str(exe)))
    after = np.array([a.xyz for a in mol.atoms])
    print(f"\n  ran={res.ran}")
    print(f"  {res.messages[0][:130]}")
    assert not res.ran, "a force-field-less minimisation was reported as real"
    assert np.allclose(before, after)
    assert any("Refusing to run" in m for m in res.messages)
    # The deck is still written, so it can be inspected.
    assert res.input_script.exists()


def test_a_deck_with_styles_does_run(tmp_path):
    """The counterpart: with styles present the refusal must not fire."""
    mol = _tiny_molecule()
    data = tmp_path / "cell.data"
    data.write_text("dummy\n")
    exe = _stub_lammps(tmp_path, len(mol.atoms), shift=3.0)
    res = relax_cell(mol, (20.0, 20.0, 20.0), data, tmp_path / "work",
                     settings=RelaxSettings(lammps_exe=str(exe)),
                     init_file="cell_system.in.init",
                     settings_file="cell_system.in.settings")
    print(f"\n  ran={res.ran}, PE={res.potential_energy}")
    assert res.ran
    assert res.potential_energy == pytest.approx(-1234.5)


def test_energy_is_read_from_the_thermo_table():
    """Header-driven, so the thermo_style column order can change freely."""
    from paaf.cell.relax import _parse_log
    log = ("Step Temp Press Density PotEng E_vdwl E_bond\n"
           "       0    0  -123.4   0.850   1234.5   10.0   5.0\n"
           "      42    0  -100.2   0.912   -987.6   -8.0   2.0\n"
           "Loop time of 0.5\n")
    pe, dens = _parse_log(log)
    print(f"\n  last thermo row -> PE {pe}, density {dens}")
    assert pe == pytest.approx(-987.6)
    assert dens == pytest.approx(0.912)


def test_missing_thermo_table_is_reported_as_unknown_not_zero():
    from paaf.cell.relax import _parse_log
    assert _parse_log("no thermo here at all\n") == (None, None)


# ================================================ restoring the force field
def test_only_style_commands_are_reissued_after_the_push_off(tmp_path):
    """The whole .in.init cannot be re-included.

    Moltemplate's init file starts with units / atom_style / boundary, and
    LAMMPS rejects those once a box exists: "Units command after simulation
    box is defined". Only the style commands may be reissued.
    """
    from paaf.cell.relax import extract_style_lines

    init = tmp_path / "cell_system.in.init"
    init.write_text(
        "# generated by moltemplate\n"
        "units real\n"
        "atom_style full\n"
        "boundary p p p\n"
        "pair_style lj/cut/coul/long 10.0 10.0\n"
        "bond_style harmonic\n"
        "angle_style harmonic\n"
        "dihedral_style opls\n"
        "improper_style harmonic\n"
        "kspace_style pppm 0.0001\n"
        "pair_modify mix geometric\n"
        "special_bonds lj/coul 0.0 0.0 0.5\n")
    lines = extract_style_lines(init)
    print("\n  reissued:\n    " + "\n    ".join(lines))
    assert not any(l.startswith(("units", "atom_style", "boundary"))
                   for l in lines), "a box-defining command would be reissued"
    assert lines[0].startswith("pair_style")
    # Order matters: pair_modify and special_bonds must follow pair_style.
    assert lines.index("pair_modify mix geometric") > 0
    assert len(lines) == 8


def test_the_deck_reissues_styles_not_the_init_file(tmp_path):
    from paaf.cell.relax import extract_style_lines

    init = tmp_path / "cell_system.in.init"
    init.write_text("units real\natom_style full\n"
                    "pair_style lj/cut/coul/long 10.0 10.0\n"
                    "dihedral_style opls\n")
    p = write_relax_input(
        tmp_path, "cell.data", RelaxSettings(),
        init_file="cell_system.in.init",
        settings_file="cell_system.in.settings",
        restore_lines=extract_style_lines(init))
    text = p.read_text()
    commands = [l.strip() for l in text.splitlines()
                if l.strip() and not l.lstrip().startswith("#")]

    # Included exactly once, before read_data, never after.
    i_read = next(i for i, l in enumerate(commands) if l.startswith("read_data"))
    includes_after = [l for l in commands[i_read:]
                      if l.startswith("include") and "in.init" in l]
    print(f"\n  .in.init includes after read_data: {includes_after}")
    assert includes_after == [], "the init file was re-included after read_data"
    squashed = [" ".join(l.split()) for l in commands]
    assert "dihedral_style opls" in squashed
    # Exactly one units command, and it comes first.
    assert squashed[0] == "units real"
    assert sum(1 for l in squashed if l.startswith("units ")) == 1
    assert sum(1 for l in squashed if l.startswith("atom_style ")) == 1


def test_push_off_is_dropped_when_the_styles_cannot_be_recovered(tmp_path):
    """Better a weaker minimisation than one left under the soft potential."""
    mol = _tiny_molecule()
    data = tmp_path / "cell.data"
    data.write_text("dummy\n")
    exe = _stub_lammps(tmp_path, len(mol.atoms))
    work = tmp_path / "work"
    work.mkdir()
    (work / "cell_system.in.init").write_text("units real\natom_style full\n")

    res = relax_cell(mol, (20.0, 20.0, 20.0), data, work,
                     settings=RelaxSettings(lammps_exe=str(exe)),
                     init_file="cell_system.in.init",
                     settings_file="cell_system.in.settings")
    text = res.input_script.read_text()
    print(f"\n  {res.messages[0][:120]}")
    assert "pair_style      soft" not in text
    assert any("push-off" in m and "disabled" in m for m in res.messages)
