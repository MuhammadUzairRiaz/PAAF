"""Regression tests for the items still open in presentation/PAAF_issues_to_fix.md
after commit bde6aeb (P03, P08, P09, P12, P23, P24, P25, P27, P29, P38, P41,
P48, P49). Each test names the issue it pins.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ P03
_DLF_IN = """units real
atom_style full
boundary p p p
bond_style hybrid harmonic
pair_style hybrid lj/cut/coul/long 10.0
kspace_style pppm 1.0e-4
special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.5
read_data lammps1.data
pair_coeff 1 1 lj/cut/coul/long 0.066 3.5 # CT CT
"""


def test_p03_dlfield_relax_deck_reapplies_pair_coefficients(tmp_path):
    from paaf.cell.relax import (RelaxSettings, extract_coeff_lines,
                                 extract_style_lines, write_relax_input)
    src = tmp_path / "lammps.in"
    src.write_text(_DLF_IN)
    coeffs = extract_coeff_lines(src)
    assert coeffs == ["pair_coeff 1 1 lj/cut/coul/long 0.066 3.5"]
    styles = dict(ln.split(None, 1) for ln in extract_style_lines(src))
    (tmp_path / "dlf_coeffs.in").write_text("\n".join(coeffs) + "\n")
    text = write_relax_input(tmp_path, "cell.data", RelaxSettings(),
                             settings_file="dlf_coeffs.in", styles=styles,
                             styles_source="DL_FIELD").read_text()
    assert text.count("read_data") == 1
    assert "pair_coeff" not in text.split("read_data")[0]
    i_read = text.index("read_data")
    i_first_inc = text.index("include         dlf_coeffs.in")
    i_soft = text.index("pair_style      soft")
    i_restore = text.index("pair_style      hybrid lj/cut/coul/long", i_soft)
    i_reinc = text.index("include         dlf_coeffs.in", i_restore)
    i_min = text.index("minimize")
    assert i_read < i_first_inc < i_soft < i_restore < i_reinc < i_min
    assert text.rfind("kspace_style    none", 0, i_soft) != -1
    assert text.find("kspace_style    pppm", i_restore) != -1


def test_p03_dlfield_relax_deck_runs_in_lammps(tmp_path):
    """The real DL_FIELD output must relax: rc 0 (skips without LAMMPS)."""
    from paaf.cell.relax import (RelaxSettings, extract_coeff_lines,
                                 extract_style_lines, find_lammps,
                                 write_relax_input)
    dlf = ROOT / "output/project/ENR_maleic_acid_ENR_bridge/_typing_product/dlf_output1"
    exe = find_lammps()
    if exe is None or not (dlf / "lammps.in").exists():
        pytest.skip("needs LAMMPS and the ENR DL_FIELD example output")
    # DL_FIELD typed ONE molecule in a 250 A vacuum box; PPPM over that
    # nearly empty grid segfaults this LAMMPS build even with DL_FIELD's own
    # unmodified lammps.in. A relaxed cell has a dense box, so use one.
    text = (dlf / "lammps1.data").read_text()
    text = re.sub(r"-125\.0+\s+125\.0+ (\w)lo (\w)hi", r"-30.0 30.0 \1lo \2hi", text)
    (tmp_path / "lammps1.data").write_text(text)
    styles = dict(ln.split(None, 1) for ln in extract_style_lines(dlf / "lammps.in"))
    (tmp_path / "dlf_coeffs.in").write_text(
        "\n".join(extract_coeff_lines(dlf / "lammps.in")) + "\n")
    deck = write_relax_input(tmp_path, "lammps1.data",
                             RelaxSettings(push_steps=20, maxiter=20, maxeval=200),
                             settings_file="dlf_coeffs.in", styles=styles)
    proc = subprocess.run([str(exe), "-in", deck.name], cwd=tmp_path,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-1500:]
    assert (tmp_path / "relaxed.data").exists()


# ------------------------------------------------------------------ P27
def test_p27_provenance_uses_the_selected_library_numbering():
    from paaf.chain_provenance import expand_by_provenance
    from paaf.structure import Atom, Molecule
    from paaf.type_guard import elements_for_ff
    chain = Molecule(atoms=[Atom(index=0, element="H", xyz=np.zeros(3))])
    out = expand_by_provenance(chain, {0: "136"}, {0: (0, 0)},
                               type_elements=elements_for_ff("oplsaa2008"))
    assert out == {0: "136"}                       # 2008 @atom:136 is an H
    assert len(elements_for_ff("loplsaa2008")) > 800   # base 2008 types merged


def test_p27_assign_passes_the_force_field_table(monkeypatch):
    import paaf.ff_assigner as fa
    seen = {}

    def fake_expand(mol, manual, **kw):
        seen["table"] = kw.get("type_elements")
        return {}
    monkeypatch.setattr(fa, "expand_manual_types", fake_expand)

    class FF:
        key = "oplsaa2008"
        atom_typer = "manual"
    with pytest.raises(RuntimeError):
        fa.assign(__import__("paaf.structure", fromlist=["Molecule"]).Molecule(),
                  FF(), manual_types={0: "136"})
    assert seen["table"].get("136") == "H"


# ------------------------------------------------------------------ P29
def test_p29_blend_packmol_honours_a_cancel(tmp_path):
    from paaf.blend_replicator import _build_blend_packmol, _find_packmol
    from paaf.cell.packing import CancelToken, PackCancelled
    if _find_packmol() is None:
        pytest.skip("packmol not installed")
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(PackCancelled):
        _build_blend_packmol([tmp_path / "a.pdb"], [1], tmp_path / "out.pdb",
                             (20.0, 20.0, 20.0), 2.0, 1, cancel=tok)


def test_p29_every_long_call_takes_a_cancel():
    import inspect
    from paaf.blend_minimise import minimise_component
    from paaf.blend_replicator import replicate_blend
    from paaf.cell.amorphous import pack_cell
    from paaf.gromacs_blend import replicate_gromacs_blend
    from paaf.gromacs_packer import energy_minimise_single_chain, pack_with_gmx_insert
    from paaf.layering import build_layered_cell
    for fn in (minimise_component, replicate_blend, pack_cell,
               replicate_gromacs_blend, pack_with_gmx_insert,
               energy_minimise_single_chain, build_layered_cell):
        assert "cancel" in inspect.signature(fn).parameters, fn.__name__
    for mod in ("cell/amorphous.py", "blend_replicator.py", "blend_minimise.py",
                "gromacs_blend.py", "gromacs_packer.py"):
        assert "subprocess.run(" not in (ROOT / "paaf" / mod).read_text(), mod


def test_p29_blend_and_layering_pages_have_a_cancel_button():
    pytest.importorskip("PyQt5")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from paaf.gui.blend_tab import BlendTab
    from paaf.gui.layering_tab import LayeringTab
    for cls in (BlendTab, LayeringTab):
        page = cls()
        assert not page.cancel_btn.isEnabled()
        page._cancel_run()                        # no run yet: harmless
    assert app is not None


# ------------------------------------------------------------------ P08
def _ester_library(tmp_path):
    from paaf.reaction import ReactionLibrary
    from paaf.reaction_smiles import build_template
    lib = ReactionLibrary()
    lib.templates.append(build_template(["[CH3:1][OH:2]", "[OH:3][C:4](=O)C"],
                                        ["[CH3:1][O:3][C:4](=O)C", "[OH2:2]"],
                                        name="ester"))
    return lib


def _methanol_acid(atoms, bonds, origin, co, oh_h):
    """Hand-placed methanol + acetic acid. ``co``: methanol C -> acid O(H)
    offset; ``oh_h``: where the acid's hydroxyl H sits (its distance to the
    methanol O decides the second created bond)."""
    from paaf.structure import Atom
    o = np.asarray(origin, float)
    k0 = len(atoms)
    pts = [("C", (0, 0, 0)), ("O", (0, 1.4, 0)), ("H", (-1, 0, 0)), ("H", (0, -1, 0)),
           ("H", (0, 0, -1)), ("H", (0, 2.4, 0)),                       # methanol
           ("O", co), ("H", oh_h), ("C", (co[0] + 1.2, 0.5, 0)),
           ("O", (co[0] + 1.2, 1.7, 0)), ("C", (co[0] + 2.5, 0, 0)),
           ("H", (co[0] + 3.5, 0, 0)), ("H", (co[0] + 2.5, -1, 0)),
           ("H", (co[0] + 2.5, 0, -1))]                                  # acid
    for el, p in pts:
        atoms.append(Atom(index=len(atoms), element=el, xyz=o + np.asarray(p, float)))
    for i, j, order in ((0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (1, 5, 1),
                        (6, 7, 1), (6, 8, 1), (8, 9, 2), (8, 10, 1), (10, 11, 1),
                        (10, 12, 1), (10, 13, 1)):
        bonds.append((k0 + i, k0 + j, float(order)))


def test_p08_a_pair_with_one_far_created_bond_does_not_stop_the_template(tmp_path):
    pytest.importorskip("rdkit")
    pytest.importorskip("openbabel")
    from paaf.structure import Molecule, write
    from paaf.xlink_engine import apply_library
    atoms, bonds = [], []
    # Pair A: C...O = 3.0 A (nearest, tried first) but the acid H is 14 A
    # from the methanol O, so the water O-H bond cannot form.
    _methanol_acid(atoms, bonds, (0, 0, 0), (3.0, 0, 0), (3.0, -12.0, 0))
    # Pair B: C...O = 3.5 A and O...H = 3.2 A — a valid event.
    _methanol_acid(atoms, bonds, (100, 0, 0), (3.5, 0, 0), (3.5, 0.9, 0))
    system = tmp_path / "sys.mol2"
    write(Molecule(atoms=atoms, bonds=bonds, name="sys"), system)
    stats = apply_library(system, _ester_library(tmp_path), tmp_path / "out.xyz",
                          cutoff=5.0)
    assert stats.events_applied == 1


# ------------------------------------------------------------------ P09
def test_p09_periodic_box_is_read_and_used(tmp_path):
    pytest.importorskip("rdkit")
    pytest.importorskip("openbabel")
    from paaf.structure import Molecule, load_structure, write
    from paaf.xlink_engine import apply_library
    atoms, bonds = [], []
    # Acid 17 A away directly, 3 A away through the periodic x face.
    _methanol_acid(atoms, bonds, (1.0, 10, 10), (17.0, 0, 0), (17.0, 0.9, 0))
    plain = tmp_path / "plain.pdb"
    write(Molecule(atoms=atoms, bonds=bonds, name="sys"), plain)
    periodic = tmp_path / "periodic.pdb"
    periodic.write_text("CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1\n"
                        + plain.read_text())
    assert load_structure(periodic).cell == (20.0, 20.0, 20.0)
    lib = _ester_library(tmp_path)
    assert apply_library(plain, lib, tmp_path / "a.xyz", cutoff=5.0).events_applied == 0
    assert apply_library(periodic, lib, tmp_path / "b.xyz", cutoff=5.0).events_applied >= 1
    # --box on the CLI does the same for a file without a box
    assert apply_library(plain, lib, tmp_path / "c.xyz", cutoff=5.0,
                         box=(20, 20, 20)).events_applied >= 1


def test_p09_lammps_data_file_loads_with_its_box(tmp_path):
    from paaf.structure import load_structure
    data = tmp_path / "packed_box.data"
    data.write_text("""LAMMPS data

3 atoms
2 bonds

2 atom types
1 bond types

-5.0 15.0 xlo xhi
0.0 20.0 ylo yhi
0.0 30.0 zlo zhi

Masses

1 12.011  # c3
2 1.008

Atoms  # full

2 1 2 0.06 1.0 0.0 0.0
1 1 1 -0.12 0.0 0.0 0.0
3 1 2 0.06 -1.0 0.0 0.0

Bonds

1 1 1 2
2 1 1 3
""")
    mol = load_structure(data)
    assert mol.cell == (20.0, 20.0, 30.0)
    assert [a.element for a in mol.atoms] == ["C", "H", "H"]
    assert sorted((i, j) for i, j, _o in mol.bonds) == [(0, 1), (0, 2)]
    assert mol.atoms[0].charge == pytest.approx(-0.12)


def test_p09_cli_accepts_box(monkeypatch):
    import paaf.cli as cli
    import paaf.reaction as reaction
    import paaf.xlink_engine as xe
    seen = {}

    def fake_apply(system, lib, out, **kw):
        seen.update(kw)
        return xe.XlinkStats()
    monkeypatch.setattr(xe, "apply_library", fake_apply)
    monkeypatch.setattr(reaction.ReactionLibrary, "load",
                        classmethod(lambda cls, p: cls()))
    rc = cli.main(["reactions-apply", "--library", "l.json", "--system", "s.data",
                   "--out", "o.xyz", "--box", "20", "20", "20"])
    assert rc == 0
    assert seen["box"] == [20.0, 20.0, 20.0]


# ------------------------------------------------------------------ P12
def test_p12_backbone_cg_bonds_start_near_the_fene_minimum(tmp_path):
    from paaf.cell.cg_model import CGSettings, build_cg_cell
    from paaf.cell.composition import Component, from_chain_counts
    from paaf.cell.grow import grow_amorphous_cell
    comp = from_chain_counts([Component(name="PE", repeat_unit="[*]CC[*]",
                                        degree_of_polymerisation=10, n_chains=2,
                                        ris_key="PE")], 0.85)
    res = grow_amorphous_cell(comp.grow_specs(), comp.box(), temperature=413.0, seed=7)
    data, inp = build_cg_cell(res, comp.grow_specs(), tmp_path, settings=CGSettings(
        units="real", mapping="backbone", angle_mode="off"))
    sigma = float(re.search(r"PE=([\d.]+)", inp.read_text()).group(1))
    text = data.read_text()
    dims = np.array([float(re.search(rf"0.0 ([\d.]+) {ax}lo", text).group(1))
                     for ax in "xyz"])
    atoms = {}
    for ln in text.split("Atoms # molecular")[1].split("Bonds")[0].strip().splitlines():
        t = ln.split()
        atoms[int(t[0])] = np.array([float(v) for v in t[3:6]])
    lengths = []
    for ln in text.split("Bonds")[1].strip().splitlines():
        _b, _t, i, j = (int(v) for v in ln.split())
        d = atoms[j] - atoms[i]
        d -= dims * np.round(d / dims)
        lengths.append(np.linalg.norm(d))
    ratio = float(np.mean(lengths)) / sigma
    assert 0.85 <= ratio <= 1.1, ratio


# ------------------------------------------------------------------ P23
def test_p23_box_packmol_false_uses_the_grid_packer(tmp_path, monkeypatch):
    import paaf.lammps_replicator as lr

    def boom(*_a, **_k):
        raise AssertionError("packmol must not run when box.packmol is false")
    monkeypatch.setattr(lr, "_pack_with_packmol", boom)
    single = tmp_path / "single.data"
    single.write_text("""chain

2 atoms
1 bonds

1 atom types
1 bond types

0.0 30.0 xlo xhi
0.0 30.0 ylo yhi
0.0 30.0 zlo zhi

Masses

1 12.011

Atoms  # full

1 1 1 0.0 0.0 0.0 0.0
2 1 1 0.0 1.5 0.0 0.0

Bonds

1 1 1 2
""")
    out = lr.replicate_single_chain(single, 2, (30.0, 30.0, 30.0),
                                    tmp_path / "packed.data", use_packmol=False)
    assert "4 atoms" in Path(out).read_text()
    assert "use_packmol=bool(getattr(cfg.box, \"packmol\", True))" in \
        (ROOT / "paaf/pipeline.py").read_text()
    assert "expand_if_polymer" not in (ROOT / "paaf/cell/grow.py").read_text()


# ------------------------------------------------------------------ P24
def test_p24_one_soft_stage_helper_writes_every_deck(tmp_path):
    from paaf.cell.relax import RelaxSettings, write_relax_input
    from paaf.cell.soft_stage import soft_stage_lines
    assert soft_stage_lines(3.0, 0, 60.0, nve_limit=0.05) == [
        "pair_style      soft 3.0",
        "pair_coeff      * * 0.0",
        "variable        prefactor equal ramp(0,60.0)",
        "fix             push all adapt 1 pair soft a * * v_prefactor",
        "fix             lim all nve/limit 0.05",
    ]
    assert soft_stage_lines(2.2, 1.0, 100.0, disable_kspace=True, pad=False)[:2] == [
        "kspace_style none", "pair_style soft 2.2"]
    init = tmp_path / "s.in.init"
    init.write_text("pair_style lj/cut 10\n")
    deck = write_relax_input(tmp_path, "c.data", RelaxSettings(), init_file=init.name,
                             settings_file="s.in.settings",
                             restore_lines=["pair_style lj/cut 10"]).read_text()
    assert "\n".join(soft_stage_lines(3.0, 0, 60.0, nve_limit=0.05)) in deck
    for mod in ("cell/soft_pushoff.py", "cell/relax.py", "cell/cg_model.py"):
        assert "soft_stage_lines" in (ROOT / "paaf" / mod).read_text(), mod


# ------------------------------------------------------------------ P38
def test_p38_hybrid_molecule_is_neutralised(tmp_path):
    from paaf.ua_hybrid import rebalance_hybrid_charges
    # PEO-like fragment: CH3(bead 1) - O(2) - CH2(bead 3) - H? no: all-atom C(4) with H(5)
    data = tmp_path / "system.data"
    data.write_text("""moltemplate

5 atoms
4 bonds

4 atom types
1 bond types

0 30 xlo xhi
0 30 ylo yhi
0 30 zlo zhi

Masses

1 15.035  # 74_bC3_aC3_dC3_iC3
2 15.999  # 180_bOS_aOS_dOS_iOS
3 14.027  # 71_bC2_aC2_dC2_iC2
4 12.011  # 181_bCT_aCT_dCT_iCT

Atoms  # full

1 1 1 0.0 0.0 0.0 0.0
2 1 2 0.0 1.4 0.0 0.0
3 1 3 0.0 2.8 0.0 0.0
4 1 4 0.0 4.2 0.0 0.0
5 1 2 0.0 5.6 0.0 0.0

Bonds

1 1 1 2
2 1 2 3
3 1 3 4
4 1 4 5
""")
    charges = tmp_path / "system.in.charges"
    charges.write_text("set type 1 charge 0.25\nset type 2 charge -0.40\n"
                       "set type 3 charge 0.25\nset type 4 charge 0.14\n")
    said = []
    worst = rebalance_hybrid_charges(data, charges, ["74", "71"], said.append)
    assert worst == pytest.approx(abs(0.25 - 0.40 + 0.25 + 0.14 - 0.40))
    rows = data.read_text().split("Atoms  # full")[1].split("Bonds")[0].split()
    qs = [float(rows[k + 3]) for k in range(0, len(rows), 7)]
    assert sum(qs) == pytest.approx(0.0, abs=1e-6)
    # only the all-atom neighbours of beads (atoms 2 and 4) were corrected
    assert qs[0] == pytest.approx(0.25) and qs[2] == pytest.approx(0.25)
    assert qs[4] == pytest.approx(-0.40)
    assert "set type" not in "\n".join(ln for ln in charges.read_text().splitlines()
                                      if not ln.startswith("#"))
    assert said


# ------------------------------------------------------------------ P41
def test_p41_no_silent_broad_exception_handlers():
    offenders = []
    for path in (ROOT / "paaf").rglob("*.py"):
        text = path.read_text()
        for m in re.finditer(r"except Exception:[^\n]*\n\s*pass\s*$", text, re.M):
            offenders.append(f"{path.relative_to(ROOT)}:{text[:m.start()].count(chr(10)) + 1}")
    assert not offenders, offenders


# ------------------------------------------------------------------ P25 / P48
def test_p25_readme_test_count_is_current():
    assert "About 1,900 tests" in (ROOT / "README.md").read_text()


def test_p48_crystal_carries_its_supercell():
    from paaf.cell.crystal import build_crystal
    mol = build_crystal(4.0, 5.0, 6.0, basis=[("Si", (0, 0, 0))], nx=2, ny=3, nz=1)
    cell = mol.cell
    assert (cell.a, cell.b, cell.c) == (8.0, 15.0, 6.0)
    assert cell.shape == "orthorhombic"


# ------------------------------------------------------------------ P49
def test_p49_aromatic_bonds_are_written_as_valid_mol2(tmp_path):
    pytest.importorskip("rdkit")
    pytest.importorskip("openbabel")
    from paaf.structure import load_smiles, load_structure, write
    from paaf.typers.oplsaa import type_oplsaa
    m = load_smiles("c1ccccc1", make_3d=True)
    m.bonds = [(i, j, 1.5 if m.atoms[i].element == m.atoms[j].element == "C" else o)
               for i, j, o in m.bonds]
    write(m, tmp_path / "bz.mol2")
    write(m, tmp_path / "bz.sdf")
    block = (tmp_path / "bz.mol2").read_text().split("@<TRIPOS>BOND")[1]
    assert {ln.split()[3] for ln in block.strip().splitlines()} <= {"1", "2", "ar"}
    assert {o for *_ij, o in load_structure(tmp_path / "bz.mol2").bonds} <= {1.0, 2.0, 1.5}
    assert {o for *_ij, o in load_structure(tmp_path / "bz.sdf").bonds} <= {1.0, 2.0, 1.5}
    types = type_oplsaa(m)
    assert {types[a.index] for a in m.atoms if a.element == "C"} == {"145"}
    assert {types[a.index] for a in m.atoms if a.element == "H"} == {"146"}


# ------------------------------------------------------------------ found while testing
@pytest.mark.parametrize("kw", [
    dict(ensemble="npt"),
    dict(ensemble="nvt", thermostat="langevin"),
    dict(ensemble="nvt"),
    dict(multistage=True, thermostat="langevin"),
])
def test_run_in_unfixes_only_fixes_it_defined(tmp_path, kw):
    """`unfix RELAX_nve` for a fix that never existed aborted every MD run."""
    from paaf.lt_writer import write_lammps_input
    text = write_lammps_input(tmp_path, **kw).read_text()
    defined = set()
    for ln in text.splitlines():
        toks = ln.split("#", 1)[0].split()
        if toks[:1] == ["fix"]:
            defined.add(toks[1])
        elif toks[:1] == ["unfix"]:
            assert toks[1] in defined, f"unfix of undefined fix {toks[1]}"
            defined.discard(toks[1])


def test_charge_check_reads_moltemplate_type_charges(tmp_path):
    """OPLS via moltemplate: data column zero, charges in system.in.charges."""
    from paaf.cell.cell_export import data_file_charges
    data = tmp_path / "system.data"
    data.write_text("x\n\nAtoms  # full\n\n1 1 1 0.0 0 0 0\n2 1 2 0.0 1 0 0\n")
    chg = tmp_path / "system.in.charges"
    chg.write_text("set type 1 charge -0.5\nset type 2 charge 0.5\n")
    assert data_file_charges(data) == (2, 0.0, 0.0)
    n, total, biggest = data_file_charges(data, chg)
    assert (n, biggest) == (2, 0.5) and total == pytest.approx(0.0)


def test_packmol_imperfect_packing_is_used_not_refused(tmp_path, monkeypatch):
    """packmol rc 173 ("ENDED WITHOUT PERFECT PACKING") still writes its best
    layout; a 5-chain PBS build at melt density was refused because of it."""
    import paaf.cell.packing as packing
    import paaf.lammps_replicator as lr
    single = tmp_path / "single.pdb"
    single.write_text("ATOM      1  C   MOL     1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    out = tmp_path / "packed.pdb"

    def fake_run(cmd, **kw):
        out.write_text(single.read_text())
        return 173, "ENDED WITHOUT PERFECT PACKING", ""
    monkeypatch.setattr(packing, "run_cancellable", fake_run)
    monkeypatch.setattr(lr, "_find_packmol", lambda explicit=None: "/bin/packmol")
    assert lr._pack_with_packmol(single, 2, (20.0, 20.0, 20.0), out,
                                 tolerance=2.0, seed=1) is True
