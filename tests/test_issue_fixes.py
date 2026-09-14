"""Regression tests for the items in presentation/PAAF_issues_to_fix.md.

Each test names the issue it pins, so a failure points straight at the entry
that describes what used to go wrong.
"""
from __future__ import annotations

import inspect
import json
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest


# ------------------------------------------------------------------ P01 / P06
def test_p01_build_prints_valid_json_and_p06_box_is_honoured(monkeypatch, capsys, tmp_path):
    import paaf.cli as cli
    seen = {}

    def fake_run(cfg):
        seen["cfg"] = cfg
        return {"config": asdict(cfg), "output_dir": Path(tmp_path)}

    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    rc = cli.main(["build", "--monomer", "x.pdb", "--box", "40", "50", "60",
                   "--output", str(tmp_path)])
    assert rc == 0
    json.loads(capsys.readouterr().out)
    box = seen["cfg"].box
    assert (box.a, box.b, box.c) == (40.0, 50.0, 60.0)
    assert box.shape == "orthorhombic"


def test_p36_legacy_box_size_maps_onto_edges():
    from paaf.config import BoxCfg
    b = BoxCfg(size=[80.0, 80.0, 80.0])
    assert (b.a, b.b, b.c) == (80.0, 80.0, 80.0)


def test_p20_density_that_looks_like_g_cm3_is_refused(capsys):
    import paaf.cli as cli
    rc = cli.main(["pack-cell", "--species", "x.xyz:1", "--density", "1.0", "--out", "y.pdb"])
    assert rc == 2
    assert "g/cm" in capsys.readouterr().err


# ------------------------------------------------------------------ P04 / P05
def _in_thread(fn, timeout=5.0):
    box = {}

    def run():
        try:
            box["value"] = fn()
        except Exception as exc:              # noqa: BLE001
            box["error"] = exc
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "call hung"
    return box


def test_p04_zero_block_sizes_raise_instead_of_hanging():
    from paaf.chain_builder import _make_sequence
    for sizes in ([0, 0], [3]):
        got = _in_thread(lambda: _make_sequence(2, 10, "block", None, sizes, 1))
        assert isinstance(got.get("error"), ValueError)


def test_p05_wrong_fraction_count_names_fractions():
    from paaf.chain_builder import _make_sequence
    with pytest.raises(ValueError, match="fractions"):
        _make_sequence(2, 10, "random", [0.5], None, 1)


# ------------------------------------------------------------------ P11
@pytest.mark.parametrize("smiles,model", [
    ("C([*])C[*]", "PE"), ("[*]C(C)C[*]", "PP"),
    ("[*]C(c1ccccc1)C[*]", "PS"), ("[*]OCCO[*]", "")])
def test_p11_ris_model_matches_equivalent_smiles(smiles, model):
    pytest.importorskip("rdkit")
    pytest.importorskip("PyQt5")
    from paaf.gui.amorphous_tab import ComponentRow
    assert ComponentRow._guess_ris(smiles) == model


# ------------------------------------------------------------------ P14 / P15
def test_p15_cancel_during_optimisation_propagates():
    pytest.importorskip("openbabel")
    from paaf import optimizer
    from paaf.cell.packing import PackCancelled
    from paaf.structure import load_smiles

    class Token:
        def is_cancelled(self):
            return True

    with pytest.raises(PackCancelled):
        optimizer.optimize(load_smiles("CCCCCCCC"), ff="MMFF94", steps=500,
                           cancel=Token())


def test_p14_optimizer_settings_reach_the_smiles_builder(monkeypatch):
    import paaf.chain_builder as cb
    sig = inspect.signature(cb.build_chain).parameters
    for name in ("optimize", "opt_ff", "opt_steps", "opt_tol", "opt_algorithm", "cancel"):
        assert name in sig


def test_optimizer_rejects_bad_steps_and_tolerance():
    pytest.importorskip("openbabel")
    from paaf import optimizer
    from paaf.structure import load_smiles
    with pytest.raises(ValueError):
        optimizer.optimize(load_smiles("CCO"), steps=0)
    with pytest.raises(ValueError):
        optimizer.optimize(load_smiles("CCO"), tol=0.0)


@pytest.mark.parametrize("ff", ["MMFF94", "MMFF94s", "UFF", "Ghemical", "GAFF"])
@pytest.mark.parametrize("alg", ["cg", "sd"])
def test_every_optimize_page_force_field_runs(ff, alg):
    pytest.importorskip("openbabel")
    from paaf import optimizer
    from paaf.structure import load_smiles
    mol = load_smiles("CCOC(=O)C")
    before = np.array([a.xyz for a in mol.atoms]).copy()
    optimizer.optimize(mol, ff=ff, steps=200, tol=1e-4, algorithm=alg)
    after = np.array([a.xyz for a in mol.atoms])
    assert np.isfinite(after).all()
    assert not np.allclose(before, after), f"{ff}/{alg} did not move any atom"


# ------------------------------------------------------------------ P16
def test_p16_find_clashes_matches_the_dense_reference():
    from paaf.geometry_repair import _rcov, find_clashes
    from paaf.structure import Atom, Molecule
    rng = np.random.default_rng(3)
    els = rng.choice(["C", "H", "O"], 300)
    atoms = [Atom(index=i, element=str(e), xyz=rng.uniform(0, 12, 3))
             for i, e in enumerate(els)]
    bonds = [(i, i + 1, 1.0) for i in range(0, 298, 3)]
    mol = Molecule(atoms=atoms, bonds=bonds, name="r")
    clashes, _bad = find_clashes(mol)

    xyz = mol.coords()
    r = np.array([_rcov(e) for e in els])
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    bonded = np.zeros((300, 300), bool)
    for i, j, _ in bonds:
        bonded[i, j] = bonded[j, i] = True
    close = (d < 1.15 * (r[:, None] + r[None])) & ~bonded
    np.fill_diagonal(close, False)
    ii, jj = np.where(np.triu(close))
    assert [(a, b) for a, b, _ in clashes] == list(zip(ii.tolist(), jj.tolist()))


# ------------------------------------------------------------------ P21
def test_p21_layer_gap_default():
    from paaf.layering import plan_regions
    assert inspect.signature(plan_regions).parameters["gap"].default == 5.0


# ------------------------------------------------------------------ P27
def test_p27_oplsaa2008_uses_its_own_numbering():
    from paaf.type_guard import check_assignment, elements_for_ff, type_elements
    t2008 = elements_for_ff("oplsaa2008")
    assert t2008.get("136") == "H"
    assert check_assignment("H", "136", t2008) is None
    assert type_elements().get("136") == "C"          # 2024 default unchanged


# ------------------------------------------------------------------ P32
def test_p32_ether_carbons_are_not_alkane_ch3():
    pytest.importorskip("openbabel")
    from paaf.structure import load_smiles
    from paaf.typers.oplsaa import type_oplsaa
    mol = load_smiles("CCOCCOCC", make_3d=False)
    types = type_oplsaa(mol)
    assert all(types[i] == "182" for i in (1, 3, 4, 6))


# ------------------------------------------------------------------ P33
def test_p33_blend_refuses_different_mixing_and_takes_tightest_kspace():
    from paaf.blend_styles import StyleBlock, StyleMismatch, merge_style_blocks
    base = {"units": "real", "atom_style": "full", "pair_style": "lj/cut/coul/long 10"}
    a = StyleBlock(directives=dict(base, pair_modify="mix geometric",
                                   kspace_style="pppm 1.0e-4"))
    b = StyleBlock(directives=dict(base, pair_modify="mix arithmetic",
                                   kspace_style="pppm 1.0e-5"))
    with pytest.raises(StyleMismatch):
        merge_style_blocks([a, b], ["a", "b"])
    b.directives["pair_modify"] = "mix geometric"
    merged, _ = merge_style_blocks([a, b], ["a", "b"])
    assert merged.directives["kspace_style"].endswith("1.0e-5")


# ------------------------------------------------------------------ P40
def test_p40_blend_without_minimise_settings_does_not_minimise():
    import paaf.blend_replicator as br
    src = inspect.getsource(br.replicate_blend)
    assert "MinimiseSettings(enabled=False)" in src


# ------------------------------------------------------------------ P07 / P35
def test_p07_p35_triclinic_box_carries_tilt(tmp_path):
    from paaf import lt_writer
    from paaf.cell.amorphous import BoxShape
    shape = BoxShape(shape="triclinic", a=30, b=30, c=30, gamma=60)
    lx, ly, lz, xy, xz, yz = shape.lammps_params()
    assert xy == pytest.approx(15.0)
    assert lx * ly * lz == pytest.approx(shape.volume_ang3())
    chain_lt = tmp_path / "X.lt"
    chain_lt.write_text("X {}\n")
    txt = lt_writer.write_system_lt(tmp_path, chain_lt, 1,
                                    box=list(shape.lammps_params())).read_text()
    assert "xy xz yz" in txt


# ------------------------------------------------------------------ reactions
def test_p30_p08_p09_p34_reaction_engine(tmp_path):
    pytest.importorskip("rdkit")
    pytest.importorskip("openbabel")
    from paaf.reaction import ReactionLibrary
    from paaf.reaction_smiles import build_template
    from paaf.structure import Atom, Molecule, load_smiles, write
    from paaf.xlink_engine import _distance, apply_library

    tmpl = build_template(["[CH3:1][OH:2]", "[OH:3][C:4](=O)C"],
                          ["[CH3:1][O:2][C:4](=O)C", "[OH2:3]"], name="ester")
    atoms, bonds, off = [], [], 0
    for smi, shift in (("CO", 0), ("CC(=O)O", 40), ("CO", 80), ("CC(=O)O", 83)):
        m = load_smiles(smi)
        for a in m.atoms:
            atoms.append(Atom(index=a.index + off, element=a.element,
                              xyz=np.asarray(a.xyz, float) + [shift, 0, 0]))
        bonds += [(i + off, j + off, o) for i, j, o in m.bonds]
        off += len(m.atoms)
    system = tmp_path / "sys.mol2"
    write(Molecule(atoms=atoms, bonds=bonds, name="sys"), system)
    lib = ReactionLibrary()
    lib.templates.append(tmpl)
    lib.save(tmp_path / "lib.json")
    stats = apply_library(system, ReactionLibrary.load(tmp_path / "lib.json"),
                          tmp_path / "out.mol2", cutoff=6.0)
    assert stats.events_applied >= 1                     # P30 + P08

    two = Molecule(atoms=[Atom(index=0, element="C", xyz=np.array([0.5, 5, 5])),
                          Atom(index=1, element="C", xyz=np.array([19.5, 5, 5]))],
                   bonds=[], name="x")
    assert _distance(two, 0, 1, np.array([20.0, 20.0, 20.0])) == pytest.approx(1.0)  # P09

    hyd = build_template(["[CH2:1]=[CH2:2]", "[H:3][H:4]"], ["[CH3:1][CH3:2]"], name="h")
    assert any(old == 2.0 and new == 1.0 for _a, _b, old, new in hyd.changed_bonds)  # P34


# ------------------------------------------------------------------ P02 / P03
def test_p02_soft_stage_switches_kspace_off(tmp_path):
    from paaf.cell.relax import RelaxSettings, write_relax_input
    init = tmp_path / "system.in.init"
    init.write_text("units real\natom_style full\npair_style lj/cut/coul/long 10.0\n"
                    "kspace_style pppm 1e-4\n")
    restore = ["pair_style lj/cut/coul/long 10.0", "kspace_style pppm 1e-4"]
    path = write_relax_input(tmp_path, "cell.data", RelaxSettings(push_off=True),
                             init_file=init.name, settings_file="system.in.settings",
                             restore_lines=restore)
    text = Path(path).read_text() if not isinstance(path, str) or Path(path).exists() \
        else (tmp_path / "relax.in").read_text()
    soft = text.index("pair_style      soft")
    assert text.rfind("kspace_style    none", 0, soft) != -1
    assert text.find("kspace_style pppm", soft) != -1


# ------------------------------------------------------------------ P29
def test_p29_cancel_kills_external_program_quickly():
    import time
    from paaf.cell.packing import PackCancelled, run_cancellable

    class Token:
        def is_cancelled(self):
            return True

    start = time.monotonic()
    with pytest.raises(PackCancelled):
        run_cancellable(["sleep", "30"], cancel=Token())
    assert time.monotonic() - start < 3.0


# ------------------------------------------------------------------ P37
def test_p37_packmol_region_is_inset_by_half_the_tolerance():
    from paaf.lammps_replicator import inset_box_region
    assert inset_box_region(2.0, 60, 60, 60) == \
        "inside box 1.0000 1.0000 1.0000 59.0000 59.0000 59.0000"


# ------------------------------------------------------------------ P39
def test_p39_module_logs_reach_the_paaf_logger():
    import logging
    from paaf.logging_utils import get_logger
    got = []

    class H(logging.Handler):
        def emit(self, record):
            got.append(record.getMessage())

    h = H()
    # Other test modules call logging.disable(CRITICAL) and leave it on.
    previous = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    logging.getLogger("paaf").addHandler(h)
    try:
        get_logger("paaf.typers.oplsaa").warning("reaches the console")
    finally:
        logging.getLogger("paaf").removeHandler(h)
        logging.disable(previous)
    assert "reaches the console" in got
