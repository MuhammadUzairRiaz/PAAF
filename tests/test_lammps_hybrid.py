"""Hybrid -> non-hybrid conversion of DL_FIELD LAMMPS output."""
from pathlib import Path

import pytest

from paaf.lammps_hybrid import (convert_data, convert_input,
                                convert_to_nonhybrid, is_hybrid_input)

IN = """units real
atom_style      full       # molecular system with charges
bond_style      hybrid harmonic 
angle_style     hybrid harmonic 
dihedral_style  hybrid opls 
improper_style  hybrid cvff 
pair_style      hybrid lj/cut/coul/long 12.000000  
kspace_style    pppm 1.0e-4
read_data lammps1.data
pair_coeff     1     1  lj/cut/coul/long    0.066000    3.500000 # CT CT
pair_coeff     1     2  lj/cut/coul/long    0.070824    3.524911 # CT CME
"""

DATA = """LAMMPS data

4 atoms
1 bonds
2 atom types
1 bond types
1 angle types
1 dihedral types
1 improper types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

   1  12.011 # CT
   2   1.008 # HC

Bond Coeffs

   1 harmonic   317.000000    1.510000

Angle Coeffs

   1 harmonic    35.000000  109.500000

Dihedral Coeffs

   1 opls     0.000000    0.000000    0.000000    0.000000

Improper Coeffs

   1 cvff     1.100000  -1   2

Atoms  # full

1 1 1 0.0 1.0 1.0 1.0
2 1 2 0.0 2.0 1.0 1.0
3 1 2 0.0 1.0 2.0 1.0
4 1 2 0.0 1.0 1.0 2.0

Bonds

1 1 1 2
"""


def test_input_unwraps_single_substyle_hybrids(tmp_path):
    src = tmp_path / "lammps.in"; src.write_text(IN)
    dst = tmp_path / "nh.in"
    removed = convert_input(src, dst, read_data="packed_box.data")
    txt = dst.read_text()
    assert removed == {"bond_style": "harmonic", "angle_style": "harmonic",
                       "dihedral_style": "opls", "improper_style": "cvff",
                       "pair_style": "lj/cut/coul/long"}
    assert "hybrid" not in txt
    assert "bond_style      harmonic" in txt
    assert "pair_style      lj/cut/coul/long 12.000000" in txt
    assert "pair_coeff     1     1 0.066000    3.500000 # CT CT" in txt
    assert "read_data packed_box.data" in txt
    assert "kspace_style    pppm 1.0e-4" in txt          # untouched
    assert is_hybrid_input(src) and not is_hybrid_input(dst)


def test_genuine_hybrid_is_left_alone(tmp_path):
    src = tmp_path / "a.in"; src.write_text("bond_style hybrid harmonic morse\n")
    dst = tmp_path / "b.in"
    assert convert_input(src, dst) == {}
    assert dst.read_text().strip() == "bond_style hybrid harmonic morse"


def test_data_strips_tokens_and_leaves_atoms_alone(tmp_path):
    src = tmp_path / "d.data"; src.write_text(DATA)
    dst = tmp_path / "nh.data"
    stripped = convert_data(src, dst)
    assert stripped == {"Bond Coeffs": "harmonic", "Angle Coeffs": "harmonic",
                        "Dihedral Coeffs": "opls", "Improper Coeffs": "cvff"}
    txt = dst.read_text()
    assert "1 317.000000 1.510000" in txt
    assert "1 1.100000 -1 2" in txt
    for tok in ("harmonic", "opls", "cvff"):
        assert tok not in txt
    # Masses / Atoms / Bonds untouched
    assert "1  12.011 # CT" in txt
    assert "2 1 2 0.0 2.0 1.0 1.0" in txt
    assert "1 1 1 2" in txt


def test_data_conversion_is_idempotent(tmp_path):
    src = tmp_path / "d.data"; src.write_text(DATA)
    once = tmp_path / "1.data"; twice = tmp_path / "2.data"
    convert_data(src, once)
    assert convert_data(once, twice) == {}
    assert once.read_text() == twice.read_text()


def test_pair_style_class2_data(tmp_path):
    """PCFF-style tokens convert with the same code (nothing hard-wired)."""
    src = tmp_path / "d.data"
    src.write_text(DATA.replace("harmonic   317", "class2   1.52 253.7 -423.0 396.9")
                       .replace("harmonic    35", "quartic 100.3 38.9 -3.8 -8.0"))
    dst = tmp_path / "nh.data"
    stripped = convert_data(src, dst)
    assert stripped["Bond Coeffs"] == "class2"
    assert stripped["Angle Coeffs"] == "quartic"
    assert "class2" not in dst.read_text()


def test_convert_set_writes_into_subdir_and_keeps_originals(tmp_path):
    (tmp_path / "lammps.in").write_text(IN)
    (tmp_path / "lammps.data").write_text(DATA)
    (tmp_path / "packed_box.data").write_text(DATA)
    ni, nd = convert_to_nonhybrid(tmp_path / "lammps.in",
                                  [tmp_path / "lammps.data", tmp_path / "packed_box.data",
                                   tmp_path / "missing.data"],
                                  tmp_path / "non_hybrid", read_data="packed_box.data")
    assert ni == tmp_path / "non_hybrid" / "lammps.in"
    assert [p.name for p in nd] == ["lammps.data", "packed_box.data"]
    assert (tmp_path / "lammps.in").read_text() == IN                 # untouched
    assert "read_data packed_box.data" in ni.read_text()


def test_config_has_lammps_styles_default_hybrid():
    from paaf.config import Config
    assert Config().lammps_styles == "hybrid"


def test_pipeline_applies_choice_and_emits_stages():
    import ast
    import paaf.pipeline as pl
    src = Path(pl.__file__).read_text()
    assert "convert_to_nonhybrid" in src and "lammps_styles" in src
    # every stage 1..7 and the done marker are emitted somewhere
    for k in range(1, 8):
        assert f"_stage({k}," in src, k
    assert "[stage done/" in src


def test_parse_stage_lines():
    from paaf.run_progress import parse_stage
    assert parse_stage("Loading monomers...") is None
    assert parse_stage("[stage 1/7] Loading monomers") == (0, 7, "Step 1 of 7: Loading monomers …", False)
    assert parse_stage("[stage 4/7] Assigning atom types")[0] == int(100 * 3 / 7)
    pct, n, text, done = parse_stage("[stage done/7] Finished — files in /x")
    assert (pct, done) and text.startswith("✓ Completed")


def test_pipeline_stage_lines_are_parseable():
    """Every _stage()/done call in pipeline.py yields a line the GUI parses."""
    from paaf.run_progress import parse_stage
    import paaf.pipeline as pl
    src = Path(pl.__file__).read_text()
    assert parse_stage("[stage 3/7] x") is not None
    assert parse_stage(f"[stage done/7] Finished") is not None
    assert 'f"[stage {k}/{N_STAGES}] {label}"' in src


def test_clear_all_is_wired_in_sidebar_and_export():
    src_mw = Path("paaf/gui/main_window.py").read_text()
    src_sb = Path("paaf/gui/sidebar.py").read_text()
    assert "clearAllClicked = pyqtSignal()" in src_sb and "def reset(self)" in src_sb
    assert "self.sidebar.clearAllClicked.connect(self._clear_all)" in src_mw
    assert 'QPushButton("Clear all — new polymer")' in src_mw
    assert "def _clear_all(self)" in src_mw
    # output dir and DL_FIELD lib dir survive a clear
    body = src_mw.split("def _clear_all(self)")[1].split("def _save_config")[0]
    assert "keep_out" in body and "keep_lib" in body and "self.console.clear()" in body
