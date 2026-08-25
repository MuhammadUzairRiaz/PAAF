"""The force field the blend is written with.

A data file carries numbers; the input file carries the styles that say what
those numbers mean. PAAF used to write the numbers and not the styles, so the
blend it produced could not be read back — LAMMPS aborts at ``read_data`` with
no bond style defined.

DL_FIELD complicates it by wrapping every style in a one-member ``hybrid`` and
tagging each coefficient line with the sub-style name. That is self-consistent
for one component and falls apart when two are merged, so it is converted
away. These tests pin both halves of the conversion, because dropping the
token from only one of the two files would produce coefficient lines with the
wrong number of columns.
"""
from __future__ import annotations

import pytest

from paaf.blend_styles import (                                # noqa: E402
    StyleBlock, StyleMismatch, dehybridise_coeff_lines,
    dehybridise_pair_coeff, dehybridise_style, find_forcefield_input,
    merge_style_blocks, read_style_block,
)

DLFIELD_IN = """# DL_FIELD generated
units           real       # in kcal/mol
atom_style      full       # molecular system with charges
boundary        p p p      # Normal periodic boundary
bond_style      hybrid harmonic
angle_style     hybrid harmonic
dihedral_style  hybrid opls
improper_style  hybrid cvff
pair_style      hybrid lj/cut/coul/long 12.000000
kspace_style    pppm 1.0e-4
special_bonds   lj 0.0 0.0 0.500000 coul 0.0 0.0 0.500000

read_data lammps1.data

pair_coeff 1 1 lj/cut/coul/long 0.066000 3.500000 # CT CT
thermo 100
run 0
"""


def _tree(tmp_path, name="enr", extra=()):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "lammps1.data").write_text("dummy\n")
    (d / "lammps.in").write_text(DLFIELD_IN)
    for fname in extra:
        (d / fname).write_text("units real\n")
    return d


# =========================================================== reading styles
def test_the_style_block_is_read_from_the_component_input():
    import io
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "lammps.in"
        p.write_text(DLFIELD_IN)
        block = read_style_block(p)
    print("\n  " + "\n  ".join(block.lines()))
    assert block.get("pair_style").startswith("hybrid lj/cut/coul/long")
    assert block.get("kspace_style") == "pppm 1.0e-4"
    assert block.get("special_bonds").startswith("lj 0.0")


def test_run_commands_after_read_data_are_not_styles():
    """thermo/run belong to a run, not to the model."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "lammps.in"
        p.write_text(DLFIELD_IN)
        block = read_style_block(p)
    assert "thermo" not in block.directives
    assert all("run 0" not in v for v in block.directives.values())


# ============================================================ finding the .in
def test_dlfield_lammps_in_is_found(tmp_path):
    d = _tree(tmp_path)
    got = find_forcefield_input(d / "lammps1.data")
    print(f"\n  -> {got}")
    assert got == d / "lammps.in"


def test_paafs_own_outputs_do_not_confuse_the_search(tmp_path):
    """The bug that silently disabled minimisation.

    After one blend run the folder also holds packed_blend.in. A plain "is
    there exactly one .in?" test then finds two, gives up, and the component
    is packed unrelaxed with nobody told.
    """
    d = _tree(tmp_path, extra=("packed_blend.in",))
    got = find_forcefield_input(d / "lammps1.data")
    print(f"\n  with packed_blend.in present -> {got}")
    assert got == d / "lammps.in"


def test_a_dlf_output_subfolder_is_searched(tmp_path):
    d = tmp_path / "pbs"
    (d / "dlf_output1").mkdir(parents=True)
    (d / "lammps1.data").write_text("dummy\n")
    (d / "dlf_output1" / "lammps.in").write_text(DLFIELD_IN)
    assert find_forcefield_input(d / "lammps1.data") == \
        d / "dlf_output1" / "lammps.in"


def test_a_file_with_no_styles_is_not_treated_as_a_force_field(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    (d / "chain.data").write_text("dummy\n")
    (d / "notes.in").write_text("# nothing useful here\n")
    assert find_forcefield_input(d / "chain.data") is None


# =============================================================== de-hybrid
@pytest.mark.parametrize("value,expect,sub", [
    ("hybrid harmonic", "harmonic", "harmonic"),
    ("hybrid opls", "opls", "opls"),
    ("hybrid lj/cut/coul/long 12.0", "lj/cut/coul/long 12.0",
     "lj/cut/coul/long"),
    ("harmonic", "harmonic", ""),                     # already plain
])
def test_a_one_member_hybrid_is_just_the_substyle(value, expect, sub):
    got, got_sub = dehybridise_style(value)
    print(f"\n  {value!r} -> {got!r} (stripped {got_sub!r})")
    assert got == expect and got_sub == sub


def test_a_genuine_hybrid_is_left_alone():
    """Two sub-styles means the wrapper is load-bearing."""
    value = "hybrid harmonic morse 10.0"
    got, sub = dehybridise_style(value)
    print(f"\n  {value!r} -> {got!r}")
    assert got == value and sub == ""


def test_the_comment_survives_the_conversion():
    got, _ = dehybridise_style("hybrid harmonic   # from DL_FIELD")
    print(f"\n  {got!r}")
    assert got.startswith("harmonic") and "DL_FIELD" in got


def test_the_substyle_token_comes_off_coefficient_lines():
    lines = ["1 harmonic 317.000000 1.510000",
             "2 harmonic 340.000000 1.090000"]
    got = dehybridise_coeff_lines(lines, "harmonic")
    print(f"\n  {got}")
    assert got == ["1 317.000000 1.510000", "2 340.000000 1.090000"]


def test_a_coefficient_line_without_a_token_is_untouched():
    lines = ["1 317.000000 1.510000"]
    assert dehybridise_coeff_lines(lines, "harmonic") == lines


def test_a_number_that_looks_like_nothing_is_not_stripped():
    """Only the exact sub-style name, only in second position."""
    lines = ["1 opls 0.0 0.0 0.662 0.0"]
    assert dehybridise_coeff_lines(lines, "harmonic") == lines


def test_pair_coeff_loses_its_substyle_too():
    line = "pair_coeff 1 2 lj/cut/coul/long 0.070824 3.524911 # CT CME"
    got = dehybridise_pair_coeff(line, "lj/cut/coul/long")
    print(f"\n  {got}")
    assert got == "pair_coeff 1 2 0.070824 3.524911 # CT CME"


# ================================================================ merging
def _block(**kw) -> StyleBlock:
    return StyleBlock(directives=dict(kw))


def test_merging_two_dlfield_components_gives_a_plain_style_block(tmp_path):
    d1, d2 = _tree(tmp_path, "enr"), _tree(tmp_path, "pbs")
    blocks = [read_style_block(d1 / "lammps.in"),
              read_style_block(d2 / "lammps.in")]
    merged, subs = merge_style_blocks(blocks, ["ENR", "PBS"])
    print("\n  " + "\n  ".join(merged.lines()))
    assert merged.get("bond_style") == "harmonic"
    assert merged.get("pair_style").startswith("lj/cut/coul/long")
    assert subs["dihedral_style"] == "opls"
    assert "hybrid" not in " ".join(merged.lines())


def test_every_style_lammps_needs_is_present(tmp_path):
    d = _tree(tmp_path)
    merged, _ = merge_style_blocks([read_style_block(d / "lammps.in")], ["A"])
    for directive in ("units", "atom_style", "pair_style", "bond_style",
                      "angle_style", "dihedral_style", "improper_style",
                      "kspace_style", "special_bonds"):
        assert directive in merged.directives, f"{directive} missing"


def test_components_typed_with_different_force_fields_are_refused():
    """Merging these would run and quietly model the wrong material."""
    a = _block(units="real", atom_style="full", bond_style="harmonic")
    b = _block(units="real", atom_style="full", bond_style="class2")
    with pytest.raises(StyleMismatch) as e:
        merge_style_blocks([a, b], ["ENR", "PBS"])
    print(f"\n  {e.value}")
    assert "bond_style" in str(e.value)
    assert "ENR" in str(e.value) and "PBS" in str(e.value)


def test_differing_comments_are_not_a_mismatch():
    a = _block(bond_style="harmonic  # from DL_FIELD")
    b = _block(bond_style="harmonic")
    merged, _ = merge_style_blocks([a, b], ["A", "B"])
    assert merged.get("bond_style").startswith("harmonic")


def test_no_readable_input_anywhere_is_refused_not_guessed():
    with pytest.raises(StyleMismatch) as e:
        merge_style_blocks([StyleBlock(), StyleBlock()], ["A", "B"])
    print(f"\n  {e.value}")
    assert "will not guess" in str(e.value)


# ================================================== the emitted file's order
def test_the_style_block_follows_dl_fields_own_order(tmp_path):
    """A blend input should read like the inputs it was built from.

    LAMMPS does not care about the order among these directives. A user
    diffing the blend against a single-component file does.
    """
    d = _tree(tmp_path)
    merged, _ = merge_style_blocks([read_style_block(d / "lammps.in")], ["A"])
    emitted = [l.split()[0] for l in merged.lines()]
    print("\n  " + " → ".join(emitted))
    reference = [l.split()[0] for l in DLFIELD_IN.splitlines()
                 if l.strip() and not l.startswith("#")
                 and l.split()[0] in emitted]
    assert emitted == reference


def test_dimension_and_timestep_are_not_dropped(tmp_path):
    """Both sat in DL_FIELD's file and were being thrown away."""
    d = tmp_path / "c"
    d.mkdir()
    (d / "lammps.in").write_text(
        "units real\ndimension 3\natom_style full\ntimestep 0.5\n"
        "bond_style hybrid harmonic\nread_data x.data\n")
    block = read_style_block(d / "lammps.in")
    print(f"\n  dimension={block.get('dimension')} "
          f"timestep={block.get('timestep')}")
    assert block.get("dimension") == "3"
    assert block.get("timestep") == "0.5"


def test_the_smaller_timestep_wins():
    """Stability is not negotiable: 2 fs may suit one component and wreck another."""
    a = _block(units="real", timestep="1.0")
    b = _block(units="real", timestep="0.5  # in fs")
    merged, _ = merge_style_blocks([a, b], ["A", "B"])
    print(f"\n  1.0 vs 0.5 -> {merged.get('timestep')}")
    assert merged.get("timestep").startswith("0.5")


def test_a_timestep_difference_is_not_an_error():
    a = _block(units="real", timestep="1.0")
    b = _block(units="real", timestep="0.5")
    merge_style_blocks([a, b], ["A", "B"])       # must not raise


def test_a_component_with_no_impropers_does_not_block_the_blend():
    """PIB says `improper_style none` because an alkane HAS no impropers.

    Blending it with PMMA (`improper_style cvff`) was refused as "typed with
    different force fields" — a wrong diagnosis of a legitimate blend. An
    absence defers to the components that have terms.
    """
    from paaf.blend_styles import StyleBlock, merge_style_blocks

    pib = StyleBlock(directives={"pair_style": "lj/cut/coul/long 12.0",
                                 "improper_style": "none"})
    pmma = StyleBlock(directives={"pair_style": "lj/cut/coul/long 12.0",
                                  "improper_style": "cvff"})
    merged, _subs = merge_style_blocks([pib, pmma], ["PIB", "PMMA"])
    print(f"\n  merged improper_style: {merged.get('improper_style')}")
    assert merged.get("improper_style") == "cvff"


def test_two_real_styles_that_differ_are_still_refused():
    """The refusal must survive the exception above."""
    import pytest as _pytest
    from paaf.blend_styles import StyleBlock, StyleMismatch, merge_style_blocks

    a = StyleBlock(directives={"improper_style": "cvff"})
    b = StyleBlock(directives={"improper_style": "harmonic"})
    with _pytest.raises(StyleMismatch):
        merge_style_blocks([a, b], ["A", "B"])


def test_all_none_stays_none():
    from paaf.blend_styles import StyleBlock, merge_style_blocks

    a = StyleBlock(directives={"improper_style": "none"})
    b = StyleBlock(directives={"improper_style": "none"})
    merged, _ = merge_style_blocks([a, b], ["A", "B"])
    assert merged.get("improper_style") == "none"
