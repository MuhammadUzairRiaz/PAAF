"""Merging GROMACS topologies must refuse anything it cannot merge safely.

Every DL_FIELD single-chain export names its molecule ``XYZ``, so a blend of
two of them collides on the name; and GROMACS keeps the LAST [atomtypes]
definition it reads, so a silent parameter clash would be simulated, not
reported. Both are handled here, and the refusals are as tested as the
merges.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from paaf.gromacs_blend import (                                 # noqa: E402
    BlendTopologyError, GromacsBlendComponent, merge_tops, parse_top,
)


def _component(tmp_path, name, atomtypes, molname="XYZ", defaults=None):
    d = tmp_path / name
    d.mkdir()
    (d / "gromacs1.itp").write_text(
        f"[ moleculetype ]\n{molname}   3\n"
        f"[ atoms ]\n1  CT  1  MOL  C1  1  0.0  12.011\n")
    (d / "gromacs.top").write_text(
        "[ defaults ]\n" + (defaults or "1  1  no  0.5  0.5") + "\n"
        "[ atomtypes ]\n" + "\n".join(atomtypes) + "\n"
        '#include "gromacs1.itp"\n'
        "[ system ]\nsingle chain\n[ molecules ]\nXYZ 1\n")
    (d / "gromacs.gro").write_text("chain\n1\n    1MOL     C1    1"
                                   "   0.000   0.000   0.000\n   5.0 5.0 5.0\n")
    return GromacsBlendComponent(name=name, gro_file=d / "gromacs.gro",
                                 count=5)


CT = "CT  6  12.0115  0.0  A  2.03e-03  3.73e-06"
HC = "HC  1   1.0080  0.0  A  1.22e-04  2.99e-08"
OS = "OS  8  15.9990  0.0  A  1.50e-03  2.10e-06"


def test_two_components_merge_into_one_topology(tmp_path):
    top = merge_tops([_component(tmp_path, "PIB", [CT, HC]),
                      _component(tmp_path, "PBS", [CT, OS])],
                     tmp_path / "out")
    sections = parse_top(top)
    print(f"\n{top.read_text()}")
    assert sections["molecules"] == ["PIB    5", "PBS    5"]
    names = {l.split()[0] for l in sections["atomtypes"]}
    assert names == {"CT", "HC", "OS"}, "shared CT must appear exactly once"
    assert sections["#include"] == ["PIB.itp", "PBS.itp"]


def test_each_moleculetype_is_renamed_after_its_component(tmp_path):
    """Both DL_FIELD exports are called XYZ; the blend must not be."""
    out = tmp_path / "out"
    merge_tops([_component(tmp_path, "PIB", [CT]),
                _component(tmp_path, "PBS", [CT])], out)
    for name in ("PIB", "PBS"):
        text = (out / f"{name}.itp").read_text()
        print(f"\n  {name}.itp: {text.splitlines()[1]!r}")
        assert text.splitlines()[1].split()[0] == name
        assert "XYZ" not in text


def test_conflicting_atomtype_parameters_are_refused(tmp_path):
    """GROMACS would keep one silently; the merge must not let it."""
    clash = "CT  6  12.0115  0.0  A  9.99e-03  9.99e-06"
    with pytest.raises(BlendTopologyError, match="different parameters"):
        merge_tops([_component(tmp_path, "A", [CT]),
                    _component(tmp_path, "B", [clash])], tmp_path / "out")


def test_disagreeing_defaults_are_refused(tmp_path):
    with pytest.raises(BlendTopologyError, match="defaults"):
        merge_tops([_component(tmp_path, "A", [CT]),
                    _component(tmp_path, "B", [CT],
                               defaults="1  2  yes  0.5  0.833")],
                   tmp_path / "out")


def test_a_missing_itp_is_refused_with_the_component_named(tmp_path):
    comp = _component(tmp_path, "A", [CT])
    (comp.gro_file.parent / "gromacs1.itp").unlink()
    with pytest.raises(BlendTopologyError, match="A.*includes"):
        merge_tops([comp], tmp_path / "out")


def test_a_missing_top_is_refused(tmp_path):
    comp = _component(tmp_path, "A", [CT])
    (comp.gro_file.parent / "gromacs.top").unlink()
    with pytest.raises(BlendTopologyError, match="no topology"):
        merge_tops([comp], tmp_path / "out")


def test_an_explicit_top_path_overrides_the_sibling(tmp_path):
    comp = _component(tmp_path, "A", [CT])
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "special.top").write_text(
        "[ defaults ]\n1  1  no  0.5  0.5\n"
        "[ atomtypes ]\n" + OS + "\n"
        '#include "special.itp"\n[ system ]\ns\n[ molecules ]\nXYZ 1\n')
    (other / "special.itp").write_text("[ moleculetype ]\nXYZ 3\n"
                                       "[ atoms ]\n1 OS 1 M O1 1 0.0 16.0\n")
    comp.top_file = other / "special.top"
    top = merge_tops([comp], tmp_path / "out")
    assert "OS" in {l.split()[0] for l in parse_top(top)["atomtypes"]}


def test_counts_reach_the_molecules_section(tmp_path):
    a = _component(tmp_path, "A", [CT]); a.count = 12
    b = _component(tmp_path, "B", [HC]); b.count = 3
    sections = parse_top(merge_tops([a, b], tmp_path / "out"))
    print(f"\n  {sections['molecules']}")
    assert sections["molecules"] == ["A    12", "B    3"]


def test_the_skeleton_top_is_named_for_what_it_is(tmp_path):
    """The project root holds a placeholder .top (type C, charge 0, mass 0).

    Picking it produced "has no #include for its moleculetype", which is
    true and useless. The refusal must say it is the skeleton and point at
    dlf_output1/gromacs.top when that exists.
    """
    d = tmp_path / "PIB"
    d.mkdir()
    (d / "PIB.top").write_text(
        "[ defaults ]\n1 1 no 0.5 0.5\n"
        "[ moleculetype ]\nMOL 3\n"
        "[ atoms ]\n1 C 1 MOL C1 1 0.0 0.0\n"
        "[ system ]\nx\n[ molecules ]\nMOL 1\n")
    (d / "gro.gro").write_text("x\n1\n    1MOL C1 1 0 0 0\n 5 5 5\n")
    dlf = d / "dlf_output1"
    dlf.mkdir()
    (dlf / "gromacs.top").write_text("real one\n")
    comp = GromacsBlendComponent(name="PIB", gro_file=d / "gro.gro",
                                 count=1, top_file=d / "PIB.top")
    with pytest.raises(BlendTopologyError) as e:
        merge_tops([comp], tmp_path / "out")
    msg = str(e.value)
    print(f"\n{msg}")
    assert "skeleton" in msg
    assert "dlf_output1" in msg, "the fix must name the usable file"
