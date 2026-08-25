"""The layering planner: sub-boxes, gaps, cells, refusals.

Geometry is the whole feature — a region off by one gap puts an interface
in the wrong place — so every rule is pinned numerically here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from paaf.layering import LayerSpec, LayeringError, plan_regions


def _ly(name, size, origin=None, count=1):
    return LayerSpec(name=name, data_file=Path("x.data"), count=count,
                     size=size, origin=origin)


def test_two_layers_stack_along_z_with_the_gap():
    regions, cell = plan_regions(
        [_ly("bottom", (40, 40, 20)), _ly("top", (40, 40, 30))], gap=5.0)
    print(f"\n  cell {cell}\n  regions {regions}")
    assert cell == (40, 40, 55)                    # 20 + 5 + 30
    assert regions[0] == (0, 0, 0, 40, 40, 20)
    assert regions[1] == (0, 0, 25, 40, 40, 55)    # starts at 20 + 5


def test_auto_layers_anchor_at_the_origin_laterally():
    """A layered film starts at the cell wall (0,0), not centred — the
    user's convention: layer 1 from 0,0, spanning its own Lx x Ly."""
    regions, cell = plan_regions(
        [_ly("wide", (60, 60, 10)), _ly("narrow", (20, 40, 10))], gap=5.0)
    x0, y0, _z0, x1, y1, _z1 = regions[1]
    print(f"\n  narrow layer x {x0}-{x1}, y {y0}-{y1} in cell {cell}")
    assert cell[0] == 60 and cell[1] == 60
    assert (x0, x1) == (0, 20)
    assert (y0, y1) == (0, 40)


def test_three_layers_two_gaps():
    _regions, cell = plan_regions(
        [_ly("a", (30, 30, 10)), _ly("b", (30, 30, 10)),
         _ly("c", (30, 30, 10))], gap=5.0)
    assert cell[2] == 10 + 5 + 10 + 5 + 10


def test_axis_choice_changes_the_stacking_direction():
    regions, cell = plan_regions(
        [_ly("a", (10, 40, 40)), _ly("b", (15, 40, 40))], axis="x", gap=5.0)
    assert cell == (30, 40, 40)
    assert regions[1][0] == 15                     # 10 + 5 along x


def test_an_explicit_origin_is_honoured_verbatim():
    regions, _cell = plan_regions(
        [_ly("base", (50, 50, 20)),
         _ly("pinned", (10, 10, 10), origin=(5, 5, 30))],
        gap=5.0, total_box=(50, 50, 60))
    assert regions[1] == (5, 5, 30, 15, 15, 40)


def test_a_layer_poking_out_of_the_cell_is_refused_with_numbers():
    with pytest.raises(LayeringError, match="spans .* but the cell is only"):
        plan_regions([_ly("big", (40, 40, 80))], total_box=(40, 40, 50))


def test_a_fixed_total_box_leaves_headroom_not_errors():
    regions, cell = plan_regions(
        [_ly("a", (40, 40, 20)), _ly("b", (40, 40, 20))],
        gap=5.0, total_box=(60, 60, 100))
    assert cell == (60, 60, 100)
    assert regions[0][0] == 0                      # anchored at the origin


def test_refusals_speak_user_language():
    with pytest.raises(LayeringError, match="axis must be x, y or z"):
        plan_regions([_ly("a", (10, 10, 10))], axis="q")
    with pytest.raises(LayeringError, match="cannot be negative"):
        plan_regions([_ly("a", (10, 10, 10))], gap=-1)
    with pytest.raises(LayeringError, match="must be positive"):
        plan_regions([_ly("a", (10, 0, 10))])
    with pytest.raises(LayeringError, match="At least one layer"):
        plan_regions([])


def test_regions_reach_the_packmol_input(tmp_path, monkeypatch):
    """The extended blend packer must write one inside-box per component."""
    import paaf.blend_replicator as br
    from paaf.blend_replicator import _build_blend_packmol

    # No packmol here — stand in a no-op executable; the input file the
    # packer writes BEFORE running it is what carries the regions.
    monkeypatch.setattr(br, "_find_packmol", lambda explicit=None: "/bin/true")
    pdbs = []
    for i in range(2):
        p = tmp_path / f"c{i}.pdb"
        p.write_text("HETATM    1  C   MOL A   1       0.000   0.000   "
                     "0.000  1.00  0.00           C\nEND\n")
        pdbs.append(p)
    # No packmol in this environment: the function may return False, but the
    # input file it wrote first is what carries the regions.
    _build_blend_packmol(pdbs, [2, 3], tmp_path / "out.pdb", (40, 40, 55),
                         2.0, 7,
                         regions=[(0, 0, 0, 40, 40, 20),
                                  (0, 0, 25, 40, 40, 55)])
    text = (tmp_path / "pack_blend.inp").read_text()
    print(f"\n{text}")
    assert "inside box 0.0000 0.0000 0.0000 40.0000 40.0000 20.0000" in text
    assert "inside box 0.0000 0.0000 25.0000 40.0000 40.0000 55.0000" in text
