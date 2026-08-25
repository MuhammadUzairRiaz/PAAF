"""The 3D view has to be usable, not merely present.

The first version drew every label at once. On a 41-atom trimer that is a pile
of overlapping white boxes — ``h5:140`` sitting on ``n2:135`` sitting on
``h3:136`` — and picking an atom out of it is harder than reading the table it
was meant to replace. It also had no way to move the structure sideways
(3Dmol puts translate on middle-drag, which nobody finds) and no way to look at
one unit without the other twenty-eight atoms in the way.

So the viewer gained three things, and this pins all three:

* label modes — all / selected+neighbours / none;
* drag-to-pan, on plain left-drag in pan mode or shift-drag at any time;
* a unit filter that hides everything except the unit being typed.

The half that lives in JavaScript is exercised by running it, in
``tests/js/viewer_harness.js``, against a stub renderer. Asserting that the
generated HTML contains the word "translate" would prove nothing about whether
a drag actually moves anything.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from paaf.gui.mol3d_view import (                                # noqa: E402
    LABEL_MODES, _neighbour_serials, build_page, build_payload,
    colors_for_types,
)
from paaf.structure import Atom, Molecule                        # noqa: E402


def _mol(n=4, name="m"):
    return Molecule(
        atoms=[Atom(index=i, element="C", xyz=np.array([float(i), 0.0, 0.0]))
               for i in range(n)],
        bonds=[(i, i + 1, 1.0) for i in range(n - 1)], name=name)


# ============================================================ the payload
def test_the_payload_carries_everything_the_page_needs():
    p = build_payload("BLOCK", labels={0: "h0"}, colors={0: "#fff"},
                      neighbours={0: [1]}, hidden=[2], label_mode="all",
                      pan_mode=True)
    assert p["molblock"] == "BLOCK"
    assert p["labelMode"] == "all"
    assert p["hidden"] == [2]
    assert p["panMode"] is True
    assert p["neighbours"] == {"0": [1]}


@pytest.mark.parametrize("mode", LABEL_MODES)
def test_every_documented_label_mode_is_accepted(mode):
    assert build_payload("B", label_mode=mode)["labelMode"] == mode


def test_an_unknown_label_mode_is_refused_rather_than_silently_ignored():
    """A typo that quietly disables labelling would be very hard to notice."""
    with pytest.raises(ValueError):
        build_payload("B", label_mode="selcted")


def test_the_payload_is_json_serialisable():
    """It crosses into the page as JSON; a stray numpy type would break it."""
    json.dumps(build_payload("B", labels={0: "a"}, hidden=[1, 2]))


def test_hidden_and_context_are_sorted_so_the_page_can_compare_them():
    """apply() decides whether to re-frame by comparing these as strings."""
    p = build_payload("B", hidden=[5, 1, 3])
    assert p["hidden"] == [1, 3, 5]


# ======================================================= neighbour lookup
def test_neighbours_come_from_bonds_not_from_distance():
    n = _neighbour_serials([_mol(4)])
    print(f"\n  {n}")
    assert n[0] == [1] and n[1] == [0, 2] and n[3] == [2]


def test_neighbours_are_offset_across_a_multi_molecule_scene():
    """Two monomers in one scene must not share atom 0's neighbours."""
    n = _neighbour_serials([_mol(3), _mol(3)])
    print(f"\n  {n}")
    assert n[0] == [1], "first molecule"
    assert n[3] == [4], "second molecule, offset by three"
    assert 2 not in n[3], "the two molecules were joined by accident"


def test_a_molecule_with_no_bonds_contributes_nothing():
    lone = Molecule(atoms=[Atom(index=0, element="C", xyz=np.zeros(3))],
                    bonds=[], name="lone")
    assert _neighbour_serials([lone]) == {}


# ============================================================== the page
def test_the_page_embeds_the_state_it_was_given():
    html = build_page("BLOCK", labels={0: "h0"}, hidden=[3],
                      label_mode="none", pan_mode=True,
                      script_src="about:blank")
    assert '"labelMode": "none"' in html
    assert '"labelContent"' in html and '"labelColour"' in html
    assert '"hidden": [3]' in html
    assert '"panMode": true' in html


def test_the_page_is_self_contained_apart_from_the_renderer():
    html = build_page("BLOCK", script_src="about:blank")
    assert html.count("<script") == 3          # renderer, webchannel, ours
    assert "about:blank" in html


def test_colours_are_stable_as_more_atoms_are_typed():
    """The picture must not reshuffle every time one atom is assigned."""
    first = colors_for_types({0: "135", 1: "136"})
    later = colors_for_types({0: "135", 1: "136", 2: "140", 3: "135"})
    print(f"\n  {first} -> {later}")
    assert later[0] == first[0] and later[1] == first[1]


# ===================================================== the page's own logic
NODE = shutil.which("node")
HARNESS = Path(__file__).parent / "js" / "viewer_harness.js"


@pytest.mark.skipif(NODE is None, reason="needs node to run the page's JS")
def test_the_viewer_javascript_behaves(tmp_path):
    """Run the real page script against a stub renderer.

    Checks, in the code that actually ships: the four label modes, the three
    label contents, the label colour, that labels have no background panel,
    that a filtered-out atom cannot be clicked, that a drag pans in pan mode
    and on shift but not otherwise, that Fit frames only what is visible, that
    clicks survive a model swap, and that the selection highlight stays a
    translucent marker rather than something mistakable for an atom.
    """
    html = build_page("MOLBLOCK", labels={0: "h0", 1: "0", 2: "t4"},
                      names={0: "C0", 1: "C1", 2: "O2"},
                      colors={0: "#e6194b"}, neighbours={0: [1, 2], 1: [0]},
                      hidden=[2], label_mode="clicked",
                      label_content="type", label_colour="#000000",
                      script_src="about:blank")
    body = html.split("<script>\nvar DATA")[1].split("new QWebChannel")[0]
    script = tmp_path / "viewer.js"
    script.write_text("var DATA" + body)

    result = subprocess.run([NODE, str(HARNESS), str(script)],
                            capture_output=True, text=True, timeout=60)
    print("\n" + result.stdout + result.stderr)
    assert result.returncode == 0, "a viewer behaviour check failed"
    assert "FAIL" not in result.stdout
    assert result.stdout.count("PASS") >= 33
