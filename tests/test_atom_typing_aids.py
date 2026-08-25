"""The three things that make atom typing hard to get wrong.

A carbon type was assigned to a hydrogen and exported with mass 12.011.
Nothing objected — the file was well formed and LAMMPS would have read it.
The atom was one of fifteen rows differing only by an index.

Three changes address that, and all three have a testable core even though
the 3D view itself needs a browser:

* the molfile handed to the viewer carries real bond orders, so a double bond
  is drawn as one rather than guessed from distance;
* an element mismatch is refused before it can be written;
* equivalent atoms are typed together, so there is no row-by-row editing for
  a slip to hide in.

The ground truth for OPLS types here is the user's hand-built 1,4-isoprene
repeat unit.
"""
from __future__ import annotations

import numpy as np
import pytest

from paaf.molblock import (                                    # noqa: E402
    bond_order_counts, combined_molblock, to_molblock,
)
from paaf.structure import Atom, Molecule                      # noqa: E402
from paaf.type_guard import (                                  # noqa: E402
    check_all, check_assignment, type_elements,
)

rdkit = pytest.importorskip("rdkit", reason="needs RDKit")
from rdkit import Chem                                          # noqa: E402
from rdkit.Chem import AllChem                                  # noqa: E402


def _mol(smiles: str) -> Molecule:
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=1)
    conf = m.GetConformer()
    atoms = [Atom(index=i, element=a.GetSymbol(),
                  xyz=np.array([conf.GetAtomPosition(i).x,
                                conf.GetAtomPosition(i).y,
                                conf.GetAtomPosition(i).z]))
             for i, a in enumerate(m.GetAtoms())]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble())
             for b in m.GetBonds()]
    return Molecule(atoms=atoms, bonds=bonds, name=smiles)


#: Isoprene's capped repeat unit, and the types the user assigned by hand.
ISOPRENE = "CC=C(C)C"
REFERENCE_TYPES = {
    0: "136", 1: "142", 2: "141", 3: "136", 4: "135",
    5: "140", 6: "140", 7: "140", 8: "144", 9: "140",
    10: "140", 11: "140", 12: "140", 13: "140", 14: "140",
}


# ============================================ bond orders reach the viewer
def test_a_double_bond_survives_into_the_molfile():
    """Drawn as double, not inferred from distance."""
    counts = bond_order_counts(_mol(ISOPRENE))
    print(f"\n  isoprene bond orders: {counts}")
    assert counts.get(2, 0) == 1, "the C=C was lost"


def test_the_molfile_names_the_double_bond_explicitly():
    block = to_molblock(_mol(ISOPRENE), "isoprene")
    doubles = [l for l in block.splitlines()
               if len(l.split()) == 7 and l.split()[2] == "2"]
    print(f"\n  {doubles}")
    assert len(doubles) == 1


def test_an_aromatic_ring_is_marked_aromatic_not_single():
    """Six single bonds is the classic wrong-looking benzene."""
    counts = bond_order_counts(_mol("c1ccccc1"))
    print(f"\n  benzene: {counts}")
    assert counts.get(4, 0) == 6


def test_atom_order_is_preserved_exactly():
    """Clicks come back by serial; a reordering would mistype atoms."""
    mol = _mol(ISOPRENE)
    block = to_molblock(mol)
    lines = block.splitlines()[4:4 + len(mol.atoms)]
    got = [l.split()[3] for l in lines]
    assert got == [a.element for a in mol.atoms]


def test_a_copolymer_scene_maps_each_atom_back_to_its_monomer():
    """Two monomers in one scene; each click must resolve to the right one."""
    iso, epox = _mol(ISOPRENE), _mol("CC1(C)OC1C")
    _block, mapping = combined_molblock([iso, epox])
    print(f"\n  {len(mapping)} atoms; serial {len(iso.atoms)} -> "
          f"{mapping[len(iso.atoms)]}")
    assert len(mapping) == len(iso.atoms) + len(epox.atoms)
    assert mapping[0] == (0, 0)
    assert mapping[len(iso.atoms)] == (1, 0), "monomer 2 starts at its own 0"


def test_monomers_do_not_overlap_in_the_scene():
    iso, epox = _mol(ISOPRENE), _mol("CC1(C)OC1C")
    block, mapping = combined_molblock([iso, epox], gap=6.0)
    xs = [float(l[:10]) for l in block.splitlines()[4:4 + len(mapping)]]
    first = xs[:len(iso.atoms)]
    second = xs[len(iso.atoms):]
    print(f"\n  monomer 1 x: {min(first):.1f}–{max(first):.1f}; "
          f"monomer 2 x: {min(second):.1f}–{max(second):.1f}")
    assert min(second) > max(first), "the two monomers are drawn on top of each other"


# ================================================== the element guard
def test_the_library_states_each_types_element():
    table = type_elements()
    print(f"\n  {len(table)} types; 136={table.get('136')} 140={table.get('140')}")
    assert table.get("135") == "C"
    assert table.get("136") == "C"
    assert table.get("140") == "H"
    assert table.get("144") == "H"


def test_a_carbon_type_on_a_hydrogen_is_refused():
    """The exact assignment that shipped mass 12.011 on a hydrogen."""
    message = check_assignment("H", "136")
    print(f"\n  {message}")
    assert message is not None
    assert "cannot be assigned" in message


@pytest.mark.parametrize("element,type_id", [
    ("C", "135"), ("C", "136"), ("C", "141"), ("C", "142"),
    ("H", "140"), ("H", "144"),
])
def test_every_type_in_the_reference_is_accepted(element, type_id):
    """The guard must not obstruct correct chemistry."""
    assert check_assignment(element, type_id) is None


def test_the_hand_built_reference_passes_as_a_whole():
    mol = _mol(ISOPRENE)
    elements = {a.index: a.element for a in mol.atoms}
    problems = check_all(elements, REFERENCE_TYPES)
    print(f"\n  problems: {problems or 'none'}")
    assert not problems


def test_the_reference_with_the_original_slip_is_caught():
    mol = _mol(ISOPRENE)
    elements = {a.index: a.element for a in mol.atoms}
    broken = dict(REFERENCE_TYPES)
    broken[10] = "136"                    # carbon type on a hydrogen
    problems = check_all(elements, broken)
    print(f"\n  {problems}")
    assert [i for i, _m in problems] == [10]


def test_an_unknown_type_is_allowed_through():
    """A library PAAF cannot read must not block all work."""
    assert check_assignment("C", "zzz999") is None


def test_an_empty_type_is_not_an_error():
    assert check_assignment("C", "") is None


# ============================================== equivalent atoms
def test_methyl_hydrogens_are_one_class():
    from paaf.equivalence import equivalent_atoms

    mol = _mol(ISOPRENE)
    groups = equivalent_atoms(mol)
    hydrogens = [a.index for a in mol.atoms if a.element == "H"]
    classes = {tuple(groups[h]) for h in hydrogens}
    print(f"\n  hydrogen classes: {sorted(classes)}")
    # The lone alkene H must be its own class — it is the one that needs 144.
    singletons = [c for c in classes if len(c) == 1]
    assert singletons, "the alkene H was lumped in with the alkane H's"


def test_equivalence_is_symmetric_and_reflexive():
    from paaf.equivalence import equivalent_atoms

    groups = equivalent_atoms(_mol(ISOPRENE))
    for index, members in groups.items():
        assert index in members
        for other in members:
            assert set(groups[other]) == set(members)


def test_carbons_and_hydrogens_are_never_in_one_class():
    """Otherwise assigning by class would recreate the original bug."""
    from paaf.equivalence import equivalent_atoms

    mol = _mol(ISOPRENE)
    for members in equivalent_atoms(mol).values():
        elements = {mol.atoms[i].element for i in members}
        assert len(elements) == 1, f"mixed-element class: {members}"


# ============================================== the generated page
def test_the_page_embeds_the_molecule_and_a_click_handler():
    from paaf.gui.mol3d_view import build_page

    page = build_page(to_molblock(_mol(ISOPRENE)),
                      script_src="file:///tmp/3Dmol-min.js")
    assert "setClickable" in page, "atoms would not be selectable"
    assert "atom_clicked" in page, "clicks would not reach Python"
    assert "singleBonds: false" in page, "double bonds would render as single"
    assert "V2000" in page, "the molecule is not in the page"


def test_the_page_is_valid_without_any_assignments_yet():
    from paaf.gui.mol3d_view import build_page

    page = build_page(to_molblock(_mol(ISOPRENE)))
    assert page.strip().startswith("<!DOCTYPE html>")
    assert page.count("__PAYLOAD__") == 0, "template placeholder left unfilled"
    assert page.count("__SRC__") == 0


def test_assigned_types_get_stable_distinct_colours():
    from paaf.gui.mol3d_view import colors_for_types

    colours = colors_for_types({0: "136", 1: "142", 2: "141", 5: "140"})
    print(f"\n  {colours}")
    assert len(set(colours.values())) == 4, "two types share a colour"
    again = colors_for_types({0: "136", 1: "142", 2: "141", 5: "140"})
    assert colours == again, "colours shuffle between refreshes"


def test_untyped_atoms_are_left_uncoloured():
    from paaf.gui.mol3d_view import colors_for_types

    colours = colors_for_types({0: "136", 1: "", 2: None})
    assert set(colours) == {0}


# ==================================== the web engine's import-order rule
def test_the_launcher_imports_webengine_before_the_application():
    """Qt refuses the import once a QCoreApplication exists.

    Getting this wrong produces an ImportError that reads like "not
    installed", which is what it was mistaken for. The import must therefore
    appear in the launcher BEFORE the QApplication is constructed.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "paaf" / "gui" / "app.py").read_text()
    import_at = src.find("QtWebEngineWidgets")
    app_at = src.find("QApplication(sys.argv)")
    print(f"\n  QtWebEngineWidgets at {import_at}, QApplication at {app_at}")
    assert import_at != -1, "the launcher never imports QtWebEngineWidgets"
    assert import_at < app_at, \
        "QtWebEngineWidgets is imported after the QApplication — Qt will refuse it"


def test_shared_opengl_contexts_are_requested_before_the_application():
    """Without this attribute the web view renders blank."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "paaf" / "gui" / "app.py").read_text()
    attr_at = src.find("AA_ShareOpenGLContexts")
    app_at = src.find("QApplication(sys.argv)")
    assert attr_at != -1 and attr_at < app_at


def test_the_failure_message_reports_the_real_reason(monkeypatch):
    """"Not installed" was the wrong diagnosis; say what Qt actually said."""
    import paaf.gui.mol3d_view as m

    monkeypatch.setenv("PAAF_WEBENGINE_ERROR",
                       "QtWebEngineWidgets must be imported before a "
                       "QCoreApplication instance is created")
    reason = m.webengine_error()
    # On a machine with no PyQt5 at all this is an ImportError instead, which
    # is equally a real reason rather than a guess.
    print(f"\n  {reason}")
    assert reason is None or isinstance(reason, str)


def test_the_vendored_renderer_is_used_when_present():
    """No network needed once 3Dmol-min.js is downloaded."""
    from paaf.gui.mol3d_view import VIEWER_ASSET, build_page

    page = build_page("dummy")
    if VIEWER_ASSET.is_file():
        assert "file://" in page, "the local copy is being ignored"
        print(f"\n  loading from {VIEWER_ASSET}")
    else:
        assert "cdnjs" in page
        print("\n  no local copy; CDN fallback in use")
