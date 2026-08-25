"""The typing dialog itself, driven offscreen, across a change of monomer.

The pure rules this depends on live in
``test_stale_types_across_monomers.py``; these check that the dialog actually
applies them, because the bug was not in the rule — there was no rule — it was
that ``_types_by_atom`` outlived the molecule it described.
"""
from __future__ import annotations

import pytest

from paaf.type_guard import check_all                            # noqa: E402

#: Polybutadiene, the molecule in the report. Twelve atoms, 0-11.
BUTADIENE = ("CC=CC", 0, 3)

#: What PBS left behind: indices valid in a 26-atom monomer, not in a 12-atom
#: one. 140 is a hydrogen type, so these would each read "? atom".
STALE = {13: "140", 14: "140", 15: "140", 16: "140", 17: "140", 18: "140"}

# ===================================================== the dialog's pruning
pytest.importorskip("rdkit", reason="needs RDKit")
PyQt5 = pytest.importorskip("PyQt5", reason="needs PyQt5")

from PyQt5.QtWidgets import QApplication                          # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _xyz(tmp_path, name, smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=1)
    AllChem.UFFOptimizeMolecule(m)
    conf = m.GetConformer()
    lines = [str(m.GetNumAtoms()), name]
    for i, a in enumerate(m.GetAtoms()):
        p = conf.GetAtomPosition(i)
        lines.append(f"{a.GetSymbol()} {p.x:.4f} {p.y:.4f} {p.z:.4f}")
    path = tmp_path / f"{name}.xyz"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _spec(path, smiles, head, tail):
    from rdkit import Chem
    from paaf.config import MonomerSpec

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))

    def cap(k):
        return [n.GetIdx() for n in m.GetAtomWithIdx(k).GetNeighbors()
                if n.GetSymbol() == "H"][0]

    return MonomerSpec(file=path, name="m", head=head, tail=tail,
                       head_h=cap(head), tail_h=cap(tail))


def _dialog(path, smiles, head, tail, initial_types):
    from paaf.ff_registry import get_ff
    from paaf.gui.atom_type_dialog import AtomTypingDialog

    ff = get_ff("oplsaa")
    lt = ff.bundled_path() if ff.bundled_lt else None
    if lt is None or not lt.exists():
        pytest.skip("the OPLS-AA .lt library is not bundled here")
    return AtomTypingDialog(monomer_files=[path], ff_lt_path=lt,
                            ff_key=ff.key, ff_inherit=ff.inherit,
                            monomer_specs=[_spec(path, smiles, head, tail)],
                            initial_types=initial_types)


def test_assignments_from_a_previous_monomer_are_dropped(app, tmp_path):
    path = _xyz(tmp_path, "butadiene", BUTADIENE[0])
    dlg = _dialog(path, *BUTADIENE, initial_types=dict(STALE))
    left = set(dlg.overrides()) & set(STALE)
    print(f"\n  {len(STALE)} stale keys in, {len(left)} still there")
    assert not left, "PBS's assignments survived into a butadiene dialog"


def test_the_user_is_told_rather_than_left_to_wonder(app, tmp_path):
    path = _xyz(tmp_path, "butadiene", BUTADIENE[0])
    dlg = _dialog(path, *BUTADIENE, initial_types=dict(STALE))
    notes = " ".join(getattr(dlg, "_notes", []))
    print(f"\n  note: {notes!r}")
    assert "cleared" in notes.lower(), \
        "assignments were discarded without saying so"


def test_the_monomer_s_own_assignments_are_untouched(app, tmp_path):
    """Pruning must take the stale keys and nothing else."""
    path = _xyz(tmp_path, "butadiene", BUTADIENE[0])
    mine = {0: "143", 1: "142"}
    dlg = _dialog(path, *BUTADIENE, initial_types={**STALE, **mine})
    kept = dlg.overrides()
    print(f"\n  kept {len(kept)} of {len(STALE) + len(mine)}")
    for key, value in mine.items():
        assert kept.get(key) == value, f"atom {key} lost its type"


def test_every_row_can_be_applied_after_the_switch(app, tmp_path):
    """The end state the user was denied: nothing blocking Apply."""
    path = _xyz(tmp_path, "butadiene", BUTADIENE[0])
    dlg = _dialog(path, *BUTADIENE, initial_types=dict(STALE))
    elements = {row[0]: row[3] for row in dlg._loaded_atoms}
    problems = check_all(elements, dlg.overrides())
    print(f"\n  {len(elements)} atoms on the table, {len(problems)} problems")
    assert not problems, f"Apply would still refuse: {problems[:3]}"


def test_a_genuine_element_clash_still_blocks_apply(app, tmp_path):
    """Clearing the noise must not clear the signal."""
    path = _xyz(tmp_path, "butadiene", BUTADIENE[0])
    dlg = _dialog(path, *BUTADIENE, initial_types={})
    elements = {row[0]: row[3] for row in dlg._loaded_atoms}
    carbon = next(k for k, e in elements.items() if e == "C")
    types = dict(dlg.overrides())
    types[carbon] = "140"                       # a hydrogen type on a carbon
    problems = check_all(elements, types)
    print(f"\n  {problems[:1]}")
    assert any(i == carbon for i, _m in problems), \
        "a hydrogen type on a carbon was allowed through"


# ============================== and the molecule itself is left alone
def test_a_hydrocarbon_is_not_given_a_carboxyl_end_cap(app, tmp_path):
    """Butadiene has no acid end, so there is nothing to cap.

    ``cap_carboxyl_end`` exists for polyesters: PAAF presents an acid end as
    an aldehyde so there is a hydrogen to remove when linking, and restores
    the -OH afterwards. Applied to a hydrocarbon it would invent an oxygen
    that is not in the user's SMILES.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from paaf.monomer import Monomer
    from paaf.structure import Atom, Molecule
    from paaf.typing_context import build_trimer

    smiles, head, tail = BUTADIENE
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=3)
    AllChem.UFFOptimizeMolecule(m)
    conf = m.GetConformer()
    mol = Molecule(
        atoms=[Atom(index=i, element=a.GetSymbol(),
                    xyz=np.array([conf.GetAtomPosition(i).x,
                                  conf.GetAtomPosition(i).y,
                                  conf.GetAtomPosition(i).z]))
               for i, a in enumerate(m.GetAtoms())],
        bonds=[(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                b.GetBondTypeAsDouble()) for b in m.GetBonds()],
        name="polybutadiene")

    def cap(k):
        return [n.GetIdx() for n in m.GetAtomWithIdx(k).GetNeighbors()
                if n.GetSymbol() == "H"][0]

    ctx = build_trimer([Monomer(mol, head, tail, [cap(head)], [cap(tail)],
                                name="polybutadiene")])
    elements = "".join(a.element for a in ctx.molecule.atoms)
    print(f"\n  trimer: {len(ctx.molecule.atoms)} atoms, cap {ctx.terminal_cap}")
    assert ctx.terminal_cap == [], "a carboxyl cap was added to a hydrocarbon"
    assert "O" not in elements, "an oxygen appeared in a C/H-only polymer"
