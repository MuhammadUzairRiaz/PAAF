"""Entering a monomer by hand must not produce a tangled, dangling structure.

Two independent faults did, and both were invisible.

1. Head/tail detection scored pairs of heavy atoms by how nearly opposite their
   C-H vectors pointed. That describes one conformer, not the molecule. For PBS
   typed in by hand as ``OCCCCOC(=O)CCC(=O)O`` it chose atoms 8 and 9 — two
   ADJACENT CH2 carbons in the middle of the succinate — because their
   hydrogens happened to splay apart. Linking there polymerises through the
   middle of the monomer. Nothing objected: two carbons bearing hydrogens is a
   perfectly good answer to the question that was being asked.

2. The trimer shown for typing placed each unit by translating it along the
   previous unit's head-to-tail direction, with no rotation. On a five-atom
   monomer that looks fine. On a folded twelve-heavy-atom one the units end up
   inside each other — 1.75 A between atoms of different units for PBS, 1.66 A
   for polystyrene — which is the tangle that appeared on screen.

There is also a case that no amount of cleverness can fix: PBS written in its
condensation form has an -OH at BOTH ends, so the only atoms with a removable
hydrogen are two oxygens, and PAAF links by deleting one H from each end. That
gives an O-O peroxide backbone. The right response is to say so, not to build
it quietly.
"""
from __future__ import annotations

import numpy as np
import pytest

from paaf.monomer import Monomer, _guess_terminal_h                # noqa: E402
from paaf.structure import Atom, Molecule                          # noqa: E402
from paaf.typing_context import build_trimer                       # noqa: E402

rdkit = pytest.importorskip("rdkit", reason="needs RDKit")
from rdkit import Chem                                             # noqa: E402
from rdkit.Chem import AllChem                                     # noqa: E402


def _mol(smiles: str, seed: int = 1) -> Molecule:
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    try:
        AllChem.UFFOptimizeMolecule(m)
    except Exception:
        pass
    conf = m.GetConformer()
    return Molecule(
        atoms=[Atom(index=i, element=a.GetSymbol(),
                    xyz=np.array([conf.GetAtomPosition(i).x,
                                  conf.GetAtomPosition(i).y,
                                  conf.GetAtomPosition(i).z]))
               for i, a in enumerate(m.GetAtoms())],
        bonds=[(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                b.GetBondTypeAsDouble()) for b in m.GetBonds()],
        name=smiles)


def _monomer(smiles: str, head: int, tail: int) -> Monomer:
    mol = _mol(smiles)
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))

    def cap(k):
        return [n.GetIdx() for n in m.GetAtomWithIdx(k).GetNeighbors()
                if n.GetSymbol() == "H"][0]

    return Monomer(mol, head, tail, [cap(head)], [cap(tail)], name=smiles)


def _bonds_between(mol: Molecule, a: int, b: int) -> int:
    """Shortest path in bonds, for asserting two atoms are at opposite ends."""
    from collections import deque
    seen = {a: 0}
    queue = deque([a])
    while queue:
        cur = queue.popleft()
        for nb in mol.neighbors(cur):
            if nb not in seen:
                seen[nb] = seen[cur] + 1
                queue.append(nb)
    return seen.get(b, -1)


# ================================================ head/tail are the ENDS
def test_pbs_typed_by_hand_does_not_link_through_its_middle():
    """The exact failure: atoms 8 and 9, two adjacent CH2 in the succinate."""
    mol = _mol("OCCCCOC(=O)CCC(=O)O")
    head, tail, _hr, _tr = _guess_terminal_h(mol)
    separation = _bonds_between(mol, head, tail)
    print(f"\n  head {mol.atoms[head].element}{head}, "
          f"tail {mol.atoms[tail].element}{tail}, {separation} bonds apart")
    assert separation >= 10, "the link atoms are not at opposite ends"
    assert abs(head - tail) > 1, "adjacent atoms were chosen again"


@pytest.mark.parametrize("smiles,expected_bonds", [
    ("OCCCCOC(=O)CCC(=O)", 9),      # PBS, polymerisable form
    ("OCCCCOC(=O)CCC(=O)O", 10),    # PBS, condensation form
    ("CCCC", 3),                    # polyethylene
    ("CC(C)CC", 3),                 # polypropylene
    ("COCCOC", 5),                  # PEG
    ("OCCO", 3),                    # ethylene glycol
])
def test_the_two_link_atoms_are_as_far_apart_as_the_molecule_allows(
        smiles, expected_bonds):
    """Chosen from the bond graph, so a folded conformer cannot mislead it."""
    mol = _mol(smiles)
    head, tail, _hr, _tr = _guess_terminal_h(mol)
    got = _bonds_between(mol, head, tail)
    print(f"\n  {smiles:22s} {mol.atoms[head].element}{head} .. "
          f"{mol.atoms[tail].element}{tail} = {got} bonds")
    assert got == expected_bonds


def test_the_choice_does_not_depend_on_the_conformer():
    """The old scoring did; that is what made it wrong."""
    picks = set()
    for seed in (1, 2, 3, 7, 11):
        mol = _mol("OCCCCOC(=O)CCC(=O)", seed=seed)
        head, tail, _hr, _tr = _guess_terminal_h(mol)
        picks.add((head, tail))
    print(f"\n  across 5 embeddings: {picks}")
    assert len(picks) == 1, f"the answer moved with the conformer: {picks}"


def test_each_link_atom_still_has_a_hydrogen_to_remove():
    """Linking deletes one H per end; an end without one cannot link."""
    for smiles in ("OCCCCOC(=O)CCC(=O)", "CCCC", "COCCOC"):
        mol = _mol(smiles)
        head, tail, head_h, tail_h = _guess_terminal_h(mol)
        assert mol.atoms[head_h[0]].element == "H"
        assert mol.atoms[tail_h[0]].element == "H"
        assert head_h[0] in mol.neighbors(head)
        assert tail_h[0] in mol.neighbors(tail)


# ============================================ the peroxide case is announced
def test_the_condensation_form_of_pbs_is_reported_not_built_silently(caplog):
    """Two -OH ends means an O-O backbone, which is never what was meant."""
    import logging
    mol = _mol("OCCCCOC(=O)CCC(=O)O")
    with caplog.at_level(logging.WARNING):
        head, tail, _hr, _tr = _guess_terminal_h(mol)
    text = caplog.text
    print(f"\n  head/tail = {mol.atoms[head].element}{head}/"
          f"{mol.atoms[tail].element}{tail}")
    assert mol.atoms[head].element == "O" and mol.atoms[tail].element == "O"
    assert "peroxide" in text.lower(), "the O-O link was not flagged"
    assert "[*]" in text, "the warning does not say what to do instead"


def test_a_normal_monomer_produces_no_such_warning(caplog):
    """The warning must mean something, so it must not fire on everything."""
    import logging
    with caplog.at_level(logging.WARNING):
        _guess_terminal_h(_mol("CCCC"))
    assert "peroxide" not in caplog.text.lower()


# ==================================== the trimer is a chain, not a tangle
POLYMERS = [
    ("PBS",          "OCCCCOC(=O)CCC(=O)", 0, 10),
    ("isoprene",     "CC=C(C)C",           0, 4),
    ("polyethylene", "CCCC",               0, 3),
    ("polystyrene",  "CC(c1ccccc1)C",      0, 4),
    ("PMMA",         "CC(C)(C(=O)OC)C",    0, 7),
    ("PEG",          "COCCOC",             0, 5),
]


def _contacts(ctx):
    """Closest approach between atoms of different units, ignoring the links.

    End-cap atoms are excluded: they are bonded to the tail unit, so their
    1.34 A bond is not a clash.
    """
    from paaf.typing_context import CAP_UNIT
    links = {frozenset(b) for b in ctx.link_bonds}
    cap = {i for i, (u, _k) in ctx.provenance.items() if u == CAP_UNIT}
    any_gap, heavy_gap = 1e9, 1e9
    for i, (ui, _a) in ctx.provenance.items():
        for j, (uj, _b) in ctx.provenance.items():
            if ui >= uj or frozenset((i, j)) in links:
                continue
            if i in cap or j in cap:
                continue
            d = float(np.linalg.norm(ctx.molecule.atoms[i].xyz
                                     - ctx.molecule.atoms[j].xyz))
            any_gap = min(any_gap, d)
            if (ctx.molecule.atoms[i].element != "H"
                    and ctx.molecule.atoms[j].element != "H"):
                heavy_gap = min(heavy_gap, d)
    return any_gap, heavy_gap


@pytest.mark.parametrize("name,smiles,head,tail", POLYMERS,
                         ids=[p[0] for p in POLYMERS])
def test_units_do_not_sit_inside_one_another(name, smiles, head, tail):
    """PBS was 1.75 A and polystyrene 1.66 A — atoms drawn inside atoms."""
    ctx = build_trimer([_monomer(smiles, head, tail)])
    any_gap, heavy_gap = _contacts(ctx)
    print(f"\n  {name:14s} closest any {any_gap:.2f} A, heavy {heavy_gap:.2f} A")
    assert heavy_gap >= 2.15, "two heavy atoms are drawn on top of each other"
    assert any_gap >= 1.2


@pytest.mark.parametrize("name,smiles,head,tail", POLYMERS,
                         ids=[p[0] for p in POLYMERS])
def test_the_junction_bonds_stay_bond_length(name, smiles, head, tail):
    """Clearance is bought with a nudge; it must stay a bond, not a gap."""
    ctx = build_trimer([_monomer(smiles, head, tail)])
    lengths = [float(np.linalg.norm(ctx.molecule.atoms[i].xyz
                                    - ctx.molecule.atoms[j].xyz))
               for i, j in ctx.link_bonds]
    print(f"\n  {name:14s} junctions {['%.2f' % v for v in lengths]}")
    assert lengths, "the units were not joined at all"
    assert max(lengths) <= 2.4, "a junction is stretched into a visible gap"
    assert min(lengths) >= 1.4


@pytest.mark.parametrize("name,smiles,head,tail", POLYMERS,
                         ids=[p[0] for p in POLYMERS])
def test_the_units_are_laid_out_in_order_along_the_chain(name, smiles, head, tail):
    """Head, then repeat, then tail — not three molecules in a heap."""
    ctx = build_trimer([_monomer(smiles, head, tail)])
    centres = {}
    for i, (unit, _k) in ctx.provenance.items():
        centres.setdefault(unit, []).append(ctx.molecule.atoms[i].xyz)
    x = {u: float(np.mean([p[0] for p in pts])) for u, pts in centres.items()}
    print(f"\n  {name:14s} unit centroids along the axis: "
          f"{[round(x[u], 1) for u in (0, 1, 2)]}")
    assert x[0] < x[1] < x[2], "the units are not in chain order"


def test_the_trimer_topology_is_unchanged_by_the_new_placement():
    """Geometry moved; the bonds and caps that typing depends on must not."""
    mono = _monomer("CC=C(C)C", 0, 4)
    ctx = build_trimer([mono])
    per_unit = len(mono.molecule.atoms)          # 15: the CAPPED repeat unit
    print(f"\n  {per_unit} atoms per unit -> {len(ctx.molecule.atoms)} in the "
          f"trimer, middle unit {len(ctx.middle)}")
    # Four caps are consumed: one from each side of each of the two junctions.
    assert len(ctx.molecule.atoms) == 3 * per_unit - 4
    assert len(ctx.link_bonds) == 2
    assert len(ctx.provenance) == len(ctx.molecule.atoms)
    assert len(ctx.middle) == per_unit - 2       # the middle loses both caps


def test_placement_is_deterministic():
    """Two runs must give the same picture, or nothing can be trusted."""
    a = build_trimer([_monomer("CC(c1ccccc1)C", 0, 4)])
    b = build_trimer([_monomer("CC(c1ccccc1)C", 0, 4)])
    for i in range(len(a.molecule.atoms)):
        assert np.allclose(a.molecule.atoms[i].xyz, b.molecule.atoms[i].xyz)
