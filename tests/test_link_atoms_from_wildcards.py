"""The chain must join where the polymerisation SMILES says it joins.

The failure
-----------
Poly(1-butene) came up in the typing dialog with a repeat unit of four
methylenes and no branch — a linear C4 backbone. That is polyethylene with a
long repeat, not poly(1-butene), and every atom type assigned against it was
assigned against a molecule that does not exist.

The library was not at fault. Its entry is ``[*]CC(CC)[*]``, which is right:
``-CH2-CH(CH2CH3)-``. What was at fault was how the two link atoms were found.
``_pick_head_tail_by_element`` reads only the ELEMENT beside each wildcard and
then looks for TERMINAL atoms of that element. Capped, ``[*]CC(CC)[*]`` is
n-butane, and its link carbons are the two INTERIOR ones — never candidates.
Two terminal carbons of the right element were available, so a plausible wrong
answer was always found and nothing ever complained.

For a polyester the rule happens to hold: the links really are at the ends. So
this went unnoticed for as long as the polymers being tested were PBS and
polyethylene.

Measured across the shipped 122-polymer database, **72 of the 103 that can be
embedded were linked at the wrong atoms** — 38 where the old rule gave a
different answer and 34 where it gave none. Every vinyl polymer in the
library: polypropylene, PMMA, poly(vinyl acetate), the acrylates, the vinyl
ethers.

The fix reads the link atoms from where the wildcards ARE, carrying that
position from the SMILES's numbering into the 3D file's by matching the two as
graphs.
"""
from __future__ import annotations

import numpy as np
import pytest

from paaf.builder import _closed_shell                          # noqa: E402
from paaf.monomer import (                                      # noqa: E402
    _capped_from_wildcards, _links_from_wildcards,
    _pick_head_tail_by_element, _wildcard_neighbour_symbols,
)
from paaf.structure import Atom, Molecule                       # noqa: E402

rdkit = pytest.importorskip("rdkit", reason="needs RDKit")
from rdkit import Chem                                          # noqa: E402
from rdkit.Chem import AllChem                                  # noqa: E402


def _mol(smiles, name="m", seed=1):
    """A 3D molecule the way PAAF builds one: from the closed-shell SMILES."""
    base = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(base)
    assert AllChem.EmbedMolecule(m, randomSeed=seed) == 0, f"{name} would not embed"
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
        name=name)


def _heavy_degree(mol, i):
    return sum(1 for j in mol.neighbors(i) if mol.atoms[j].element != "H")


# ======================================================= the reported case
def test_poly_1_butene_links_through_its_backbone_not_its_methyls():
    """``[*]CC(CC)[*]``: the links are the two INTERIOR carbons of butane."""
    poly = "[*]CC(CC)[*]"
    mol = _mol(_closed_shell(poly), "poly-1-butene")
    head, tail, _hh, _th = _links_from_wildcards(mol, poly)
    print(f"\n  links {head}/{tail}, heavy degrees "
          f"{_heavy_degree(mol, head)}/{_heavy_degree(mol, tail)}")
    assert tail in mol.neighbors(head), \
        "the two link atoms must be bonded — the repeat unit is two carbons"
    # One link is a chain-end methyl of butane, the other is interior. In the
    # polymer they become the backbone CH2 and the branched CH.
    degrees = sorted((_heavy_degree(mol, head), _heavy_degree(mol, tail)))
    assert degrees == [1, 2], f"unexpected link environment {degrees}"


def test_the_old_rule_gets_poly_1_butene_wrong():
    """Pinning the bug itself, so the fix cannot quietly be undone.

    If this ever starts passing "correctly", the element rule has changed and
    the fallback path needs looking at again.
    """
    poly = "[*]CC(CC)[*]"
    mol = _mol(_closed_shell(poly), "poly-1-butene")
    symbols = _wildcard_neighbour_symbols(poly)
    old = _pick_head_tail_by_element(mol, *symbols)
    new = _links_from_wildcards(mol, poly)
    print(f"\n  element rule: {old[:2]}   wildcard rule: {new[:2]}")
    assert {old[0], old[1]} != {new[0], new[1]}, \
        "the element rule now agrees; this test has lost its subject"
    assert old[1] not in mol.neighbors(old[0]), \
        "the element rule picked the two ends of butane, four bonds apart"


def test_the_repeat_unit_has_a_methine_and_a_branch():
    """The chemistry the user was shown, and what it should have been."""
    from paaf.chem_env import atom_group
    from paaf.monomer import Monomer
    from paaf.typing_context import build_trimer

    poly = "[*]CC(CC)[*]"
    mol = _mol(_closed_shell(poly), "poly-1-butene")
    head, tail, hh, th = _links_from_wildcards(mol, poly)
    ctx = build_trimer([Monomer(mol, head, tail, hh, th, name="poly-1-butene")])

    groups = {}
    for trimer_index, (unit, monomer_index) in sorted(ctx.provenance.items()):
        if unit == 1:                              # the repeat unit
            groups.setdefault(monomer_index,
                              atom_group(ctx.molecule, trimer_index))
    carbons = {i: g for i, g in groups.items() if mol.atoms[i].element == "C"}
    print(f"\n  repeat unit: {carbons}")
    assert any("methine" in g for g in carbons.values()), \
        "no CH — the backbone came out unbranched, as polyethylene"
    assert any("methyl" in g for g in carbons.values()), \
        "no CH3 — the ethyl branch is missing"


# ============================================== the rest of the vinyl family
#: (name, polymerisation SMILES, how many carbons lie between the links)
VINYLS = [
    ("polypropylene",     "[*]CC(C)[*]"),
    ("PMMA",              "[*]CC(C)(C(=O)OC)[*]"),
    ("poly(vinyl acetate)", "[*]CC(OC(C)=O)[*]"),
    ("poly(methyl acrylate)", "[*]CC(C(=O)OC)[*]"),
    ("poly(vinyl methyl ether)", "[*]CC(OC)[*]"),
    ("PVC",               "[*]CC(Cl)[*]"),
    ("polystyrene",       "[*]CC([*])c1ccccc1"),
]


@pytest.mark.parametrize("name,poly", VINYLS, ids=[v[0] for v in VINYLS])
def test_a_vinyl_polymer_links_through_two_adjacent_carbons(name, poly):
    """Every vinyl polymer has a two-carbon backbone repeat, by definition."""
    mol = _mol(_closed_shell(poly), name)
    head, tail, _hh, _th = _links_from_wildcards(mol, poly)
    print(f"\n  {name}: links {head}/{tail}")
    assert mol.atoms[head].element == "C" and mol.atoms[tail].element == "C"
    assert tail in mol.neighbors(head), \
        f"{name}: the links are not bonded, so the backbone is not -C-C-"


# =============================================== polyesters are not disturbed
#: These were already right, and are the reason the bug survived so long.
ALREADY_RIGHT = [
    ("polyethylene", "[*]CC[*]"),
    ("PBS",  "[*]OCCCCOC(=O)CCC(=O)[*]"),
    ("PLA",  "[*]OC(C)C(=O)[*]"),
    ("PCL",  "[*]OCCCCCC(=O)[*]"),
    ("PEO",  "[*]CCO[*]"),
]


@pytest.mark.parametrize("name,poly", ALREADY_RIGHT,
                         ids=[p[0] for p in ALREADY_RIGHT])
def test_the_polymers_that_were_already_right_still_are(name, poly):
    mol = _mol(_closed_shell(poly), name)
    new = _links_from_wildcards(mol, poly)
    assert new is not None, f"{name}: the wildcard rule refused"
    symbols = _wildcard_neighbour_symbols(poly)
    old = _pick_head_tail_by_element(mol, *symbols)
    print(f"\n  {name}: was {old[:2] if old else None}, now {new[:2]}")
    if old is not None:
        assert {new[0], new[1]} == {old[0], old[1]}, \
            f"{name}: the links moved on a polymer that was building correctly"


# ================================================== the cap hydrogens
@pytest.mark.parametrize("name,poly", VINYLS + ALREADY_RIGHT,
                         ids=[p[0] for p in VINYLS + ALREADY_RIGHT])
def test_each_cap_is_a_hydrogen_on_its_own_link_atom(name, poly):
    """The cap is the H the junction consumes; on the wrong atom it is a hole.

    Ethane is the case that exposed this: all six hydrogens share one
    environment, so a cap could be paired with a hydrogen of the OTHER carbon.
    Eight library polymers were refused outright because of it.
    """
    mol = _mol(_closed_shell(poly), name)
    head, tail, head_h, tail_h = _links_from_wildcards(mol, poly)
    print(f"\n  {name}: caps {head_h} on {head}, {tail_h} on {tail}")
    assert mol.atoms[head_h[0]].element == "H"
    assert mol.atoms[tail_h[0]].element == "H"
    assert head_h[0] in mol.neighbors(head), "head cap is on another atom"
    assert tail_h[0] in mol.neighbors(tail), "tail cap is on another atom"
    assert head_h[0] != tail_h[0], "one hydrogen cannot cap both ends"


# ================================================ the whole shipped library
def test_every_library_polymer_resolves_its_links_from_the_wildcards():
    """The sweep that measured the damage, kept as a floor.

    Falling back to the element rule is not an error in itself — it is there
    for exactly that — but every polymer that CAN be read from its wildcards
    should be, because the fallback is the thing that was guessing.
    """
    import csv
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "paaf" / "data" / \
        "polymer_database.csv"
    if not path.exists():
        pytest.skip("the polymer database is not shipped here")

    embeddable = resolved = 0
    failures = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            poly = (row.get("smiles") or "").strip()
            name = (row.get("polymer name") or "").strip()
            if not poly or "[*]" not in poly:
                continue
            try:
                mol = _mol(_closed_shell(poly), name)
            except Exception:
                continue                      # cannot be embedded; not our topic
            embeddable += 1
            if _links_from_wildcards(mol, poly) is not None:
                resolved += 1
            else:
                failures.append(name)
    print(f"\n  {resolved}/{embeddable} resolved from wildcards")
    assert embeddable > 90, "the database did not load as expected"
    assert not failures, f"fell back to guessing for: {failures[:6]}"


# ============================== the real path: a file plus a .polysmi sidecar
def _write_xyz(tmp_path, name, smiles):
    mol = _mol(smiles, name)
    lines = [str(len(mol.atoms)), name]
    for a in mol.atoms:
        lines.append(f"{a.element} {a.xyz[0]:.4f} {a.xyz[1]:.4f} {a.xyz[2]:.4f}")
    path = tmp_path / f"{name}.xyz"
    path.write_text("\n".join(lines) + "\n")
    return path


#: ``load_structure`` reads files through OpenBabel, so these two run on a
#: full install and skip where it is missing. They are the only tests here
#: that touch the disk, and the only ones that prove the wiring.
needs_openbabel = pytest.mark.skipif(
    __import__("importlib").util.find_spec("openbabel") is None,
    reason="reading a structure file needs OpenBabel")


@needs_openbabel
def test_monomer_from_file_uses_the_sidecar_wildcards(tmp_path):
    """This is how the GUI actually gets here.

    The unit tests above call ``_links_from_wildcards`` directly, so they stay
    green even if nothing calls it. ``monomer_from_file`` reading the
    ``.polysmi`` sidecar is the path a library pick really takes.
    """
    from paaf.monomer import monomer_from_file

    poly = "[*]CC(CC)[*]"
    path = _write_xyz(tmp_path, "poly1butene", _closed_shell(poly))
    (tmp_path / "poly1butene.xyz.polysmi").write_text(poly + "\n")

    mono = monomer_from_file(path)
    mol = mono.molecule
    print(f"\n  head {mono.head_index}, tail {mono.tail_index}, "
          f"removes {mono.head_removes}/{mono.tail_removes}")
    assert mono.tail_index in mol.neighbors(mono.head_index), \
        "monomer_from_file still joins through the wrong atoms"
    assert mono.head_removes and mono.tail_removes
    assert mol.atoms[mono.head_removes[0]].element == "H"
    assert mono.head_removes[0] in mol.neighbors(mono.head_index)
    assert mono.tail_removes[0] in mol.neighbors(mono.tail_index)


@needs_openbabel
def test_explicit_indices_still_win_over_the_sidecar(tmp_path):
    """A user who states head and tail means it."""
    from paaf.monomer import monomer_from_file

    poly = "[*]CC(CC)[*]"
    path = _write_xyz(tmp_path, "poly1butene", _closed_shell(poly))
    (tmp_path / "poly1butene.xyz.polysmi").write_text(poly + "\n")

    mono = monomer_from_file(path, head=1, tail=4)     # 1-based, so 0 and 3
    print(f"\n  head {mono.head_index}, tail {mono.tail_index}")
    assert (mono.head_index, mono.tail_index) == (0, 3)


# ========================================================= negative controls
def test_a_smiles_without_two_wildcards_is_refused():
    mol = _mol("CCCC", "butane")
    assert _links_from_wildcards(mol, "CCCC") is None, "no wildcards at all"
    assert _links_from_wildcards(mol, "[*]CCCC") is None, "only one wildcard"


def test_a_structure_that_is_not_the_smiles_is_refused():
    """The fallback exists for this; silently mapping would be worse.

    The curated PBS entry is a real instance: its stored monomer is the acid,
    while capping the wildcards gives the aldehyde PAAF links through.
    """
    mol = _mol("CCCCCC", "hexane")
    assert _links_from_wildcards(mol, "[*]CC(CC)[*]") is None, \
        "a six-carbon structure was matched to a four-carbon SMILES"


def test_the_wildcard_becomes_the_cap_hydrogen():
    """The mechanism, stated on its own: [*] does not vanish, it becomes H."""
    built = _capped_from_wildcards("[*]CC(CC)[*]")
    assert built is not None
    reference, caps, links = built
    print(f"\n  caps at {caps}, links at {links}, "
          f"{reference.GetNumAtoms()} atoms")
    assert len(caps) == 2 and len(links) == 2
    for cap, link in zip(caps, links):
        assert reference.GetAtomWithIdx(cap).GetSymbol() == "H"
        neighbours = [n.GetIdx()
                      for n in reference.GetAtomWithIdx(cap).GetNeighbors()]
        assert neighbours == [link], "the cap is not on its link atom"
