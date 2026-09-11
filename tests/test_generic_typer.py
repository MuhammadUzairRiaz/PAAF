"""Automatic atom typing for every bundled Moltemplate force field."""
from __future__ import annotations

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

from paaf.ff_registry import get_ff  # noqa: E402
from paaf.lt_parser import parse_atom_types  # noqa: E402
from paaf.structure import Atom, Molecule  # noqa: E402
from paaf.typers.generic import (FF_MAPS, UNITED_ATOM_KEYS, classify,  # noqa: E402
                                 supported_keys, type_generic)

POLYMERS = {
    "PE": "CCCCCCCC",
    "PS": "CC(c1ccccc1)CC(c1ccccc1)C",
    "PMMA": "CC(C)(C(=O)OC)CC(C)(C(=O)OC)C",
    "PBS": "OCCCCOC(=O)CCC(=O)OCCCCOC(=O)CCC(=O)O",
    "nylon6": "CCCCCC(=O)NCCCCCC(=O)NC",
    "PDMS": "C[Si](C)(O[Si](C)(C)O[Si](C)(C)C)C",
    "PEO": "COCCOCCOC",
    "PVC": "CC(Cl)CC(Cl)C",
    "PAN": "CC(C#N)CC(C#N)C",
    "ENR": "CC1OC1CC=C(C)CC",
    "PET-like": "COC(=O)c1ccc(cc1)C(=O)OCCO",
}


def _mol(smi: str) -> Molecule:
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    mol = Molecule(name=smi)
    for a in m.GetAtoms():
        mol.atoms.append(Atom(index=a.GetIdx(), element=a.GetSymbol(), xyz=np.zeros(3)))
    for b in m.GetBonds():
        mol.bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()))
    return mol


@pytest.mark.parametrize("key", supported_keys())
def test_every_mapped_id_exists_in_bundled_lt(key):
    table, fallback = FF_MAPS[key]
    ids = {t.ff_id for t in parse_atom_types(str(get_ff(key).bundled_path()))}
    if key.startswith("lopls"):        # L-OPLS imports the OPLS-AA library it refines
        base = "oplsaa2008" if "2008" in key else "oplsaa"
        ids |= {t.ff_id for t in parse_atom_types(str(get_ff(base).bundled_path()))}
    used = set(table.values()) | set(fallback.values())
    assert used <= ids, sorted(used - ids)


@pytest.mark.parametrize("key", ["compass_published", "dreiding", "gaff", "gaff2", "oplsaa", "oplsaa2008"])
@pytest.mark.parametrize("name", list(POLYMERS))
def test_all_atom_ffs_type_every_atom_of_common_polymers(key, name):
    mol = _mol(POLYMERS[name])
    t = type_generic(mol, key)
    # Honest gaps: COMPASS-published has no Cl, GAFF has no Si.
    allowed_missing = {("compass_published", "PVC"): {"Cl"}, ("gaff", "PDMS"): {"Si"},
                       ("gaff2", "PDMS"): {"Si"}}.get((key, name), set())
    missing = [mol.atoms[i].element for i in range(len(mol.atoms)) if i not in t]
    assert set(missing) <= allowed_missing, missing
    ids = {x.ff_id for x in parse_atom_types(str(get_ff(key).bundled_path()))}
    assert set(t.values()) <= ids


def test_element_of_type_matches_atom():
    """A suggestion must never put a carbon type on an oxygen etc."""
    for key in ("compass_published", "dreiding", "gaff", "oplsaa", "oplsaa2008"):
        info = {x.ff_id: x.element for x in parse_atom_types(str(get_ff(key).bundled_path()))}
        for smi in POLYMERS.values():
            mol = _mol(smi)
            for i, tid in type_generic(mol, key).items():
                el = info.get(tid, "")
                if el:
                    assert el == mol.atoms[i].element, (key, smi, i, tid)


def test_chemistry_is_recognised():
    env = {i: k for i, (k, _) in classify(_mol("CCOC(=O)CC")).items()}
    keys = set(env.values())
    assert {"ester_C", "ester_Odb", "ester_Os", "alkane_CH3", "alkane_CH2_X"} <= keys
    env = {i: k for i, (k, _) in classify(_mol("CC(=O)NC")).items()}
    assert {"amide_C", "amide_Odb", "amide_NH", "amide_H"} <= set(env.values())
    env = {i: k for i, (k, _) in classify(_mol("c1ccccc1O")).items()}
    assert {"arom_CH", "arom_C_O", "phenol_O", "phenol_H", "arom_H"} <= set(env.values())


def test_specific_expectations():
    pbs = _mol("CCOC(=O)CCC(=O)OCC")
    c = type_generic(pbs, "compass_published")
    d = type_generic(pbs, "dreiding")
    g = type_generic(pbs, "gaff")
    carbonyl = [i for i, a in enumerate(pbs.atoms)
                if a.element == "C" and any(pbs.atoms[j].element == "O" and
                                            any(o == 2.0 for x, y, o in pbs.bonds if {x, y} == {i, j})
                                            for j in pbs.neighbors(i))]
    assert carbonyl and all(c[i] == "c3prime" for i in carbonyl)
    assert all(d[i] == "C_2_b2" for i in carbonyl)
    assert all(g[i] == "c" for i in carbonyl)
    # GAFF: H on a C bearing one O -> h1
    for i, a in enumerate(pbs.atoms):
        if a.element == "H":
            c_ = pbs.neighbors(i)[0]
            n_o = sum(1 for j in pbs.neighbors(c_) if pbs.atoms[j].element == "O")
            assert g[i] == ("h1" if n_o == 1 else "hc")


def test_united_atom_types_heavy_atoms_only():
    for key in UNITED_ATOM_KEYS:
        t = type_generic(_mol("CCCCCC"), key)
        mol = _mol("CCCCCC")
        assert all(mol.atoms[i].element != "H" for i in t)
        assert len(t) == 6


def test_assigner_routes_moltemplate_ffs_to_generic(monkeypatch):
    from paaf import ff_assigner
    mol = _mol("CCCCC")
    for key in ("compass_published", "dreiding"):
        ff = get_ff(key)
        ff_assigner.assign(mol, ff, dl_lib_dir=None)      # would raise before: needs dl_lib_dir
        ids = {x.ff_id for x in parse_atom_types(str(ff.bundled_path()))}
        assert all(a.ff_type in ids for a in mol.atoms)
    # GAFF without antechamber falls back to the built-in typer
    monkeypatch.setattr(ff_assigner, "_which", lambda exe: None)
    ff_assigner.assign(mol, get_ff("gaff"))
    assert {a.ff_type for a in mol.atoms} == {"c3", "hc"}
    # manual-strategy UA FF: heavy atoms filled, H left untyped (absorbed at export)
    for a in mol.atoms:
        a.ff_type = None
    ff_assigner.assign(mol, get_ff("trappe_ua"))
    assert all(a.ff_type for a in mol.atoms if a.element != "H")
    assert all(not a.ff_type for a in mol.atoms if a.element == "H")


def test_dialog_uses_generic_typer_for_non_opls():
    from pathlib import Path
    src = Path("paaf/gui/atom_type_dialog.py").read_text()
    assert "def _typer(self)" in src and "type_generic" in src
    assert src.count("from ..typers.oplsaa import type_oplsaa") == 1   # only inside _typer


def test_opls_smarts_rules_and_fallbacks_have_right_element():
    """Every id the OPLS-AA SMARTS typer can emit belongs to the element it types.

    Amide/amine/thiol/sulfone/halide ids used to be from another numbering
    (177 -> an ether O for the amide carbon; 739 -> a carbon for the amine N).
    """
    import re
    from paaf.typers import oplsaa
    info = {t.ff_id: t.element for t in parse_atom_types(str(get_ff("oplsaa").bundled_path()))}
    sym = {"c": "C", "cl": "Cl", "br": "Br", "si": "Si"}
    for smarts, tid, desc in oplsaa._RULES:
        m = re.match(r"\[?(Cl|Br|Si|[A-Za-z])", smarts)
        el = m.group(1); el = sym.get(el.lower(), el.upper() if len(el) == 1 else el)
        if el == "H" and smarts.startswith("[H]"):
            el = "H"
        assert info.get(tid) == el, (smarts, tid, desc, info.get(tid))
    for el, tid in oplsaa._ELEMENT_FALLBACK.items():
        assert info.get(tid) == el, (el, tid, info.get(tid))
