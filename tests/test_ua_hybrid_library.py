"""UA/AA mixing across the whole polymer library.

For every library polymer: build a chain, type it with OPLS-AA, then turn
every carbon that has an OPLS-UA counterpart into a bead (alternating
tagged 'UA:' and plain ids, as a user might pick them from either table),
absorb, and check that (a) exactly the right hydrogens vanished, (b) every
remaining atom has a type of its own element, (c) bead masses are right.
A subset is then run through moltemplate.sh to prove every bonded term
across the UA/AA boundary resolves.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from paaf import lt_writer  # noqa: E402
from paaf.builder import list_library  # noqa: E402
from paaf.chain_builder import build_chain  # noqa: E402
from paaf.ff_registry import get_ff  # noqa: E402
from paaf.lt_parser import parse_atom_types  # noqa: E402
from paaf.monomer import Monomer, _capped_from_wildcards  # noqa: E402
from paaf.structure import Atom, Molecule  # noqa: E402
from paaf.typers.generic import classify, type_generic  # noqa: E402
from paaf.ua_hybrid import (UA_PREFIX, apply_hybrid, bead_table,  # noqa: E402
                            implicit_ua_library, patch_data_masses)

LIB = Path(__file__).resolve().parents[1] / "ff_libraries" / "moltemplate"
UA_BEADS = bead_table(LIB / "oplsua_2024.lt")

# environment -> OPLS-UA bead (only carbons the UA library can represent)
ENV_TO_UA = {"alkane_CH3": "68", "alkane_CH2": "71", "alkane_CH": "73", "alkane_C": "76",
             "alkane_CH3_ar": "68", "alkane_CH2_ar": "71", "alkane_CH_ar": "73",
             "alkene_CH2": "72", "alkene_CH": "74", "alkene_C": "77", "arom_CH": "75",
             "alkane_CH3_X": "109", "alkane_CH2_X": "110", "alkane_CH_X": "106",
             "alkane_C_X": "107"}


def _monomer(poly_smiles, name):
    got = _capped_from_wildcards(poly_smiles)
    if got is None:
        return None
    m, caps, links = got
    if AllChem.EmbedMolecule(m, randomSeed=3) < 0:
        return None
    xyz = m.GetConformer().GetPositions()
    mol = Molecule(name=name)
    for a in m.GetAtoms():
        mol.atoms.append(Atom(index=a.GetIdx(), element=a.GetSymbol(), xyz=np.array(xyz[a.GetIdx()]),
                              name=f"{a.GetSymbol()}{a.GetIdx()+1}"))
    for b in m.GetBonds():
        mol.bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()))
    return Monomer(mol, links[0], links[1], [caps[0]], [caps[1]], name=name)


def _records():
    return [r for r in list_library("all") if r.smiles.count("[*]") == 2]


def _ua_typed_chain(rec, n=4):
    mon = _monomer(rec.smiles, rec.name)
    if mon is None:
        return None, None
    chain = build_chain([mon], n=n, backend="simple")
    t = type_generic(chain, "oplsaa")
    for a in chain.atoms:
        a.ff_type = t.get(a.index)
    env = {i: k for i, (k, _) in classify(chain).items()}
    expected_removed = 0
    beads = 0
    for a in chain.atoms:
        if a.element != "C":
            continue
        ua = ENV_TO_UA.get(env.get(a.index, ""))
        if not ua:
            continue
        n_h = sum(1 for j in chain.neighbors(a.index) if chain.atoms[j].element == "H")
        if n_h < UA_BEADS[ua][0]:
            continue
        a.ff_type = (UA_PREFIX + ua) if beads % 2 else ua
        expected_removed += UA_BEADS[ua][0]
        beads += 1
    return chain, (expected_removed, beads)


@pytest.mark.parametrize("rec", _records(), ids=lambda r: r.pid or r.name)
def test_library_polymer_absorbs_exactly_its_bead_hydrogens(rec, tmp_path):
    chain, exp = _ua_typed_chain(rec)
    if chain is None:
        pytest.skip("RDKit cannot embed this repeat unit")
    expected_removed, beads = exp
    n_before = len(chain.atoms)
    if beads == 0:
        assert implicit_ua_library(get_ff("oplsaa"), chain) is None
        return
    assert implicit_ua_library(get_ff("oplsaa"), chain) == "oplsua_2024"
    res = apply_hybrid(chain, get_ff("oplsaa"), get_ff("oplsua_2024"), tmp_path)
    assert res.removed_h == expected_removed and res.n_beads == beads
    assert len(res.chain.atoms) == n_before - expected_removed
    # every atom typed, type element == atom element, no UA: tag left
    elements = {t.ff_id: t.element for t in parse_atom_types(str(LIB / "oplsaa2024.lt"))}
    for a in res.chain.atoms:
        assert a.ff_type and not a.ff_type.startswith(UA_PREFIX), a
        el = elements.get(a.ff_type)
        assert el in (None, a.element), (rec.name, a.index, a.ff_type, el, a.element)
        if a.ff_type.startswith("UA_"):
            assert a.element == "C"
    for tid, mass in res.bead_masses.items():
        assert mass == pytest.approx(UA_BEADS[tid.replace("UA_", "")][1])
    assert (tmp_path / "paaf_ua_bridge.lt").exists()
    # bonds only between surviving atoms, connectivity intact
    n = len(res.chain.atoms)
    assert all(0 <= i < n and 0 <= j < n for i, j, _ in res.chain.bonds)
    assert len(res.chain.bonds) == len(chain.bonds) - expected_removed


MOLTEMPLATE_SUBSET = ["PE", "PP", "PS", "PMMA", "PBS", "PDMS", "PEO", "PVC", "PAN",
                      "polybutadiene", "PET", "nylon-6", "polyisoprene", "PLA", "PVA"]


def _find(name):
    for r in _records():
        for cand in (r.pid, r.name):
            if cand and cand.lower().replace(" ", "") == name.lower().replace(" ", ""):
                return r
    for r in _records():
        if name.lower() in (r.name or "").lower():
            return r
    return None


@pytest.mark.skipif(not (Path.home() / ".local/bin/moltemplate.sh").exists()
                    and shutil.which("moltemplate.sh") is None, reason="no moltemplate.sh")
@pytest.mark.parametrize("name", MOLTEMPLATE_SUBSET)
def test_moltemplate_resolves_mixed_chain(name, tmp_path, monkeypatch):
    rec = _find(name)
    if rec is None:
        pytest.skip(f"{name} not in library")
    monkeypatch.setenv("PATH", str(Path.home() / ".local/bin") + os.pathsep + os.environ.get("PATH", ""))
    from paaf.moltemplate_runner import run_moltemplate
    chain, exp = _ua_typed_chain(rec, n=3)
    if chain is None or exp[1] == 0:
        pytest.skip("no beads possible")
    res = apply_hybrid(chain, get_ff("oplsaa"), get_ff("oplsua_2024"), tmp_path)
    for f in LIB.glob("oplsaa2024.lt"):
        shutil.copy(f, tmp_path / f.name)
    assert res.lt_include == "paaf_ua_bridge.lt"
    lt = lt_writer.write_chain_lt(res.chain, tmp_path, get_ff("oplsaa"), name="poly",
                                  inherit=res.inherit, lt_include=res.lt_include,
                                  bond_type=res.bond_type)
    sysl = lt_writer.write_system_lt(tmp_path, lt, n_chains=1, box=[60, 60, 60])
    data = run_moltemplate(sysl, work_dir=tmp_path)
    patch_data_masses(data, res.bead_masses)
    txt = data.read_text()
    g = lambda k: int(re.search(rf"(\d+)\s+{k}", txt).group(1))
    assert g("atoms") == len(res.chain.atoms)
    assert g("bonds") == len(res.chain.bonds)
    log = (tmp_path / "moltemplate.log").read_text()
    assert "Error" not in log
