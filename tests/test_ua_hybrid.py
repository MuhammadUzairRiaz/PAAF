"""United-atom / all-atom mixing: hydrogens absorbed, files consistent."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from paaf import lt_writer  # noqa: E402
from paaf.ff_registry import get_ff  # noqa: E402
from paaf.structure import Atom, Molecule  # noqa: E402
from paaf.typers.generic import type_generic  # noqa: E402
from paaf.ua_hybrid import (UA_PREFIX, absorb_hydrogens, apply_hybrid,  # noqa: E402
                            bead_h_count, bead_table, patch_data_masses,
                            plan_absorption, write_bridge_lt)

LIB = Path(__file__).resolve().parents[1] / "ff_libraries" / "moltemplate"


def _mol(smi="CCCCCCCC", name="octane"):
    m = Chem.AddHs(Chem.MolFromSmiles(smi)); AllChem.EmbedMolecule(m, randomSeed=1)
    xyz = m.GetConformer().GetPositions(); mol = Molecule(name=name)
    for a in m.GetAtoms():
        mol.atoms.append(Atom(index=a.GetIdx(), element=a.GetSymbol(),
                              xyz=np.array(xyz[a.GetIdx()]), name=f"{a.GetSymbol()}{a.GetIdx()+1}"))
    for b in m.GetBonds():
        mol.bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()))
    return mol


def test_bead_h_count_reads_library_text():
    assert bead_h_count("CH3", "CH3") == 3
    assert bead_h_count("71", "CH2 (SP3) ALKANES", "C") == 2
    assert bead_h_count("73", "CH (SP3) ISOBUTANE", "C") == 1
    assert bead_h_count("76", "C (SP3) NEOPENTANE", "C") == 0
    assert bead_h_count("66", "CH4 66-77: JACS", "C") == 4
    assert bead_h_count("78", "O ALCOHOLS", "O") is None
    assert bead_h_count("CT", "CH2 all-atom C: alkanes", "C") == 2   # text says CH2; caller decides


def test_bead_tables_of_both_ua_libraries():
    tr = bead_table(LIB / "trappe1998.lt")
    assert tr["CH2"][0] == 2 and abs(tr["CH2"][1] - 14.17) < 0.01
    assert tr["CH3"][0] == 3
    ua = bead_table(LIB / "oplsua_2024.lt")
    assert ua["71"][0] == 2 and abs(ua["71"][1] - 14.027) < 1e-3
    assert ua["68"][0] == 3 and ua["76"][0] == 0
    assert "78" not in ua                                    # alcohol O is not a bead


def test_plan_and_absorb():
    mol = _mol()
    beads = bead_table(LIB / "trappe1998.lt")
    plan = plan_absorption(mol, {0: "UA:CH3", 1: "UA:CH2"}, beads)
    assert not plan.problems and len(plan.remove) == 5
    new, remap = absorb_hydrogens(mol, plan)
    assert len(new.atoms) == len(mol.atoms) - 5
    assert len(new.bonds) == len(mol.bonds) - 5
    assert len({a.name for a in new.atoms}) == len(new.atoms)
    # CH3 on a CH2 carbon is refused
    bad = plan_absorption(mol, {1: "UA:CH3"}, beads)
    assert bad.problems and "absorbs 3" in bad.problems[0]


def _typed(primary, ua_key, ua_atoms):
    mol = _mol(); t = type_generic(mol, primary)
    for a in mol.atoms:
        a.ff_type = t.get(a.index)
    for i in ua_atoms:
        nh = sum(1 for j in mol.neighbors(i) if mol.atoms[j].element == "H")
        tid = {3: "CH3", 2: "CH2"}[nh] if ua_key == "trappe_ua" else {3: "68", 2: "71"}[nh]
        mol.atoms[i].ff_type = (UA_PREFIX + tid) if ua_key != primary else tid
    return mol


def test_same_family_hybrid_needs_no_bridge(tmp_path):
    mol = _typed("oplsaa", "oplsua_2024", [0, 1, 2, 3])
    res = apply_hybrid(mol, get_ff("oplsaa"), get_ff("oplsua_2024"), tmp_path)
    assert res.removed_h == 9 and res.n_beads == 4 and res.inherit is None
    assert {a.ff_type for a in res.chain.atoms if a.element == "C"} == {"68", "71", "135", "136"}
    assert res.bead_masses == {"68": pytest.approx(15.035), "71": pytest.approx(14.027)}
    assert not (tmp_path / "paaf_ua_bridge.lt").exists()


def test_bridge_reopens_aa_namespace(tmp_path):
    mol = _typed("oplsaa", "trappe_ua", [0, 1])
    res = apply_hybrid(mol, get_ff("oplsaa"), get_ff("trappe_ua"), tmp_path)
    assert res.inherit == "OPLSAA" and res.lt_include == "paaf_ua_bridge.lt"
    txt = (tmp_path / "paaf_ua_bridge.lt").read_text()
    assert 'import "oplsaa2024.lt"' in txt and txt.count("OPLSAA {") == 1
    assert "@atom:UA_CH2 14.1707" in txt and "@atom:UA_CH3 15.2507" in txt
    assert "replace{ @atom:UA_CH2 @atom:UA_CH2_bCT_aCT_dCT_iCT }" in txt
    assert re.search(r"pair_coeff @atom:UA_CH2 @atom:UA_CH2\s+0\.091412 3\.9500", txt)
    assert "set type @atom:UA_CH3 charge 0.0" in txt
    assert {a.ff_type for a in res.chain.atoms if a.element == "C"} >= {"UA_CH3", "UA_CH2"}


def test_ua_only_requires_every_carbon_to_be_a_bead(tmp_path):
    mol = _typed("trappe_ua", "trappe_ua", list(range(8)))
    res = apply_hybrid(mol, get_ff("trappe_ua"), get_ff("trappe_ua"), tmp_path)
    assert len(res.chain.atoms) == 8 and res.bond_type == "saturated"
    mol2 = _typed("trappe_ua", "trappe_ua", list(range(8)))
    mol2.atoms[3].ff_type = None
    with pytest.raises(RuntimeError, match="hydrogen"):
        apply_hybrid(mol2, get_ff("trappe_ua"), get_ff("trappe_ua"), tmp_path)


def test_chain_lt_explicit_bond_type_and_overrides(tmp_path):
    mol = _typed("trappe_ua", "trappe_ua", list(range(8)))
    res = apply_hybrid(mol, get_ff("trappe_ua"), get_ff("trappe_ua"), tmp_path)
    p = lt_writer.write_chain_lt(res.chain, tmp_path, get_ff("trappe_ua"), name="poly",
                                 bond_type=res.bond_type)
    txt = p.read_text()
    assert "write('Data Bonds')" in txt and "@bond:saturated" in txt
    p2 = lt_writer.write_chain_lt(res.chain, tmp_path, get_ff("oplsaa"), name="poly2",
                                  inherit="OPLSAA", lt_include="paaf_ua_bridge.lt")
    t2 = p2.read_text()
    assert 'import "paaf_ua_bridge.lt"' in t2 and "poly2 inherits OPLSAA" in t2


def test_patch_data_masses(tmp_path):
    d = tmp_path / "system.data"
    d.write_text("LAMMPS\n\n2 atoms\n\n2 atom types\n\nMasses\n\n1 12.011  # 71_bC2_aC2_dC2_iC2\n"
                 "2 12.011  # 136_bCT_aCT_dCT_iCT\n\nAtoms  # full\n\n1 1 1 0 0 0 0\n2 1 2 0 0 0 0\n")
    assert patch_data_masses(d, {"71": 14.027}) == 1
    assert "1 14.0270  # 71_bC2_aC2_dC2_iC2" in d.read_text()
    assert "2 12.011  # 136_bCT_aCT_dCT_iCT" in d.read_text()


@pytest.mark.skipif(shutil.which("moltemplate.sh") is None
                    and not (Path.home() / ".local/bin/moltemplate.sh").exists(),
                    reason="moltemplate.sh not installed")
@pytest.mark.parametrize("primary,ua,atoms,n_atoms", [
    ("oplsaa", "oplsua_2024", [0, 1, 2, 3], 17),
    ("oplsaa", "trappe_ua", [0, 1, 2, 3], 17),
    ("trappe_ua", "trappe_ua", list(range(8)), 8),
])
def test_moltemplate_resolves_every_bonded_term(tmp_path, monkeypatch, primary, ua, atoms, n_atoms):
    import os
    monkeypatch.setenv("PATH", str(Path.home() / ".local/bin") + os.pathsep + os.environ.get("PATH", ""))
    from paaf.moltemplate_runner import run_moltemplate
    mol = _typed(primary, ua, atoms)
    res = apply_hybrid(mol, get_ff(primary), get_ff(ua), tmp_path)
    for f in LIB.glob("*.lt"):
        shutil.copy(f, tmp_path / f.name)
    lt = lt_writer.write_chain_lt(res.chain, tmp_path, get_ff(primary), name="poly",
                                  inherit=res.inherit, lt_include=res.lt_include,
                                  bond_type=res.bond_type)
    sysl = lt_writer.write_system_lt(tmp_path, lt, n_chains=1, box=[30, 30, 30])
    data = run_moltemplate(sysl, work_dir=tmp_path)
    patch_data_masses(data, res.bead_masses)
    txt = data.read_text()
    g = lambda k: int(re.search(rf"(\d+)\s+{k}", txt).group(1))
    assert g("atoms") == n_atoms and g("bonds") == n_atoms - 1
    assert g("angles") > 0 and g("dihedrals") > 0
    # every atom's type has a mass line and bead masses are > 12
    masses = {int(l.split()[0]): float(l.split()[1])
              for l in txt.split("Masses")[1].split("Atoms")[0].splitlines() if l.strip()}
    atom_lines = [l for l in txt.split("Atoms")[1].split("Bonds")[0].splitlines()
                  if l.strip() and not l.strip().startswith("#")]
    assert len(atom_lines) == n_atoms
    n_heavy_beads = sum(1 for l in atom_lines if masses[int(l.split()[2])] > 12.5)
    assert n_heavy_beads == len(atoms)


def test_gui_wiring_present():
    src = Path("paaf/gui/advanced_typing_dialog.py").read_text()
    assert "class AdvancedTypingDialog(AtomTypingDialog)" in src
    assert "_absorbed_h_keys" in src and "UA_PREFIX" in src
    mw = Path("paaf/gui/main_window.py").read_text()
    assert "Advanced: mix united-atom + all-atom" in mw
    assert "ua_secondary_key=" in mw and "def _open_advanced_typing_dialog" in mw
    from paaf.config import ForceFieldCfg
    assert ForceFieldCfg().ua_secondary_key is None
    pl = Path("paaf/pipeline.py").read_text()
    assert "apply_hybrid(chain, ff, _ua_ff, out_dir, _p)" in pl
    assert "patch_data_masses(Path(data_file), _hyb.bead_masses)" in pl


def test_plain_ua_block_type_from_aa_table_also_absorbs(tmp_path):
    """User's polybutadiene case: head atom typed '74' (no UA: tag) kept its H."""
    from paaf.ua_hybrid import implicit_ua_library
    mol = _mol("CC=CCCC=CC", "pb")
    t = type_generic(mol, "oplsaa")
    for a in mol.atoms:
        a.ff_type = t.get(a.index)
    # every carbon a UA bead; some tagged, some picked straight from the AA table
    for a in mol.atoms:
        if a.element != "C":
            continue
        nh = sum(1 for j in mol.neighbors(a.index) if mol.atoms[j].element == "H")
        sp2 = any(o == 2.0 for i, j, o in mol.bonds if a.index in (i, j))
        tid = {3: "68", 2: "71"}[nh] if not sp2 else "74"
        a.ff_type = ("UA:" + tid) if a.index % 2 else tid
    assert implicit_ua_library(get_ff("oplsaa"), mol) == "oplsua_2024"
    res = apply_hybrid(mol, get_ff("oplsaa"), get_ff("oplsua_2024"), tmp_path)
    assert not [a for a in res.chain.atoms if a.element == "H"]
    assert res.bead_masses["74"] == pytest.approx(13.019)
    assert {a.ff_type for a in res.chain.atoms} == {"68", "71", "74"}


def test_plain_ua_block_types_absorb_with_trappe_secondary_and_without_any(tmp_path):
    """Same behaviour whichever table the bead came from, whichever secondary."""
    from paaf.ua_hybrid import primary_own_beads
    assert "74" in primary_own_beads(get_ff("oplsaa"))
    assert primary_own_beads(get_ff("compass_published")) == {}
    mol = _typed("oplsaa", "trappe_ua", [0, 1])        # UA:CH3, UA:CH2 tagged
    mol.atoms[2].ff_type = "71"                         # plain OPLS-UA CH2 from AA table
    res = apply_hybrid(mol, get_ff("oplsaa"), get_ff("trappe_ua"), tmp_path)
    assert res.removed_h == 3 + 2 + 2
    assert res.bead_masses["71"] == pytest.approx(14.027)
    assert res.bead_masses["UA_CH2"] == pytest.approx(14.171, abs=1e-3)
    mw = Path("paaf/gui/main_window.py").read_text()
    assert "AdvancedTypingDialog as AtomTypingDialog" in mw
