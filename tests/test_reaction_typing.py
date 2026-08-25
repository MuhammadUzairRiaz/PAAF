"""Reaction export must find dl_field from the lib dir and feed it an xyz.

The esterification test case wrote reactant/product structures but ZERO
typed .data files, with _typing_* folders left completely empty. Two causes,
both already fixed in the amorphous-cell path and missed here: the GUI hands
over .../dl_f_4.13/lib but the executable is one level up, and DL_FIELD
cannot read PAAF's mol2 files.
"""
from __future__ import annotations

import stat
from pathlib import Path

import pytest

pytest.importorskip("rdkit", reason="reaction building requires RDKit")

from paaf.reaction_export import export_reaction        # noqa: E402


@pytest.fixture()
def fake_install(tmp_path, monkeypatch):
    """A dl_field install root with a lib/ subfolder, binary in the root."""
    monkeypatch.delenv("DL_FIELD_EXE", raising=False)
    root = tmp_path / "dl_f_4.13"
    (root / "lib").mkdir(parents=True)
    exe = root / "dl_field"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return root


def _export(tmp_path, dl_dir):
    said = []
    export_reaction(
        "esterification",
        ["CC(=O)[OH:1]", "[C:2]CO"],
        ["CC(=O)O[C:2]", "O"],
        tmp_path / "rx",
        ff_key="opls2005_dl",
        dl_field_dir=dl_dir,
        run_typing=True,
        progress=said.append,
    )
    return said


def test_the_lib_dir_finds_the_executable_one_level_up(tmp_path,
                                                       fake_install):
    """Pointing at lib/ (as the GUI field says to) must still find dl_field
    in the install root — proven by the control file getting written."""
    _export(tmp_path, fake_install / "lib")
    controls = list((tmp_path / "rx").rglob("*.control"))
    print(f"\n  control files: {[str(c.relative_to(tmp_path)) for c in controls]}")
    assert controls, ("dl_field was never invoked — the executable was not "
                      "found from the lib dir")


def test_dlfield_is_fed_the_xyz_not_the_mol2(tmp_path, fake_install):
    _export(tmp_path, fake_install / "lib")
    for ctrl in (tmp_path / "rx").rglob("*.control"):
        cfg = [l for l in ctrl.read_text().splitlines()
               if "configuration file" in l][0]
        print(f"  {cfg.strip()}")
        assert ".xyz" in cfg and ".mol2" not in cfg


def test_mapped_atoms_keep_their_hydrogens():
    """[C:2]CO must build ETHANOL. SMILES brackets suppress implicit
    hydrogens, so the mapped carbon parsed with 3 radical electrons and no
    Hs; DL_FIELD then refused the one-neighbour carbon (detect_new_key,
    atype = aliphatic). A map number tags an atom — it must not strip its
    hydrogens."""
    from collections import Counter

    from paaf.reaction_export import build_3d_side

    mol, _maps = build_3d_side(["CC(=O)[OH:1]", "[C:2]CO"], "reactant")
    counts = Counter(a.element for a in mol.atoms)
    print(f"\n  reactant side: {dict(counts)}")
    assert counts == {"C": 4, "O": 3, "H": 10}, \
        "acetic acid + ethanol must have every hydrogen"

    mol, _maps = build_3d_side(["CC(=O)O[C:2]", "O"], "product")
    counts = Counter(a.element for a in mol.atoms)
    print(f"  product side : {dict(counts)}")
    assert counts == {"C": 3, "O": 3, "H": 8}


def test_explicit_h_counts_on_mapped_atoms_are_respected():
    """[CH2:3] says two hydrogens and must stay two — the fix only restores
    hydrogens the user never suppressed on purpose."""
    from paaf.reaction_export import build_3d_side

    mol, _ = build_3d_side(["[CH2:3]=C"], "reactant")
    h = sum(1 for a in mol.atoms if a.element == "H")
    print(f"\n  ethylene H count: {h}")
    assert h == 4


def test_reaction_folder_names_are_path_safe(tmp_path, fake_install):
    """Example names like "ENR + maleic acid + ENR (bridge)" must not become
    folders with spaces — DL_FIELD reads paths only to the first space."""
    said = []
    export_reaction(
        "ENR + maleic acid + ENR (bridge)",
        ["CC(=O)[OH:1]", "[C:2]CO"], ["CC(=O)O[C:2]", "O"],
        tmp_path / "rx", ff_key="opls2005_dl",
        dl_field_dir=fake_install / "lib", run_typing=True,
        progress=said.append)
    made = [p.name for p in (tmp_path / "rx").iterdir() if p.is_dir()]
    print(f"\n  folder made: {made}")
    assert made == ["ENR_maleic_acid_ENR_bridge"]
    for name in made:
        assert " " not in name and "(" not in name


def test_spacey_structure_paths_never_reach_the_control_file(tmp_path,
                                                            monkeypatch):
    """Even if the surrounding path has spaces, the control file must point
    at something DL_FIELD's space-splitting parser can read."""
    import stat as _stat

    from paaf.dlfield_runner import run_dlfield

    monkeypatch.delenv("DL_FIELD_EXE", raising=False)
    root = tmp_path / "dl"
    root.mkdir()
    exe = root / "dl_field"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(exe.stat().st_mode | _stat.S_IEXEC)
    monkeypatch.setenv("DL_FIELD_EXE", str(exe))

    ugly = tmp_path / "ENR + maleic (bridge)"
    ugly.mkdir()
    xyz = ugly / "product.xyz"
    xyz.write_text("2\nx\nC 0 0 0\nH 1.09 0 0\n")
    work = ugly / "_typing"
    try:
        run_dlfield(structure_path=xyz, ff_key="opls2005_dl", work_dir=work)
    except Exception:
        pass  # fake binary produces nothing; the control file is the point
    ctrl = list(work.glob("*.control"))[0].read_text()
    cfg = [l for l in ctrl.splitlines() if "configuration file" in l][0]
    path_part = cfg.split("*")[0].strip()
    print(f"\n  control points at: {path_part!r}")
    assert " " not in path_part and "(" not in path_part
    assert (work / path_part).exists(), \
        "the local copy the control references must exist in the work dir"
