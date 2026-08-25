"""Smoke tests that don't require OpenBabel / mbuild / moltemplate."""
from __future__ import annotations

import numpy as np

from paaf.chain_builder import simple_backend
from paaf.dlfield_converter import parse_par, to_moltemplate
from paaf.ff_registry import get_ff, list_ffs
from paaf.monomer import Monomer
from paaf.structure import Atom, Molecule


def _ethylene() -> Monomer:
    # H2C=CH2, indices 0..5: C0, C1, H2, H3, H4, H5
    atoms = [
        Atom(0, "C", np.array([0.0, 0.0, 0.0])),
        Atom(1, "C", np.array([1.34, 0.0, 0.0])),
        Atom(2, "H", np.array([-0.5, 0.9, 0.0])),
        Atom(3, "H", np.array([-0.5, -0.9, 0.0])),
        Atom(4, "H", np.array([1.84, 0.9, 0.0])),
        Atom(5, "H", np.array([1.84, -0.9, 0.0])),
    ]
    bonds = [(0, 1, 2), (0, 2, 1), (0, 3, 1), (1, 4, 1), (1, 5, 1)]
    mol = Molecule(atoms=atoms, bonds=bonds, name="ETH")
    # head=C0 (drop H2), tail=C1 (drop H4)
    return Monomer(mol, head_index=0, tail_index=1, head_removes=[2], tail_removes=[4], name="ETH")


def test_registry_has_all_ffs():
    keys = {ff.key for ff in list_ffs()}
    for k in ("oplsaa", "gaff", "pcff", "compass", "cvff", "trappe_ua",
              "dreiding", "charmm36", "amber_gaff"):
        assert k in keys, k
    assert get_ff("pcff").kind == "dlfield"


def test_simple_backend_builds_chain():
    m = _ethylene()
    chain = simple_backend([m], sequence=[0, 0, 0])
    # Each monomer has 6 atoms; per internal monomer removes 2 caps, total
    # atoms = 6 + (6-2) + (6-1) not exactly — we just check the range.
    assert 12 <= len(chain.atoms) <= 18
    assert any(a.element == "C" for a in chain.atoms)


def test_dlfield_converter_roundtrip(tmp_path):
    src = tmp_path / "toy.par"
    src.write_text(
        "UNIT kcal/mol\n"
        "POTENTIAL TOY\n"
        "BOND       b0    K2   K3   K4    Remark\n"
        "cA   cB   1.50  300.0  -500.0  400.0  test\n"
        "cA   hA   1.10  340.0  -600.0  300.0  test\n"
        "END BOND\n"
        "ANGLE      t0   K2  Remark\n"
        "cA cA cB   109.5  60.0 test\n"
        "END ANGLE\n"
        "VDW        eps   sigma\n"
        "cA  0.066  3.50\n"
        "hA  0.030  2.42\n"
        "END VDW\n"
    )
    dl = parse_par(src)
    assert dl.ff == "TOY"
    assert len(dl.bonds) == 2
    assert len(dl.angles) == 1
    assert len(dl.vdw) == 2
    out = to_moltemplate(dl, tmp_path / "toy.lt")
    text = out.read_text()
    assert "TOY {" in text
    assert "bond_coeff" in text
    assert "pair_coeff" in text


def test_config_roundtrip(tmp_path):
    from paaf.config import Config, MonomerSpec
    cfg = Config()
    cfg.monomers = [MonomerSpec(file="foo.pdb", head=1, tail=10)]
    p = cfg.save(tmp_path / "cfg.yaml")
    cfg2 = Config.load(p)
    assert cfg2.monomers[0].head == 1
