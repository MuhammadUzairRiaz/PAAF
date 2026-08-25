"""Tests for the force-field-independent chemical-environment classifier.

These build tiny :class:`paaf.structure.Molecule` graphs by hand (no OpenBabel
/ RDKit needed) and check that:

* every heavy atom gets an authentic chemistry label (epoxide, aromatic,
  alkene vs carbonyl, ester/amide carbonyl, methyl/methylene);
* every hydrogen is labelled by the environment of its PARENT carbon
  (the key new behaviour — "H on epoxide CH2" not just "H (aliphatic C-H)");
* the generic per-FF type suggester ranks a force field's own library.
"""
from __future__ import annotations

import numpy as np
import pytest

from paaf.structure import Atom, Molecule
from paaf.chem_env import (
    atom_group, all_groups, all_groups_with_links, suggest_ff_type,
)
from paaf.lt_parser import AtomTypeInfo


def _mol(elements, bonds):
    """Build a Molecule from element symbols and (i, j, order) bonds."""
    atoms = [Atom(index=i, element=e, xyz=np.zeros(3)) for i, e in enumerate(elements)]
    return Molecule(atoms=atoms, bonds=[(i, j, float(o)) for i, j, o in bonds])


# --------------------------------------------------------------------- epoxide
def test_epoxide_and_its_hydrogens():
    # Ethylene oxide: C0-C1-O2 ring, each carbon carries 2 H.
    m = _mol(
        ["C", "C", "O", "H", "H", "H", "H"],
        [(0, 1, 1), (0, 2, 1), (1, 2, 1),
         (0, 3, 1), (0, 4, 1), (1, 5, 1), (1, 6, 1)],
    )
    g = all_groups(m)
    assert g[0] == "epoxide CH2"
    assert g[2] == "epoxide -O- (oxirane)"
    # Hydrogens reflect the epoxide carbon they sit on.
    assert g[3] == "H on epoxide CH2"
    assert g[5] == "H on epoxide CH2"


# --------------------------------------------------------------------- alkane
def test_propane_methyl_vs_methylene_hydrogens():
    # C0H3 - C1H2 - C2H3
    m = _mol(
        ["C", "C", "C", "H", "H", "H", "H", "H", "H", "H", "H"],
        [(0, 1, 1), (1, 2, 1),
         (0, 3, 1), (0, 4, 1), (0, 5, 1),   # methyl H on C0
         (1, 6, 1), (1, 7, 1),              # methylene H on C1
         (2, 8, 1), (2, 9, 1), (2, 10, 1)], # methyl H on C2
    )
    g = all_groups(m)
    assert g[0] == "CH3 (methyl)"
    assert g[1] == "CH2 (methylene)"
    assert g[3] == "H on CH3 (methyl)"
    assert g[6] == "H on CH2 (methylene)"


# --------------------------------------------------------------------- alkene
def test_ethene_terminal_alkene():
    # H2C=CH2
    m = _mol(
        ["C", "C", "H", "H", "H", "H"],
        [(0, 1, 2), (0, 2, 1), (0, 3, 1), (1, 4, 1), (1, 5, 1)],
    )
    g = all_groups(m)
    assert g[0] == "=CH2 (terminal alkene)"
    assert g[2] == "H on =CH2 (terminal alkene)"


# --------------------------------------------------------------------- aromatic
def test_benzene_is_aromatic_not_alkene():
    # Kekulé benzene: alternating double bonds, each C bears one H.
    m = _mol(
        ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
        [(0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 2), (5, 0, 1),
         (0, 6, 1), (1, 7, 1), (2, 8, 1), (3, 9, 1), (4, 10, 1), (5, 11, 1)],
    )
    g = all_groups(m)
    assert g[0] == "aromatic CH"          # not "=CH-"
    assert g[6] == "H on aromatic CH"


# --------------------------------------------------------------- carbonyls
def test_ester_carbonyl_distinct_from_alkene():
    # Methyl acetate: CH3-C(=O)-O-CH3
    #  idx0 C(methyl) 1 C(carbonyl) 2 O(=O) 3 O(ester) 4 C(methyl)
    m = _mol(
        ["C", "C", "O", "O", "C", "H", "H", "H", "H", "H", "H"],
        [(0, 1, 1), (1, 2, 2), (1, 3, 1), (3, 4, 1),
         (0, 5, 1), (0, 6, 1), (0, 7, 1),
         (4, 8, 1), (4, 9, 1), (4, 10, 1)],
    )
    g = all_groups(m)
    assert g[1] == "ester/acid carbonyl C (O-C=O)"
    assert g[2] == "=O (carbonyl)"
    assert g[3] == "O- (ether / ester -O-)"


def test_amide_carbonyl():
    # Formamide: H-C(=O)-NH2   idx0 C 1 O(=O) 2 N 3 H(on C) 4,5 H(on N)
    m = _mol(
        ["C", "O", "N", "H", "H", "H"],
        [(0, 1, 2), (0, 2, 1), (0, 3, 1), (2, 4, 1), (2, 5, 1)],
    )
    g = all_groups(m)
    assert g[0] == "amide carbonyl C (N-C=O)"


# --------------------------------------------------------------- FF suggester
def test_generic_ff_suggester_ranks_library():
    types = [
        AtomTypeInfo("135", "C", "CT", 0.0, "CH3 alkane sp3"),
        AtomTypeInfo("136", "C", "CT", 0.0, "CH2 alkane sp3"),
        AtomTypeInfo("145", "C", "CA", 0.0, "aromatic benzene C"),
        AtomTypeInfo("180", "O", "OS", 0.0, "ether O"),
        AtomTypeInfo("154", "O", "OH", 0.0, "alcohol O"),
    ]
    # aromatic carbon -> benzene type
    top = suggest_ff_type("aromatic CH", "C", types)
    assert top and top[0][0] == "145"
    # methyl carbon -> CH3 alkane type
    top = suggest_ff_type("CH3 (methyl)", "C", types)
    assert top and top[0][0] == "135"
    # epoxide oxygen -> ether O
    top = suggest_ff_type("epoxide -O- (oxirane)", "O", types)
    assert top and top[0][0] == "180"


def test_polyisoprene_connection_carbons_become_backbone_ch2():
    # Standalone (H-capped) cis-1,4-polyisoprene repeat unit is 2-methyl-2-butene:
    #   C0H3 - C1(=C2)  ,  C2(-C3 methyl branch) - C4H3
    # idx: 0 C(cap CH3, head link), 1 C(=CH-), 2 C(>C=), 3 C(methyl branch),
    #      4 C(cap CH3, tail link)
    m = _mol(
        ["C", "C", "C", "C", "C",
         "H", "H", "H",      # on C0 (head cap)
         "H",                # on C1 (vinyl)
         "H", "H", "H",      # on C3 (methyl branch)
         "H", "H", "H"],     # on C4 (tail cap)
        [(0, 1, 1), (1, 2, 2), (2, 3, 1), (2, 4, 1),
         (0, 5, 1), (0, 6, 1), (0, 7, 1),
         (1, 8, 1),
         (3, 9, 1), (3, 10, 1), (3, 11, 1),
         (4, 12, 1), (4, 13, 1), (4, 14, 1)],
    )
    # Without link info the caps look like plain methyls.
    plain = all_groups(m)
    assert plain[0] == "CH3 (methyl)"
    assert plain[4] == "CH3 (methyl)"

    # With head=C0 (cap H idx5) and tail=C4 (cap H idx12) marked as links,
    # those carbons are backbone CH2 and the branch methyl (C3) is unchanged.
    linked = all_groups_with_links(m, {0: 5, 4: 12})
    assert "CH2" in linked[0] and "backbone chain-link" in linked[0]
    assert "CH2" in linked[4] and "backbone chain-link" in linked[4]
    assert linked[3] == "CH3 (methyl)"        # genuine branch methyl unchanged
    # A remaining H on the connection carbon reports the backbone environment.
    assert linked[6].startswith("H on CH2") and "backbone" in linked[6]
    # The cap H that is consumed by the inter-monomer bond is flagged.
    assert "removed when chain-linked" in linked[5]


def test_epoxidized_isoprene_keeps_epoxide_and_links():
    # [*]CC1(C)OC1C[*] capped standalone: two backbone CH2 caps + epoxide ring.
    #   idx0 C(head cap) - idx1 C(epoxide, +methyl idx2) - O idx3 -
    #        idx4 C(epoxide, H) - idx5 C(tail cap)
    m = _mol(
        ["C", "C", "C", "O", "C", "C",
         "H", "H", "H",        # C0 head cap (3H)
         "H", "H", "H",        # C2 methyl branch
         "H",                  # C4 epoxide CH
         "H", "H", "H"],       # C5 tail cap (3H)
        [(0, 1, 1), (1, 2, 1), (1, 3, 1), (3, 4, 1), (1, 4, 1), (4, 5, 1),
         (0, 6, 1), (0, 7, 1), (0, 8, 1),
         (2, 9, 1), (2, 10, 1), (2, 11, 1),
         (4, 12, 1),
         (5, 13, 1), (5, 14, 1), (5, 15, 1)],
    )
    linked = all_groups_with_links(m, {0: 6, 5: 13})
    # epoxide chemistry preserved
    assert linked[3] == "epoxide -O- (oxirane)"
    assert linked[1].startswith("epoxide")
    assert linked[4].startswith("epoxide")
    # both connection carbons are backbone CH2
    assert "CH2" in linked[0] and "backbone" in linked[0]
    assert "CH2" in linked[5] and "backbone" in linked[5]


def test_backwards_compatible_delegators():
    # The dialog keeps thin static delegators to chem_env; ensure they work.
    pytest.importorskip("PyQt5")
    from paaf.gui.atom_type_dialog import AtomTypingDialog  # noqa: WPS433
    lbl = AtomTypingDialog._group_label("C", ["H", "H", "H", "C"])
    assert lbl == "CH3 (methyl)"
