"""Tests for lammps_replicator (no packmol required — uses grid fallback)."""
from __future__ import annotations

from pathlib import Path


_TOY_DATA = """LAMMPS toy 3-atom chain

3 atoms
2 bonds

1 atom types
1 bond types

-10.0 10.0 xlo xhi
-10.0 10.0 ylo yhi
-10.0 10.0 zlo zhi

Masses

1 12.011

Pair Coeffs

1 0.066 3.50

Bond Coeffs

1 300.0 1.50

Atoms

1 1 1  0.000000  0.000  0.000  0.000
2 1 1  0.000000  1.500  0.000  0.000
3 1 1  0.000000  3.000  0.000  0.000

Bonds

1 1 1 2
2 1 2 3
"""


def test_replicator_expands_atoms_and_bonds(tmp_path):
    from paaf.lammps_replicator import replicate_single_chain
    src = tmp_path / "single.data"; src.write_text(_TOY_DATA)
    out = tmp_path / "packed_box.data"
    replicate_single_chain(src, n_chains=4, box_edges=(30.0, 30.0, 30.0),
                          out_data_file=out, seed=1234)
    text = out.read_text()
    # 4 chains x 3 atoms = 12 atoms
    assert "12 atoms" in text
    # 4 chains x 2 bonds = 8 bonds
    assert "8 bonds" in text
    # Verify the box was written (line is like "0.0 30.0000 xlo xhi")
    assert "30.0000 xlo xhi" in text
    assert "30.0000 ylo yhi" in text
    assert "30.0000 zlo zhi" in text
    # Coefficients preserved verbatim
    assert "300.0 1.50" in text
    # Atom section has 12 rows with mol-ids 1..4
    for molid in range(1, 5):
        assert f" {molid} 1 " in text or f"{molid} 1 " in text
