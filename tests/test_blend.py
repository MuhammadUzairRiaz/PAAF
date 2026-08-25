"""Smoke tests for the multi-component blend replicator.

We monkey-patch the packmol invocation so the test runs without needing
packmol installed — it produces a fake PDB whose coordinates are just a
translation of the input, mimicking what packmol would return.
"""
from __future__ import annotations

from pathlib import Path


_TOY_PBS = """LAMMPS toy PBS-like 4-atom chain

4 atoms
3 bonds

1 atom types
1 bond types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 12.0

Pair Coeffs

1 0.06 3.50

Bond Coeffs

1 300.0 1.50

Atoms

1 1 1 0.0  0.0 0.0 0.0
2 1 1 0.0  1.5 0.0 0.0
3 1 1 0.0  3.0 0.0 0.0
4 1 1 0.0  4.5 0.0 0.0

Bonds

1 1 1 2
2 1 2 3
3 1 3 4
"""

_TOY_MAH = """LAMMPS toy MAH-like 3-atom molecule

3 atoms
2 bonds

2 atom types
1 bond types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 12.0
2 16.0

Pair Coeffs

1 0.07 3.60
2 0.15 3.10

Bond Coeffs

1 400.0 1.35

Atoms

1 1 2 -0.5  0.0 0.0 0.0
2 1 1  0.0  1.2 0.0 0.0
3 1 2 -0.5  2.4 0.0 0.0

Bonds

1 1 1 2
2 1 2 3
"""


def test_blend_replicator_merges_type_offsets_and_topology(tmp_path, monkeypatch):
    from paaf import blend_replicator as br
    pbs_data = tmp_path / "pbs.data"; pbs_data.write_text(_TOY_PBS)
    mah_data = tmp_path / "mah.data"; mah_data.write_text(_TOY_MAH)

    # Fake packmol: write a PDB with 5 * 4 + 3 * 3 = 29 atoms whose
    # coordinates are simple grid translations of the input xyz.
    def _fake_pack(pdb_files, counts, out_pdb, box_edges, tolerance, seed,
                   packmol_path=None):
        # Just concatenate input atoms shifted by index — enough for parsing.
        lines = ["REMARK fake packmol output"]
        serial = 1
        for pdb, n in zip(pdb_files, counts):
            coords = []
            for line in pdb.read_text().splitlines():
                if line.startswith("HETATM"):
                    parts = line.split()
                    # Grab the three float-with-dot tokens
                    floats = [float(t) for t in parts if "." in t]
                    if len(floats) >= 3:
                        coords.append((floats[0], floats[1], floats[2]))
            for chain_i in range(n):
                shift = chain_i * 5.0
                for x, y, z in coords:
                    lines.append(
                        f"HETATM{serial:5d}  C   MOL     1    "
                        f"{x + shift:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
                    )
                    serial += 1
        lines.append("END")
        out_pdb.write_text("\n".join(lines) + "\n")
        return True

    monkeypatch.setattr(br, "_build_blend_packmol", _fake_pack)

    components = [
        br.BlendComponent(name="PBS", data_file=pbs_data, count=5),
        br.BlendComponent(name="MAH", data_file=mah_data, count=3),
    ]
    out = tmp_path / "packed_blend.data"
    br.replicate_blend(components, box_edges=(30.0, 30.0, 30.0),
                       out_data_file=out)
    text = out.read_text()

    # 5*4 + 3*3 = 29 atoms
    assert "29 atoms" in text
    # 5*3 + 3*2 = 21 bonds
    assert "21 bonds" in text
    # Merged atom types = 1 (PBS) + 2 (MAH) = 3
    assert "3 atom types" in text
    # Bond types = 1 + 1 = 2
    assert "2 bond types" in text
    # PBS atom-type 1 stays as 1; MAH atom-types 1 & 2 become 2 & 3 after offset
    # (Masses section must have all three)
    lines = text.splitlines()
    mass_idx = lines.index("Masses")
    mass_block = [l.strip() for l in lines[mass_idx + 2:mass_idx + 5]]
    assert mass_block[0].startswith("1 12.0")   # PBS carbon (offset 0)
    assert mass_block[1].startswith("2 12.0")   # MAH carbon (offset +1)
    assert mass_block[2].startswith("3 16.0")   # MAH oxygen (offset +1)


def test_cumulative_offsets_are_correct():
    from paaf.blend_replicator import _cumulative_offsets, BlendComponent
    from pathlib import Path
    a = BlendComponent(name="A", data_file=Path("a"), count=1)
    a.ntypes = {"atom": 3, "bond": 2, "angle": 4, "dihedral": 5, "improper": 1}
    b = BlendComponent(name="B", data_file=Path("b"), count=1)
    b.ntypes = {"atom": 5, "bond": 3, "angle": 6, "dihedral": 7, "improper": 2}
    off = _cumulative_offsets([a, b])
    assert off[0] == {"atom": 0, "bond": 0, "angle": 0, "dihedral": 0, "improper": 0}
    assert off[1] == {"atom": 3, "bond": 2, "angle": 4, "dihedral": 5, "improper": 1}
