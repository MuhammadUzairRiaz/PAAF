"""Tests for exporting a reaction as typed LAMMPS .data files.

The 3D-embedding path needs RDKit and the typing path needs dl_field, so the
heavy tests skip when those are absent. The dataclass/plumbing behaviour is
always checked.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from paaf.reaction_export import (
    ExportedSide, ReactionExport, available, build_3d_side,
)


def test_result_dataclasses_summarise_sensibly():
    exp = ReactionExport(name="esterification", folder=Path("/tmp/x"))
    exp.reactant = ExportedSide("reactant", n_molecules=2, n_atoms=17)
    exp.product = ExportedSide("product", n_molecules=2, n_atoms=17)
    assert "esterification" in exp.summary()
    assert "17" in exp.summary()
    # Untyped by default — the summary must say so rather than imply success.
    assert not exp.typed
    assert "3D structures only" in exp.summary()


@pytest.mark.skipif(not available(), reason="RDKit not installed")
def test_build_3d_side_embeds_and_separates_molecules():
    """Two reactants must get real coordinates and must not overlap."""
    import numpy as np

    mol, maps = build_3d_side(["CC(=O)[OH:1]", "[C:2]CO"], label="reactant")
    coords = mol.coords()

    # Real 3D coordinates, not the zeros the graph-only path uses.
    assert coords.shape[0] == len(mol.atoms)
    assert np.abs(coords).max() > 0.1

    # Both atom maps survived the embed.
    assert set(maps) == {1, 2}
    assert all(0 <= i < len(mol.atoms) for i in maps.values())

    # The two species are separated along x, not sitting on top of each other.
    assert coords[:, 0].max() - coords[:, 0].min() > 3.0


@pytest.mark.skipif(not available(), reason="RDKit not installed")
def test_build_3d_side_handles_a_single_molecule():
    mol, maps = build_3d_side(["O"], label="product")
    assert len(mol.atoms) == 3          # water, with explicit hydrogens
    assert maps == {}


@pytest.mark.skipif(not available(), reason="RDKit not installed")
def test_export_writes_structures_even_without_dlfield(tmp_path):
    """Typing may be unavailable; the 3D structures must still be written."""
    from paaf.reaction_export import export_reaction

    exp = export_reaction(
        name="esterification",
        reactants=["CC(=O)[OH:1]", "[C:2]CO"],
        products=["CC(=O)O[C:2]", "O"],
        out_dir=tmp_path,
        run_typing=False,            # skip dl_field
        optimize=False,              # skip OpenBabel
    )
    assert exp.folder.exists()
    assert (exp.folder / "reactant.xyz").exists()
    assert (exp.folder / "product.xyz").exists()
    assert exp.reactant.n_molecules == 2
    assert exp.product.n_molecules == 2
    assert not exp.typed             # honest about what did not happen
    assert exp.messages              # and says why
