"""Solvation — surround a solute with solvent molecules to a target density.

Delegates to :func:`pack_cell` so the packmol / mbuild / grid backends are
reused. A curated set of solvents is shipped as SMILES so the user doesn't
need to supply a solvent structure explicitly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..logging_utils import get_logger
from ..structure import Molecule, load_smiles
from .amorphous import PackSpec, pack_cell

log = get_logger(__name__)


SOLVENTS = {
    "water":        {"smiles": "O",                    "density": 997.0,  "name": "H2O"},
    "methanol":     {"smiles": "CO",                   "density": 792.0,  "name": "MeOH"},
    "ethanol":      {"smiles": "CCO",                  "density": 789.0,  "name": "EtOH"},
    "acetone":      {"smiles": "CC(=O)C",              "density": 784.0,  "name": "acetone"},
    "chloroform":   {"smiles": "ClC(Cl)Cl",            "density": 1489.0, "name": "CHCl3"},
    "dcm":          {"smiles": "ClCCl",                "density": 1330.0, "name": "CH2Cl2"},
    "dmso":         {"smiles": "CS(=O)C",              "density": 1100.0, "name": "DMSO"},
    "thf":          {"smiles": "C1CCOC1",              "density": 889.0,  "name": "THF"},
    "toluene":      {"smiles": "Cc1ccccc1",            "density": 866.0,  "name": "toluene"},
    "benzene":      {"smiles": "c1ccccc1",             "density": 876.0,  "name": "benzene"},
    "cyclohexane":  {"smiles": "C1CCCCC1",             "density": 779.0,  "name": "cyclohexane"},
    "hexane":       {"smiles": "CCCCCC",               "density": 655.0,  "name": "hexane"},
    "ethyl_acetate":{"smiles": "CCOC(=O)C",            "density": 902.0,  "name": "EtOAc"},
    "dmf":          {"smiles": "CN(C)C=O",             "density": 944.0,  "name": "DMF"},
}


def solvate(
    solute: Molecule | str | Path,
    solvent: str = "water",
    n_solvent: int = 500,
    density_kg_m3: Optional[float] = None,
    box_ang: Optional[float] = None,
    out_path: Optional[str | Path] = None,
    seed: int = 12345,
):
    """Add `n_solvent` copies of `solvent` around `solute` in a periodic box.

    - `solute` may be a Molecule or a file path.
    - `solvent` is a key from :data:`SOLVENTS` (e.g. ``"water"``).
    - Either provide the target `density_kg_m3` (default: solvent density) or
      `box_ang`.

    Returns ``(packed_molecule, box_side_ang)``.
    """
    if solvent not in SOLVENTS:
        raise KeyError(f"Unknown solvent {solvent!r}. Available: {sorted(SOLVENTS)}")
    s = SOLVENTS[solvent]
    solvent_mol = load_smiles(s["smiles"], name=s["name"])
    from ..structure import load_structure
    if isinstance(solute, (str, Path)):
        solute_mol = load_structure(solute)
    else:
        solute_mol = solute
    specs = [
        PackSpec(molecule=solute_mol, count=1, name=solute_mol.name),
        PackSpec(molecule=solvent_mol, count=n_solvent, name=s["name"]),
    ]
    packed, box_shape = pack_cell(
        specs,
        density_kg_m3=density_kg_m3 or s["density"],
        box_ang=box_ang,
        out_path=out_path,
        seed=seed,
    )
    # Backward compatibility: solvate() historically returned (Molecule, float).
    return packed, box_shape.a
