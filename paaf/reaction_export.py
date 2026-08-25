"""Export a validated reaction scheme as typed LAMMPS ``.data`` files.

The Reaction-scheme tab produces a *graph* template (which bonds form and
break). The reactive-MD engine — ``reaction_engine_local.py`` and its
``reaction_mapper.py`` — needs something stronger: a folder per reaction
holding fully typed structures::

    <output>/<project>/reactions/<name>/
        reactant.xyz      3D, energy-minimised
        product.xyz
        reactant.data     LAMMPS: atom types, charges, bonds/angles/dihedrals
        product.data
        reaction.json     the graph template for reference

This module bridges that gap. For each side it:

1. **Embeds 3D coordinates** with RDKit (ETKDG), then MMFF-minimises. Several
   molecules on one side are embedded separately and translated apart so they
   do not overlap — an esterification's reactants are two distinct species, and
   its products are the ester *plus* water.
2. **Optimises** with the same OpenBabel routine the Optimize step uses, so the
   geometry matches the rest of the pipeline.
3. **Types and exports** through ``dl_field``, which writes LAMMPS ``.data``
   directly, honouring the force field chosen on the Force-field page.

Atom-map numbers are preserved through the whole path, so the exported files
line up with the template atom-for-atom.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .logging_utils import get_logger
from .structure import Atom, Molecule

log = get_logger(__name__)

__all__ = ["ExportedSide", "ReactionExport", "export_reaction",
           "build_3d_side", "available"]


def available() -> bool:
    """True when RDKit is importable (required for 3D embedding)."""
    try:
        import rdkit  # noqa: F401
        return True
    except Exception:
        return False


# ===================================================================== results
@dataclass
class ExportedSide:
    """One side (reactants or products) after 3D build + typing."""
    label: str                       # "reactant" | "product"
    n_molecules: int = 0
    n_atoms: int = 0
    structure: Optional[Path] = None   # the .xyz written
    data_file: Optional[Path] = None   # the LAMMPS .data, when typing ran
    map_indices: Dict[int, int] = field(default_factory=dict)
    note: str = ""


@dataclass
class ReactionExport:
    name: str
    folder: Path
    reactant: Optional[ExportedSide] = None
    product: Optional[ExportedSide] = None
    template_file: Optional[Path] = None
    typed: bool = False
    messages: List[str] = field(default_factory=list)

    def summary(self) -> str:
        r = self.reactant.n_atoms if self.reactant else 0
        p = self.product.n_atoms if self.product else 0
        kind = "typed LAMMPS .data" if self.typed else "3D structures only"
        return f"{self.name}: {r} → {p} atoms, {kind}, in {self.folder}"


# ================================================================== 3D build
def _embed_one(smiles: str, seed: int = 0xF00D):
    """SMILES → RDKit mol with 3D coordinates and explicit hydrogens."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {smiles!r}")
    # A mapped atom is written in brackets — [C:2] — and SMILES brackets
    # mean "hydrogens fully specified", so [C:2] parses as a carbon with NO
    # hydrogens. The esterification example built an ethanol missing its
    # whole methyl group, and DL_FIELD refused the one-neighbour carbon
    # ("atype = aliphatic"). A map number tags an atom, it does not change
    # its chemistry: restore implicit hydrogens on mapped atoms that carry
    # no explicit H count. [CH2:2] etc. still means exactly what it says.
    changed = False
    for a in mol.GetAtoms():
        if a.GetAtomMapNum() and a.GetNoImplicit() and \
                a.GetNumExplicitHs() == 0:
            a.SetNoImplicit(False)
            # RDKit balances the missing valence with radical electrons
            # ([C:2] parses with 3 of them); they satisfy the valence model,
            # so implicit Hs stay at zero unless they are cleared as well.
            a.SetNumRadicalElectrons(0)
            changed = True
    if changed:
        Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        # Fall back to plain distance-geometry for awkward fragments.
        if AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True) != 0:
            raise RuntimeError(f"3D embedding failed for {smiles!r}")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
        except Exception:
            log.warning("Force-field cleanup failed for %s; using raw embed.", smiles)
    return mol


def _extent(coords: np.ndarray) -> float:
    if len(coords) == 0:
        return 0.0
    return float(np.max(coords.max(axis=0) - coords.min(axis=0)))


def build_3d_side(smiles_list: Sequence[str], label: str = "reactant",
                  gap: float = 4.0, seed: int = 0xF00D):
    """Build one side of a reaction as a single :class:`Molecule`.

    Each molecule is embedded independently, then translated along +x so the
    species sit side by side with ``gap`` Å of clearance instead of on top of
    one another. Returns ``(molecule, {map_number: atom_index})``.
    """
    atoms: List[Atom] = []
    bonds: List[tuple] = []
    maps: Dict[int, int] = {}
    offset = 0
    x_cursor = 0.0

    for k, smi in enumerate(smiles_list):
        rdmol = _embed_one(smi, seed=seed + k)
        conf = rdmol.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i))
                           for i in range(rdmol.GetNumAtoms())], dtype=float)
        # Sit this molecule to the right of the previous one.
        coords[:, 0] += x_cursor - coords[:, 0].min()
        x_cursor = coords[:, 0].max() + gap

        for a in rdmol.GetAtoms():
            i = a.GetIdx()
            gi = i + offset
            atoms.append(Atom(index=gi, element=a.GetSymbol(),
                              xyz=coords[i], name=f"{a.GetSymbol()}{gi + 1}"))
            n = a.GetAtomMapNum()
            if n:
                maps[n] = gi
        for b in rdmol.GetBonds():
            bonds.append((b.GetBeginAtomIdx() + offset,
                          b.GetEndAtomIdx() + offset,
                          float(b.GetBondTypeAsDouble())))
        offset += rdmol.GetNumAtoms()

    mol = Molecule(atoms=atoms, bonds=bonds, name=label)
    log.info("Built %s: %d molecule(s), %d atoms, %d mapped",
             label, len(smiles_list), len(atoms), len(maps))
    return mol, maps


# ================================================================== typing
def _type_with_dlfield(structure: Path, work_dir: Path, ff_key: str,
                       dl_field_dir: Optional[Path],
                       emit: Callable[[str], None]) -> Optional[Path]:
    """Run dl_field on one structure and return the LAMMPS ``.data`` it wrote."""
    from .dlfield_runner import run_dlfield
    from .cell.cell_export import _dlfield_root

    # The GUI field holds .../dl_f_4.13/lib; the executable lives one level
    # up. Without this the reaction export failed with "dl_field executable
    # not found" before writing so much as a control file, while the same
    # installation typed amorphous cells happily.
    dl_field_dir = _dlfield_root(dl_field_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    emit(f"    dl_field ({ff_key}) on {structure.name} …")
    result = run_dlfield(
        structure_path=structure,
        ff_key=ff_key,
        work_dir=work_dir,
        dl_field_dir=dl_field_dir,
        output_engine="lammps",
    )
    # DLFieldResult's field is ``return_code``, not ``returncode`` — the old
    # getattr default of 1 marked every successful run as failed (rc=?).
    rc = getattr(result, "return_code", getattr(result, "returncode", 1))
    if rc != 0:
        emit(f"    dl_field failed (rc={rc})")
        return None

    # dl_field writes into dlf_output1/lammps<N>.data
    candidates = sorted((work_dir / "dlf_output1").glob("lammps*.data"))
    if not candidates:
        candidates = sorted(work_dir.glob("lammps*.data"))
    if not candidates:
        emit("    dl_field produced no lammps*.data")
        return None
    return candidates[0]


# =================================================================== top level
def export_reaction(
    name: str,
    reactants: Sequence[str],
    products: Sequence[str],
    out_dir: str | Path,
    ff_key: str = "opls2005_dl",
    dl_field_dir: Optional[str | Path] = None,
    optimize: bool = True,
    opt_ff: str = "UFF",
    opt_steps: int = 2000,
    run_typing: bool = True,
    template: Optional[object] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> ReactionExport:
    """Build, optimise, type and export one reaction.

    Writes ``reactant.xyz`` / ``product.xyz`` always, and
    ``reactant.data`` / ``product.data`` when ``run_typing`` succeeds.
    """
    emit = progress or (lambda _m: None)
    # Folder-safe name. Example names like "ENR + maleic acid + ENR
    # (bridge)" made a folder with spaces, and DL_FIELD's Fortran parser
    # reads the config path only up to the first space — "User
    # configuration file format not recognise." on a perfectly good xyz.
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "reaction"
    folder = Path(out_dir) / safe
    folder.mkdir(parents=True, exist_ok=True)
    export = ReactionExport(name=name, folder=folder)

    if not available():
        raise RuntimeError(
            "RDKit is required to build 3D structures. "
            "Install it with:  conda install -c conda-forge rdkit")

    from . import structure as structure_mod

    sides = (("reactant", list(reactants)), ("product", list(products)))
    for label, smiles_list in sides:
        emit(f"  {label}: embedding {len(smiles_list)} molecule(s) in 3D …")
        mol, maps = build_3d_side(smiles_list, label=label)

        if optimize:
            emit(f"  {label}: optimising ({opt_ff}) …")
            try:
                from . import optimizer
                optimizer.optimize(mol, ff=opt_ff, steps=opt_steps)
            except Exception as exc:
                emit(f"  {label}: optimisation skipped ({exc})")

        xyz = folder / f"{label}.xyz"
        mol.to_xyz(xyz)
        # A mol2 keeps explicit bond orders, which dl_field perceives better
        # than a bare xyz (no connectivity) for anything with a ring or C=O.
        mol2 = folder / f"{label}.mol2"
        try:
            structure_mod.write(mol, mol2)
        except Exception:
            mol2 = xyz

        side = ExportedSide(
            label=label, n_molecules=len(smiles_list), n_atoms=len(mol.atoms),
            structure=xyz, map_indices=maps,
        )
        emit(f"  {label}: {len(mol.atoms)} atoms -> {xyz.name}")

        if run_typing:
            # XYZ, not mol2 — same lesson as the amorphous cell: DL_FIELD
            # cannot read PAAF's mol2 ("Can't locate any sensible
            # information in config file") and perceives its own bonds from
            # geometry anyway.
            try:
                data = _type_with_dlfield(
                    xyz, folder / f"_typing_{label}", ff_key,
                    Path(dl_field_dir) if dl_field_dir else None, emit)
            except Exception as exc:
                data = None
                emit(f"  {label}: typing failed — {exc}")
            if data is not None:
                dest = folder / f"{label}.data"
                shutil.copy2(data, dest)
                side.data_file = dest
                emit(f"  {label}: typed -> {dest.name}")
            else:
                side.note = "typing did not produce a .data file"

        setattr(export, label, side)

    export.typed = bool(export.reactant and export.reactant.data_file
                        and export.product and export.product.data_file)

    if template is not None:
        tpl = folder / "reaction.json"
        payload = template.to_json() if hasattr(template, "to_json") else dict(template)
        payload["reactants_smiles"] = list(reactants)
        payload["products_smiles"] = list(products)
        tpl.write_text(json.dumps(payload, indent=2) + "\n")
        export.template_file = tpl

    if not export.typed:
        export.messages.append(
            "Structures were written, but LAMMPS .data files were not produced. "
            "Check that dl_field is installed and the DL_FIELD lib path is set "
            "on the Force-field page.")
    emit(export.summary())
    return export
