"""Layers / interface / trilayer builder.

Materials-Studio-style layered system builder. Given a list of layer sources
(each a Molecule, an xyz/pdb/mol2 file, or a crystal preset name), stack them
along the +Z axis with configurable gap sizes between layers.

Typical uses::

    # Polymer / graphene / polymer trilayer for tribology or friction studies
    build_layers([
        LayerSpec(file="pbs_slab.pdb", name="PBS_bottom"),
        LayerSpec(preset="graphene", supercell=(6, 6, 1), name="graphene"),
        LayerSpec(file="pbs_slab.pdb", name="PBS_top"),
    ], gap_ang=4.0)

    # Substrate + polymer melt for adhesion studies
    build_layers([
        LayerSpec(preset="Cu",  supercell=(6, 6, 3), name="Cu_substrate"),
        LayerSpec(file="polymer_melt.pdb",           name="melt"),
    ], gap_ang=3.0)

    # Simple bilayer (interface)
    build_layers([a, b], gap_ang=2.5)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..logging_utils import get_logger
from ..structure import Atom, Molecule, load_structure, write

log = get_logger(__name__)


@dataclass
class LayerSpec:
    """One layer in the stack.

    Provide exactly one of ``file``, ``molecule`` or ``preset``.
    """
    file: Optional[str] = None
    molecule: Optional[Molecule] = None
    preset: Optional[str] = None
    supercell: Tuple[int, int, int] = (1, 1, 1)   # only used with preset
    name: str = ""
    lateral_offset: Tuple[float, float] = (0.0, 0.0)   # shift in x, y (Å)
    thickness_override: Optional[float] = None
    # Optional gap that comes *before* this layer (Å). If None, the top-level
    # gap_ang from build_layers() is used.
    gap_before: Optional[float] = None


def _resolve(spec: LayerSpec) -> Molecule:
    if spec.molecule is not None:
        return spec.molecule
    if spec.file:
        mol = load_structure(spec.file, name=spec.name or Path(spec.file).stem)
        return mol
    if spec.preset:
        from .crystal import build_preset
        return build_preset(spec.preset, *spec.supercell)
    raise ValueError("LayerSpec needs file, molecule, or preset")


def _bounds(mol: Molecule) -> Tuple[np.ndarray, np.ndarray]:
    coords = mol.coords()
    if len(coords) == 0:
        return np.zeros(3), np.zeros(3)
    return coords.min(axis=0), coords.max(axis=0)


def build_layers(
    layers: Sequence[LayerSpec],
    gap_ang: float = 3.0,
    center_xy: bool = True,
    name: str = "layers",
    out_path: Optional[str | Path] = None,
) -> Tuple[Molecule, Tuple[float, float, float]]:
    """Stack `layers` along the Z axis.

    Each layer is translated so its lowest atom sits ``gap_ang`` above the
    highest atom of the previous layer (a layer may override the gap via
    ``LayerSpec.gap_before``). The lateral (xy) offset per layer is applied
    on top of that translation. If ``center_xy`` is true, all layers are
    centered on the largest layer's xy footprint.

    Returns ``(stacked_molecule, box_xyz)`` where ``box_xyz`` is the (X, Y, Z)
    extent that comfortably contains the stack (min → max span + a small
    padding on Z equal to ``gap_ang``).
    """
    if not layers:
        raise ValueError("At least one layer is required")

    resolved: List[Molecule] = [_resolve(s) for s in layers]
    if len(resolved) == 1:
        log.warning("Only one layer supplied — output is just that layer.")

    # Determine common xy footprint (max extent) for centering
    max_x = max((_bounds(m)[1][0] - _bounds(m)[0][0]) for m in resolved)
    max_y = max((_bounds(m)[1][1] - _bounds(m)[0][1]) for m in resolved)

    all_atoms: List[Atom] = []
    all_bonds: List[Tuple[int, int, float]] = []
    z_cursor = 0.0
    offset = 0
    log.info("Stacking %d layer(s), default gap = %g Å", len(layers), gap_ang)

    for i, (spec, mol) in enumerate(zip(layers, resolved)):
        lo, hi = _bounds(mol)
        layer_thickness = float(hi[2] - lo[2])
        if spec.thickness_override is not None:
            layer_thickness = spec.thickness_override
        gap = spec.gap_before if spec.gap_before is not None else gap_ang
        if i == 0:
            gap = 0.0

        # Translate so the layer's bottom sits at z_cursor + gap.
        shift = np.zeros(3)
        shift[2] = (z_cursor + gap) - lo[2]

        if center_xy:
            shift[0] = -lo[0] + (max_x - (hi[0] - lo[0])) / 2 + spec.lateral_offset[0]
            shift[1] = -lo[1] + (max_y - (hi[1] - lo[1])) / 2 + spec.lateral_offset[1]
        else:
            shift[0] = spec.lateral_offset[0]
            shift[1] = spec.lateral_offset[1]

        for a in mol.atoms:
            all_atoms.append(Atom(
                index=a.index + offset,
                element=a.element,
                xyz=a.xyz + shift,
                name=a.name, charge=a.charge, ff_type=a.ff_type,
            ))
        for i0, j0, o in mol.bonds:
            all_bonds.append((i0 + offset, j0 + offset, o))
        offset += len(mol.atoms)

        z_cursor = z_cursor + gap + layer_thickness
        log.info(
            "  layer %d (%s): %d atoms, gap_before=%.2f Å, thickness=%.2f Å, z_top=%.2f Å",
            i, spec.name or f"layer_{i}", len(mol.atoms), gap, layer_thickness, z_cursor,
        )

    # Renumber to 0..N-1
    for k, a in enumerate(all_atoms):
        a.index = k

    stacked = Molecule(atoms=all_atoms, bonds=all_bonds, name=name)
    box = (max_x, max_y, z_cursor + gap_ang)   # add gap padding above top layer
    log.info("Stack complete: %d atoms, box = %.2f x %.2f x %.2f Å",
             len(all_atoms), *box)

    if out_path:
        write(stacked, out_path)
        log.info("Wrote layered system -> %s", out_path)
    return stacked, box
