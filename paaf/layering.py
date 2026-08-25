"""Layered systems: each component packed into its OWN sub-box.

Blend methodology (typed LAMMPS components in, one consolidated data file
out, force fields merged with per-component type offsets) — but where a
blend mixes every species through one shared box, a layered build gives
each component a slab of its own:

    z ^  +---------------------------+
      |  |        layer 3            |   its own Lx x Ly x Lz
      |  +---------------------------+ ← gap (default 5 Å)
      |  |        layer 2            |
      |  +---------------------------+ ← gap
      |  |        layer 1            |
      +--+---------------------------+--> x

Layers stack along a chosen axis, separated by a configurable gap. Any
layer may instead pin its own origin explicitly, which allows side-by-side
or embedded arrangements, not just a stack. The total cell is cubic or
orthorhombic; it can be given, or computed from the layers.

The heavy lifting (packmol with per-structure ``inside box`` regions,
force-field merging, bond-length validation) is
:func:`paaf.blend_replicator.replicate_blend` with ``regions=...``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .blend_replicator import BlendComponent, replicate_blend
from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["LayerSpec", "LayeringError", "plan_regions", "build_layered_cell"]

_AXES = {"x": 0, "y": 1, "z": 2}


class LayeringError(ValueError):
    """A layout that cannot be built, explained in user terms."""


@dataclass
class LayerSpec:
    """One layer: a typed component plus the sub-box it is packed into."""
    name: str
    data_file: Path                       # typed LAMMPS data (single chain)
    count: int                            # copies packed into this layer
    size: Tuple[float, float, float]      # the layer's own Lx, Ly, Lz (Å)
    #: Explicit lower-corner origin (Å). ``None`` = auto: stacked along the
    #: chosen axis after the previous layer plus the gap, centred on the
    #: other two axes.
    origin: Optional[Tuple[float, float, float]] = None
    forcefield_input: Optional[Path] = None   # lammps.in with styles/coeffs


def plan_regions(
    layers: List[LayerSpec],
    axis: str = "z",
    gap: float = 5.0,
    total_box: Optional[Tuple[float, float, float]] = None,
) -> Tuple[List[Tuple[float, float, float, float, float, float]],
           Tuple[float, float, float]]:
    """Compute each layer's ``(x0,y0,z0,x1,y1,z1)`` region and the cell.

    Auto-placed layers stack along ``axis``, ``gap`` Å apart, centred on the
    two perpendicular axes of the final cell. When ``total_box`` is None the
    cell is sized to hold everything: stacked extent along the axis, maximum
    layer size across it. Explicit origins are honoured verbatim, and any
    region that pokes out of the cell is refused with the numbers shown.
    """
    if not layers:
        raise LayeringError("At least one layer is required.")
    ax = _AXES.get(str(axis).lower())
    if ax is None:
        raise LayeringError(f"Stacking axis must be x, y or z, not {axis!r}.")
    if gap < 0:
        raise LayeringError("The gap between layers cannot be negative.")
    for ly in layers:
        if min(ly.size) <= 0:
            raise LayeringError(
                f"Layer '{ly.name}': all three dimensions must be positive "
                f"(got {ly.size}).")

    # ---- the cell ----------------------------------------------------
    if total_box is None:
        stacked = sum(ly.size[ax] for ly in layers if ly.origin is None)
        n_auto = sum(1 for ly in layers if ly.origin is None)
        stacked += gap * max(0, n_auto - 1)
        cell = [0.0, 0.0, 0.0]
        cell[ax] = stacked
        for other in range(3):
            if other == ax:
                continue
            cell[other] = max(ly.size[other] for ly in layers)
        # Explicit-origin layers can extend the automatic cell.
        for ly in layers:
            if ly.origin is not None:
                for k in range(3):
                    cell[k] = max(cell[k], ly.origin[k] + ly.size[k])
        total_box = (cell[0], cell[1], cell[2])
    if min(total_box) <= 0:
        raise LayeringError(f"The total cell must be positive, got {total_box}.")

    # ---- regions -----------------------------------------------------
    regions: List[Tuple[float, float, float, float, float, float]] = []
    cursor = 0.0
    for ly in layers:
        if ly.origin is not None:
            o = list(ly.origin)
        else:
            # Auto-placed layers anchor at 0 on the two lateral axes — a
            # layered film starts at the cell wall, not centred as a
            # floating island. (Set Lx/Ly equal to the cell edge to span
            # the full cross-section; pin an origin for islands.)
            o = [0.0, 0.0, 0.0]
            o[ax] = cursor
            cursor += ly.size[ax] + gap
        hi = [o[k] + ly.size[k] for k in range(3)]
        for k, axis_name in enumerate("xyz"):
            if o[k] < -1e-9 or hi[k] > total_box[k] + 1e-9:
                raise LayeringError(
                    f"Layer '{ly.name}' spans {o[k]:.1f}–{hi[k]:.1f} Å on "
                    f"{axis_name} but the cell is only "
                    f"{total_box[k]:.1f} Å. Enlarge the cell, shrink the "
                    f"layer, or move its origin.")
        regions.append((o[0], o[1], o[2], hi[0], hi[1], hi[2]))
    return regions, total_box


def build_layered_cell(
    layers: List[LayerSpec],
    out_data_file: Path,
    axis: str = "z",
    gap: float = 5.0,
    total_box: Optional[Tuple[float, float, float]] = None,
    out_input_file: Optional[Path] = None,
    seed: int = 12345,
    tolerance: float = 2.0,
    packmol_path: Optional[str] = None,
    minimise: Optional[object] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """Pack every layer into its own region and write ONE LAMMPS data file.

    Same guarantees as the Blend tool: per-component type offsets, merged
    coefficient sections, refused style conflicts, bond-length validation.
    """
    emit = progress or (lambda _m: None)
    regions, cell = plan_regions(layers, axis=axis, gap=gap,
                                 total_box=total_box)
    for ly, reg in zip(layers, regions):
        emit(f"  layer '{ly.name}': {ly.count} copies in "
             f"[{reg[0]:.1f},{reg[3]:.1f}] x [{reg[1]:.1f},{reg[4]:.1f}] x "
             f"[{reg[2]:.1f},{reg[5]:.1f}] Å")
    emit(f"  total cell: {cell[0]:.1f} x {cell[1]:.1f} x {cell[2]:.1f} Å, "
         f"gap {gap:g} Å along {axis}")

    comps = [BlendComponent(name=ly.name, data_file=Path(ly.data_file),
                            count=int(ly.count),
                            forcefield_input=(Path(ly.forcefield_input)
                                              if ly.forcefield_input else None))
             for ly in layers]
    return replicate_blend(
        comps, cell, Path(out_data_file), out_input_file=out_input_file,
        seed=seed, tolerance=tolerance, packmol_path=packmol_path,
        minimise=minimise, progress=progress, regions=regions)
