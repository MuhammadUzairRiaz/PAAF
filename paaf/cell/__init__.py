"""Materials-Studio-style system builders.

Modules:
    amorphous  — pack multiple molecules into a periodic box at a target density
                 (Amorphous Cell / Blends analogue).
    crystal    — build periodic crystals from a Bravais lattice + basis, plus a
                 curated preset library (SC/BCC/FCC/diamond/NaCl/CsCl/wurtzite/
                 zinc-blende/graphene/graphite/α-quartz).
    surface    — cleave a crystal along a Miller plane (hkl) and add vacuum for
                 slab-model simulations.
    nanotube   — armchair/zigzag/chiral carbon (or generic single-element)
                 nanotube from (n, m).
    solvate    — surround a solute with a solvent grid packed to a target density.
    fragments  — Avogadro-style SMILES library of common functional groups + rings.
"""
from .amorphous import pack_cell, PackSpec       # noqa: F401
from .crystal import (                          # noqa: F401
    build_crystal, build_preset, CRYSTAL_PRESETS, list_presets,
)
from .surface import cleave_surface              # noqa: F401
from .nanotube import build_nanotube             # noqa: F401
from .solvate import solvate                     # noqa: F401
from .fragments import list_fragments, get_fragment, FRAGMENTS  # noqa: F401
from .layers import build_layers, LayerSpec      # noqa: F401
