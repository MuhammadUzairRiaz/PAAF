"""Materials-Studio-style system builders.

Modules:
    amorphous  — pack multiple molecules into a periodic box at a target density
                 (Amorphous Cell / Blends analogue).
    fragments  — Avogadro-style SMILES library of common functional groups + rings.
"""
from .amorphous import pack_cell, PackSpec       # noqa: F401
from .fragments import list_fragments, get_fragment, FRAGMENTS  # noqa: F401
