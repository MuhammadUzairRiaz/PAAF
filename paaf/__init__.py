"""PAAF — Polymer Auto-Assembly Framework
=========================================

An automated polymer builder + reactive-MD + Moltemplate / DL_FIELD /
LAMMPS / GROMACS system generator.

Given a monomer structure (xyz/pdb/mol2/sdf/mol) produced by Avogadro or any
other program, PAAF:

1. Loads and optionally optimizes the monomer with OpenBabel
   (MMFF94 → UFF fallback chain — never crashes on typing failures).
2. Detects head/tail connection atoms (or accepts them from the user).
3. Builds a homopolymer or copolymer chain of the desired length via mbuild.
4. Optimizes the chain and packs N copies into a cubic / orthorhombic /
   triclinic periodic cell (packmol / mbuild / grid backends).
5. Assigns force-field atom types (OPLS-AA/UA, GAFF/GAFF2, TraPPE-UA,
   PCFF, COMPASS, CVFF, DREIDING, CHARMM36, GROMOS 54A7, MARTINI, SDK, ...).
6. Writes Moltemplate .lt files and, on demand, invokes ``moltemplate.sh``
   (for native FFs) or ``dl_field`` (for DL_FIELD-based FFs) to emit
   LAMMPS ``.data``/``.in`` or GROMACS ``.gro``/``.top``/``.mdp`` files.

PAAF also ships an Amorphous cell builder (Theodorou-Suter chain growth, with
copolymer sequences and LAMMPS/GROMACS relaxation), a Materials-Studio-style
System Builder (crystal, surface/slab, nanotube, layers/interface, solvate)
and a reaction learner for iterative crosslinking.

The Python import name is kept as ``paaf`` for backwards
compatibility with earlier releases; the product / brand name is **PAAF**.
"""
from .__version__ import __version__

__product_name__ = "PAAF"
__full_name__ = "Polymer Auto-Assembly Framework"

__all__ = ["__version__", "__product_name__", "__full_name__"]
