"""Reaction schemes defined as SMILES + atom-map numbers.

Instead of uploading pre-built 3D reactant/product structures, the user writes
the reaction as an equation:

    reactant 1 :  CC(=O)[OH:1]
    reactant 2 :  [C:2]CO
    product    :  CC(=O)O[C:2]

The bracketed ``[X:n]`` tokens are **atom-map numbers**. The same number on
both sides means "this is the same atom before and after the reaction". They
give an exact anchor at the reaction centre; :func:`paaf.reaction.complete_mapping`
then extends that anchor over the spectator atoms by graph propagation, and
:func:`paaf.reaction.extract_template` diffs the two graphs into the same
:class:`~paaf.reaction.ReactionTemplate` the crosslinking engine already
consumes.

Because the anchor is explicit, this path does not need the chemistry-specific
correction hacks the purely-geometric mapper required (e.g. the hard-coded
"ENR-MAH-PBS epoxide vs acid OH" disambiguation) — the user simply maps the
epoxide oxygen.

Validation error codes (surfaced verbatim in the GUI):

===========  ==========================================================
``E-SMI-04``  a SMILES string could not be parsed
``E-MAP-01``  a map number appears on one side but not the other
``E-MAP-02``  a map number is used twice on the same side
``E-MAP-03``  no atom-map numbers at all
``W-BND-02``  the maps parse, but no bond is formed
``W-STE-03``  an unmapped stereocentre — chirality is not tracked
===========  ==========================================================
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .logging_utils import get_logger
from .structure import Atom, Molecule

log = get_logger(__name__)

__all__ = [
    "Issue",
    "SchemeResult",
    "parse_atom_maps",
    "available",
    "validate_scheme",
    "build_template",
]


# ``[C:1]``, ``[OH:12]``, ``[C@@H:3]``, ``[N+:2]`` … capture the map number.
_MAP_TOKEN = re.compile(r"\[([^\[\]:]+):(\d+)\]")
# A stereocentre marker inside a bracket atom.
_STEREO = re.compile(r"\[[^\[\]]*@[^\[\]]*\]")


def available() -> bool:
    """True when RDKit is importable (needed to actually build the template)."""
    try:
        import rdkit  # noqa: F401
        return True
    except Exception:
        return False


# ===================================================================== issues
@dataclass
class Issue:
    """One validation finding. ``level`` is ``"error"`` or ``"warning"``."""
    code: str
    level: str
    message: str
    # Which input the issue belongs to: ("reactant", i) or ("product", None)
    where: str = ""
    index: Optional[int] = None

    @property
    def is_error(self) -> bool:
        return self.level == "error"

    def as_dict(self) -> dict:
        return {"code": self.code, "level": self.level, "message": self.message,
                "where": self.where, "index": self.index}


@dataclass
class SchemeResult:
    """Outcome of validating (and optionally building) a reaction scheme."""
    ok: bool = False
    issues: List[Issue] = field(default_factory=list)
    # map number -> (present on reactants?, present on product?)
    map_state: Dict[int, Tuple[bool, bool]] = field(default_factory=dict)
    template: Optional[object] = None          # ReactionTemplate when built
    n_bonds_formed: int = 0
    n_bonds_broken: int = 0
    n_atoms_deleted: int = 0
    n_atoms_added: int = 0
    bonds_formed: List[Tuple[int, int]] = field(default_factory=list)
    bonds_broken: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.is_error]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if not i.is_error]

    def summary(self) -> str:
        if not self.ok:
            return f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        return (f"{self.n_bonds_formed} bond(s) formed · "
                f"{self.n_bonds_broken} broken · "
                f"{self.n_atoms_deleted} atom(s) deleted")


# ================================================================ map parsing
def parse_atom_maps(smiles: str) -> Dict[int, int]:
    """Return ``{map_number: occurrence_order}`` for one SMILES string.

    Dependency-free (pure regex), so the GUI can highlight and count map
    tokens live while the user types, without RDKit.

    >>> parse_atom_maps("CC(=O)[OH:1]")
    {1: 0}
    >>> sorted(parse_atom_maps("[O:2]C[C:3](=O)O"))
    [2, 3]
    """
    out: Dict[int, int] = {}
    for order, m in enumerate(_MAP_TOKEN.finditer(smiles or "")):
        num = int(m.group(2))
        out.setdefault(num, order)
    return out


def _duplicate_maps(smiles: str) -> List[int]:
    """Map numbers used more than once inside a single SMILES string."""
    seen: Dict[int, int] = {}
    for m in _MAP_TOKEN.finditer(smiles or ""):
        num = int(m.group(2))
        seen[num] = seen.get(num, 0) + 1
    return sorted(n for n, c in seen.items() if c > 1)


def _has_unmapped_stereocentre(smiles: str) -> bool:
    """True if a stereo-marked atom carries no atom-map number."""
    for m in _STEREO.finditer(smiles or ""):
        if ":" not in m.group(0):
            return True
    return False


# ============================================================ RDKit → Molecule
def _rdkit_mol(smiles: str, add_hs: bool = True):
    """Parse one SMILES into an RDKit mol with explicit hydrogens."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if add_hs:
        mol = Chem.AddHs(mol)
    return mol


def _to_internal(rdmol, name: str = "MOL", offset: int = 0
                 ) -> Tuple[Molecule, Dict[int, int]]:
    """Convert an RDKit mol to a PAAF :class:`Molecule`.

    Returns ``(molecule, {map_number: internal_index})``. Indices are shifted by
    ``offset`` so several reactants can be concatenated into one graph.
    Coordinates are zeros — the template is a pure graph diff, so geometry is
    not needed (and would only add an embedding failure mode).
    """
    atoms: List[Atom] = []
    maps: Dict[int, int] = {}
    for a in rdmol.GetAtoms():
        i = a.GetIdx() + offset
        atoms.append(Atom(index=i, element=a.GetSymbol(), xyz=np.zeros(3),
                          name=f"{a.GetSymbol()}{i + 1}"))
        n = a.GetAtomMapNum()
        if n:
            maps[n] = i
    bonds = [(b.GetBeginAtomIdx() + offset, b.GetEndAtomIdx() + offset,
              float(b.GetBondTypeAsDouble())) for b in rdmol.GetBonds()]
    return Molecule(atoms=atoms, bonds=bonds, name=name), maps


def _combine(smiles_list: Sequence[str], side: str = "R"
             ) -> Tuple[Molecule, Dict[int, int]]:
    """Concatenate several SMILES into one disconnected graph.

    Used for BOTH sides: a reaction may consume several molecules and produce
    several (an esterification yields the ester *and* water). Atom indices are
    offset per molecule so the combined graph is unambiguous.
    """
    all_atoms: List[Atom] = []
    all_bonds: List[Tuple[int, int, float]] = []
    all_maps: Dict[int, int] = {}
    offset = 0
    for k, smi in enumerate(smiles_list):
        rdmol = _rdkit_mol(smi)
        if rdmol is None:
            raise ValueError(f"{side}{k + 1}")
        mol, maps = _to_internal(rdmol, name=f"{side}{k + 1}", offset=offset)
        all_atoms.extend(mol.atoms)
        all_bonds.extend(mol.bonds)
        all_maps.update(maps)
        offset += len(mol.atoms)
    return Molecule(atoms=all_atoms, bonds=all_bonds,
                    name="reactants" if side == "R" else "products"), all_maps


# ================================================================== validation
def _static_checks(reactants: Sequence[str], products: Sequence[str]
                   ) -> Tuple[List[Issue], Dict[int, Tuple[bool, bool]]]:
    """RDKit-free checks: map pairing, duplicates, stereo. Always runnable."""
    # A bare str is a valid Sequence[str] but iterates CHARACTERS, which would
    # silently check every letter as if it were a molecule. Normalise first.
    if isinstance(reactants, str):
        reactants = [reactants]
    if isinstance(products, str):
        products = [products]
    issues: List[Issue] = []

    r_maps: Dict[int, int] = {}
    for k, smi in enumerate(reactants):
        for dup in _duplicate_maps(smi):
            issues.append(Issue(
                "E-MAP-02", "error",
                f"Map number :{dup} is used more than once in reactant {k + 1}. "
                "Every number must be unique per side.",
                where="reactant", index=k))
        r_maps.update(parse_atom_maps(smi))
        if _has_unmapped_stereocentre(smi):
            issues.append(Issue(
                "W-STE-03", "warning",
                f"Stereocentre in reactant {k + 1} is unmapped, so chirality is "
                "not tracked through the reaction. Map it if the product must "
                "stay isotactic.",
                where="reactant", index=k))

    p_maps: Dict[int, int] = {}
    seen_p: Dict[int, int] = {}
    for k, smi in enumerate(products):
        for dup in _duplicate_maps(smi):
            issues.append(Issue(
                "E-MAP-02", "error",
                f"Map number :{dup} is used more than once in product {k + 1}.",
                where="product", index=k))
        for n in parse_atom_maps(smi):
            if n in seen_p:
                issues.append(Issue(
                    "E-MAP-02", "error",
                    f"Map number :{n} appears in product {seen_p[n] + 1} and "
                    f"product {k + 1}. Each number may only be used once per side.",
                    where="product", index=k))
            seen_p[n] = k
        p_maps.update(parse_atom_maps(smi))
        if _has_unmapped_stereocentre(smi):
            issues.append(Issue(
                "W-STE-03", "warning",
                f"Stereocentre in product {k + 1} is unmapped, so chirality is "
                "not tracked through the reaction.",
                where="product", index=k))

    if not r_maps and not p_maps:
        issues.append(Issue(
            "E-MAP-03", "error",
            "No atom-map numbers found. Tag the atoms whose bonds change, "
            "e.g. write [C:1] and [O:2], using the same number on both sides.",
            where="product"))

    # Pairing state per map number.
    state: Dict[int, Tuple[bool, bool]] = {}
    for n in sorted(set(r_maps) | set(p_maps)):
        state[n] = (n in r_maps, n in p_maps)

    # A map present on the reactant but absent from the product is a LEAVING
    # GROUP — legitimate and common (that's how water leaves an esterification),
    # so it is not an error. The reverse (product-only) has no anchor and is.
    for n, (in_r, in_p) in state.items():
        if in_p and not in_r:
            issues.append(Issue(
                "E-MAP-01", "error",
                f"Map number :{n} appears on the product but not on any "
                "reactant, so PAAF cannot tell where that atom came from. "
                f"Add [X:{n}] to the correct reactant, or remove it from the products.",
                where="product"))
    return issues, state


def validate_scheme(reactants: Sequence[str], products: Sequence[str],
                    name: str = "reaction") -> SchemeResult:
    """Validate a reaction scheme and, if valid, build its template.

    Runs the RDKit-free structural checks first so the GUI always gets useful
    feedback; only parses with RDKit (and builds the template) when those pass
    and RDKit is installed.
    """
    reactants = [s.strip() for s in reactants if (s or "").strip()]
    if isinstance(products, str):          # tolerate the old single-product call
        products = [products]
    products = [s.strip() for s in products if (s or "").strip()]

    res = SchemeResult()
    if not reactants:
        res.issues.append(Issue("E-SMI-04", "error",
                                "Add at least one reactant molecule.",
                                where="reactant", index=0))
    if not products:
        res.issues.append(Issue("E-SMI-04", "error",
                                "Enter at least one product SMILES.",
                                where="product"))
    if res.issues:
        return res

    issues, state = _static_checks(reactants, products)
    res.issues.extend(issues)
    res.map_state = state

    if not available():
        res.issues.append(Issue(
            "E-SMI-04", "error",
            "RDKit is required to parse SMILES and build the reaction "
            "template. Install it with:  conda install -c conda-forge rdkit",
            where="product"))
        return res

    # ---- RDKit parse (per input, so the error names the offending field)
    from rdkit import Chem
    for k, smi in enumerate(reactants):
        if Chem.MolFromSmiles(smi) is None:
            res.issues.append(Issue(
                "E-SMI-04", "error",
                f"Invalid SMILES in reactant {k + 1} — RDKit could not parse "
                f"{smi!r}.", where="reactant", index=k))
    for k, smi in enumerate(products):
        if Chem.MolFromSmiles(smi) is None:
            res.issues.append(Issue(
                "E-SMI-04", "error",
                f"Invalid SMILES in product {k + 1} — RDKit could not parse "
                f"{smi!r}.", where="product", index=k))
    if res.errors:
        return res

    # ---- build graphs + seed the mapping from the atom-map anchors
    r_mol, r_maps = _combine(reactants, side="R")
    p_mol, p_maps = _combine(products, side="P")

    seed = {r_maps[n]: p_maps[n] for n in sorted(set(r_maps) & set(p_maps))}
    log.info("Reaction %s: %d reactant atoms, %d product atoms, %d map anchors",
             name, len(r_mol.atoms), len(p_mol.atoms), len(seed))

    from .reaction import complete_mapping, extract_template
    mapping, evidence = complete_mapping(r_mol, p_mol, seed=seed)
    tmpl = extract_template(
        r_mol, p_mol, mapping, evidence, name=name,
        reactant=" . ".join(reactants), product=" . ".join(products),
    )

    res.template = tmpl
    res.bonds_formed = [tuple(b) for b in tmpl.created_bonds]
    res.bonds_broken = [tuple(b) for b in tmpl.deleted_bonds]
    res.n_bonds_formed = len(tmpl.created_bonds)
    res.n_bonds_broken = len(tmpl.deleted_bonds)
    res.n_atoms_deleted = len(tmpl.deleted_atoms)
    res.n_atoms_added = len(tmpl.new_product_atoms)

    if res.n_bonds_formed == 0:
        res.issues.append(Issue(
            "W-BND-02", "warning",
            "No bonds formed with the current maps. Check that at least two "
            "surviving map numbers become adjacent in the product.",
            where="product"))

    res.ok = not res.errors
    return res


def build_template(reactants: Sequence[str], products: Sequence[str],
                   name: str = "reaction"):
    """Validate and return the :class:`~paaf.reaction.ReactionTemplate`.

    Raises :class:`ValueError` listing the error codes if validation fails.
    """
    res = validate_scheme(reactants, products, name=name)
    if not res.ok or res.template is None:
        detail = "; ".join(f"{i.code}: {i.message}" for i in res.errors)
        raise ValueError(f"Reaction scheme is not valid — {detail}")
    return res.template
