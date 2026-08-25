"""Molecule / polymer designer.

Turns SMILES (or a name from the built-in library) into a 3D structure that
downstream modules can use as a monomer or as a whole molecule.

Two libraries ship with the tool:

- A small curated dictionary of common polymers with human-readable keys
  (PE, PP, PS, PBS, PLA, ...) used by :func:`get_recipe`.
- A larger 100+ polymer database loaded from
  ``paaf/data/polymer_database.csv`` at import time.  Each entry
  carries SMILES (with ``[*]`` connection points), PID, Tg, density and
  literature references so the GUI Builder tab can present them as a
  filter-able table.

Both libraries are merged so a single :func:`get_recipe` call resolves keys
from either source.

Entry points:

- :func:`build_from_smiles`  — SMILES  -> 3D structure, save to file.
- :func:`build_recipe`       — resolve name -> SMILES -> 3D file.
- :func:`search_library`     — text search across the full DB.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .logging_utils import get_logger
from .structure import Molecule, load_smiles, write

log = get_logger(__name__)


# =============================================================== data
@dataclass(frozen=True)
class MonomerRecipe:
    name: str
    smiles: str              # SMILES with [*] connection points (for polymerization)
    monomer_smiles: str      # closed-shell single-unit SMILES (for standalone build)
    description: str
    pid: str = ""            # e.g. "W01_P001"
    tg_k: Optional[float] = None
    tg_experimental_k: Optional[float] = None
    tg_source: str = ""
    density_kg_m3: Optional[float] = None
    density_experimental_kg_m3: Optional[float] = None
    density_source: str = ""
    equilibrated: bool = False
    tags: List[str] = field(default_factory=list)


# =============================================================== curated
_CURATED: Dict[str, MonomerRecipe] = {
    "PE":    MonomerRecipe("PE",  "[*]CC[*]",            "CC",           "Polyethylene",
                           tags=["curated"]),
    "PP":    MonomerRecipe("PP",  "[*]CC([*])C",         "CCC",          "Polypropylene",
                           tags=["curated"]),
    "PS":    MonomerRecipe("PS",  "[*]CC([*])c1ccccc1",  "CCc1ccccc1",   "Polystyrene",
                           tags=["curated"]),
    "PMMA":  MonomerRecipe("PMMA","[*]CC([*])(C)C(=O)OC","CC(C)C(=O)OC", "Poly(methyl methacrylate)",
                           tags=["curated"]),
    "PVC":   MonomerRecipe("PVC", "[*]CC([*])Cl",        "CCCl",         "Poly(vinyl chloride)",
                           tags=["curated"]),
    "PVA":   MonomerRecipe("PVA", "[*]CC([*])O",         "CCO",          "Poly(vinyl alcohol)",
                           tags=["curated"]),
    "PAN":   MonomerRecipe("PAN", "[*]CC([*])C#N",       "CCC#N",        "Polyacrylonitrile",
                           tags=["curated"]),
    "PLA":   MonomerRecipe("PLA", "[*]OC(C)C(=O)[*]",    "OC(C)C(=O)O",  "Poly(lactic acid)",
                           tags=["curated"]),
    "PGA":   MonomerRecipe("PGA", "[*]OCC(=O)[*]",       "OCC(=O)O",     "Poly(glycolic acid)",
                           tags=["curated"]),
    "PCL":   MonomerRecipe("PCL", "[*]OCCCCCC(=O)[*]",   "OCCCCCC(=O)O", "Poly(caprolactone)",
                           tags=["curated"]),
    "PBS":   MonomerRecipe("PBS", "[*]OCCCCOC(=O)CCC(=O)[*]",
                           "OCCCCOC(=O)CCC(=O)O", "Poly(butylene succinate)",
                           tags=["curated"]),
    "PET":   MonomerRecipe("PET", "[*]OCCOC(=O)c1ccc(cc1)C(=O)[*]",
                           "OCCOC(=O)c1ccc(cc1)C(=O)O",
                           "Poly(ethylene terephthalate)", tags=["curated"]),
    "PC":    MonomerRecipe("PC",  "[*]OC(=O)Oc1ccc(cc1)C(C)(C)c1ccc(cc1)[*]",
                           "OC(=O)Oc1ccc(cc1)C(C)(C)c1ccc(cc1)O",
                           "Polycarbonate (bisphenol-A)", tags=["curated"]),
    "PDMS":  MonomerRecipe("PDMS","[*][Si]([*])(C)C",    "[SiH2](C)C",   "Polydimethylsiloxane",
                           tags=["curated"]),
    "PEO":   MonomerRecipe("PEO", "[*]CCO[*]",           "CCOC",         "Poly(ethylene oxide)",
                           tags=["curated"]),
    "PU":    MonomerRecipe("PU",  "[*]NC(=O)O[*]",       "NC(=O)OC",     "Polyurethane (generic linkage)",
                           tags=["curated"]),
    "PA6":   MonomerRecipe("PA6", "[*]NCCCCCC(=O)[*]",   "NCCCCCC(=O)O", "Nylon 6",
                           tags=["curated"]),
    "PBAT":  MonomerRecipe("PBAT","[*]OCCCCOC(=O)c1ccc(cc1)C(=O)[*]",
                           "OCCCCOC(=O)c1ccc(cc1)C(=O)O",
                           "Poly(butylene adipate-co-terephthalate)", tags=["curated"]),
    "ENR":   MonomerRecipe("ENR", "[*]C/C=C(C)\\C[*]",   "CC=C(C)C",
                           "Natural rubber (cis-1,4-polyisoprene, unsaturated precursor of ENR)",
                           tags=["curated"]),
    "ENR_EPOXIDE": MonomerRecipe(
        "ENR_EPOXIDE",
        "[*]CC1(C)OC1C[*]",       # oxirane (3-membered epoxide ring)
        "CC1(C)OC1C",             # closed-shell 2,3-epoxy-2-methylbutane
        "Epoxidized natural rubber (true epoxide oxirane form)",
        tags=["curated"],
    ),
    "MAH":   MonomerRecipe("MAH", "O=C(O)/C=C\\C(=O)O",  "O=C(O)/C=C\\C(=O)O",
                           "Maleic acid / anhydride hydrolyzed", tags=["curated"]),
}


# =============================================================== DB loader
_DB_PATH = Path(__file__).resolve().parent / "data" / "polymer_database.csv"


def _to_float(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s in {"-", "N/A", "NA", "n/a"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _slug(name: str) -> str:
    """Turn a polymer name into a short, uppercased key: "Poly(methyl acrylate)" -> "POLYMETHYLACRYLATE"."""
    s = re.sub(r"[^A-Za-z0-9]+", "", name).upper()
    return s or "MOL"


def _closed_shell(smiles: str) -> str:
    """Turn a polymerization SMILES with [*] wildcards into a plausible closed-shell
    single-repeat-unit SMILES by capping wildcards with hydrogens.
    """
    # Two conventions appear: [*] (bracket) and * (bare wildcard). Both mean
    # "connection point". Cap both by removing the wildcard atom (implicit H
    # will then satisfy valence).
    s = smiles.replace("[*]", "").replace("*", "")
    # Clean up leftover parentheses/brackets from patterns like "([*])".
    s = s.replace("()", "").replace("[]", "")
    return s or smiles


def load_database(path: Path = _DB_PATH) -> List[MonomerRecipe]:
    """Load the polymer database CSV shipped with the package."""
    if not path.exists():
        log.warning("Polymer database %s not found — DB entries will be unavailable.", path)
        return []
    out: List[MonomerRecipe] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("pid_true") or "").strip()
            smiles = (row.get("smiles") or "").strip()
            name = (row.get("polymer name") or row.get("polymer_name") or "").strip()
            if not (pid and smiles and name):
                continue
            key = pid.upper()   # e.g. W01_P001 as the canonical key
            out.append(MonomerRecipe(
                name=key,
                smiles=smiles,
                monomer_smiles=_closed_shell(smiles),
                description=name,
                pid=pid,
                tg_k=_to_float(row.get("Tg (K)")),
                tg_experimental_k=_to_float(row.get("Tg (experimental) (K)")),
                tg_source=(row.get("Tg source") or "").strip(),
                density_kg_m3=_to_float(row.get("density (kg/m^3)")),
                density_experimental_kg_m3=_to_float(row.get("density (experimental) (kg/m^3)")),
                density_source=(row.get("density source") or "").strip(),
                equilibrated=(row.get("Equilibrated?", "").strip().upper() == "TRUE"),
                tags=["database"],
            ))
    log.info("Loaded %d polymers from database %s", len(out), path.name)
    return out


# =============================================================== merged LIBRARY
def _build_registry() -> Dict[str, MonomerRecipe]:
    reg: Dict[str, MonomerRecipe] = dict(_CURATED)
    for r in load_database():
        # index by PID (primary), and also by the polymer name slug if free
        reg[r.pid.upper()] = r
        slug = _slug(r.description)
        if slug and slug not in reg:
            reg[slug] = r
    for r in _load_user_polymers():
        # Last, so a user entry with the same id deliberately shadows a
        # shipped one. Overriding a library polymer you disagree with is a
        # reasonable thing to want, and the alternative — silently ignoring
        # what the user added — is not.
        reg[r.pid.upper()] = r
        slug = _slug(r.description)
        if slug and slug not in reg:
            reg[slug] = r
    return reg


def _load_user_polymers() -> List[MonomerRecipe]:
    """The user's own polymers, as library recipes.

    Kept outside the package (see :mod:`paaf.custom_library`) so that
    reinstalling PAAF does not delete them.
    """
    try:
        from .custom_library import load_custom
        entries = load_custom()
    except Exception as exc:                          # pragma: no cover
        log.warning("Custom polymers could not be loaded (%s); the shipped "
                    "library is unaffected.", exc)
        return []
    return [MonomerRecipe(name=e.pid.upper(), smiles=e.smiles,
                          monomer_smiles=_closed_shell(e.smiles),
                          description=e.name, pid=e.pid,
                          tags=["custom"])
            for e in entries.values()]


LIBRARY: Dict[str, MonomerRecipe] = _build_registry()


def reload_library() -> None:
    """Rebuild ``LIBRARY`` in place after the user adds or removes a polymer.

    In place, because the GUI and the pipeline both hold a reference to this
    dict; rebinding the module global would leave them looking at the old one.
    """
    fresh = _build_registry()
    LIBRARY.clear()
    LIBRARY.update(fresh)


# =============================================================== API
def list_library(source: str = "all") -> List[MonomerRecipe]:
    """Return recipes filtered by source tag: 'all' | 'curated' | 'database'."""
    if source == "all":
        # Deduplicate by (pid or name) so a DB entry aliased under multiple keys
        # doesn't show up twice.
        seen: set[str] = set()
        out: List[MonomerRecipe] = []
        for r in LIBRARY.values():
            key = r.pid or r.name
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out
    return [r for r in list_library("all") if source in r.tags]


def get_recipe(name: str) -> MonomerRecipe:
    """Resolve any of: 'PBS', 'W01_P001', 'polyethylene', 'poly(butyl acrylate)'."""
    if not name:
        raise KeyError("empty polymer name")
    # Try direct upper-case match (covers curated keys and W01_PXXX PIDs).
    key = name.strip().upper()
    if key in LIBRARY:
        return LIBRARY[key]
    # Try slugified description match.
    slug = _slug(name)
    if slug in LIBRARY:
        return LIBRARY[slug]
    # Case-insensitive substring search over descriptions as a fallback.
    matches = [
        r for r in list_library("all")
        if name.lower() in r.description.lower()
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(
            f"Ambiguous polymer name {name!r}. Candidates: "
            + ", ".join(f"{m.pid or m.name} ({m.description})" for m in matches[:8])
        )
    raise KeyError(
        f"Unknown polymer {name!r}. Try build-molecule --list, or pass a PID "
        f"like W01_P001."
    )


def search_library(query: str) -> List[MonomerRecipe]:
    """Case-insensitive substring search across name, PID and description."""
    q = query.lower().strip()
    if not q:
        return []
    return [
        r for r in list_library("all")
        if q in r.name.lower() or q in r.description.lower() or q in (r.pid or "").lower()
    ]


# =============================================================== build 3D
def build_from_smiles(
    smiles: str,
    out_path: str | Path,
    add_h: bool = True,
    optimize: bool = True,
    opt_ff: str = "MMFF94",
    opt_steps: int = 2000,
    name: str = "MOL",
) -> Path:
    """SMILES -> 3D -> file. Output extension controls format.

    If optimization fails (e.g. MMFF94 can't type a specific molecule), we
    warn and fall through to writing the un-optimized 3D coordinates rather
    than losing the whole build.
    """
    mol = load_smiles(smiles, name=name, add_h=add_h, make_3d=True)
    if optimize:
        from .optimizer import optimize as _opt
        try:
            _opt(mol, ff=opt_ff, steps=opt_steps, fallback=True, strict=False)
        except Exception as exc:
            # Optimizer already handles most failures internally, but guard
            # against anything unexpected so the build still delivers a file.
            log.warning("Optimizer raised %s for %s; keeping unoptimized 3D structure.",
                       exc, name)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(mol, out)
    log.info("Built %s (%d atoms) -> %s", name, len(mol.atoms), out)
    return out


def build_recipe(
    name: str,
    out_path: str | Path,
    optimize: bool = True,
    use_polymerization_smiles: bool = False,
) -> Path:
    """Build a monomer from any library entry.

    If ``use_polymerization_smiles`` is True the SMILES with ``[*]`` connection
    points is used verbatim — useful when downstream code (e.g. mbuild) wants
    to see the connection markers. Otherwise a closed-shell single-unit
    SMILES is built and 3D-optimized.
    """
    r = get_recipe(name)
    smi = r.smiles if use_polymerization_smiles else _polymerisable_form(r)
    return build_from_smiles(smi, out_path, optimize=optimize, name=r.name)


def _polymerisable_form(recipe: MonomerRecipe) -> str:
    """The single-unit SMILES that can actually be linked into a chain.

    Not simply ``monomer_smiles``. A few curated entries write the chemically
    tidy standalone molecule there — PBS as ``OCCCCOC(=O)CCC(=O)O``, with a
    real carboxylic acid — and that form cannot be polymerised by PAAF, whose
    linking removes ONE hydrogen from each end. The acid's carbon has no
    hydrogen to remove, so head/tail detection falls through to a geometry
    guess and picks two carbons in the middle of the monomer.
    """
    if recipe.smiles and "[*]" in recipe.smiles:
        return _closed_shell(recipe.smiles)
    return recipe.monomer_smiles
