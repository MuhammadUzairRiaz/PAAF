"""Registry of supported force fields.

Each FF is described by:
- kind: "moltemplate_native" | "dlfield" | "gaff" | "trappe"
- lt_include: the Moltemplate .lt file to `import` (as shipped with moltemplate)
- inherit: the base object monomers should inherit from (e.g. OPLSAA)
- atom_typer: how atoms are typed (rule-based, DL_FIELD .sf, antechamber, ...)
- notes: human-readable

If the .lt file ships with Moltemplate we also bundle a copy in
``ff_libraries/moltemplate/`` so the pipeline can drop it next to the
generated system.lt and moltemplate.sh finds it locally without depending
on a user's Moltemplate install having them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

BUNDLED_MOLTEMPLATE = Path(__file__).resolve().parent.parent / "ff_libraries" / "moltemplate"
BUNDLED_WATER = BUNDLED_MOLTEMPLATE  # water models live in the same directory


@dataclass
class ForceField:
    key: str
    display_name: str
    kind: str
    lt_include: Optional[str] = None    # filename to `import` from .lt files
    inherit: Optional[str] = None       # base object name (OPLSAA, GAFF, ...)
    atom_typer: str = "openbabel"       # openbabel | dlfield_sf | antechamber | manual
    dlfield_par: Optional[str] = None   # e.g. "PCFF.par"
    dlfield_sf: Optional[str] = None    # e.g. "PCFF.sf"
    bundled_lt: Optional[str] = None    # path (relative to ff_libraries/moltemplate)
    united_atom: bool = False
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    def bundled_path(self) -> Optional[Path]:
        if not self.bundled_lt:
            return None
        p = BUNDLED_MOLTEMPLATE / self.bundled_lt
        return p if p.exists() else None


REGISTRY: Dict[str, ForceField] = {
    # ==================== native (shipped with Moltemplate) ====================
    "oplsaa": ForceField(
        key="oplsaa",
        display_name="OPLS-AA 2024 (Moltemplate native)",
        kind="moltemplate_native",
        lt_include="oplsaa2024.lt",
        bundled_lt="oplsaa2024.lt",
        inherit="OPLSAA",
        atom_typer="openbabel",
        notes="Jorgensen et al. 2024 OPLS-AA parameters (J. Phys. Chem. B).",
        tags=["all-atom", "organic", "latest"],
    ),
    "oplsaa2008": ForceField(
        key="oplsaa2008",
        display_name="OPLS-AA 2008 (Jorgensen)",
        kind="moltemplate_native",
        lt_include="oplsaa2008.lt",
        bundled_lt="oplsaa2008.lt",
        inherit="OPLSAA",
        atom_typer="openbabel",
        notes="Legacy OPLS-AA 2008 parameter set — use for reproducibility of older studies.",
        tags=["all-atom", "organic"],
    ),
    "loplsaa": ForceField(
        key="loplsaa",
        display_name="L-OPLS-AA 2024 (long-chain refit)",
        kind="moltemplate_native",
        lt_include="loplsaa2024.lt",
        bundled_lt="loplsaa2024.lt",
        inherit="OPLSAA",
        atom_typer="openbabel",
        notes="Siu et al. 2012 refit of OPLS-AA torsions for long alkanes (2024 update).",
        tags=["all-atom", "polymer"],
    ),
    "loplsaa2008": ForceField(
        key="loplsaa2008",
        display_name="L-OPLS-AA 2008",
        kind="moltemplate_native",
        lt_include="loplsaa2008.lt",
        bundled_lt="loplsaa2008.lt",
        inherit="OPLSAA",
        atom_typer="openbabel",
        notes="L-OPLS on top of OPLS-AA 2008.",
        tags=["all-atom", "polymer"],
    ),
    "trappe_ua": ForceField(
        key="trappe_ua",
        display_name="TraPPE-UA (Moltemplate native)",
        kind="moltemplate_native",
        lt_include="trappe1998.lt",
        bundled_lt="trappe1998.lt",
        inherit="TraPPE",
        atom_typer="manual",
        united_atom=True,
        notes="TraPPE 1998 united-atom for alkanes; hydrogens absorbed into CH_n pseudo-atoms.",
        tags=["united-atom", "hydrocarbons"],
    ),
    "oplsua_2024": ForceField(
        key="oplsua_2024",
        display_name="OPLS-UA 2024 (extracted from OPLS-AA 2024)",
        kind="moltemplate_native",
        lt_include="oplsua_2024.lt",
        bundled_lt="oplsua_2024.lt",
        inherit="OPLSUA_2024",
        atom_typer="manual",
        united_atom=True,
        notes="Standalone united-atom OPLS force field auto-derived by "
              "paaf.oplsua_extractor from the UA block "
              "(types 66-134) of oplsaa2024.lt. Contains CH/CH2/CH3/CH4 "
              "pseudo-atoms for alkanes, aromatics, alcohols, thioethers, "
              "ethers, nitriles, DMSO, DMF, chlorocarbons, plus polar-H "
              "atoms where OPLS keeps them explicit. All bond/angle/"
              "dihedral parameters filtered to those referencing UA "
              "equivalence classes only.",
        tags=["united-atom", "organic", "polymer"],
    ),
    "gaff": ForceField(
        key="gaff",
        display_name="GAFF (Moltemplate native, antechamber for typing)",
        kind="gaff",
        lt_include="gaff.lt",
        bundled_lt="gaff.lt",
        inherit="GAFF",
        atom_typer="antechamber",
        notes="General AMBER FF; requires ambertools' antechamber for atom typing.",
        tags=["all-atom", "organic"],
    ),
    "gaff2": ForceField(
        key="gaff2",
        display_name="GAFF2 (Moltemplate native, antechamber)",
        kind="gaff",
        lt_include="gaff2.lt",
        bundled_lt="gaff2.lt",
        inherit="GAFF2",
        atom_typer="antechamber",
        notes="Updated GAFF; needs ambertools >=20.",
        tags=["all-atom", "organic"],
    ),
    "dreiding": ForceField(
        key="dreiding",
        display_name="DREIDING (Moltemplate native)",
        kind="moltemplate_native",
        lt_include="dreiding.lt",
        bundled_lt="dreiding.lt",
        inherit="DREIDING",
        atom_typer="dlfield_sf",
        dlfield_par="DREIDING.par",
        dlfield_sf="DREIDING.sf",
        notes="Generic all-atom FF for organics/inorganics (Mayo et al. 1990).",
        tags=["all-atom", "generic"],
    ),
    "compass_published": ForceField(
        key="compass_published",
        display_name="COMPASS (public parameters, Moltemplate native)",
        kind="moltemplate_native",
        lt_include="compass_published.lt",
        bundled_lt="compass_published.lt",
        inherit="COMPASS",
        atom_typer="dlfield_sf",
        dlfield_par="COMPASS.par",
        dlfield_sf="COMPASS.sf",
        notes="Publicly available COMPASS parameters shipped with Moltemplate. "
              "For full COMPASS coverage use `compass` (DL_FIELD converted).",
        tags=["all-atom", "polymer", "class-II"],
    ),
    "sdk": ForceField(
        key="sdk",
        display_name="SDK coarse-grained lipids/cholesterol",
        kind="moltemplate_native",
        lt_include="sdk.lt",
        bundled_lt="sdk.lt",
        inherit="SDK",
        atom_typer="manual",
        notes="Shinoda-DeVane-Klein CG model (Shinoda et al. 2010).",
        tags=["coarse-grained", "lipid"],
    ),
    "martini": ForceField(
        key="martini",
        display_name="MARTINI 2.0 (coarse-grained)",
        kind="moltemplate_native",
        lt_include="martini.lt",
        bundled_lt="martini.lt",
        inherit="MARTINI",
        atom_typer="manual",
        notes="MARTINI 2.0 coarse-grained FF for lipids/proteins/ions/cholesterol.",
        tags=["coarse-grained", "biomolecular"],
    ),
    "drymartini": ForceField(
        key="drymartini",
        display_name="Dry MARTINI (implicit solvent)",
        kind="moltemplate_native",
        lt_include="drymartini.lt",
        bundled_lt="drymartini.lt",
        inherit="DRYMARTINI",
        atom_typer="manual",
        notes="MARTINI variant with implicit solvent.",
        tags=["coarse-grained", "implicit-solvent"],
    ),

    # ---- from DL_FIELD ----
    "pcff": ForceField(
        key="pcff",
        display_name="PCFF (converted from DL_FIELD)",
        kind="dlfield",
        lt_include=None,   # generated on the fly
        inherit="PCFF",
        atom_typer="dlfield_sf",
        dlfield_par="PCFF.par",
        dlfield_sf="PCFF.sf",
        notes="Class-II polymer FF (quartic bonds/angles + cross terms). "
              "Converted by paaf.dlfield_converter into a Moltemplate .lt.",
        tags=["all-atom", "polymer", "class-II"],
    ),
    "compass": ForceField(
        key="compass",
        display_name="COMPASS (converted from DL_FIELD)",
        kind="dlfield",
        inherit="COMPASS",
        atom_typer="dlfield_sf",
        dlfield_par="COMPASS.par",
        dlfield_sf="COMPASS.sf",
        notes="Condensed-phase optimised molecular potentials for atomistic simulation studies.",
        tags=["all-atom", "polymer", "class-II"],
    ),
    "cvff": ForceField(
        key="cvff",
        display_name="CVFF (converted from DL_FIELD)",
        kind="dlfield",
        inherit="CVFF",
        atom_typer="dlfield_sf",
        dlfield_par="CVFF.par",
        dlfield_sf="CVFF.sf",
        notes="Consistent Valence FF (class-I) for organics.",
        tags=["all-atom", "organic"],
    ),
    "opls2005_dl": ForceField(
        key="opls2005_dl",
        display_name="OPLS 2005 (DL_FIELD)",
        kind="dlfield",
        inherit="OPLS2005",
        atom_typer="dlfield_sf",
        dlfield_par="OPLS2005.par",
        dlfield_sf="OPLS2005.sf",
        notes="Alternative OPLS 2005 params via DL_FIELD.",
        tags=["all-atom", "organic"],
    ),
    "opls_ua": ForceField(
        key="opls_ua",
        display_name="OPLS-UA (DL_FIELD)",
        kind="dlfield",
        inherit="OPLS_UA",
        atom_typer="dlfield_sf",
        dlfield_par="OPLS_UA.par",
        dlfield_sf="OPLS_UA.sf",
        united_atom=True,
        notes="United-atom OPLS from DL_FIELD.",
        tags=["united-atom", "organic"],
    ),
    "trappe_eh": ForceField(
        key="trappe_eh",
        display_name="TraPPE-EH (DL_FIELD)",
        kind="dlfield",
        inherit="TraPPE_EH",
        atom_typer="dlfield_sf",
        dlfield_par="TRAPPE_EH.par",
        dlfield_sf="TRAPPE_EH.sf",
        notes="Explicit-hydrogen TraPPE from DL_FIELD.",
        tags=["all-atom", "organic"],
    ),
    "charmm36": ForceField(
        key="charmm36",
        display_name="CHARMM36 (DL_FIELD)",
        kind="dlfield",
        inherit="CHARMM36",
        atom_typer="dlfield_sf",
        dlfield_par="CHARMM36_prot.par",
        dlfield_sf="CHARMM36_prot.sf",
        notes="Protein CHARMM36 from DL_FIELD.",
        tags=["all-atom", "biomolecular"],
    ),
    "amber_gaff": ForceField(
        key="amber_gaff",
        display_name="AMBER GAFF (DL_FIELD)",
        kind="dlfield",
        inherit="AMBER_GAFF",
        atom_typer="dlfield_sf",
        dlfield_par="AMBER16_gaff.par",
        dlfield_sf="AMBER16_gaff.sf",
        notes="AMBER's GAFF via DL_FIELD.",
        tags=["all-atom", "organic"],
    ),
    "gromos_54a7": ForceField(
        key="gromos_54a7",
        display_name="GROMOS 54A7 (DL_FIELD)",
        kind="dlfield",
        inherit="GROMOS54A7",
        atom_typer="dlfield_sf",
        dlfield_par="GROMOS_G54A7.par",
        dlfield_sf="GROMOS_G54A7.sf",
        united_atom=True,
        notes="United-atom GROMOS.",
        tags=["united-atom", "biomolecular"],
    ),
}


def list_ffs() -> List[ForceField]:
    return list(REGISTRY.values())


def get_ff(key: str) -> ForceField:
    if key not in REGISTRY:
        raise KeyError(f"Unknown force field {key!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[key]


def register(ff: ForceField) -> None:
    REGISTRY[ff.key] = ff


# ==================== Water / solvent add-on templates ====================
# These are Moltemplate .lt files that define a water molecule inheriting from
# a companion FF. They are added to the box via `system.lt` when the user asks
# to solvate; they are not standalone FFs on their own.
WATER_MODELS: Dict[str, str] = {
    "spce":                    "spce.lt",
    "spce_amber":              "spce_amber.lt",
    "spce_dreiding":           "spce_dreiding.lt",
    "spce_oplsaa2024":         "spce_oplsaa2024.lt",
    "spce_oplsaa":             "spce_oplsaa.lt",
    "spc_oplsaa2024":          "spc_oplsaa2024.lt",
    "spc_oplsaa2008":          "spc_oplsaa2008.lt",
    "tip3p_1983":              "tip3p_1983.lt",
    "tip3p_1983_charmm":       "tip3p_1983_charmm.lt",
    "tip3p_1983_oplsaa2024":   "tip3p_1983_oplsaa2024.lt",
    "tip3p_1983_oplsaa2008":   "tip3p_1983_oplsaa2008.lt",
    "tip3p_2004":              "tip3p_2004.lt",
    "tip3p_2004_oplsaa2024":   "tip3p_2004_oplsaa2024.lt",
    "tip5p_oplsaa2024":        "tip5p_oplsaa2024.lt",
    "tip5p_oplsaa2008":        "tip5p_oplsaa2008.lt",
}


def water_lt_path(name: str) -> Optional[Path]:
    fname = WATER_MODELS.get(name)
    if not fname:
        return None
    p = BUNDLED_MOLTEMPLATE / fname
    return p if p.exists() else None


# =====================================================================
# Every remaining DL_FIELD force field, registered from one table
# =====================================================================
# DL_FIELD 4.13 ships 32 parameter sets in its ``lib``; only ten were
# hand-written above, so the rest could not be selected at all. Rather than
# copy the same twelve lines thirty times, the descriptions in
# :mod:`paaf.dlfield_styles` — which already carry the exact ``POTENTIAL``
# string each control file needs — are turned into registry entries here.
#
# Entries written by hand above are left alone: they carry curated notes and
# tags this loop cannot reproduce.

def _register_dlfield_schemes() -> int:
    """Add a ForceField for every DL_FIELD scheme not already registered."""
    from .dlfield_styles import DLFIELD_SCHEMES

    _CATEGORY_TAGS = {
        "polymer":      ["all-atom", "polymer"],
        "biomolecular": ["all-atom", "biomolecular"],
        "liquid":       ["liquid", "small-molecule"],
        "inorganic":    ["inorganic", "ionic"],
        "general":      ["all-atom"],
    }
    added = 0
    for scheme in DLFIELD_SCHEMES.values():
        if scheme.key in REGISTRY:
            continue
        tags = list(_CATEGORY_TAGS.get(scheme.category, ["all-atom"]))
        if scheme.mix == "sixthpower":
            tags.append("class-II")
        note = (f"DL_FIELD scheme '{scheme.dl_key}' ({scheme.par}). "
                f"Bonded styles are read from the generated data file; "
                f"pair_style {scheme.pair_style}, mixing {scheme.mix}.")
        REGISTRY[scheme.key] = ForceField(
            key=scheme.key,
            display_name=scheme.display,
            kind="dlfield",
            inherit=scheme.dl_key,
            atom_typer="dlfield_sf",
            dlfield_par=scheme.par,
            dlfield_sf=scheme.par.replace(".par", ".sf"),
            notes=note + (" " + scheme.note if scheme.note else ""),
            tags=tags,
        )
        added += 1
    return added


_N_DLFIELD_ADDED = _register_dlfield_schemes()


def dlfield_keys() -> List[str]:
    """Registry keys backed by DL_FIELD, in display order."""
    return sorted((k for k, f in REGISTRY.items() if f.kind == "dlfield"),
                  key=lambda k: REGISTRY[k].display_name)
