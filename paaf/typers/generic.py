"""Force-field-agnostic atom typing for every bundled Moltemplate library.

Two-step design, the same one DL_FIELD uses internally:

1. **Chemical environment** (force-field independent). An ordered list of
   SMARTS rules labels each atom with an environment key such as
   ``alkane_CH2``, ``ester_C``, ``arom_CH``, ``ether_O`` or ``amide_NH``.
   The first matching rule wins, so specific rules come before generic ones.

2. **Per-force-field map** from environment key to that library's own atom
   type id: ``alkane_CH2`` -> OPLS ``136``, COMPASS ``c4``, DREIDING ``C_3``,
   GAFF ``c3``, TraPPE ``CH2``, OPLS-UA ``71``. Every id in every map is
   checked by the test-suite against the bundled ``.lt`` file, so a
   suggestion is always a type moltemplate will accept.

The OPLS-AA family keeps its dedicated typer (:mod:`paaf.typers.oplsaa`);
this module covers everything else and is the fallback for it.

United-atom libraries (TraPPE-UA, OPLS-UA) only carry heavy-atom types: the
hydrogens get no type here and are reported, because those force fields do
not model them at all.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..logging_utils import get_logger

log = get_logger(__name__)

# ------------------------------------------------------------------ rules
# (SMARTS, environment key, description). Atom 0 of the SMARTS is the typed
# atom. Order matters: first match wins.
ENV_RULES: List[Tuple[str, str, str]] = [
    # epoxide
    ("[OX2r3]([#6])[#6]",                "epoxide_O",   "epoxide O (3-ring)"),
    ("[CX4r3;$(C1OC1)]",                 "epoxide_C",   "epoxide sp3 C"),
    # carboxylic acid
    ("[CX3](=O)[OX2H]",                  "acid_C",      "carboxylic acid C"),
    ("[OX1]=[CX3][OX2H]",                "acid_Odb",    "carboxylic acid C=O"),
    ("[OX2H][CX3](=O)",                  "acid_OH",     "carboxylic acid -OH O"),
    ("[H][OX2][CX3](=O)",                "acid_H",      "carboxylic acid H"),
    # ester
    ("[CX3](=O)[OX2][#6]",               "ester_C",     "ester carbonyl C"),
    ("[OX1]=[CX3][OX2][#6]",             "ester_Odb",   "ester C=O"),
    ("[OX2]([CX3]=O)[#6]",               "ester_Os",    "ester -O- (sp3)"),
    # amide
    ("[CX3](=O)[NX3]",                   "amide_C",     "amide carbonyl C"),
    ("[OX1]=[CX3][NX3]",                 "amide_Odb",   "amide C=O"),
    ("[NX3H0]([CX3]=O)([#6])[#6]",       "amide_N",     "tertiary amide N"),
    ("[NX3H1]([CX3]=O)[#6]",             "amide_NH",    "secondary amide N-H"),
    ("[NX3H2][CX3]=O",                   "amide_NH2",   "primary amide NH2"),
    ("[H][NX3][CX3]=O",                  "amide_H",     "amide N-H hydrogen"),
    # ketone / aldehyde
    ("[CX3H1](=O)[#6]",                  "aldehyde_C",  "aldehyde C"),
    ("[H][CX3]=O",                       "aldehyde_H",  "aldehyde H"),
    ("[CX3](=O)([#6])[#6]",              "ketone_C",    "ketone C"),
    ("[OX1]=[CX3]([#6])[#6]",            "ketone_O",    "ketone O"),
    ("[OX1]=[CX3H1][#6]",                "aldehyde_O",  "aldehyde O"),
    # nitrile
    ("[NX1]#[CX2]",                      "nitrile_N",   "nitrile N"),
    ("[CX2]#[NX1]",                      "nitrile_C",   "nitrile C"),
    # alcohol / phenol / ether / siloxane
    ("[OX2H][c]",                        "phenol_O",    "phenol O"),
    ("[H][OX2][c]",                      "phenol_H",    "phenol H"),
    ("[OX2H][CX4]",                      "alcohol_O",   "alcohol -OH O"),
    ("[H][OX2][CX4]",                    "alcohol_H",   "alcohol -OH H"),
    ("[OX2]([Si])[Si]",                  "siloxane_O",  "siloxane Si-O-Si"),
    ("[OX2]([Si])[#6]",                  "siloxane_O",  "Si-O-C oxygen"),
    ("[OX2]([c])[#6]",                   "aryl_ether_O","aryl ether O"),
    ("[OX2]([CX4])[CX4]",                "ether_O",     "ether -O- (sp3)"),
    # silicon
    ("[Si]",                             "Si",          "silicon (sp3)"),
    # aromatic carbons
    ("[cX3H1]",                          "arom_CH",     "aromatic C-H"),
    ("[cX3]([OX2])",                     "arom_C_O",    "aromatic C bonded to O"),
    ("[cX3]([NX3])",                     "arom_C_N",    "aromatic C bonded to N"),
    ("[cX3H0]",                          "arom_C",      "aromatic C (substituted)"),
    ("[H][c]",                           "arom_H",      "aromatic H"),
    ("[nX2]",                            "arom_N",      "aromatic N (pyridine)"),
    ("[nX3H1]",                          "arom_NH",     "aromatic N-H (pyrrole)"),
    ("[H][n]",                           "arom_NH_H",   "aromatic N-H hydrogen"),
    # alkenes
    ("[CX3H2]=[CX3]",                    "alkene_CH2",  "terminal =CH2"),
    ("[CX3H1]=[CX3]",                    "alkene_CH",   "vinyl =CH-"),
    ("[CX3H0]=[CX3]",                    "alkene_C",    "alkene =C< (no H)"),
    ("[H][CX3]=[CX3]",                   "alkene_H",    "alkene C-H"),
    ("[CX2]#[CX2]",                      "alkyne_C",    "alkyne C"),
    ("[H][CX2]#C",                       "alkyne_H",    "alkyne H"),
    # halides
    ("[F][CX4]",                         "F_alkyl",     "F on sp3 C"),
    ("[Cl][CX4]",                        "Cl_alkyl",    "Cl on sp3 C"),
    ("[Br][CX4]",                        "Br_alkyl",    "Br on sp3 C"),
    ("[I][CX4]",                         "I_alkyl",     "I on sp3 C"),
    ("[F][c,$(C=C)]",                    "F_aryl",      "F on sp2 C"),
    ("[Cl][c,$(C=C)]",                   "Cl_aryl",     "Cl on sp2 C"),
    ("[Br][c,$(C=C)]",                   "Br_aryl",     "Br on sp2 C"),
    # sulfur
    ("[SX2H][#6]",                       "thiol_S",     "thiol -SH S"),
    ("[H][SX2]",                         "thiol_H",     "thiol H"),
    ("[SX2]([#6])[#6]",                  "sulfide_S",   "sulfide S"),
    ("[SX2]([#6])[SX2]",                 "disulfide_S", "disulfide S"),
    ("[SX4](=O)(=O)",                    "sulfone_S",   "sulfone S"),
    ("[OX1]=[SX4]",                      "sulfone_O",   "sulfone =O"),
    ("[SX1]=[#6]",                       "thione_S",    "C=S sulfur"),
    # amines (sp3 N)
    ("[NX3H2][c]",                       "aniline_N",   "aniline NH2"),
    ("[NX3H2][CX4]",                     "amine_NH2",   "primary amine"),
    ("[NX3H1]([#6])[#6]",                "amine_NH",    "secondary amine"),
    ("[NX3H0]([#6])([#6])[#6]",          "amine_N",     "tertiary amine"),
    ("[H][NX3]",                         "amine_H",     "amine N-H"),
    ("[NX3](=O)=O",                      "nitro_N",     "nitro N"),
    ("[OX1][NX3]",                       "nitro_O",     "nitro O"),
    ("[NX2]=[CX3]",                      "imine_N",     "imine N"),
    ("[CX3]=[NX2]",                      "imine_C",     "imine C"),
    # sp3 carbons — by substitution and heteroatom neighbours
    ("[CX4H3][OX2,NX3,SX2,F,Cl,Br,I]",   "alkane_CH3_X","CH3 on heteroatom"),
    ("[CX4H2]([OX2,NX3,SX2,F,Cl,Br,I])", "alkane_CH2_X","CH2 on heteroatom"),
    ("[CX4H1]([OX2,NX3,SX2,F,Cl,Br,I])", "alkane_CH_X", "CH on heteroatom"),
    ("[CX4H0]([OX2,NX3,SX2,F,Cl,Br,I])", "alkane_C_X",  "quaternary C on heteroatom"),
    ("[CX4H3][c]",                       "alkane_CH3_ar","CH3 on aromatic ring"),
    ("[CX4H2]([c])",                     "alkane_CH2_ar","CH2 on aromatic ring"),
    ("[CX4H1]([c])",                     "alkane_CH_ar", "CH on aromatic ring"),
    ("[CX4H4]",                          "methane_C",   "methane"),
    ("[CX4H3]",                          "alkane_CH3",  "sp3 CH3"),
    ("[CX4H2]",                          "alkane_CH2",  "sp3 CH2"),
    ("[CX4H1]",                          "alkane_CH",   "sp3 CH"),
    ("[CX4H0]",                          "alkane_C",    "sp3 C (quaternary)"),
    # hydrogens
    ("[H][CX4][OX2,NX3,F,Cl,Br,I]",      "alkane_H_X",  "H on C bearing a heteroatom"),
    ("[H][CX4]",                         "alkane_H",    "H on sp3 C"),
    ("[H][Si]",                          "Si_H",        "H on Si"),
    ("[H][H]",                           "H2",          "H2"),
    # phosphorus / boron
    ("[PX4](=O)",                        "phosphate_P", "phosphate P"),
    ("[P]",                              "P",           "phosphorus"),
    ("[B]",                              "B",           "boron"),
]

# ------------------------------------------------------------------ maps
# Keys that mean "same as": resolve chains so tables stay short.
_ALIAS: Dict[str, str] = {
    "alkane_CH3_X": "alkane_CH3", "alkane_CH2_X": "alkane_CH2",
    "alkane_CH_X": "alkane_CH", "alkane_C_X": "alkane_C",
    "alkane_CH3_ar": "alkane_CH3", "alkane_CH2_ar": "alkane_CH2",
    "alkane_CH_ar": "alkane_CH", "alkane_H_X": "alkane_H",
    "acid_C": "ester_C", "acid_Odb": "ester_Odb", "acid_OH": "alcohol_O",
    "acid_H": "alcohol_H", "phenol_O": "alcohol_O", "phenol_H": "alcohol_H",
    "aryl_ether_O": "ether_O", "aldehyde_C": "ketone_C",
    "aldehyde_O": "ketone_O", "aldehyde_H": "alkene_H",
    "amide_Odb": "ketone_O", "amide_NH2": "amide_NH", "arom_C_O": "arom_C",
    "arom_C_N": "arom_C", "arom_NH_H": "amine_H", "aniline_N": "amine_NH2",
    "disulfide_S": "sulfide_S", "thione_S": "sulfide_S", "imine_C": "alkene_C",
    "F_aryl": "F_alkyl", "Cl_aryl": "Cl_alkyl", "Br_aryl": "Br_alkyl",
    "epoxide_C": "alkane_CH_X", "epoxide_O": "ether_O", "siloxane_O": "ether_O",
    "methane_C": "alkane_CH3", "alkyne_H": "alkene_H", "Si_H": "alkane_H",
    "thiol_H": "alcohol_H", "sulfone_O": "ketone_O", "nitro_O": "ketone_O",
    "alkene_CH2": "alkene_CH", "phosphate_P": "P",
}

# COMPASS (compass_published.lt) — Sun 1998 public subset.
_COMPASS: Dict[str, str] = {
    "alkane_CH3": "c4", "alkane_CH2": "c4", "alkane_CH": "c43", "alkane_C": "c44",
    "alkane_CH3_X": "c41o", "alkane_CH2_X": "c4o", "alkane_CH_X": "c43o",
    "alkane_C_X": "c4o", "epoxide_C": "c43o",
    "arom_CH": "c3a", "arom_C": "c3a",
    "alkene_CH": "c2=", "alkene_C": "c2=", "alkyne_C": "c1o",
    "ester_C": "c3prime", "ketone_C": "c3prime", "amide_C": "c3prime",
    "nitrile_C": "c1o",
    "alkane_H": "h1", "arom_H": "h1", "alkene_H": "h1", "alcohol_H": "h1o",
    "amine_H": "h1", "amide_H": "h1", "H2": "h1h",
    "ether_O": "o2e", "ester_Os": "o2s", "alcohol_O": "o2h", "siloxane_O": "o2z",
    "ester_Odb": "o1=", "ketone_O": "o1=", "acid_Odb": "o1=", "epoxide_O": "o2e",
    "nitro_O": "o1n",
    "amide_N": "n3m", "amide_NH": "n3m", "amine_N": "n3m", "amine_NH": "n3m",
    "amine_NH2": "n3m", "imine_N": "n2=", "arom_N": "n2=", "nitrile_N": "n1n",
    "nitro_N": "n3o",
    "Si": "si4c", "sulfide_S": "s2=", "thiol_S": "s2=", "sulfone_S": "s2=",
    "P": "p4=",
}
_COMPASS_FALLBACK = {"C": "c4", "H": "h1", "O": "o2", "N": "n3m", "Si": "si4",
                     "S": "s2=", "P": "p4="}

# DREIDING (Mayo 1990) as bundled (Bone 2020). C_2/C_2_b2 and O_2/O_2_b2
# pairs encode which sp2 atoms share the double bond: carbonyl C=O uses the
# _b2 pair so an adjacent C=C (plain C_2) is not also read as a double bond.
_DREIDING: Dict[str, str] = {
    "alkane_CH3": "C_3", "alkane_CH2": "C_3", "alkane_CH": "C_3", "alkane_C": "C_3",
    "arom_CH": "C_R", "arom_C": "C_R",
    "alkene_CH": "C_2", "alkene_C": "C_2", "alkyne_C": "C_1", "nitrile_C": "C_1",
    "ester_C": "C_2_b2", "ketone_C": "C_2_b2", "amide_C": "C_2_b2",
    "ester_Odb": "O_2_b2", "ketone_O": "O_2_b2",
    "alkane_H": "H", "arom_H": "H", "alkene_H": "H",
    "alcohol_H": "H_HB", "amine_H": "H_HB", "amide_H": "H_HB", "H2": "H",
    "ether_O": "O_3", "ester_Os": "O_3", "alcohol_O": "O_3_hd", "siloxane_O": "O_3",
    "amide_N": "N_R_d2", "amide_NH": "N_R_d2_hd", "amine_N": "N_3",
    "amine_NH": "N_3_hd", "amine_NH2": "N_3_hd", "imine_N": "N_2_d1",
    "arom_N": "N_R_d1", "arom_NH": "N_R_d2_hd", "nitrile_N": "N_1",
    "nitro_N": "N_2_b2_d2",
    "Si": "Si_3", "sulfide_S": "S_3", "thiol_S": "S_3", "sulfone_S": "S_3",
    "F_alkyl": "F", "Cl_alkyl": "Cl", "Br_alkyl": "Br", "I_alkyl": "I",
    "P": "P_3_d3", "B": "B_3",
}
_DREIDING_FALLBACK = {"C": "C_3", "H": "H", "O": "O_3", "N": "N_3", "Si": "Si_3",
                      "S": "S_3", "F": "F", "Cl": "Cl", "Br": "Br", "I": "I",
                      "P": "P_3_d3", "B": "B_3", "Na": "Na", "Ca": "Ca",
                      "Fe": "Fe", "Zn": "Zn"}

# GAFF / GAFF2 (Wang 2004). h1/h2/h3 count electron-withdrawing neighbours
# of the parent carbon — handled in _refine_gaff.
_GAFF: Dict[str, str] = {
    "alkane_CH3": "c3", "alkane_CH2": "c3", "alkane_CH": "c3", "alkane_C": "c3",
    "arom_CH": "ca", "arom_C": "ca",
    "alkene_CH": "c2", "alkene_C": "c2", "alkyne_C": "c1", "nitrile_C": "c1",
    "ester_C": "c", "ketone_C": "c", "amide_C": "c", "imine_C": "c2",
    "alkane_H": "hc", "alkane_H_X": "h1", "arom_H": "ha", "alkene_H": "ha",
    "alcohol_H": "ho", "amine_H": "hn", "amide_H": "hn", "thiol_H": "hs",
    "ether_O": "os", "ester_Os": "os", "epoxide_O": "os", "alcohol_O": "oh",
    "ester_Odb": "o", "ketone_O": "o", "nitro_O": "o", "siloxane_O": "os",
    "amide_N": "n", "amide_NH": "n", "amine_N": "n3", "amine_NH": "n3",
    "amine_NH2": "n3", "aniline_N": "nh", "imine_N": "n2", "arom_N": "nb",
    "arom_NH": "na", "nitrile_N": "n1", "nitro_N": "no",
    "sulfide_S": "ss", "thiol_S": "sh", "sulfone_S": "s6", "thione_S": "s",
    "F_alkyl": "f", "Cl_alkyl": "cl", "Br_alkyl": "br", "I_alkyl": "i",
    "P": "p5", "phosphate_P": "p5",
}
_GAFF_FALLBACK = {"C": "c3", "H": "hc", "O": "os", "N": "n3", "S": "ss",
                  "F": "f", "Cl": "cl", "Br": "br", "I": "i", "P": "p5"}

# TraPPE-UA 1998: heavy-atom pseudo-atoms only.
_TRAPPE: Dict[str, str] = {"alkane_CH3": "CH3", "alkane_CH2": "CH2",
                           "methane_C": "CH4"}

# OPLS-UA 2024 (as extracted in oplsua_2024.lt).
_OPLSUA: Dict[str, str] = {
    "alkane_CH3": "68", "alkane_CH2": "71", "alkane_CH": "73", "alkane_C": "76",
    "alkene_CH2": "72", "alkene_CH": "74", "alkene_C": "77", "arom_CH": "75",
    "alkane_CH3_X": "109", "alkane_CH2_X": "110", "alkane_CH_X": "106",
    "alkane_C_X": "107", "ether_O": "108", "ester_Os": "108",
    "alcohol_O": "78", "alcohol_H": "79",
    "sulfide_S": "84", "thiol_S": "83", "thiol_H": "87", "disulfide_S": "85",
    "nitrile_N": "94", "nitrile_C": "95", "amide_C": "131", "ketone_O": "129",
    "amide_N": "130", "amide_NH": "130", "methane_C": "66",
}

# OPLS-AA fallback map (used only when the dedicated typer is unavailable).
_OPLSAA: Dict[str, str] = {
    "alkane_CH3": "135", "alkane_CH2": "136", "alkane_CH": "137",
    "alkane_C": "139", "alkane_H": "140", "methane_C": "138",
    "arom_CH": "145", "arom_C": "148", "arom_H": "146",
    "alkene_CH": "142", "alkene_C": "141", "alkene_H": "144",
    "ester_C": "465", "ester_Odb": "466", "ester_Os": "467",
    "acid_C": "267", "acid_OH": "268", "acid_Odb": "269", "acid_H": "270",
    "amide_C": "177", "amide_Odb": "178", "amide_NH": "179", "amide_N": "180",
    "amide_H": "183", "nitrile_N": "753", "nitrile_C": "754",
    "alcohol_O": "154", "alcohol_H": "155", "ether_O": "180", "epoxide_O": "180",
    "F_alkyl": "719", "Cl_alkyl": "264", "Br_alkyl": "722", "I_alkyl": "725",
    "thiol_S": "142", "sulfide_S": "202", "sulfone_S": "473", "sulfone_O": "474",
    "amine_NH2": "739", "amine_NH": "740", "amine_N": "741", "amine_H": "742",
    "Si": "500", "P": "440",
}
_OPLSAA_FALLBACK = {"H": "140", "C": "135", "N": "739", "O": "154", "S": "202",
                    "F": "719", "Cl": "264", "Br": "722", "I": "725",
                    "P": "440", "Si": "500"}

FF_MAPS: Dict[str, Tuple[Dict[str, str], Dict[str, str]]] = {
    "compass_published": (_COMPASS, _COMPASS_FALLBACK),
    "dreiding":          (_DREIDING, _DREIDING_FALLBACK),
    "gaff":              (_GAFF, _GAFF_FALLBACK),
    "gaff2":             (_GAFF, _GAFF_FALLBACK),
    "trappe_ua":         (_TRAPPE, {}),
    "oplsua_2024":       (_OPLSUA, {}),
    "oplsaa":            (_OPLSAA, _OPLSAA_FALLBACK),
    "oplsaa2008":        (_OPLSAA, _OPLSAA_FALLBACK),
}

UNITED_ATOM_KEYS = {"trappe_ua", "oplsua_2024"}


def supported_keys() -> List[str]:
    return sorted(FF_MAPS)


def is_supported(ff_key: str) -> bool:
    return ff_key in FF_MAPS


# ------------------------------------------------------------- environment
_compiled = None


def _rules():
    global _compiled
    if _compiled is None:
        from rdkit import Chem
        out = []
        for smarts, key, desc in ENV_RULES:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                log.warning("generic typer: bad SMARTS %r", smarts)
                continue
            out.append((patt, key, desc))
        _compiled = out
    return _compiled


def _rdkit_mol(mol):
    """RDKit graph from a PAAF Molecule (explicit H, bond orders kept)."""
    from rdkit import Chem
    rw = Chem.RWMol()
    for a in mol.atoms:
        try:
            at = Chem.Atom(a.element)
        except Exception:
            at = Chem.Atom(0)
        at.SetNoImplicit(True)
        rw.AddAtom(at)
    bt = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for i, j, o in mol.bonds:
        try:
            order = float(o)
        except Exception:
            order = 1.0
        b = Chem.BondType.AROMATIC if abs(order - 1.5) < 1e-6 else bt.get(int(round(order)), Chem.BondType.SINGLE)
        if rw.GetBondBetweenAtoms(int(i), int(j)) is None:
            rw.AddBond(int(i), int(j), b)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                         | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                         | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                         | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                         catchErrors=True)
    return m


def classify(mol) -> Dict[int, Tuple[str, str]]:
    """``{atom_index: (env_key, description)}`` for every atom a rule matches.

    Bond orders come from the Molecule; when they are all single (a chain
    read back from PDB/xyz) aromatic rings and C=O are still recognised
    because RDKit re-perceives them from the graph where it can. Chains
    built by PAAF carry recovered bond orders, so this is the normal case.
    """
    m = _rdkit_mol(mol)
    out: Dict[int, Tuple[str, str]] = {}
    for patt, key, desc in _rules():
        try:
            matches = m.GetSubstructMatches(patt, uniquify=False, maxMatches=1_000_000)
        except Exception:
            continue
        for match in matches:
            i = match[0]
            if i not in out:
                out[i] = (key, desc)
    return out


def _lookup(key: str, table: Dict[str, str]) -> Optional[str]:
    seen = set()
    while key and key not in seen:
        if key in table:
            return table[key]
        seen.add(key)
        key = _ALIAS.get(key, "")
    return None


def _refine_gaff(mol, envs, types):
    """GAFF h1/h2/h3: H on sp3 C with 1/2/3 electron-withdrawing neighbours."""
    ewg = {"N", "O", "F", "Cl", "Br", "I", "S"}
    for i, a in enumerate(mol.atoms):
        if a.element != "H" or types.get(i) not in ("hc", "h1"):
            continue
        nb = mol.neighbors(i)
        if not nb:
            continue
        c = nb[0]
        if mol.atoms[c].element != "C":
            continue
        n_ewg = sum(1 for j in mol.neighbors(c) if mol.atoms[j].element in ewg)
        if n_ewg >= 1:
            types[i] = {1: "h1", 2: "h2"}.get(n_ewg, "h3")


def type_generic(mol, ff_key: str) -> Dict[int, str]:
    """``{atom_index: type_id}`` for ``ff_key``; atoms with no type are absent."""
    if ff_key not in FF_MAPS:
        raise KeyError(f"no generic typing table for force field {ff_key!r}")
    table, fallback = FF_MAPS[ff_key]
    envs = classify(mol)
    types: Dict[int, str] = {}
    missing = []
    for i, a in enumerate(mol.atoms):
        key = envs.get(i, ("", ""))[0]
        t = _lookup(key, table) if key else None
        if t is None:
            t = fallback.get(a.element)
        if t is None:
            if not (ff_key in UNITED_ATOM_KEYS and a.element == "H"):
                missing.append(i)
            continue
        types[i] = t
    if ff_key.startswith("gaff"):
        _refine_gaff(mol, envs, types)
    if missing:
        log.warning("generic typer (%s): %d atom(s) without a type, e.g. %s",
                    ff_key, len(missing),
                    ", ".join(f"{mol.atoms[i].element}{i+1}" for i in missing[:6]))
    return types


def describe(mol, ff_key: str) -> Dict[int, Tuple[str, str, str]]:
    """``{atom_index: (type_id, env_key, description)}`` — for the dialog."""
    envs = classify(mol)
    types = type_generic(mol, ff_key)
    return {i: (t, envs.get(i, ("", ""))[0], envs.get(i, ("", ""))[1])
            for i, t in types.items()}
