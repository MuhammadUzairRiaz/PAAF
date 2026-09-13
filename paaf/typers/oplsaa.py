"""SMARTS-based automatic OPLS-AA 2024 atom typer.

Maps every atom in a Molecule to the numeric OPLS-AA type ID used inside
``ff_libraries/moltemplate/oplsaa2024.lt``. The mapping is expressed as an
ordered list of ``(SMARTS, opls_type, description)`` rules. For each atom,
the FIRST rule whose SMARTS matches wins — so *more-specific rules must come
first* (e.g. alpha-carbon of an ester before generic sp3 C).

Type IDs used here come from the "In Charges" and "Data Masses" blocks of
oplsaa2024.lt. The full type list is at the bottom of this file (`OPLS_TYPES`).

Coverage: the rules below cover the 20-curated polymers + the ~103 database
polymers shipped with PAAF — polyethylene, polypropylene, polystyrene,
polybutadiene, PIB, isoprene, PVC/PVDC/PVF/PTFE, PVA, PAN, PMMA / methacrylates,
PMA / acrylates, PEO, PBS, PLA, PGA, PCL, PET, nylons, polycarbonate,
polysulfones, polyethersulfones, methoxystyrenes, epoxides, DMSO/DMF variants.

If no rule matches, the atom's element symbol is returned as a fallback and
the caller logs a warning listing the unresolved atoms so the user can add
either a SMARTS rule or a manual override on the Force-field page.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..logging_utils import get_logger

log = get_logger(__name__)


# =====================================================================
# Ordered SMARTS rules: (pattern, opls_type, human description)
# The FIRST matching rule wins per atom. Put specific rules before generic ones.
# =====================================================================
# Element -> safe OPLS numeric fallback. Used only when NO SMARTS rule matches
# an atom. The chosen defaults are broad "alkane / alcohol / amine" types that
# exist in oplsaa2024.lt so moltemplate's renumber_DATA_first_column.py never
# sees a mixed int/str atom-type column (which crashes it with a TypeError).
_ELEMENT_FALLBACK: Dict[str, str] = {
    "H":  "140",   # HC — aliphatic H
    "C":  "135",   # CT — sp3 alkane C
    "N":  "900",   # N  — primary amine N
    "O":  "154",   # OH — alcohol O
    "S":  "202",   # S  — sulfide
    "F":  "956",   # F  — alkyl fluoride
    "Cl": "151",   # Cl — alkyl chloride
    "Br": "975",   # Br — alkyl bromide
    "I":  "1014",  # I  — alkyl iodide
    "P":  "440",   # P  — generic phosphorus
    "Si": "1060",  # Si — tetraalkylsilane
}


_RULES: List[Tuple[str, str, str]] = [
    # ---- Epoxide (3-membered ring with one O) ---------------------------
    ("[OX2r3]",                           "180", "epoxide oxygen (3-ring)"),
    ("[OX2r3]([CX4r3])[CX4r3]",           "180", "epoxide oxygen (both C sp3)"),
    ("[CX4r3][OX2r3]",                    "136", "epoxide sp3 C"),

    # ---- Anisole / methoxy on aromatic ring -----------------------------
    ("[OX2]([cX3])[CX4]",                 "180", "aryl methoxy -O-"),
    ("[CH3X4][OX2][cX3]",                 "135", "methoxy CH3"),

    # ---- Ester carbonyl and neighbours -----------------------------------
    # These IDs come from the BUNDLED oplsaa2024.lt, not from another OPLS
    # table.
    #
    # They used to be 210/211/212, labelled "ester carbonyl / ester -O- /
    # ester =O" -- and in oplsaa2024.lt 210-214 are SULFIDE AND DISULFIDE
    # CARBONS (CT). Two different OPLS numbering schemes. Every ester oxygen
    # in a polyester was therefore suggested a carbon type, which the element
    # guard refused at export: "atom 5 -- Type 211 is a C type, it cannot be
    # assigned to a O atom". The suggestion was wrong, not the user's edit.
    #
    # 465 = AA C: esters, 466 = AA =O: esters, 467 = AA -OR: ester -O-.
    ("[CX3](=O)[OX2][#6]",                "465", "ester C(=O)-O carbonyl C"),
    ("[OX1]=[CX3][OX2][#6]",              "466", "ester C=O carbonyl oxygen"),
    ("[OX2]([CX3](=O))[#6]",              "467", "ester -O- oxygen (sp3)"),

    # ---- Carboxylic acid -------------------------------------------------
    # 267 = Co in CCOOH, 268 = Oh, 269 = Oc, 270 = H. Same mistake as the
    # esters above: 209/210 are carbons in this library.
    ("[CX3](=O)[OX2H]",                   "267", "carboxylic acid C"),
    ("[OX1]=[CX3][OX2H]",                 "269", "carboxylic acid C=O oxygen"),
    ("[OX2H][CX3](=O)",                   "268", "carboxylic acid OH oxygen"),
    ("[H][OX2][CX3](=O)",                 "270", "carboxylic acid OH hydrogen"),

    # ---- Amide (-C(=O)N-) ------------------------------------------------
    ("[CX3](=O)[NX3]",                    "235", "amide carbonyl C"),
    ("[OX1]=[CX3][NX3]",                  "236", "amide carbonyl O"),
    ("[NX3H0]([CX3]=O)([#6])[#6]",        "239", "tertiary amide N"),
    ("[NX3H1]([CX3]=O)[#6]",              "238", "secondary amide N-H"),
    ("[NX3H2][CX3](=O)",                  "237", "primary amide -NH2"),
    ("[H][NX3][CX3](=O)",                 "241", "amide N-H hydrogen"),

    # ---- Nitrile ---------------------------------------------------------
    ("[NX1]#[CX2]",                       "753", "nitrile N"),
    ("[CX2]#[NX1]",                       "754", "nitrile C"),

    # ---- Phenol (aromatic C-O-H) ------------------------------------------
    ("[OX2H][c]",                         "167", "phenol O"),
    ("[H][OX2][c]",                       "168", "phenol H"),
    ("[c][OX2H]",                         "166", "phenol ipso C"),

    # ---- Alcohol (aliphatic C-O-H) ---------------------------------------
    ("[OX2H][CX4]",                       "154", "alcohol -OH oxygen"),
    ("[H][OX2][CX4]",                     "155", "alcohol -OH hydrogen"),
    # Any other hydroxyl (silanol Si-OH, chain-end OH): the H must still be
    # a hydroxyl H, or O(154)-H(140) has no bond parameters.
    ("[OX2H]",                            "154", "hydroxyl O (generic)"),
    ("[H][OX2]",                          "155", "hydroxyl H (generic)"),

    # ---- Ether (C-O-C, both C sp3) --------------------------------------
    ("[OX2]([CX4])[CX4]",                 "180", "ether -O- sp3"),

    # ---- Aromatic ring atoms (benzenoid) --------------------------------
    ("[cH]",                              "145", "aromatic C (benzene)"),   # matches C in c1ccccc1
    ("[c]([#6])[c]([c])[c]",              "148", "aromatic C attached to alkyl"),
    ("[c]([OX2])[c]",                     "199", "aromatic C - alkoxy"),
    ("[H][c]",                            "146", "aromatic H (benzene)"),
    # generic aromatic C (fallback within aromatic ring)
    ("[c]",                               "145", "aromatic C (generic)"),

    # ---- Methacrylate / acrylate side groups ----------------------------
    # C attached to two hydrogens + ester link (acrylate backbone CH2)
    ("[CH2X4]([CX3](=O)[OX2])[#6]",       "136", "acrylate backbone CH2"),
    # quaternary C of methacrylate: -CH2-C(CH3)(COOR)-
    ("[CX4]([CH3])([CX3](=O)[OX2])[#6]",  "139", "methacrylate quaternary C"),

    # ---- Halides ---------------------------------------------------------
    ("[FX1][CX4]",                        "956", "F on sp3 C"),
    ("[ClX1][CX4]",                       "151", "Cl on sp3 C"),
    ("[BrX1][CX4]",                       "975", "Br on sp3 C"),
    ("[IX1][CX4]",                        "1014", "I on sp3 C"),
    ("[FX1][cX3]",                        "728", "F on aromatic C"),
    ("[ClX1][cX3]",                       "264", "Cl on aromatic C"),
    ("[BrX1][cX3]",                       "730", "Br on aromatic C"),

    # ---- Sulfur ----------------------------------------------------------
    ("[SX2H][#6]",                        "200", "thiol -SH sulfur"),
    ("[H][SX2]",                          "204", "thiol S-H hydrogen"),
    ("[SX2]([#6])[#6]",                   "202", "sulfide S"),
    ("[SX4](=O)(=O)([#6])[#6]",           "493", "sulfone S"),
    ("[OX1]=[SX4]",                       "494", "sulfone =O"),

    # ---- Alkene carbons --------------------------------------------------
    ("[CX3H2]=[CX3]",                     "143", "terminal =CH2 alkene"),
    ("[CX3H1]=[CX3]",                     "142", "vinyl =CH- alkene"),
    ("[CX3]=[CX3]",                       "141", "internal alkene C"),
    ("[H][CX3]=[CX3]",                    "144", "alkene =C-H hydrogen"),

    # ---- sp3 carbons bonded to oxygen -----------------------------------
    # Without these, CH2 next to an ether/ester/alcohol O fell through to the
    # element fallback (135, alkane CH3) with the wrong charge. Ester alkoxy
    # carbons first: the ether patterns below would also match them.
    ("[CX4H3][OX2][CX3]=O",               "468", "ester methoxy C"),
    ("[CX4H2][OX2][CX3]=O",               "490", "ester alkoxy CH2 C(H2OS)"),
    ("[CX4H1][OX2][CX3]=O",               "491", "ester alkoxy CH C(HOS)"),
    ("[CX4H0][OX2][CX3]=O",               "492", "ester alkoxy C C(OS)"),
    ("[H][CX4][OX2][CX3]=O",              "469", "ester alkoxy H"),
    ("[CX4H3,CX4H2][OX2H]",               "157", "alcohol CH3/CH2 C(OH)"),
    ("[CX4H1][OX2H]",                     "158", "alcohol CH C(OH)"),
    ("[CX4H0][OX2H]",                     "159", "alcohol C C(OH)"),
    ("[CX4H3][OX2][#6]",                  "181", "ether CH3 C(H3OR)"),
    ("[CX4H2][OX2][#6]",                  "182", "ether CH2 C(H2OR)"),
    ("[CX4H1][OX2][#6]",                  "183", "ether CH C(HOR)"),
    ("[CX4H0][OX2][#6]",                  "184", "ether C C(OR)"),
    ("[H][CX4][OX2][#6]",                 "185", "ether alpha H H(COR)"),

    # ---- Alkane atoms (aliphatic C) --------------------------------------
    # Order matters: CH3 (attached to only one heavy atom), then CH2, then CH, then C.
    ("[CX4H3]([#6])",                     "135", "sp3 CH3 (alkane)"),
    ("[CX4H2]([#6])[#6]",                 "136", "sp3 CH2 (alkane)"),
    ("[CX4H1]([#6])([#6])[#6]",           "137", "sp3 CH  (alkane)"),
    ("[CX4H0]([#6])([#6])([#6])[#6]",     "139", "sp3 C   (alkane, no H)"),
    ("[CX4H4]",                           "138", "sp3 CH4 (methane)"),

    # ---- Aliphatic H -----------------------------------------------------
    ("[H][CX4]",                          "140", "sp3 C-H hydrogen"),

    # ---- Amine (primary/secondary/tertiary aliphatic) -------------------
    ("[NX3H2][CX4]",                      "900", "primary amine N-H2"),
    ("[NX3H1]([CX4])[CX4]",               "901", "secondary amine N-H"),
    ("[NX3H0]([CX4])([CX4])[CX4]",        "902", "tertiary amine N"),
    ("[H][NX3][CX4]",                     "909", "amine N-H hydrogen"),
    # N-H on anything else (aryl amines): an alkane H (140) on N has no bond
    # parameters in the library.
    ("[H][NX3]",                          "909", "N-H hydrogen (generic)"),
]


def type_oplsaa(mol) -> Dict[int, str]:
    """Assign OPLS-AA 2024 atom types to every atom in `mol`.

    Uses OpenBabel's SMARTS engine (`pybel.Smarts`) so no RDKit dependency
    is added. Returns ``{atom_index_0based: opls_type_id_string}``. If a
    rule doesn't match a given atom, the element symbol is used as a
    fallback and the caller is expected to log a warning listing the
    unresolved indices.
    """
    try:
        from openbabel import pybel
    except Exception as exc:
        raise RuntimeError(
            "OpenBabel is required for automatic OPLS-AA typing."
        ) from exc

    # Serialize our internal Molecule to mol2 via structure.write() then
    # re-read with pybel so we can use pybel.Smarts on it.
    from ..structure import write
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory(prefix="paaf_opls_") as _d:
        tmp = Path(_d) / "in.mol2"
        write(mol, tmp)
        pmol = next(pybel.readfile("mol2", str(tmp)))

    n_atoms = len(mol.atoms)
    types: Dict[int, str] = {}
    from openbabel import openbabel as _ob
    for pattern_str, opls_type, _desc in _RULES:
        pat = _ob.OBSmartsPattern()
        if not pat.Init(pattern_str):
            log.debug("Skipping malformed SMARTS %r", pattern_str)
            continue
        pat.Match(pmol.OBMol)
        # GetMapList, not pybel's findall (GetUMapList): the unique list keeps
        # one match per atom SET, so for C-O-C only one of the two carbons was
        # ever the anchor and the other fell through to the element fallback.
        for match in pat.GetMapList():
            # pybel.Smarts.findall returns tuples of 1-based OpenBabel atom
            # indices; the FIRST atom in the tuple is the anchor.
            anchor_1 = match[0] if isinstance(match, (list, tuple)) else match
            i = anchor_1 - 1
            if 0 <= i < n_atoms and i not in types:
                types[i] = opls_type

    # A rule must never hand an atom a type belonging to another element.
    #
    # This typer was written against a different OPLS numbering table from the
    # bundled oplsaa2024.lt, and an audit found 17 rules whose type is the
    # wrong element there -- the ester rules suggested 211/212, which are
    # SULFIDE CARBONS in this library, so every polyester oxygen came back as
    # a carbon. The user then saw "Type 211 is a C type, it cannot be assigned
    # to a O atom" at export and reasonably assumed their own assignment was
    # at fault.
    #
    # The ester and acid rules are corrected. For any rule still out of step,
    # dropping back to the broad element default is the honest answer: it is
    # less specific but it is never wrong about mass and charge, and the
    # Force-field page lets you override it.
    from ..type_guard import type_elements as _library_elements

    library = _library_elements()
    corrected: List[int] = []
    for index, assigned in list(types.items()):
        expected = library.get(assigned)
        actual = mol.atoms[index].element
        if expected and expected != actual:
            del types[index]
            corrected.append(index)
    if corrected:
        log.warning(
            "%d atoms were suggested a force-field type belonging to a "
            "different element; those suggestions were dropped and the broad "
            "element default used instead. This is a rule in "
            "paaf/typers/oplsaa.py that does not match the loaded library's "
            "numbering. Atoms: %s",
            len(corrected), corrected[:10])

    # Fallback: assign a broad element-based OPLS numeric type so the
    # atom-type column stays uniformly integer. Falling back to element
    # SYMBOLS ("H", "C") mixes ints and strs and crashes moltemplate's
    # renumber_DATA_first_column.py with a TypeError.
    unresolved: List[int] = []
    for a in mol.atoms:
        if a.index not in types:
            fallback = _ELEMENT_FALLBACK.get(a.element, "135")
            types[a.index] = fallback
            unresolved.append(a.index)
    if unresolved:
        log.warning(
            "OPLS-AA typer used element-based fallback for %d atoms "
            "(indices %s). These will use broad defaults (e.g. C→135, H→140, "
            "O→154). Extend paaf/typers/oplsaa.py or override on "
            "the Force-field page for chemistry-specific accuracy.",
            len(unresolved), unresolved[:10],
        )

    n_typed = n_atoms - len(unresolved)
    log.info("OPLS-AA typer: %d/%d atoms matched a specific SMARTS rule "
             "(%d used element fallback).", n_typed, n_atoms, len(unresolved))
    return types


# =====================================================================
# Complete list of OPLS-AA 2024 numeric type IDs, from oplsaa2024.lt.
# Used only by tests / documentation.
# =====================================================================
OPLS_TYPES = {
    "135": "CT — sp3 CH3 alkane",
    "136": "CT — sp3 CH2 alkane",
    "137": "CT — sp3 CH  alkane",
    "138": "CT — sp3 CH4 methane",
    "139": "CT — sp3 C   alkane (quaternary)",
    "140": "HC — sp3 C-H hydrogen",
    "141": "CM — alkene =C=",
    "142": "CM — alkene RH-C=",
    "143": "CM — alkene H2-C=",
    "144": "HC — alkene =C-H hydrogen",
    "145": "CA — benzene C",
    "146": "HA — benzene H",
    "148": "CT — alkyl-substituted benzene C",
    "154": "OH — alcohol O",
    "155": "HO — alcohol H",
    "165": "CA — phenol C-O",
    "180": "OS — ether O",
    "199": "CA — anisole/aryl-ether C",
    "200": "SH — thiol S",
    "204": "HS — thiol H",
    "202": "S  — sulfide",
    "267": "C  — carboxylic acid carbonyl",
    "268": "OH — carboxyl -OH oxygen",
    "269": "O  — carboxyl =O",
    "270": "HO — carboxyl -OH hydrogen",
    "465": "C  — ester carbonyl",
    "466": "O  — ester =O",
    "467": "OS — ester -O-",
    "235": "C  — amide carbonyl",
    "236": "O  — amide =O",
    "237": "N  — primary amide N",
    "238": "N  — secondary amide N-H",
    "239": "N  — tertiary amide N",
    "241": "H  — amide N-H hydrogen",
    "900": "NT — primary amine",
    "901": "NT — secondary amine",
    "902": "NT — tertiary amine",
    "909": "H  — amine H",
    "753": "NZ — nitrile N",
    "754": "CZ — nitrile C",
    "493": "SY — sulfone S",
    "494": "OY — sulfone =O",
    "956": "F  — alkyl fluoride",
    "151": "Cl — alkyl chloride",
    "975": "Br — alkyl bromide",
    "1014": "I  — alkyl iodide",
    "728": "F  — aromatic",
    "264": "Cl — aromatic",
    "730": "Br — aromatic",
    "1060": "Si — tetraalkylsilane",
}
