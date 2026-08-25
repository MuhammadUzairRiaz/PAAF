"""Avogadro-style fragment / functional-group library.

Each entry gives a chemistry class, a name, and a SMILES snippet. The
GUI Builder tab presents them as a browsable palette; clicking a fragment
appends its SMILES to the current input so the user can compose molecules
without typing SMILES from scratch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Fragment:
    key: str
    name: str
    smiles: str
    category: str


FRAGMENTS: Dict[str, Fragment] = {
    # ---- Alkyl groups
    "methyl":       Fragment("methyl",       "Methyl -CH3",           "C",         "alkyl"),
    "ethyl":        Fragment("ethyl",        "Ethyl -CH2CH3",         "CC",        "alkyl"),
    "propyl":       Fragment("propyl",       "n-Propyl",              "CCC",       "alkyl"),
    "isopropyl":    Fragment("isopropyl",    "iso-Propyl",            "C(C)C",     "alkyl"),
    "butyl":        Fragment("butyl",        "n-Butyl",               "CCCC",      "alkyl"),
    "tbutyl":       Fragment("tbutyl",       "t-Butyl",               "C(C)(C)C",  "alkyl"),
    "vinyl":        Fragment("vinyl",        "Vinyl -CH=CH2",         "C=C",       "alkyl"),
    "allyl":        Fragment("allyl",        "Allyl",                 "CC=C",      "alkyl"),
    "cyclopropyl":  Fragment("cyclopropyl",  "Cyclopropyl",           "C1CC1",     "alkyl"),
    "cyclohexyl":   Fragment("cyclohexyl",   "Cyclohexyl",            "C1CCCCC1",  "alkyl"),
    # ---- Oxygen groups
    "hydroxyl":     Fragment("hydroxyl",     "Hydroxyl -OH",          "O",         "oxygen"),
    "methoxy":      Fragment("methoxy",      "Methoxy -OMe",          "OC",        "oxygen"),
    "ethoxy":       Fragment("ethoxy",       "Ethoxy -OEt",           "OCC",       "oxygen"),
    "carbonyl":     Fragment("carbonyl",     "Carbonyl C=O",          "C=O",       "oxygen"),
    "carboxyl":     Fragment("carboxyl",     "Carboxyl -COOH",        "C(=O)O",    "oxygen"),
    "ester_methyl": Fragment("ester_methyl", "Methyl ester -COOMe",   "C(=O)OC",   "oxygen"),
    "aldehyde":     Fragment("aldehyde",     "Aldehyde -CHO",         "C=O",       "oxygen"),
    "acetate":      Fragment("acetate",      "Acetate -OC(=O)CH3",    "OC(=O)C",   "oxygen"),
    "epoxide":      Fragment("epoxide",      "Epoxide",               "C1CO1",     "oxygen"),
    # ---- Nitrogen groups
    "amino":        Fragment("amino",        "Amino -NH2",            "N",         "nitrogen"),
    "methylamine":  Fragment("methylamine",  "Methylamine -NHMe",     "NC",        "nitrogen"),
    "dimethylamine":Fragment("dimethylamine","Dimethylamine -NMe2",   "N(C)C",     "nitrogen"),
    "amide":        Fragment("amide",        "Amide -C(=O)NH2",       "C(=O)N",    "nitrogen"),
    "nitrile":      Fragment("nitrile",      "Nitrile -C#N",          "C#N",       "nitrogen"),
    "nitro":        Fragment("nitro",        "Nitro -NO2",            "[N+](=O)[O-]", "nitrogen"),
    "azide":        Fragment("azide",        "Azide -N3",             "N=[N+]=[N-]",  "nitrogen"),
    # ---- Sulfur / phosphorus
    "thiol":        Fragment("thiol",        "Thiol -SH",             "S",         "sulfur"),
    "sulfonate":    Fragment("sulfonate",    "Sulfonate -SO3H",       "S(=O)(=O)O", "sulfur"),
    "sulfoxide":    Fragment("sulfoxide",    "Sulfoxide S=O",         "S=O",       "sulfur"),
    "phosphate":    Fragment("phosphate",    "Phosphate -OP(=O)(OH)2","OP(=O)(O)O","phosphorus"),
    # ---- Halogens
    "fluoro":       Fragment("fluoro",       "Fluoro -F",             "F",         "halogen"),
    "chloro":       Fragment("chloro",       "Chloro -Cl",            "Cl",        "halogen"),
    "bromo":        Fragment("bromo",        "Bromo -Br",             "Br",        "halogen"),
    "iodo":         Fragment("iodo",         "Iodo -I",               "I",         "halogen"),
    "trifluoromethyl": Fragment("trifluoromethyl", "Trifluoromethyl -CF3", "C(F)(F)F", "halogen"),
    # ---- Aromatic rings
    "phenyl":       Fragment("phenyl",       "Phenyl -C6H5",          "c1ccccc1",  "aromatic"),
    "pyridyl_2":    Fragment("pyridyl_2",    "2-Pyridyl",             "c1ccccn1",  "aromatic"),
    "pyridyl_3":    Fragment("pyridyl_3",    "3-Pyridyl",             "c1cccnc1",  "aromatic"),
    "pyridyl_4":    Fragment("pyridyl_4",    "4-Pyridyl",             "c1ccncc1",  "aromatic"),
    "furyl":        Fragment("furyl",        "Furyl",                 "c1ccoc1",   "aromatic"),
    "thienyl":      Fragment("thienyl",      "Thienyl",               "c1ccsc1",   "aromatic"),
    "pyrrolyl":     Fragment("pyrrolyl",     "Pyrrolyl",              "c1cc[nH]c1","aromatic"),
    "imidazolyl":   Fragment("imidazolyl",   "Imidazolyl",            "c1cnc[nH]1","aromatic"),
    "naphthyl":     Fragment("naphthyl",     "Naphthyl",              "c1ccc2ccccc2c1", "aromatic"),
    "benzyl":       Fragment("benzyl",       "Benzyl -CH2C6H5",       "Cc1ccccc1", "aromatic"),
    # ---- Common linkers
    "amide_link":   Fragment("amide_link",   "Amide linker -C(=O)NH-",  "C(=O)N",   "linker"),
    "ester_link":   Fragment("ester_link",   "Ester linker -C(=O)O-",   "C(=O)O",   "linker"),
    "ether_link":   Fragment("ether_link",   "Ether linker -O-",        "O",        "linker"),
    "urea_link":    Fragment("urea_link",    "Urea -NHC(=O)NH-",        "NC(=O)N",  "linker"),
    "urethane_link":Fragment("urethane_link","Urethane -OC(=O)NH-",     "OC(=O)N",  "linker"),
    "carbonate_link":Fragment("carbonate_link","Carbonate -OC(=O)O-",   "OC(=O)O",  "linker"),
    "siloxane_link":Fragment("siloxane_link","Siloxane -Si(CH3)2-O-",   "[Si](C)(C)O", "linker"),
}


def list_fragments() -> List[Fragment]:
    return list(FRAGMENTS.values())


def get_fragment(key: str) -> Fragment:
    if key not in FRAGMENTS:
        raise KeyError(f"Unknown fragment {key!r}")
    return FRAGMENTS[key]
