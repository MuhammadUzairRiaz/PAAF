"""Standalone periodic-table dataset.

Kept outside `paaf.gui` so it can be imported without PyQt5
(useful for CLI tools, tests, and downstream code that just needs
element info).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Element:
    Z: int
    symbol: str
    name: str
    mass: float
    row: int          # 1-based row in the standard extended layout
    col: int          # 1-based column
    category: str


CATEGORY_COLORS: Dict[str, Tuple[str, str]] = {
    "alkali metal":         ("#111827", "#fca5a5"),
    "alkaline earth metal": ("#111827", "#fdba74"),
    "transition metal":     ("#111827", "#fde68a"),
    "post-transition metal":("#111827", "#a7f3d0"),
    "metalloid":            ("#111827", "#93c5fd"),
    "nonmetal":             ("#111827", "#c4b5fd"),
    "halogen":              ("#111827", "#f9a8d4"),
    "noble gas":            ("#111827", "#e0e7ff"),
    "lanthanide":           ("#111827", "#fbcfe8"),
    "actinide":             ("#111827", "#ddd6fe"),
    "unknown":              ("#111827", "#d1d5db"),
}


_ELEMENTS_RAW = [
    (1,  "H",  "Hydrogen",       1.008,   1, 1,  "nonmetal"),
    (2,  "He", "Helium",         4.003,   1, 18, "noble gas"),
    (3,  "Li", "Lithium",        6.94,    2, 1,  "alkali metal"),
    (4,  "Be", "Beryllium",      9.012,   2, 2,  "alkaline earth metal"),
    (5,  "B",  "Boron",          10.81,   2, 13, "metalloid"),
    (6,  "C",  "Carbon",         12.011,  2, 14, "nonmetal"),
    (7,  "N",  "Nitrogen",       14.007,  2, 15, "nonmetal"),
    (8,  "O",  "Oxygen",         15.999,  2, 16, "nonmetal"),
    (9,  "F",  "Fluorine",       18.998,  2, 17, "halogen"),
    (10, "Ne", "Neon",           20.180,  2, 18, "noble gas"),
    (11, "Na", "Sodium",         22.990,  3, 1,  "alkali metal"),
    (12, "Mg", "Magnesium",      24.305,  3, 2,  "alkaline earth metal"),
    (13, "Al", "Aluminium",      26.982,  3, 13, "post-transition metal"),
    (14, "Si", "Silicon",        28.085,  3, 14, "metalloid"),
    (15, "P",  "Phosphorus",     30.974,  3, 15, "nonmetal"),
    (16, "S",  "Sulfur",         32.06,   3, 16, "nonmetal"),
    (17, "Cl", "Chlorine",       35.45,   3, 17, "halogen"),
    (18, "Ar", "Argon",          39.948,  3, 18, "noble gas"),
    (19, "K",  "Potassium",      39.098,  4, 1,  "alkali metal"),
    (20, "Ca", "Calcium",        40.078,  4, 2,  "alkaline earth metal"),
    (21, "Sc", "Scandium",       44.956,  4, 3,  "transition metal"),
    (22, "Ti", "Titanium",       47.867,  4, 4,  "transition metal"),
    (23, "V",  "Vanadium",       50.942,  4, 5,  "transition metal"),
    (24, "Cr", "Chromium",       51.996,  4, 6,  "transition metal"),
    (25, "Mn", "Manganese",      54.938,  4, 7,  "transition metal"),
    (26, "Fe", "Iron",           55.845,  4, 8,  "transition metal"),
    (27, "Co", "Cobalt",         58.933,  4, 9,  "transition metal"),
    (28, "Ni", "Nickel",         58.693,  4, 10, "transition metal"),
    (29, "Cu", "Copper",         63.546,  4, 11, "transition metal"),
    (30, "Zn", "Zinc",           65.38,   4, 12, "transition metal"),
    (31, "Ga", "Gallium",        69.723,  4, 13, "post-transition metal"),
    (32, "Ge", "Germanium",      72.630,  4, 14, "metalloid"),
    (33, "As", "Arsenic",        74.922,  4, 15, "metalloid"),
    (34, "Se", "Selenium",       78.971,  4, 16, "nonmetal"),
    (35, "Br", "Bromine",        79.904,  4, 17, "halogen"),
    (36, "Kr", "Krypton",        83.798,  4, 18, "noble gas"),
    (37, "Rb", "Rubidium",       85.468,  5, 1,  "alkali metal"),
    (38, "Sr", "Strontium",      87.62,   5, 2,  "alkaline earth metal"),
    (39, "Y",  "Yttrium",        88.906,  5, 3,  "transition metal"),
    (40, "Zr", "Zirconium",      91.224,  5, 4,  "transition metal"),
    (41, "Nb", "Niobium",        92.906,  5, 5,  "transition metal"),
    (42, "Mo", "Molybdenum",     95.95,   5, 6,  "transition metal"),
    (43, "Tc", "Technetium",     98.0,    5, 7,  "transition metal"),
    (44, "Ru", "Ruthenium",      101.07,  5, 8,  "transition metal"),
    (45, "Rh", "Rhodium",        102.906, 5, 9,  "transition metal"),
    (46, "Pd", "Palladium",      106.42,  5, 10, "transition metal"),
    (47, "Ag", "Silver",         107.868, 5, 11, "transition metal"),
    (48, "Cd", "Cadmium",        112.414, 5, 12, "transition metal"),
    (49, "In", "Indium",         114.818, 5, 13, "post-transition metal"),
    (50, "Sn", "Tin",            118.710, 5, 14, "post-transition metal"),
    (51, "Sb", "Antimony",       121.760, 5, 15, "metalloid"),
    (52, "Te", "Tellurium",      127.60,  5, 16, "metalloid"),
    (53, "I",  "Iodine",         126.904, 5, 17, "halogen"),
    (54, "Xe", "Xenon",          131.293, 5, 18, "noble gas"),
    (55, "Cs", "Caesium",        132.905, 6, 1,  "alkali metal"),
    (56, "Ba", "Barium",         137.327, 6, 2,  "alkaline earth metal"),
    (57, "La", "Lanthanum",      138.905, 8, 3,  "lanthanide"),
    (58, "Ce", "Cerium",         140.116, 8, 4,  "lanthanide"),
    (59, "Pr", "Praseodymium",   140.908, 8, 5,  "lanthanide"),
    (60, "Nd", "Neodymium",      144.242, 8, 6,  "lanthanide"),
    (61, "Pm", "Promethium",     145.0,   8, 7,  "lanthanide"),
    (62, "Sm", "Samarium",       150.36,  8, 8,  "lanthanide"),
    (63, "Eu", "Europium",       151.964, 8, 9,  "lanthanide"),
    (64, "Gd", "Gadolinium",     157.25,  8, 10, "lanthanide"),
    (65, "Tb", "Terbium",        158.925, 8, 11, "lanthanide"),
    (66, "Dy", "Dysprosium",     162.500, 8, 12, "lanthanide"),
    (67, "Ho", "Holmium",        164.930, 8, 13, "lanthanide"),
    (68, "Er", "Erbium",         167.259, 8, 14, "lanthanide"),
    (69, "Tm", "Thulium",        168.934, 8, 15, "lanthanide"),
    (70, "Yb", "Ytterbium",      173.045, 8, 16, "lanthanide"),
    (71, "Lu", "Lutetium",       174.967, 8, 17, "lanthanide"),
    (72, "Hf", "Hafnium",        178.49,  6, 4,  "transition metal"),
    (73, "Ta", "Tantalum",       180.948, 6, 5,  "transition metal"),
    (74, "W",  "Tungsten",       183.84,  6, 6,  "transition metal"),
    (75, "Re", "Rhenium",        186.207, 6, 7,  "transition metal"),
    (76, "Os", "Osmium",         190.23,  6, 8,  "transition metal"),
    (77, "Ir", "Iridium",        192.217, 6, 9,  "transition metal"),
    (78, "Pt", "Platinum",       195.084, 6, 10, "transition metal"),
    (79, "Au", "Gold",           196.967, 6, 11, "transition metal"),
    (80, "Hg", "Mercury",        200.592, 6, 12, "transition metal"),
    (81, "Tl", "Thallium",       204.383, 6, 13, "post-transition metal"),
    (82, "Pb", "Lead",           207.2,   6, 14, "post-transition metal"),
    (83, "Bi", "Bismuth",        208.980, 6, 15, "post-transition metal"),
    (84, "Po", "Polonium",       209.0,   6, 16, "metalloid"),
    (85, "At", "Astatine",       210.0,   6, 17, "halogen"),
    (86, "Rn", "Radon",          222.0,   6, 18, "noble gas"),
    (87, "Fr", "Francium",       223.0,   7, 1,  "alkali metal"),
    (88, "Ra", "Radium",         226.0,   7, 2,  "alkaline earth metal"),
    (89, "Ac", "Actinium",       227.0,   9, 3,  "actinide"),
    (90, "Th", "Thorium",        232.038, 9, 4,  "actinide"),
    (91, "Pa", "Protactinium",   231.036, 9, 5,  "actinide"),
    (92, "U",  "Uranium",        238.029, 9, 6,  "actinide"),
    (93, "Np", "Neptunium",      237.0,   9, 7,  "actinide"),
    (94, "Pu", "Plutonium",      244.0,   9, 8,  "actinide"),
    (95, "Am", "Americium",      243.0,   9, 9,  "actinide"),
    (96, "Cm", "Curium",         247.0,   9, 10, "actinide"),
    (97, "Bk", "Berkelium",      247.0,   9, 11, "actinide"),
    (98, "Cf", "Californium",    251.0,   9, 12, "actinide"),
    (99, "Es", "Einsteinium",    252.0,   9, 13, "actinide"),
    (100,"Fm", "Fermium",        257.0,   9, 14, "actinide"),
    (101,"Md", "Mendelevium",    258.0,   9, 15, "actinide"),
    (102,"No", "Nobelium",       259.0,   9, 16, "actinide"),
    (103,"Lr", "Lawrencium",     262.0,   9, 17, "actinide"),
    (104,"Rf", "Rutherfordium",  267.0,   7, 4,  "transition metal"),
    (105,"Db", "Dubnium",        270.0,   7, 5,  "transition metal"),
    (106,"Sg", "Seaborgium",     271.0,   7, 6,  "transition metal"),
    (107,"Bh", "Bohrium",        270.0,   7, 7,  "transition metal"),
    (108,"Hs", "Hassium",        277.0,   7, 8,  "transition metal"),
    (109,"Mt", "Meitnerium",     276.0,   7, 9,  "unknown"),
    (110,"Ds", "Darmstadtium",   281.0,   7, 10, "unknown"),
    (111,"Rg", "Roentgenium",    282.0,   7, 11, "unknown"),
    (112,"Cn", "Copernicium",    285.0,   7, 12, "transition metal"),
    (113,"Nh", "Nihonium",       286.0,   7, 13, "post-transition metal"),
    (114,"Fl", "Flerovium",      289.0,   7, 14, "post-transition metal"),
    (115,"Mc", "Moscovium",      290.0,   7, 15, "post-transition metal"),
    (116,"Lv", "Livermorium",    293.0,   7, 16, "post-transition metal"),
    (117,"Ts", "Tennessine",     294.0,   7, 17, "halogen"),
    (118,"Og", "Oganesson",      294.0,   7, 18, "noble gas"),
]

ELEMENTS: List[Element] = [Element(*e) for e in _ELEMENTS_RAW]
BY_SYMBOL: Dict[str, Element] = {e.symbol: e for e in ELEMENTS}
