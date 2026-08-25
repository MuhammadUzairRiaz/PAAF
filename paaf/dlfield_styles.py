"""LAMMPS styles for every DL_FIELD force field, read rather than guessed.

The problem with tables
-----------------------
PAAF used to carry a hand-written table saying which LAMMPS styles each
DL_FIELD force field needs. It was wrong for PCFF, and wrong in a way that
runs. The table claimed::

    bond_style class2   angle_style class2
    dihedral_style class2   improper_style class2

because "PCFF is a class-II force field" is true of the *published* PCFF. It
is not what DL_FIELD emits. A real DL_FIELD PCFF data file says::

    Bond Coeffs
       1 class2     1.520200   253.706700  -423.037000   396.900000
    Angle Coeffs
       1 quartic  100.318200    38.863100    -3.832300    -7.980200
    Dihedral Coeffs
       1 fourier 3    1.834100 1 180.000000 ...
    Improper Coeffs
       1 inversion/harmonic    46.926400    0.000000

DL_FIELD deliberately avoids ``angle_style class2``, because that form
requires ``BondBond`` and ``BondAngle`` cross-term sections — which DL_FIELD
does not write. Declaring ``angle_style class2`` over a file with no cross
terms makes LAMMPS demand sections that do not exist.

Reading instead of guessing
---------------------------
Every DL_FIELD coefficient line names its own sub-style in the second column.
That is the force field telling us what it is, in the file itself. So the
bonded styles are *read* from the data file, which is authoritative, needs no
maintenance, and works for all 32 shipped force fields — including any added
later.

What still needs a table
------------------------
Only what is genuinely absent from the data file:

* ``pair_style`` — DL_FIELD writes ``pair_coeff`` into the ``.in``, not a
  ``Pair Coeffs`` section, so the functional form is not recoverable from the
  data file alone.
* ``pair_modify mix`` — class-II force fields mix **sixthpower**, OPLS mixes
  **geometric**, AMBER/CHARMM mix **arithmetic**. Getting this wrong changes
  every cross-species interaction in a blend and is invisible in the output.
* ``kspace_style`` and ``special_bonds`` — exclusion rules are part of the
  force-field definition, not of the topology.

Those are per-scheme constants, and the table below carries them. When a
component's own ``.in`` is available it wins over all of this, because it was
written by DL_FIELD for that exact system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["DLFieldScheme", "DLFIELD_SCHEMES", "scheme_for", "infer_bonded_styles",
           "styles_for_data", "list_schemes", "POLYMER_SCHEMES"]


# =====================================================================
@dataclass(frozen=True)
class DLFieldScheme:
    """One DL_FIELD force field.

    ``dl_key`` is the exact string DL_FIELD expects in a control file. It is
    **not** the ``POTENTIAL`` line of the ``.par``, and it is not the file
    name: DL_FIELD 4.13 matches it in ``get_par_file`` with ``strcmp`` against
    a fixed list of lower-case names. ``PCFF`` fails where ``pcff`` works, and
    an unmatched name aborts the run with ``Unknown FF in parameter``.

    ``supported`` is ``False`` for a parameter file that ships in ``lib`` but
    has no key in that list — ``OPLS2020.par`` is the case. Offering it would
    produce a control file DL_FIELD rejects outright.
    """
    key: str                        # PAAF registry key
    dl_key: str                     # what goes in the control file
    display: str
    par: str
    category: str = "general"
    pair_style: str = "lj/cut/coul/long 12.0"
    mix: str = "geometric"
    kspace_style: str = "pppm 1.0e-4"
    special_bonds: str = "lj/coul 0.0 0.0 0.5"
    note: str = ""
    supported: bool = True


# --- the four non-bonded conventions the shipped force fields use -----
_CLASS2 = dict(pair_style="lj/class2/coul/long 12.0", mix="sixthpower",
               special_bonds="lj/coul 0.0 0.0 1.0",
               note="Class-II: 9-6 LJ and sixth-power mixing. Using 12-6 or "
                    "geometric mixing here changes every interaction.")
_OPLS = dict(pair_style="lj/cut/coul/long 12.0", mix="geometric",
             special_bonds="lj/coul 0.0 0.0 0.5")
_AMBER = dict(pair_style="lj/cut/coul/long 10.0", mix="arithmetic",
              special_bonds="amber")
_CHARMM = dict(pair_style="lj/charmm/coul/long 10.0 12.0", mix="arithmetic",
               special_bonds="charmm")
_GROMOS = dict(pair_style="lj/cut/coul/cut 14.0", mix="geometric",
               kspace_style="", special_bonds="lj/coul 0.0 0.0 1.0")
_UA = dict(pair_style="lj/cut 14.0", mix="geometric", kspace_style="",
           special_bonds="lj/coul 0.0 0.0 0.0")
_INORGANIC = dict(pair_style="born/coul/long 12.0", mix="geometric",
                  special_bonds="lj/coul 0.0 0.0 1.0",
                  note="Ionic: Born-Mayer-Huggins with full Coulomb, no "
                       "intramolecular exclusions.")


def _s(key, dl_key, display, par, category, base=None, **kw) -> DLFieldScheme:
    merged = dict(base or {})
    merged.update(kw)
    return DLFieldScheme(key=key, dl_key=dl_key, display=display, par=par,
                         category=category, **merged)


#: Every force field in DL_FIELD 4.13's ``lib``. ``dl_key`` values come from
#: each file's ``POTENTIAL`` line, which is what the control file must match.
DLFIELD_SCHEMES: Dict[str, DLFieldScheme] = {s.key: s for s in [
    # ---- class-II, the polymer work-horses ---------------------------
    _s("pcff", "pcff", "PCFF (DL_FIELD)", "PCFF.par", "polymer", _CLASS2),
    _s("compass", "compass", "COMPASS (DL_FIELD)", "COMPASS.par", "polymer",
       _CLASS2),

    # ---- OPLS family --------------------------------------------------
    _s("opls2005_dl", "opls2005", "OPLS-2005 (DL_FIELD)", "OPLS2005.par",
       "polymer", _OPLS),
    _s("opls2020_dl", "opls2020", "OPLS-2020 (DL_FIELD)", "OPLS2020.par",
       "polymer", _OPLS, supported=False,
       note="Ships in lib/ but DL_FIELD 4.13 has no control-file key for it."),
    _s("opls_aam", "opls_aam", "OPLS-AA/M (DL_FIELD)", "OPLS_AAM.par",
       "polymer", _OPLS),
    _s("opls_cl_p", "opls_cl_p", "OPLS ionic liquids (DL_FIELD)",
       "OPLS_CL_P.par", "liquid", _OPLS),
    _s("opls_des", "opls_des", "OPLS deep-eutectic solvents (DL_FIELD)",
       "OPLS_DES.par", "liquid", _OPLS),
    _s("opls_ua", "opls2005", "OPLS united-atom (DL_FIELD)", "OPLS2005.par",
       "polymer", _OPLS),

    # ---- other organics ----------------------------------------------
    _s("cvff", "cvff", "CVFF (DL_FIELD)", "CVFF.par", "polymer", _OPLS,
       special_bonds="lj/coul 0.0 0.0 1.0"),
    _s("dreiding_dl", "dreiding", "DREIDING (DL_FIELD)", "DREIDING.par",
       "polymer", _OPLS, special_bonds="lj/coul 0.0 0.0 1.0"),
    _s("misc_ff", "misc_ff", "Miscellaneous (DL_FIELD)", "MISC_FF.par",
       "general", _OPLS),

    # ---- united-atom / coarse ----------------------------------------
    _s("trappe_eh", "trappe_eh", "TraPPE explicit-H (DL_FIELD)",
       "TRAPPE_EH.par", "liquid", _UA),
    _s("trappe_ua_dl", "trappe_ua", "TraPPE united-atom (DL_FIELD)",
       "TRAPPE_UA.par", "liquid", _UA),

    # ---- AMBER --------------------------------------------------------
    _s("amber", "amber", "AMBER (DL_FIELD)", "AMBER.par", "biomolecular",
       _AMBER),
    _s("amber_gaff", "amber16_gaff", "GAFF / AMBER16 (DL_FIELD)",
       "AMBER16_gaff.par", "polymer", _AMBER),
    _s("amber25_gaff", "amber25_gaff", "GAFF / AMBER25 (DL_FIELD)",
       "AMBER25_gaff.par", "polymer", _AMBER),

    # ---- CHARMM -------------------------------------------------------
    _s("charmm", "charmm", "CHARMM (DL_FIELD)", "CHARMM.par", "biomolecular",
       _CHARMM),
    _s("charmm19", "charmm19", "CHARMM19 (DL_FIELD)", "CHARMM19.par",
       "biomolecular", _CHARMM),
    _s("charmm22_prot", "charmm22_prot", "CHARMM22 protein (DL_FIELD)",
       "CHARMM22_prot.par", "biomolecular", _CHARMM),
    _s("charmm36", "charmm36_prot", "CHARMM36 protein (DL_FIELD)",
       "CHARMM36_prot.par", "biomolecular", _CHARMM),
    _s("charmm36_carb", "charmm36_carb", "CHARMM36 carbohydrate (DL_FIELD)",
       "CHARMM36_carb.par", "biomolecular", _CHARMM),
    _s("charmm36_cgenff", "charmm36_cgenff", "CHARMM36 CGenFF (DL_FIELD)",
       "CHARMM36_cgenff.par", "polymer", _CHARMM),
    _s("charmm36_lipid", "charmm36_lipid", "CHARMM36 lipid (DL_FIELD)",
       "CHARMM36_lipid.par", "biomolecular", _CHARMM),
    _s("charmm36_nucl", "charmm36_nucl", "CHARMM36 nucleic acid (DL_FIELD)",
       "CHARMM36_nucl.par", "biomolecular", _CHARMM),

    # ---- GROMOS -------------------------------------------------------
    _s("gromos_54a7", "g54a7", "GROMOS 54A7 (DL_FIELD)", "GROMOS_G54A7.par",
       "biomolecular", _GROMOS),

    # ---- inorganic ----------------------------------------------------
    _s("inorganic_binary_halide", "inorganic_binary_halide",
       "Binary halides (DL_FIELD)", "INORGANIC_binary_halides.par",
       "inorganic", _INORGANIC),
    _s("inorganic_binary_misc", "inorganic_binary_misc",
       "Binary miscellaneous (DL_FIELD)", "INORGANIC_binary_misc.par",
       "inorganic", _INORGANIC),
    _s("inorganic_binary_oxide", "inorganic_binary_oxide",
       "Binary oxides (DL_FIELD)", "INORGANIC_binary_oxides.par",
       "inorganic", _INORGANIC),
    _s("inorganic_clay", "inorganic_clay", "Clay (DL_FIELD)",
       "INORGANIC_clay.par", "inorganic", _INORGANIC),
    _s("inorganic_glass", "inorganic_glass", "Glass (DL_FIELD)",
       "INORGANIC_glass.par", "inorganic", _INORGANIC),
    _s("inorganic_ternary_oxide", "inorganic_ternary_oxide",
       "Ternary oxides (DL_FIELD)", "INORGANIC_ternary_oxides.par",
       "inorganic", _INORGANIC),
    _s("inorganic_zeolite", "inorganic_zeolite", "Zeolite (DL_FIELD)",
       "INORGANIC_zeolite.par", "inorganic", _INORGANIC),
    _s("inorganic_zeolite_hs", "inorganic_zeolite_hs",
       "Zeolite, Hill-Sauer (DL_FIELD)", "INORGANIC_zeolite_Hill_Sauer.par",
       "inorganic", _INORGANIC),
]}

#: The subset worth putting in front of someone building a polymer.
POLYMER_SCHEMES = [k for k, s in DLFIELD_SCHEMES.items()
                   if s.category == "polymer"]


#: Exactly the names ``get_par_file`` in DL_FIELD 4.13 accepts by ``strcmp``.
DL_FIELD_EXACT_KEYS = frozenset({
    "charmm36_prot", "charmm22_prot", "charmm36_lipid", "charmm36_cgenff",
    "charmm36_nucl", "charmm36_carb", "charmm", "amber", "amber16_gaff",
    "amber25_gaff", "trappe_eh", "trappe_ua", "oplsaa", "opls_cl_p",
    "opls2005", "opls_des", "opls_aam", "dreiding", "pcff", "compass",
    "cvff", "charmm19", "g54a7", "misc_ff", "inorganic",
})

#: …and the substrings it accepts by ``strstr``, for the inorganic sets.
DL_FIELD_SUBSTRING_KEYS = ("binary_halide", "binary_oxide", "binary_misc",
                           "zeolite_hs", "zeolite", "ternary_oxide",
                           "glass", "clay")


def dl_field_accepts(dl_key: str) -> bool:
    """Would DL_FIELD 4.13 recognise this control-file value?

    An unrecognised name is not a soft failure: ``get_par_file`` prints
    ``Unknown FF in parameter`` and calls ``exit(0)``, so the run dies with a
    zero status and no output. Checking here turns that into a message that
    names the problem.
    """
    if not dl_key:
        return False
    return (dl_key in DL_FIELD_EXACT_KEYS
            or any(s in dl_key for s in DL_FIELD_SUBSTRING_KEYS))


def list_schemes(category: str = "", supported_only: bool = False
                 ) -> List[DLFieldScheme]:
    """Every scheme, or every scheme in one category, in display order."""
    items = [s for s in DLFIELD_SCHEMES.values()
             if (not category or s.category == category)
             and (not supported_only or s.supported)]
    return sorted(items, key=lambda s: (s.category, s.display))


def scheme_for(key: str) -> Optional[DLFieldScheme]:
    """Look up by PAAF key, DL_FIELD key, or ``.par`` stem — case-insensitive."""
    if not key:
        return None
    probe = key.strip().lower()
    for s in DLFIELD_SCHEMES.values():
        if probe in (s.key.lower(), s.dl_key.lower(),
                     s.par.lower().replace(".par", "")):
            return s
    return None


# =====================================================================
#: Which ``*_style`` each coefficient section names.
_SECTION_STYLE = (
    ("Bond Coeffs", "bond_style"),
    ("Angle Coeffs", "angle_style"),
    ("Dihedral Coeffs", "dihedral_style"),
    ("Improper Coeffs", "improper_style"),
    ("Pair Coeffs", "pair_style"),
)


def infer_bonded_styles(data_file: Path) -> Dict[str, str]:
    """Bonded styles read out of a DL_FIELD data file's own coefficients.

    Each coefficient line is ``<type-id> <substyle> <numbers…>``, so the
    sub-style is simply the second column when it is not a number. Reading it
    beats any table: it is what the force field actually wrote, for this
    system, with this version of DL_FIELD.

    A section whose lines are bare numbers carries no sub-style; it is left
    out rather than guessed at.
    """
    from .lammps_replicator import parse_lammps_data, _clean

    try:
        _header, sections = parse_lammps_data(Path(data_file))
    except Exception as exc:                                # pragma: no cover
        log.warning("Could not parse %s: %s", data_file, exc)
        return {}

    styles: Dict[str, str] = {}
    for section, directive in _SECTION_STYLE:
        lines = _clean(sections.get(section))
        names = set()
        for line in lines:
            parts = line.split("#")[0].split()
            if len(parts) >= 2 and not _is_number(parts[1]):
                names.add(parts[1])
        if len(names) == 1:
            styles[directive] = names.pop()
        elif len(names) > 1:
            # Genuinely mixed: the hybrid wrapper is load-bearing here.
            styles[directive] = "hybrid " + " ".join(sorted(names))
            log.info("%s: %s uses %d sub-styles; keeping hybrid",
                     Path(data_file).name, section, len(names))
    return styles


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def force_field_scheme_in(data_file: Path) -> str:
    """The scheme DL_FIELD recorded in the data file's header comment.

    DL_FIELD writes ``# Force field scheme: pcff`` at the top. When present,
    that is a more reliable statement of what was used than anything the
    caller passes in.
    """
    try:
        with Path(data_file).open(errors="replace") as fh:
            for _ in range(20):
                line = fh.readline()
                if not line:
                    break
                if "force field scheme" in line.lower():
                    return line.split(":", 1)[-1].strip()
    except OSError:
        pass
    return ""


def styles_for_data(data_file: Path, scheme_key: str = "") -> Dict[str, str]:
    """A complete style block for a DL_FIELD data file.

    Bonded styles come from the file. Non-bonded settings — which the data
    file cannot carry, because DL_FIELD puts ``pair_coeff`` in the ``.in`` —
    come from the scheme table, chosen by the header comment when the caller
    did not name one.
    """
    styles = infer_bonded_styles(data_file)
    scheme = scheme_for(scheme_key) or scheme_for(
        force_field_scheme_in(data_file))

    if scheme is None:
        log.warning("%s: force field not identified; pair_style, mixing rule "
                    "and exclusions are unknown", Path(data_file).name)
        return styles

    styles.setdefault("pair_style", scheme.pair_style)
    if scheme.kspace_style:
        styles.setdefault("kspace_style", scheme.kspace_style)
    if scheme.special_bonds:
        styles.setdefault("special_bonds", scheme.special_bonds)
    if scheme.mix:
        styles.setdefault("pair_modify", f"mix {scheme.mix}")
    return styles
