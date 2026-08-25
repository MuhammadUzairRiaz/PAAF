"""LAMMPS styles for DL_FIELD force fields.

The bug being pinned: PAAF carried a hand-written table claiming PCFF needs
``class2`` for bonds, angles, dihedrals and impropers, on the reasoning that
PCFF is a class-II force field. What DL_FIELD writes is ``class2`` bonds with
``quartic`` angles, ``fourier`` dihedrals and ``inversion/harmonic``
impropers — because ``angle_style class2`` requires BondBond and BondAngle
cross-term sections DL_FIELD does not emit.

The fix is not a better table. Every DL_FIELD coefficient line names its own
sub-style in column 2, so the styles are read out of the data file. These
tests use a fragment of a real DL_FIELD PCFF file.
"""
from __future__ import annotations

import pytest

from paaf.dlfield_styles import (                            # noqa: E402
    DLFIELD_SCHEMES, force_field_scheme_in, infer_bonded_styles, list_schemes,
    scheme_for, styles_for_data,
)

# Verbatim from DL_FIELD 4.13, scheme pcff — the shapes are the point.
PCFF_DATA = """# LAMMPS data file. Produced from DL_FIELD 4.13
# Molecular Group 1: XYZ
# Force field scheme: pcff

    3 atoms
    2 bonds

    2 atom types
    2 bond types
    1 angle types
    1 dihedral types
    1 improper types

   -75.0 75.0 xlo xhi
   -75.0 75.0 ylo yhi
   -75.0 75.0 zlo zhi

Masses

   1   12.011500   # c_1
   2    1.007970   # hc

Bond Coeffs

   1 class2     1.520200   253.706700  -423.037000   396.900000
   2 class2     1.368300   367.148100  -794.790800  1055.231900

Angle Coeffs

   1 quartic   100.318200    38.863100    -3.832300    -7.980200

Dihedral Coeffs

   1 fourier 3      1.834100 1   180.000000      2.060300 2   180.000000      -0.019500 3   180.000000

Improper Coeffs

   1 inversion/harmonic    46.926400    0.000000

Atoms

   1 1 1  0.702000  1.0 1.0 1.0 # c_1
   2 1 2 -0.106000  2.0 1.0 1.0 # hc
   3 1 2 -0.106000  3.0 1.0 1.0 # hc

Bonds

   1 1 1 2
   2 2 2 3
"""

OPLS_DATA = PCFF_DATA.replace("scheme: pcff", "scheme: opls2005") \
    .replace("class2     1.520200   253.706700  -423.037000   396.900000",
             "harmonic   340.000000     1.090000") \
    .replace("class2     1.368300   367.148100  -794.790800  1055.231900",
             "harmonic   317.000000     1.510000") \
    .replace("quartic   100.318200    38.863100    -3.832300    -7.980200",
             "harmonic    35.000000   109.500000") \
    .replace("fourier 3      1.834100 1   180.000000      2.060300 2   "
             "180.000000      -0.019500 3   180.000000",
             "opls 0.000000 0.000000 0.300000 0.000000") \
    .replace("inversion/harmonic    46.926400    0.000000",
             "cvff  1.100000 -1 2")


@pytest.fixture
def pcff(tmp_path):
    p = tmp_path / "lammps1.data"
    p.write_text(PCFF_DATA)
    return p


# ================================================ reading, not guessing
def test_pcff_bonded_styles_are_read_from_the_file(pcff):
    """The exact case the old table got wrong."""
    got = infer_bonded_styles(pcff)
    print("\n  " + "\n  ".join(f"{k:<15} {v}" for k, v in got.items()))
    assert got == {
        "bond_style": "class2",
        "angle_style": "quartic",
        "dihedral_style": "fourier",
        "improper_style": "inversion/harmonic",
    }


def test_the_old_table_disagrees_with_the_file(pcff):
    """Kept as a standing reminder of why reading beats guessing."""
    from paaf.cell.relax import _DLFIELD_STYLES

    read = infer_bonded_styles(pcff)
    guessed = _DLFIELD_STYLES["pcff"]
    wrong = {k: (guessed[k], read[k]) for k in read
             if k in guessed and guessed[k] != read[k]}
    print("\n  directive        table said -> file says")
    for k, (g, r) in wrong.items():
        print(f"  {k:<15} {g:<12} -> {r}")
    assert len(wrong) == 3, "the table used to be wrong on angle/dihedral/improper"


def test_an_opls_file_reads_differently(pcff, tmp_path):
    """Same reader, no force-field-specific branching."""
    p = tmp_path / "opls.data"
    p.write_text(OPLS_DATA)
    got = infer_bonded_styles(p)
    print(f"\n  {got}")
    assert got["bond_style"] == "harmonic"
    assert got["dihedral_style"] == "opls"
    assert got["improper_style"] == "cvff"


def test_the_scheme_is_taken_from_the_header_comment(pcff):
    assert force_field_scheme_in(pcff) == "pcff"


def test_a_file_with_no_coefficients_yields_nothing(tmp_path):
    p = tmp_path / "bare.data"
    p.write_text("bare\n\n1 atoms\n\n1 atom types\n\n"
                 "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n"
                 "Atoms\n\n1 1 1 0.0 0.0 0.0 0.0\n")
    assert infer_bonded_styles(p) == {}


# ============================================== the non-bonded half
def test_class2_gets_sixthpower_mixing_and_a_9_6_pair_style(pcff):
    """Mixing rule is invisible in the output and changes every cross term."""
    got = styles_for_data(pcff)
    print("\n  " + "\n  ".join(f"{k:<15} {v}" for k, v in got.items()))
    assert got["pair_style"].startswith("lj/class2")
    assert got["pair_modify"] == "mix sixthpower"


def test_opls_gets_geometric_mixing(tmp_path):
    p = tmp_path / "opls.data"
    p.write_text(OPLS_DATA)
    got = styles_for_data(p)
    print(f"\n  {got['pair_style']} / {got['pair_modify']}")
    assert got["pair_modify"] == "mix geometric"
    assert got["pair_style"].startswith("lj/cut")


def test_the_scheme_can_be_named_explicitly(pcff):
    got = styles_for_data(pcff, "compass")
    assert got["pair_modify"] == "mix sixthpower"
    # Bonded styles still come from the file, not from the named scheme.
    assert got["angle_style"] == "quartic"


# ==================================================== the scheme table
def test_every_shipped_force_field_is_registered():
    """DL_FIELD 4.13 ships 32 .par files; ten used to be reachable."""
    print(f"\n  {len(DLFIELD_SCHEMES)} schemes")
    for s in list_schemes():
        print(f"    {s.category:<13} {s.key:<26} {s.dl_key}")
    assert len(DLFIELD_SCHEMES) >= 32


@pytest.mark.parametrize("probe,expect", [
    ("pcff", "pcff"), ("PCFF", "pcff"), ("COMPASS", "compass"),
    ("opls2005_dl", "opls2005"), ("OPLS2020", "opls2020"),
    ("charmm36_cgenff", "charmm36_cgenff"), ("G54A7", "g54a7"),
])
def test_a_scheme_is_found_by_any_of_its_names(probe, expect):
    """Lookup is forgiving about case; what it returns is not.

    A user may type PCFF, the registry says pcff, and the .par is PCFF.par —
    all three must resolve. The control-file value that comes back is always
    the lower-case string DL_FIELD compares with strcmp.
    """
    s = scheme_for(probe)
    assert s is not None and s.dl_key == expect


def test_the_control_file_key_matches_the_par_file():
    """A mismatch here makes DL_FIELD reject the run with no useful message."""
    for s in DLFIELD_SCHEMES.values():
        assert s.par.endswith(".par")
        assert s.dl_key, f"{s.key} has no DL_FIELD key"


def test_every_scheme_becomes_a_selectable_force_field():
    from paaf.ff_registry import REGISTRY, dlfield_keys

    keys = set(dlfield_keys())
    print(f"\n  {len(keys)} DL_FIELD force fields in the registry")
    missing = [k for k in DLFIELD_SCHEMES if k not in keys]
    assert not missing, f"not selectable: {missing}"
    assert REGISTRY["pcff"].kind == "dlfield"


def test_the_runner_asks_dl_field_for_the_right_scheme():
    from paaf.dlfield_runner import _dl_ff_key

    for key in ("charmm36_carb", "inorganic_zeolite_hs", "cvff", "compass"):
        got = _dl_ff_key(key)
        print(f"\n  {key} -> {got}")
        assert got == DLFIELD_SCHEMES[key].dl_key


# ================================== the control-file key DL_FIELD accepts
def test_every_scheme_uses_a_key_dl_field_actually_accepts():
    """``get_par_file`` matches with strcmp against lower-case names.

    An unrecognised name is not a soft failure: DL_FIELD prints "Unknown FF
    in parameter" and calls exit(0), so the run dies with a zero status and
    no output at all.
    """
    from paaf.dlfield_styles import dl_field_accepts

    bad = [(s.key, s.dl_key) for s in DLFIELD_SCHEMES.values()
           if s.supported and not dl_field_accepts(s.dl_key)]
    print(f"\n  {len(DLFIELD_SCHEMES)} schemes, {len(bad)} unacceptable")
    assert not bad, f"DL_FIELD would reject: {bad}"


def test_the_key_is_lower_case():
    """PCFF fails where pcff works — strcmp, not strcasecmp."""
    for s in DLFIELD_SCHEMES.values():
        assert s.dl_key == s.dl_key.lower(), f"{s.key} -> {s.dl_key}"


def test_a_parameter_file_with_no_control_key_is_refused():
    """OPLS2020.par ships in lib/ but DL_FIELD 4.13 cannot select it."""
    from paaf.dlfield_runner import _dl_ff_key

    assert DLFIELD_SCHEMES["opls2020_dl"].supported is False
    with pytest.raises(ValueError) as e:
        _dl_ff_key("opls2020_dl")
    print(f"\n  {e.value}")
    assert "no control-file key" in str(e.value)


def test_cvff_end_to_end_naming():
    """The question that prompted this: does choosing CVFF work?"""
    from paaf.dlfield_runner import _dl_ff_key
    from paaf.ff_registry import REGISTRY

    assert _dl_ff_key("cvff") == "cvff"
    assert REGISTRY["cvff"].kind == "dlfield"
    print(f"\n  cvff -> control key {_dl_ff_key('cvff')!r}, "
          f"par {DLFIELD_SCHEMES['cvff'].par}")
