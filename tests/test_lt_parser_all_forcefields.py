"""The atom-type picker must list types + parameters for EVERY bundled
moltemplate force field, not only the OPLS family."""
from pathlib import Path

import pytest

from paaf.lt_parser import (_element_from_name_and_mass, _pair_type_name,
                            parse_atom_types)

LIB = Path(__file__).resolve().parents[1] / "ff_libraries" / "moltemplate"
FFS = ["oplsaa2024", "oplsaa2008", "oplsaa", "loplsaa2024", "oplsua_2024",
       "compass_published", "gaff", "gaff2", "dreiding", "trappe1998", "sdk",
       "martini"]


@pytest.mark.parametrize("ff", FFS)
def test_every_forcefield_lists_types_with_parameters(ff):
    types = parse_atom_types(str(LIB / f"{ff}.lt"))
    assert len(types) >= 3, ff
    with_params = [t for t in types if t.params]
    assert len(with_params) >= 0.5 * len(types), (ff, len(with_params), len(types))
    assert all(t.ff_id for t in types)
    ids = [t.ff_id for t in types]
    assert len(ids) == len(set(ids))          # no duplicates from merging sections


def test_compass_types_have_element_mass_and_lj():
    types = {t.ff_id: t for t in parse_atom_types(str(LIB / "compass_published.lt"))}
    c4 = types["c4"]
    assert c4.element == "C" and abs(c4.mass - 12.011) < 0.01
    assert c4.params.startswith("eps=0.062") and "sigma=3.854" in c4.params
    assert types["o2e"].element == "O"


def test_gaff_ca_is_carbon_not_calcium():
    types = {t.ff_id: t for t in parse_atom_types(str(LIB / "gaff.lt"))}
    assert types["ca"].element == "C"
    assert types["cl"].element == "Cl"
    assert _element_from_name_and_mass("si", 28.09) == "Si"
    assert _element_from_name_and_mass("CH2", 14.027) == "C"       # united atom
    assert _element_from_name_and_mass("BP4", 72.0) == ""          # CG bead


def test_opls_family_unchanged():
    types = {t.ff_id: t for t in parse_atom_types(str(LIB / "oplsaa2024.lt"))}
    t = types["135"]
    assert t.element == "C" and t.key == "CT" and abs(t.charge + 0.18) < 1e-6
    assert "eps=" in t.params
    # oplsaa2008 comments are free text: element now comes from the mass
    t8 = {t.ff_id: t for t in parse_atom_types(str(LIB / "oplsaa2008.lt"))}
    assert t8["13"].element == "C"


def test_pair_token_names():
    assert _pair_type_name("80") == "80"
    assert _pair_type_name("*~pc3a~b*~a*~d*~i*") == "c3a"
    assert _pair_type_name("BP4_bBP4_aBP4_dBP4_iBP4") == "BP4"
