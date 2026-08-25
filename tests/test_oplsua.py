"""Tests for the OPLS-UA extractor."""
from __future__ import annotations

from pathlib import Path

import pytest

from paaf.oplsua_extractor import (
    extract, default_source, default_dest, DEFAULT_UA_RANGE,
)
from paaf.ff_registry import get_ff, list_ffs


def test_ua_range_covers_documented_block():
    """The file itself labels types 66-134 as UA; the default range must match."""
    assert DEFAULT_UA_RANGE == set(range(66, 135))
    assert 66 in DEFAULT_UA_RANGE and 134 in DEFAULT_UA_RANGE
    assert 65 not in DEFAULT_UA_RANGE and 135 not in DEFAULT_UA_RANGE


def test_extracted_lt_exists_and_has_ua_object():
    """Running the extractor should produce a well-formed .lt file with the
    UA object wrapper and no OPLSAA references at the object level."""
    dest = default_dest()
    if not dest.exists():
        extract(default_source(), dest)
    assert dest.exists(), f"Expected {dest} to exist"
    text = dest.read_text()
    # Object opens and closes as OPLSUA_2024
    assert "OPLSUA_2024 {" in text
    assert "} # OPLSUA_2024" in text
    # UA atom types are present
    for i in (66, 67, 71, 75, 96):
        assert f"@atom:{i}" in text, f"UA atom {i} missing from extracted FF"
    # All-atom types outside the UA block MUST NOT appear as pair_coeff subjects
    # (they may still appear in comments so we scan literal `pair_coeff @atom:1_` etc)
    for i in (1, 6, 135, 140, 145):
        assert f"pair_coeff @atom:{i}_" not in text, \
            f"All-atom type {i} leaked into OPLS-UA pair_coeff"


def test_extracted_lt_has_bond_angle_dihedral_coeffs():
    dest = default_dest()
    if not dest.exists():
        extract(default_source(), dest)
    text = dest.read_text()
    assert text.count("bond_coeff") >= 50
    assert text.count("angle_coeff") >= 50
    assert text.count("dihedral_coeff") >= 10


def test_extracted_lt_has_init_block():
    """The `In Init` section (units, styles, kspace) must be preserved so the
    UA .lt is a drop-in Moltemplate FF."""
    dest = default_dest()
    if not dest.exists():
        extract(default_source(), dest)
    text = dest.read_text()
    assert 'write_once("In Init")' in text
    assert "units real" in text
    assert "atom_style full" in text
    assert "dihedral_style" in text


def test_oplsua_registered_in_ff_registry():
    keys = {ff.key for ff in list_ffs()}
    assert "oplsua_2024" in keys
    ff = get_ff("oplsua_2024")
    assert ff.united_atom is True
    assert ff.inherit == "OPLSUA_2024"
    assert ff.lt_include == "oplsua_2024.lt"
