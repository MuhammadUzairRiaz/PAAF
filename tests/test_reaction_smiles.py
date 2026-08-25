"""Tests for SMILES + atom-map reaction schemes.

The atom-map parsing and all structural validation are dependency-free and
always tested. The end-to-end template build needs RDKit, so those tests skip
when it isn't installed.
"""
from __future__ import annotations

import numpy as np
import pytest

from paaf.reaction_smiles import (
    Issue, parse_atom_maps, _duplicate_maps, _has_unmapped_stereocentre,
    _static_checks, validate_scheme, available,
)


# ------------------------------------------------------------- map parsing
def test_parse_atom_maps():
    assert parse_atom_maps("CC(=O)[OH:1]") == {1: 0}
    assert sorted(parse_atom_maps("[O:2]C[C:3](=O)O")) == [2, 3]
    assert parse_atom_maps("CCO") == {}          # no maps at all
    # multi-digit map numbers and decorated bracket atoms
    assert sorted(parse_atom_maps("[C@@H:12]C[N+:7]")) == [7, 12]


def test_duplicate_maps_detected():
    assert _duplicate_maps("[C:1]C[O:1]") == [1]
    assert _duplicate_maps("[C:1]C[O:2]") == []


def test_unmapped_stereocentre_detection():
    assert _has_unmapped_stereocentre("C[C@@H](O)C(=O)[OH:1]") is True
    assert _has_unmapped_stereocentre("C[C@@H:5](O)C") is False
    assert _has_unmapped_stereocentre("CCO") is False


# --------------------------------------------------------------- validation
def _codes(issues):
    return sorted(i.code for i in issues)


def test_leaving_group_is_not_an_error():
    """A map on the reactant but absent from the product is a leaving group —
    that is how water leaves an esterification, so it must NOT be an error."""
    issues, state = _static_checks(["CC(=O)[OH:1]", "[C:2]CO"],
                                   ["CC(=O)O[C:2]", "O"])
    assert [i for i in issues if i.is_error] == []
    assert state[1] == (True, False)      # :1 leaves
    assert state[2] == (True, True)       # :2 survives


def test_product_only_map_is_an_error():
    """A map that appears only on the product has no anchor -> E-MAP-01."""
    issues, _ = _static_checks(["[O:2]CC"], ["CC[O:2][C:3]"])
    assert "E-MAP-01" in _codes([i for i in issues if i.is_error])


def test_duplicate_map_is_an_error():
    issues, _ = _static_checks(["[C:1]C[O:1]"], ["[C:1]CO"])
    assert "E-MAP-02" in _codes([i for i in issues if i.is_error])


def test_no_maps_at_all_is_an_error():
    issues, _ = _static_checks(["CCO"], ["CCOC"])
    assert "E-MAP-03" in _codes([i for i in issues if i.is_error])


def test_unmapped_stereocentre_is_a_warning_not_an_error():
    issues, _ = _static_checks(["C[C@@H](O)C(=O)[OH:1]"],
                               ["C[C@@H](O)C(=O)[O:1]C"])
    warn = [i for i in issues if not i.is_error]
    assert "W-STE-03" in _codes(warn)
    assert [i for i in issues if i.is_error] == []


def test_empty_inputs_report_errors_without_crashing():
    res = validate_scheme([], "")
    assert not res.ok and res.errors
    res = validate_scheme(["CCO"], [])
    assert not res.ok and res.errors


def test_issue_helpers():
    e = Issue("E-MAP-01", "error", "boom")
    w = Issue("W-BND-02", "warning", "hmm")
    assert e.is_error and not w.is_error
    assert e.as_dict()["code"] == "E-MAP-01"


# ------------------------------------------------------------- RDKit e2e
@pytest.mark.skipif(not available(), reason="RDKit not installed")
def test_esterification_end_to_end():
    """Acid + alcohol -> ester + water: exactly one C–O bond forms."""
    res = validate_scheme(["CC(=O)[OH:1]", "[C:2]CO"], ["CC(=O)O[C:2]", "O"],
                          name="esterification")
    assert res.ok, [i.message for i in res.errors]
    assert res.n_bonds_formed >= 1
    assert res.template is not None
    assert res.template.name == "esterification"


@pytest.mark.skipif(not available(), reason="RDKit not installed")
def test_invalid_smiles_reports_e_smi_04():
    res = validate_scheme(["C[C@@H](O)C1(=O)[OH:1]"], ["CC(=O)O[C:1]"])
    assert not res.ok
    assert "E-SMI-04" in sorted(i.code for i in res.errors)


# ----------------------------------------------------- multi-product schemes
def test_multiple_products_are_supported():
    """Esterification really yields TWO molecules: the ester and water."""
    issues, state = _static_checks(["CC(=O)[OH:1]", "[C:2]CO"],
                                   ["CC(=O)O[C:2]", "O"])
    assert [i for i in issues if i.is_error] == []
    assert state[2] == (True, True)      # :2 survives into product 1


def test_a_map_reused_across_two_products_is_an_error():
    issues, _ = _static_checks(["[C:1]CC"], ["[C:1]C", "[C:1]O"])
    assert "E-MAP-02" in _codes([i for i in issues if i.is_error])


def test_a_bare_string_is_treated_as_one_molecule_not_characters():
    """Guard: str is a Sequence[str] that would otherwise iterate letters."""
    from_list, _ = _static_checks(["[C:1]CC"], ["[C:1]CO"])
    from_str, _ = _static_checks("[C:1]CC", "[C:1]CO")
    assert _codes(from_list) == _codes(from_str)
