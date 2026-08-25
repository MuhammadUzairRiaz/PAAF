"""Every built-in reaction example must validate, forever.

The examples browser promises "paste-and-run" schemes; a single broken
SMILES would teach a beginner that the tool is broken. So each entry is
driven through the REAL validator here, plus schema checks the dialog
relies on.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("rdkit", reason="validation requires RDKit")

from paaf.reaction_examples import CATEGORIES, EXAMPLES, by_category
from paaf.reaction_smiles import validate_scheme

logging.disable(logging.CRITICAL)


def test_there_are_more_than_fifty_examples():
    print(f"\n  {len(EXAMPLES)} examples in {len(CATEGORIES)} categories")
    assert len(EXAMPLES) > 50
    assert len(CATEGORIES) >= 8


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e.key)
def test_every_example_validates(example):
    res = validate_scheme([s for _l, s in example.reactants],
                          [s for _l, s in example.products])
    errs = [i.text for i in res.errors]
    assert not errs, f"{example.key}: {errs}"


def test_keys_are_unique_and_fields_filled():
    keys = [e.key for e in EXAMPLES]
    assert len(keys) == len(set(keys))
    for e in EXAMPLES:
        assert e.name and e.description and e.tags, e.key
        assert e.reactants and e.products, e.key
        for label, smi in e.reactants + e.products:
            assert label and smi, e.key


def test_the_users_enr_chemistry_is_covered():
    keys = {e.key for e in EXAMPLES}
    for needed in ("enr_mah_enr", "enr_mah_pbs", "enr_pbs_direct"):
        assert needed in keys


def test_by_category_covers_everything():
    total = sum(len(v) for v in by_category().values())
    assert total == len(EXAMPLES)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e.key)
def test_every_example_builds_in_3d(example):
    """Workable means buildable: both sides of every example must embed to
    a clash-free 3D structure through the real export builder."""
    import numpy as np

    from paaf.reaction_export import build_3d_side

    for side, rows in (("reactant", example.reactants),
                       ("product", example.products)):
        mol, _maps = build_3d_side([s for _l, s in rows], side)
        xyz = np.array([a.xyz for a in mol.atoms])
        d = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
        np.fill_diagonal(d, 9e9)
        assert d.min() > 0.5, \
            f"{example.key}/{side}: coincident atoms at {d.min():.2f} A"


def test_no_example_ships_untypeable_byproducts():
    """H2 and the hydrogen halides are not in organic force fields; an
    example whose product side contains them would export structures that
    typing then refuses."""
    banned = {"[H][H]", "[ClH:", "[BrH:", "[FH:", "[IH:"}
    for e in EXAMPLES:
        for _label, smi in e.products:
            for b in banned:
                assert b not in smi, f"{e.key}: {smi}"
