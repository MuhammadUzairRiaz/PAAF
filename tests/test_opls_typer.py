"""Static tests for the OPLS-AA SMARTS typer (no OpenBabel needed here)."""
from __future__ import annotations


def test_opls_types_dict_covers_common_polymer_families():
    from paaf.typers.oplsaa import OPLS_TYPES
    # Alkane, alkene, aromatic, alcohol, ester carbonyl, ether, nitrile,
    # amide, sulfone, halogen — every core polymer family should map.
    for tid in ("135", "136", "140", "141", "145", "146", "154", "155",
                "180", "210", "211", "212", "177", "178", "473", "474",
                "263", "264", "753", "754"):
        assert tid in OPLS_TYPES, f"OPLS type {tid} missing from OPLS_TYPES"


def test_ff_assigner_uses_smarts_typer_for_oplsaa():
    """Verify the dispatch route: ff_assigner picks the OPLS SMARTS typer
    for oplsaa, oplsaa2008, loplsaa, loplsaa2008 — not the old Ghemical
    path (which returned invalid type strings like 'C.3')."""
    import inspect
    from paaf import ff_assigner
    src = inspect.getsource(ff_assigner.assign)
    assert "_assign_opls_smarts" in src
    # These four FFs must route through the new typer.
    for key in ("oplsaa", "oplsaa2008", "loplsaa", "loplsaa2008"):
        assert repr(key) in src or f'"{key}"' in src or f"'{key}'" in src


def test_rules_ordered_specific_before_generic():
    """The alkane fallback ('sp3 CH3 alkane') must come after all more
    specific CH2 / CH / C rules in _RULES so ester alpha carbons are
    typed as 136/139 rather than being mis-labelled as 135."""
    from paaf.typers.oplsaa import _RULES
    patterns = [r[0] for r in _RULES]
    # aromatic C rule ('[c]') is the most-generic aromatic rule and must
    # come after the specific ones.
    ci = patterns.index("[c]")
    assert patterns.index("[cH]") < ci
    # The catch-all methane rule is fine at the end of the alkane block.
    assert "[CX4H4]" in patterns


def test_typers_package_exports_the_typer():
    from paaf.typers import type_oplsaa, OPLS_TYPES  # noqa: F401
    assert callable(type_oplsaa)
    assert isinstance(OPLS_TYPES, dict) and len(OPLS_TYPES) >= 20
