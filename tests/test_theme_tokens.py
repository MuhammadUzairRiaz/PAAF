"""Tests for the design-token system and the single light theme.

These are pure-Python (no Qt needed for the token checks) and assert the
concrete values from the *PAAF UI redesign v1* spec, so a future edit that
silently drifts off-spec fails loudly.
"""
from __future__ import annotations

import pytest

from paaf.gui import tokens as T


@pytest.fixture(autouse=True)
def _restore_light():
    """Every test starts and ends on the light palette."""
    T.set_theme("light")
    yield
    T.set_theme("light")


# ------------------------------------------------------------ spec values
def test_light_palette_matches_the_spec():
    assert T.BG_PAGE == "#F1F5F9"        # NOT the old #F8FAFC
    assert T.BG_SURFACE == "#FFFFFF"
    assert T.BG_SUNKEN == "#F8FAFC"      # the old page colour is now sunken
    assert T.BG_CHROME == "#0F1B2D"      # bluer/darker than the old #0F172A
    assert T.PRIMARY == "#2563EB"
    assert T.MAP_TOKEN == "#0F766E"      # new semantic role


def test_state_text_shades_differ_from_fills():
    """The spec adds separate *text* shades because the fill colours fail
    4.5:1 as small text on white."""
    assert T.SUCCESS_TEXT != T.SUCCESS
    assert T.WARN_TEXT != T.WARN
    assert T.DANGER_TEXT != T.DANGER


def test_metrics_are_fixed_by_the_spec():
    assert T.H_CONTROL == 26
    assert T.H_TABLE_ROW == 28
    assert T.H_TABLE_HEADER == 26
    assert T.H_SIDEBAR_ITEM == 30
    assert (T.R_CONTROL, T.R_CARD, T.R_PILL) == (3, 6, 10)
    # Type scale is fixed to exactly these five sizes.
    assert sorted([T.FS_OVERLINE, T.FS_CAPTION, T.FS_BODY,
                   T.FS_SECTION, T.FS_DISPLAY]) == [10, 11, 12, 13, 20]


def test_badge_colors_map_levels_to_the_palette():
    assert T.badge_colors("error")[0] == T.DANGER_TINT
    assert T.badge_colors("ok")[0] == T.SUCCESS_TINT
    assert T.badge_colors("warn")[0] == T.WARN_TINT
    # Unknown levels get the neutral treatment, never a crash.
    assert T.badge_colors("nonsense")[0] == T.BG_SUNKEN


def test_only_the_light_palette_ships():
    assert list(T.PALETTES) == ["light"]
    T.set_theme("anything-else")
    assert T.active_theme() == "light"


# ------------------------------------------------------------- stylesheet
def test_stylesheet_renders_and_honours_the_spec():
    pytest.importorskip("PyQt5", reason="theme module imports Qt lazily")
    from paaf.gui import theme

    qss = theme.build_qss()
    assert qss
    assert "box-shadow" not in qss             # depth from planes only
    assert "alternate-background-color" not in qss
    assert "border-radius: 3px" in qss         # controls
    assert "border-radius: 6px" in qss         # cards
    assert not hasattr(theme, "DARK_QSS")      # the dark theme was removed
