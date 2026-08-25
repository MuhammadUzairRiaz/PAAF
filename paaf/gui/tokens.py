"""PAAF design tokens — the single source of truth for the UI.

Transcribed from the *PAAF UI redesign v1* design system
(``docs/design/PAAF_Redesign_v1.dc.html``). Every value here is expressible in
Qt Style Sheets; nothing depends on CSS features Qt lacks (no grid, no
animation, no blur).

Depth is three background planes plus border weight — **there are no shadows**::

    PAGE  →  SURFACE  →  SUNKEN

One palette: the spec's light theme — white surfaces on a cool grey page with
a dark chrome sidebar. Colours are bound onto this module by :func:`set_theme`
so ``from . import tokens as T`` stays the single import everywhere.

Notable deviations from the pre-redesign palette (documented in the spec):

* sidebar ``#0F1B2D`` (was ``#0F172A``) — bluer/darker so it reads as chrome
* page ``#F1F5F9`` (was ``#F8FAFC``) — the old page colour was too close to
  white, leaving cards with no edge; ``#F8FAFC`` is now the *sunken* plane
* success/warn/danger gain distinct **text** shades, because the original fill
  colours fail 4.5:1 as small text on white
* a new semantic role — **map token teal** — for atom-map numbers, which are
  neither a state nor a primary action and must not be confused with either
"""
from __future__ import annotations

import sys
from typing import Dict

# ============================================================= palettes
LIGHT: Dict[str, str] = {
    # background planes
    "BG_PAGE":          "#F1F5F9",
    "BG_SURFACE":       "#FFFFFF",
    "BG_SUNKEN":        "#F8FAFC",
    "BG_CHROME":        "#0F1B2D",
    "BG_CHROME_RAISED": "#16233A",
    # borders
    "BORDER":           "#E2E8F0",
    "BORDER_STRONG":    "#CBD5E1",
    # text
    "TEXT":             "#0F172A",
    "TEXT_SECONDARY":   "#475569",
    "TEXT_MUTED":       "#64748B",
    "TEXT_DISABLED":    "#94A3B8",
    "TEXT_ON_CHROME":   "#CBD5E1",
    # primary
    "PRIMARY":          "#2563EB",
    "PRIMARY_HOVER":    "#1D4ED8",
    "PRIMARY_TINT":     "#EFF6FF",
    "PRIMARY_BORDER":   "#BFDBFE",
    # state — fill/border vs text are deliberately different shades
    "SUCCESS":          "#16A34A",
    "SUCCESS_TEXT":     "#15803D",
    "SUCCESS_TINT":     "#ECFDF5",
    "SUCCESS_BORDER":   "#A7F3D0",
    "WARN":             "#D97706",
    "WARN_TEXT":        "#92400E",
    "WARN_TINT":        "#FFFBEB",
    "WARN_BORDER":      "#FDE68A",
    "DANGER":           "#DC2626",
    "DANGER_TEXT":      "#B91C1C",
    "DANGER_TINT":      "#FEF2F2",
    "DANGER_BORDER":    "#FCA5A5",
    # atom-map tokens (new semantic role)
    "MAP_TOKEN":        "#0F766E",
    "MAP_TINT":         "#ECFDF5",
}

PALETTES = {"light": LIGHT}
_ACTIVE = "light"


def set_theme(name: str = "light") -> str:
    """Bind the palette. Only ``"light"`` exists — the dark theme was removed.

    Kept as a function (rather than plain constants) so the tokens stay the one
    place colours are defined, and so a second palette could be reintroduced
    without touching any widget.
    """
    global _ACTIVE
    name = (name or "light").lower()
    if name not in PALETTES:
        name = "light"
    _ACTIVE = name
    mod = sys.modules[__name__]
    for key, value in PALETTES[name].items():
        setattr(mod, key, value)
    return name


def active_theme() -> str:
    return _ACTIVE


# ============================================================== typography
# The scale is fixed to these five sizes only.
FS_DISPLAY  = 20   # page title,      weight 600
FS_SECTION  = 13   # card/section,    weight 600
FS_BODY     = 12   # body / controls, weight 400 (600 = "strong")
FS_CAPTION  = 11   # captions, hints
FS_OVERLINE = 10   # uppercase column headers / overlines, weight 600

FONT_UI   = '"SF Pro Text", "Segoe UI", "Helvetica Neue", Arial'
# Platform-native first. Consolas is Windows-only, and listing it first made
# Qt walk its whole alias table on macOS and Linux before falling through —
# ~40 ms of startup spent, plus a warning on every launch.
FONT_MONO = '"SF Mono", Menlo, "DejaVu Sans Mono", Consolas, monospace'

# ================================================================= spacing
SPACE = (4, 8, 12, 16, 24, 32)
GAP_FORM_ROW   = 8    # between form rows
GAP_LABEL      = 8    # label -> field
PAD_CARD       = 14   # inside a card
PAD_PAGE       = 20   # page margin
GAP_SECTION    = 16   # between sections

# ================================================================== sizing
H_CONTROL      = 26   # input, combo, spin, button
H_TABLE_ROW    = 28
H_TABLE_HEADER = 26
H_SIDEBAR_ITEM = 30
H_ACTION_BAR   = 48

W_SIDEBAR      = 232
W_RIGHT_RAIL   = 340
W_NAME_COL     = 150
W_MAPS_COL     = 66
W_REMOVE_COL   = 24

# ================================================================== radius
R_CONTROL = 3
R_CARD    = 6
R_PILL    = 10


def badge_colors(level: str):
    """Return ``(fill, border, text)`` for a status badge in the active theme.

    ``level`` is one of ``ok``/``success``, ``warn``, ``error``/``danger``,
    ``info``, or anything else for a neutral badge.
    """
    level = (level or "").lower()
    if level in ("ok", "success", "valid", "done"):
        return SUCCESS_TINT, SUCCESS_BORDER, SUCCESS_TEXT
    if level in ("warn", "warning"):
        return WARN_TINT, WARN_BORDER, WARN_TEXT
    if level in ("error", "danger", "invalid"):
        return DANGER_TINT, DANGER_BORDER, DANGER_TEXT
    if level == "info":
        return PRIMARY_TINT, PRIMARY_BORDER, PRIMARY_HOVER
    return BG_SUNKEN, BORDER, TEXT_MUTED


# Bind the default palette at import time.
set_theme("light")
