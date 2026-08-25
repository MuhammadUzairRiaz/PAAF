"""Tooltip text that wraps, kept free of Qt so it can be tested anywhere.

Qt lays a PLAIN-text tooltip out on a single line however long it is. The
"Cell size" help is four sentences, so it rendered as a full-width grey band
across the window, covering the very controls it explained. Rich text wraps,
so the width has to be carried by the TEXT — no stylesheet can supply it.
"""
from __future__ import annotations

from html import escape

__all__ = ["wrap_tooltip"]

#: Wide enough for a sentence, narrow enough to read as a card.
DEFAULT_WIDTH_PX = 340


def wrap_tooltip(text: str, width_px: int = DEFAULT_WIDTH_PX) -> str:
    """Return ``text`` as width-constrained HTML.

    Text that is already HTML is returned untouched, so applying this twice
    is harmless — worth having, because the call sites were rewritten in bulk.
    """
    text = (text or "").strip()
    if not text:
        return ""
    if text.startswith("<"):
        return text
    return (f'<div style="max-width:{int(width_px)}px; white-space:normal;">'
            f'{escape(text)}</div>')
