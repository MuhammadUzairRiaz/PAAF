"""A tooltip must be a small card, not a stripe across the window.

Qt lays PLAIN-text tooltips out on one line however long they are. The
"Cell size" help is four sentences, so it rendered as a full-width grey band
over the page — covering the very controls it explained. Rich text wraps, so
the width has to come from the TEXT; no stylesheet can fix it.
"""
from __future__ import annotations

import pytest

from paaf.gui.tooltips import wrap_tooltip


def test_long_text_is_given_a_width():
    out = wrap_tooltip("word " * 80)
    assert out.startswith("<div"), "plain text would not wrap"
    assert "max-width" in out


def test_the_text_survives():
    assert "skeletal" in wrap_tooltip("Roughly how many skeletal atoms")


def test_html_is_left_alone_so_it_is_safe_to_apply_twice():
    once = wrap_tooltip("some help")
    assert wrap_tooltip(once) == once


def test_markup_in_the_text_cannot_break_the_tooltip():
    """A stray < in help text would otherwise swallow the rest of it."""
    out = wrap_tooltip("use a < b to compare")
    assert "&lt;" in out


def test_empty_stays_empty():
    assert wrap_tooltip("") == ""
    assert wrap_tooltip(None) == ""


@pytest.mark.parametrize("width", [200, 340, 500])
def test_the_width_is_honoured(width):
    assert f"max-width:{width}px" in wrap_tooltip("text", width)
