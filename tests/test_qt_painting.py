"""Guard against the classic Qt stylesheet-background trap.

Qt will NOT paint a ``background:`` stylesheet rule on a **QWidget subclass**
unless that subclass reimplements ``paintEvent`` and draws ``PE_Widget``
itself. Plain ``QWidget()`` *instances* are fine; subclasses are not.

This bug is invisible in code review and in headless tests — it only shows up
on screen, as a widget that silently renders transparent. It shipped once (the
sidebar rendered white-on-white), so it gets a test.

Any QWidget subclass in paaf.gui that sets a background on itself must
implement paintEvent.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

GUI_DIR = Path(__file__).resolve().parent.parent / "paaf" / "gui"


def _classes_needing_paint_event(path: Path):
    """Yield (class_name, has_paint_event) for QWidget subclasses in `path`
    that set a background on *self*."""
    tree = ast.parse(path.read_text(errors="replace"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
        if "QWidget" not in bases:
            continue

        src = ast.get_source_segment(path.read_text(errors="replace"), node) or ""
        # Does the class style ITSELF with a background?
        styles_self = "self.setStyleSheet(" in src and "background" in src
        if not styles_self:
            continue
        has_paint = any(
            isinstance(n, ast.FunctionDef) and n.name == "paintEvent"
            for n in node.body
        )
        yield node.name, has_paint


def test_widget_subclasses_that_paint_a_background_implement_paintevent():
    offenders = []
    for path in sorted(GUI_DIR.glob("*.py")):
        for cls, has_paint in _classes_needing_paint_event(path):
            if not has_paint:
                offenders.append(f"{path.name}::{cls}")
    assert not offenders, (
        "These QWidget subclasses set a background but do not implement "
        "paintEvent, so Qt will render them TRANSPARENT on screen: "
        + ", ".join(offenders)
        + "\n\nAdd:\n"
        "    def paintEvent(self, ev):\n"
        "        opt = QStyleOption(); opt.initFrom(self)\n"
        "        p = QPainter(self)\n"
        "        self.style().drawPrimitive(QStyle.PE_Widget, opt, p, self)"
    )


def test_known_chrome_widgets_are_covered():
    """Belt and braces: the specific widgets that broke must stay fixed."""
    checks = {
        "sidebar.py": ["class Sidebar", "class _NavItem"],
        "page.py": ["class Card"],
    }
    for fname, classes in checks.items():
        src = (GUI_DIR / fname).read_text(errors="replace")
        assert src.count("def paintEvent") >= len(classes), (
            f"{fname} is missing a paintEvent for one of {classes}")
        assert "PE_Widget" in src


@pytest.mark.parametrize("needle", [
    "padding: 7px 14px",     # tab labels must not be clipped
    "border-radius: 8px",    # radio indicators must be circular
])
def test_rendered_stylesheet_contains_layout_fixes(needle):
    pytest.importorskip("PyQt5", reason="theme imports Qt lazily")
    from paaf.gui import theme
    assert needle in theme.LIGHT_QSS


# ------------------------------------------------ text-clipping regressions
def _rule_body(qss: str, selector: str) -> str:
    i = qss.index(selector)
    return qss[i:qss.index("}", i) + 1]


def test_selected_tab_does_not_change_font_weight():
    """Qt sizes each tab with the tab bar's NORMAL-weight font.

    Bolding only the selected tab makes its text wider than the width Qt
    allocated, so the label overflows and is clipped — the tab then reading
    "SMILES + library + fragment" with its first characters cut off. Colour
    plus an underline mark the active tab without changing metrics.
    """
    pytest.importorskip("PyQt5", reason="theme imports Qt lazily")
    from paaf.gui import theme
    body = _rule_body(theme.LIGHT_QSS, "QTabBar::tab:selected")
    assert "font-weight" not in body
    # The active tab must still be distinguishable.
    assert "border-bottom" in body and "color" in body


def test_no_state_rule_resizes_a_control():
    """Any :hover/:selected/:checked rule that changes font-weight or padding
    without compensating will visibly reflow its widget."""
    pytest.importorskip("PyQt5", reason="theme imports Qt lazily")
    import re
    from paaf.gui import theme

    offenders = []
    for m in re.finditer(r"([^\{\}]*:(?:selected|checked|hover)[^\{\}]*)\{([^\}]*)\}",
                         theme.LIGHT_QSS):
        selector, body = m.group(1).strip(), m.group(2)
        if "font-weight" in body and "QTabBar" in selector:
            offenders.append(selector)
    assert not offenders, (
        "These tab state rules change font weight and will clip their labels: "
        + ", ".join(offenders))
