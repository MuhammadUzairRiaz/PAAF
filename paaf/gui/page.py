"""Reusable page chrome for the redesign.

Every screen in the *PAAF UI redesign v1* spec is assembled from the same
handful of parts, so they live here once instead of being re-invented per page:

* :func:`page_header` — breadcrumb ▸ title ▸ description ▸ status badge
* :class:`Card`       — flat card with a sunken header bar and optional
  header-right widgets (the spec's QGroupBox-with-a-drawn-title-bar)
* :func:`action_bar`  — 48px bar: blocking explanation on the left, exactly one
  primary button right-most
* :func:`badge`, :func:`button`, :func:`overline`, :func:`caption`,
  :func:`stat_row`, :func:`hline`

Two rules from the spec are enforced structurally rather than by convention:

1. **Scoped stylesheets.** Every container rule is written as
   ``QWidget#objectName { … }``. An unscoped QSS rule set on a parent cascades
   to *every* descendant, which would draw a border around each child label.
2. **State is never colour alone.** :func:`badge` always renders a glyph or a
   word next to its colour.

PAAF ships a single light theme; :meth:`Card.retheme` simply (re)applies the
card's inline styling and is called from its constructor.
"""
from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QStyle,
    QStyleOption, QVBoxLayout, QWidget,
)

from . import tokens as T
from .tooltips import wrap_tooltip

_UID = [0]


def _uid(prefix: str) -> str:
    _UID[0] += 1
    return f"{prefix}{_UID[0]}"


# ================================================================= atoms
def fit_tabs_to_current(tabs) -> None:
    """Make a QTabWidget's height follow its CURRENT page, not its tallest.

    Qt sizes the underlying QStackedWidget to the maximum sizeHint over ALL
    pages, so a short step (a small settings form) inherits the height of
    the tallest step and the page scrolls through empty space to reach the
    footer. Ignoring the size hints of the hidden pages makes the tab
    widget hug whichever page is showing.
    """
    def _sync(_index=None):
        cur = tabs.currentWidget()
        for i in range(tabs.count()):
            page = tabs.widget(i)
            if page is None:
                continue
            if page is cur:
                page.setSizePolicy(QSizePolicy.Expanding,
                                   QSizePolicy.Preferred)
            else:
                page.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        if cur is not None:
            cur.adjustSize()
        tabs.updateGeometry()

    tabs.currentChanged.connect(_sync)
    _sync()

def fit_stack_to_current(stack) -> None:
    """QStackedWidget version of :func:`fit_tabs_to_current` — same disease
    (height = tallest page), same cure."""
    def _sync(_index=None):
        cur = stack.currentWidget()
        for i in range(stack.count()):
            page = stack.widget(i)
            if page is None:
                continue
            if page is cur:
                page.setSizePolicy(QSizePolicy.Expanding,
                                   QSizePolicy.Preferred)
            else:
                page.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        if cur is not None:
            cur.adjustSize()
        stack.updateGeometry()

    stack.currentChanged.connect(_sync)
    _sync()


def hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {T.BORDER}; border: none; color: {T.BORDER};")
    return f


def overline(text: str) -> QLabel:
    """10px uppercase column header / section label."""
    lb = QLabel(text.upper())
    lb.setStyleSheet(
        f"color: {T.TEXT_MUTED}; font-size: {T.FS_OVERLINE}px;"
        f" font-weight: 600; letter-spacing: 1px; background: transparent;"
        f" border: none;")
    return lb


def caption(text: str, color: Optional[str] = None) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(True)
    lb.setStyleSheet(
        f"color: {color or T.TEXT_MUTED}; font-size: {T.FS_CAPTION}px;"
        f" background: transparent; border: none;")
    return lb


def label(text: str, size: int = T.FS_BODY, color: Optional[str] = None,
          bold: bool = False, mono: bool = False) -> QLabel:
    lb = QLabel(text)
    fam = f"font-family: {T.FONT_MONO};" if mono else ""
    weight = "font-weight: 600;" if bold else ""
    lb.setStyleSheet(
        f"color: {color or T.TEXT}; font-size: {size}px; {weight} {fam}"
        f" background: transparent; border: none;")
    return lb


def badge(text: str, level: str = "neutral") -> QLabel:
    """Pill badge. Always carries a word or glyph — never colour alone."""
    fill, border, fg = T.badge_colors(level)
    lb = QLabel(text)
    lb.setStyleSheet(
        f"background: {fill}; border: 1px solid {border}; color: {fg};"
        f" border-radius: {T.R_PILL}px; padding: 2px 8px;"
        f" font-size: {T.FS_CAPTION}px; font-weight: 600;")
    lb.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
    return lb


def button(text: str, kind: str = "secondary") -> QPushButton:
    """A themed button. ``kind``: secondary | primary | danger | ghost.

    Destructive actions are OUTLINED, never filled — a filled red next to a
    filled blue primary is a misclick risk in a dense action bar.
    """
    b = QPushButton(text)
    b.setFixedHeight(T.H_CONTROL)
    b.setCursor(Qt.PointingHandCursor)
    if kind == "primary":
        b.setObjectName("primary")
    elif kind == "danger":
        b.setObjectName("danger")
    elif kind == "ghost":
        b.setObjectName("ghost")
    return b


def stat_row(key: str, value: str, value_color: Optional[str] = None,
             mono: bool = True) -> QWidget:
    """A right-aligned key/value line for summary cards."""
    w = QWidget()
    w.setStyleSheet("background: transparent; border: none;")
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(T.GAP_LABEL)
    h.addWidget(label(key, T.FS_BODY, T.TEXT_SECONDARY))
    h.addStretch(1)
    v = label(value, T.FS_BODY, value_color or T.TEXT, bold=True, mono=mono)
    v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    h.addWidget(v)
    return w


# ================================================================== card
class Card(QWidget):
    """Flat card: 1px border, 6px radius, sunken header bar.

    ``body`` is the content layout; ``header`` is the header row layout, so a
    page can drop a count label, a badge or a button on the right of the title.
    """

    def __init__(self, title: str = "", subtitle: str = "",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._name = _uid("card")
        self._head_name = _uid("cardHead")
        self._body_name = _uid("cardBody")
        self.setObjectName(self._name)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._head_w = QWidget()
        self._head_w.setObjectName(self._head_name)
        self.header = QHBoxLayout(self._head_w)
        self.header.setContentsMargins(T.PAD_CARD, 8, T.PAD_CARD, 8)
        self.header.setSpacing(T.GAP_LABEL)
        self._title_lb = None
        if title:
            self._title_lb = label(title, T.FS_SECTION, bold=True)
            self.header.addWidget(self._title_lb)
        self._sub_lb = None
        if subtitle:
            self._sub_lb = caption(subtitle)
            self.header.addWidget(self._sub_lb)
        self.header.addStretch(1)
        if title or subtitle:
            outer.addWidget(self._head_w)
        else:
            self._head_w.hide()

        body_w = QWidget()
        body_w.setObjectName(self._body_name)
        self.body = QVBoxLayout(body_w)
        self.body.setContentsMargins(T.PAD_CARD, T.PAD_CARD, T.PAD_CARD, T.PAD_CARD)
        self.body.setSpacing(T.GAP_FORM_ROW)
        outer.addWidget(body_w)
        self._body_w = body_w

        self.retheme()


    def paintEvent(self, ev):
        """Qt does NOT paint a stylesheet background on a QWidget *subclass*
        unless the subclass draws PE_Widget itself. Without this the widget is
        transparent and any `background:` rule is silently ignored."""
        opt = QStyleOption()
        opt.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, p, self)

    def retheme(self) -> None:
        self.setStyleSheet(
            f"QWidget#{self._name} {{ background: {T.BG_SURFACE};"
            f" border: 1px solid {T.BORDER}; border-radius: {T.R_CARD}px; }}")
        self._head_w.setStyleSheet(
            f"QWidget#{self._head_name} {{ background: {T.BG_SUNKEN};"
            f" border: none; border-bottom: 1px solid {T.BORDER};"
            f" border-top-left-radius: {T.R_CARD}px;"
            f" border-top-right-radius: {T.R_CARD}px; }}")
        self._body_w.setStyleSheet(
            f"QWidget#{self._body_name} {{ background: transparent;"
            f" border: none; }}")


# ================================================================ header
def page_header(title: str, description: str = "", breadcrumb: str = "",
                status: Optional[tuple] = None) -> QWidget:
    """Breadcrumb ▸ title ▸ description, with an optional status badge.

    ``status`` is ``(text, level)``, e.g. ``("12 atoms untyped", "warn")``.
    """
    w = QWidget()
    w.setStyleSheet("background: transparent; border: none;")
    v = QVBoxLayout(w)
    v.setContentsMargins(T.PAD_PAGE, 10, T.PAD_PAGE, 6)
    v.setSpacing(2)

    if breadcrumb:
        v.addWidget(caption(breadcrumb, T.TEXT_MUTED))

    row = QHBoxLayout()
    row.setSpacing(T.GAP_LABEL)
    t = QLabel(title)
    t.setStyleSheet(
        f"color: {T.TEXT}; font-size: {T.FS_DISPLAY}px; font-weight: 600;"
        f" letter-spacing: -0.2px; background: transparent; border: none;")
    row.addWidget(t)
    row.addStretch(1)
    if status:
        row.addWidget(badge(status[0], status[1]))
    v.addLayout(row)

    if description:
        v.addWidget(caption(description, T.TEXT_SECONDARY))
    return w


# ============================================================= action bar
def action_bar(blocking_text: str = "",
               buttons: Optional[List[QPushButton]] = None) -> QWidget:
    """48px action bar.

    Spec rule: exactly one primary button per screen, always right-most, with
    the blocking condition stated in plain text to its left. Pass the buttons
    left-to-right; the primary should be last.
    """
    bar = QWidget()
    name = _uid("actionBar")
    bar.setObjectName(name)
    bar.setFixedHeight(T.H_ACTION_BAR)
    bar.setStyleSheet(
        f"QWidget#{name} {{ background: {T.BG_SURFACE};"
        f" border: 1px solid {T.BORDER}; border-radius: {T.R_CARD}px; }}")
    h = QHBoxLayout(bar)
    h.setContentsMargins(T.PAD_CARD, 0, T.PAD_CARD, 0)
    h.setSpacing(T.GAP_LABEL)
    lb = label(blocking_text, T.FS_BODY, T.TEXT_MUTED)
    lb.setObjectName("actionBarBlocker")
    h.addWidget(lb)
    h.addStretch(1)
    for b in (buttons or []):
        h.addWidget(b)
    bar._blocker = lb           # so pages can update the explanation
    return bar

