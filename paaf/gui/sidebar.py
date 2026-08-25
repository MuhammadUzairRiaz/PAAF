"""Sidebar navigation — pipeline stepper + independent tools.

Implements the sidebar from the *PAAF UI redesign v1* spec: the six-step
pipeline reads as guided progress (what's done, where I am, what's next), while
the independent tools sit in their own section below.

    PAAF
    <project name>
    Step 1 of 6 · Builder
    ─────────────────────
    PIPELINE
      1  Builder          NOW
      2  Chain            ready
      3  Optimize         locked
      ...
    TOOLS
      Reactions
      System builder
      Blend
    ─────────────────────
    Pipeline progress
    1 of 6 complete

Drop-in replacement for the previous ``QListWidget``: it keeps
``currentRowChanged`` / ``setCurrentRow`` / ``currentRow`` so the rest of the
main window is unchanged. Row indices still address ``NAV_ITEMS``.

State is never colour alone — every step carries a number, a word
("NOW"/"ready"/"locked"/"done"), or a ✓/✕ glyph.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QSizePolicy, QStyle, QStyleOption,
    QVBoxLayout, QWidget,
)

from . import tokens as T

# Step states, in the order they appear in a run.
DONE, NOW, READY, LOCKED, ERROR = "done", "now", "ready", "locked", "error"


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {T.BG_CHROME_RAISED}; border: none;")
    return f


class _NavItem(QWidget):
    """One clickable sidebar row: [number] label [state]."""

    clicked = pyqtSignal(int)

    def __init__(self, row: int, text: str, number: Optional[int] = None,
                 tooltip: str = "", parent=None):
        super().__init__(parent)
        self.row = row
        self._number = number
        self._selected = False
        self._state = READY if number else ""
        self._result = ""
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(T.H_SIDEBAR_ITEM)
        if tooltip:
            self.setToolTip(tooltip)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 12, 0)
        lay.setSpacing(8)

        self.num_lb = QLabel("" if number is None else str(number))
        self.num_lb.setFixedWidth(12)
        lay.addWidget(self.num_lb)

        self.text_lb = QLabel(text)
        lay.addWidget(self.text_lb, 1)

        self.state_lb = QLabel("")
        lay.addWidget(self.state_lb)

        self._restyle()

    # -- state -------------------------------------------------------
    def set_state(self, state: str) -> None:
        self._state = state
        self._restyle()

    def set_result(self, text: str) -> None:
        """Short result shown on a completed step, e.g. 'n=50' or 'MMFF94'."""
        self._result = text or ""
        self._restyle()

    def set_selected(self, on: bool) -> None:
        self._selected = bool(on)
        self._restyle()

    def _restyle(self) -> None:
        """Spec: active item = #1D4ED8 fill + 3px #60A5FA left edge + NOW.
        Done = green check + result value. Locked = muted + the word 'locked'.
        Tools carry a grey TOOL marker instead of a step number."""
        if self._selected:
            bg, left = T.PRIMARY_HOVER, "#60A5FA"
            fg, weight = "#FFFFFF", "600"
        else:
            bg, left = "transparent", "transparent"
            fg, weight = T.TEXT_ON_CHROME, "400"
            if self._state == LOCKED:
                fg = T.TEXT_DISABLED
        self.setObjectName("navItemRoot")
        self.setStyleSheet(
            f"QWidget#navItemRoot {{ background: {bg};"
            f" border-left: 3px solid {left}; }}")
        self.text_lb.setStyleSheet(
            f"color: {fg}; font-size: {T.FS_BODY}px; font-weight: {weight};"
            f" background: transparent; border: none;")

        # Index column: step number, a green check when done, or TOOL.
        if self._number is None:
            self.num_lb.setText("")
        elif self._state == DONE:
            self.num_lb.setText("✓")
        else:
            self.num_lb.setText(str(self._number))
        num_color = "#FFFFFF" if self._selected else T.TEXT_DISABLED
        if self._state == DONE and not self._selected:
            num_color = T.SUCCESS
        self.num_lb.setStyleSheet(
            f"color: {num_color}; font-family: {T.FONT_MONO};"
            f" font-size: {T.FS_CAPTION}px; background: transparent;"
            f" border: none;")

        # Trailing marker. Done steps show their RESULT value (n=50, MMFF94)
        # when one has been recorded, which is more useful than the word 'done'.
        if self._number is None:
            word, color = "TOOL", T.TEXT_DISABLED
        else:
            word, color = {
                NOW:    ("NOW", "#FFFFFF" if self._selected else "#93C5FD"),
                DONE:   (self._result or "done", T.SUCCESS),
                READY:  ("ready", T.TEXT_MUTED),
                LOCKED: ("locked", T.TEXT_DISABLED),
                ERROR:  ("✕ error", T.DANGER_TEXT),
            }.get(self._state, ("", T.TEXT_MUTED))
        self.state_lb.setText(word)
        self.state_lb.setStyleSheet(
            f"color: {color}; font-size: {T.FS_OVERLINE}px; font-weight: 600;"
            f" letter-spacing: 0.5px; background: transparent; border: none;"
            f" font-family: {T.FONT_MONO};")


    def paintEvent(self, ev):
        """Qt does NOT paint a stylesheet background on a QWidget *subclass*
        unless the subclass draws PE_Widget itself. Without this the widget is
        transparent and any `background:` rule is silently ignored."""
        opt = QStyleOption()
        opt.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, p, self)

    # -- interaction --------------------------------------------------
    def mousePressEvent(self, ev):
        self.clicked.emit(self.row)
        super().mousePressEvent(ev)

    def enterEvent(self, ev):
        if not self._selected:
            self.setStyleSheet(
                f"QWidget#navItemRoot {{ background: {T.BG_CHROME_RAISED};"
                f" border-left: 3px solid transparent; }}")
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self._restyle()
        super().leaveEvent(ev)


class Sidebar(QWidget):
    """Pipeline stepper + tools. API-compatible with the old QListWidget."""

    currentRowChanged = pyqtSignal(int)
    projectClicked = pyqtSignal()      # user clicked the project name block

    def __init__(self, nav_items: Sequence[Tuple[str, str]],
                 pipeline_indices: Sequence[int], parent=None):
        super().__init__(parent)
        self._items: List[_NavItem] = []
        self._pipeline = list(pipeline_indices)
        self._current = -1
        self._done: set = set()

        self.setObjectName("sidebarRoot")
        self.setStyleSheet(
            f"QWidget#sidebarRoot {{ background: {T.BG_CHROME}; border: none; }}")
        self.setFixedWidth(T.W_SIDEBAR)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 12, 0, 12)
        v.setSpacing(0)

        # ---- brand / project block (clicking it edits the project)
        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(16, 0, 16, 0)
        brand_row.setSpacing(8)
        try:
            from .branding import logo_pixmap
            pm = logo_pixmap(20)
            if pm is not None:
                mark = QLabel()
                mark.setPixmap(pm)
                mark.setFixedSize(20, 20)
                mark.setStyleSheet("background: transparent; border: none;")
                brand_row.addWidget(mark)
        except Exception:
            pass
        brand = QLabel("PAAF")
        brand.setStyleSheet(
            f"color: #FFFFFF; font-size: {T.FS_SECTION}px; font-weight: 600;"
            f" letter-spacing: 1px; background: transparent;")
        brand_row.addWidget(brand)
        brand_row.addStretch(1)
        v.addLayout(brand_row)

        self.project_lb = QLabel("untitled project")
        self.project_lb.setCursor(Qt.PointingHandCursor)
        self.project_lb.setToolTip(
            "Click to change the project name and output folder")
        self.project_lb.setStyleSheet(
            f"color: #94A3B8; font-size: {T.FS_CAPTION}px;"
            f" padding: 2px 16px 0 16px; background: transparent;")
        # QLabel has no clicked signal; forward the press.
        self.project_lb.mousePressEvent = (
            lambda ev: self.projectClicked.emit())
        v.addWidget(self.project_lb)

        self.step_lb = QLabel("")
        self.step_lb.setStyleSheet(
            f"color: #64748B; font-size: {T.FS_OVERLINE}px; font-weight: 600;"
            f" letter-spacing: 0.5px; padding: 6px 16px 10px 16px;"
            f" background: transparent;")
        v.addWidget(self.step_lb)
        v.addWidget(_sep())

        # ---- pipeline section
        v.addWidget(self._section("Pipeline"))
        for step_no, row in enumerate(self._pipeline, start=1):
            name, sub = nav_items[row]
            it = _NavItem(row, name, number=step_no, tooltip=sub)
            it.clicked.connect(self.setCurrentRow)
            self._items.append(it)
            v.addWidget(it)

        # ---- tools section
        tool_rows = [i for i in range(len(nav_items)) if i not in self._pipeline]
        if tool_rows:
            v.addWidget(self._section("Tools"))
            for row in tool_rows:
                name, sub = nav_items[row]
                it = _NavItem(row, name, number=None, tooltip=sub)
                it.clicked.connect(self.setCurrentRow)
                self._items.append(it)
                v.addWidget(it)

        v.addStretch(1)

        # ---- progress card
        v.addWidget(_sep())
        card = QWidget()
        card.setObjectName("navProgress")
        card.setStyleSheet(
            f"QWidget#navProgress {{ background: {T.BG_CHROME_RAISED};"
            f" border-radius: {T.R_CARD}px; }}")
        cv = QVBoxLayout(card)
        cv.setContentsMargins(12, 8, 12, 8)
        cv.setSpacing(2)
        title = QLabel("Pipeline progress")
        title.setStyleSheet(
            f"color: #94A3B8; font-size: {T.FS_OVERLINE}px; font-weight: 600;"
            f" letter-spacing: 1px; background: transparent;")
        cv.addWidget(title)
        self.progress_lb = QLabel(f"0 of {len(self._pipeline)} complete")
        self.progress_lb.setStyleSheet(
            f"color: #FFFFFF; font-size: {T.FS_BODY}px; background: transparent;")
        cv.addWidget(self.progress_lb)
        wrap = QWidget()
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(12, 10, 12, 0)
        wl.addWidget(card)
        v.addWidget(wrap)

        self._refresh_states()


    def paintEvent(self, ev):
        """Qt does NOT paint a stylesheet background on a QWidget *subclass*
        unless the subclass draws PE_Widget itself. Without this the widget is
        transparent and any `background:` rule is silently ignored."""
        opt = QStyleOption()
        opt.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, p, self)

    # ------------------------------------------------------------ helpers
    def _section(self, text: str) -> QLabel:
        lb = QLabel(text.upper())
        lb.setObjectName("navSection")
        lb.setStyleSheet(
            f"color: {T.TEXT_DISABLED}; font-size: {T.FS_OVERLINE}px;"
            f" font-weight: 600; letter-spacing: 1px;"
            f" padding: 12px 16px 4px 16px; background: transparent;")
        return lb

    def _item_for_row(self, row: int) -> Optional[_NavItem]:
        for it in self._items:
            if it.row == row:
                return it
        return None

    # -------------------------------------------- QListWidget-compatible API
    def setCurrentRow(self, row: int) -> None:
        if row == self._current or not (0 <= row < len(self._items) + 8):
            if row == self._current:
                return
        self._current = row
        for it in self._items:
            it.set_selected(it.row == row)
        self._refresh_states()
        self.currentRowChanged.emit(row)

    def currentRow(self) -> int:
        return self._current

    def count(self) -> int:
        return len(self._items)

    # ------------------------------------------------------------- state
    def set_project_name(self, name: str) -> None:
        self.project_lb.setText(name or "untitled project")

    def set_step_result(self, row: int, text: str) -> None:
        """Record the headline result of a completed step (spec: done steps
        show a value like ``n=50`` or ``MMFF94`` rather than just 'done')."""
        it = self._item_for_row(row)
        if it is not None:
            it.set_result(text)

    def mark_done(self, row: int, done: bool = True) -> None:
        """Mark a pipeline step complete (drives the progress card)."""
        if done:
            self._done.add(row)
        else:
            self._done.discard(row)
        self._refresh_states()

    def _refresh_states(self) -> None:
        """Recompute every step's state: done / now / ready / locked."""
        for pos, row in enumerate(self._pipeline):
            it = self._item_for_row(row)
            if it is None:
                continue
            if row == self._current:
                it.set_state(NOW)
            elif row in self._done:
                it.set_state(DONE)
            else:
                # A step is "ready" when the one before it is either complete
                # or the step the user is on right now; anything further out is
                # "locked". The first step is always reachable.
                prev = self._pipeline[pos - 1] if pos else None
                reachable = (pos == 0
                             or prev in self._done
                             or prev == self._current)
                it.set_state(READY if reachable else LOCKED)

        n_done = len(self._done)
        self.progress_lb.setText(f"{n_done} of {len(self._pipeline)} complete")
        if self._current in self._pipeline:
            step = self._pipeline.index(self._current) + 1
            it = self._item_for_row(self._current)
            label = it.text_lb.text() if it else ""
            self.step_lb.setText(
                f"STEP {step} OF {len(self._pipeline)} · {label.upper()}")
        else:
            it = self._item_for_row(self._current)
            self.step_lb.setText(
                f"TOOL · {it.text_lb.text().upper()}" if it else "")
