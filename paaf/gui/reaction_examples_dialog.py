"""Browser for the built-in reaction examples.

Fifty-plus worked, pre-validated schemes (paaf.reaction_examples), shown
with their mapped SMILES and a plain-language explanation of every tag.
"Use this example" hands the selected entry back to the Reaction scheme,
which loads it into a reaction block ready to validate or edit.
"""
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea,
    QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from . import tokens as T
from ..reaction_examples import EXAMPLES, ReactionExample, by_category


def _lb(text: str, size: int, color: str, bold: bool = False,
        mono: bool = False) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(True)
    lb.setTextInteractionFlags(Qt.TextSelectableByMouse)
    font = f"font-family: {T.FONT_MONO};" if mono else ""
    weight = "font-weight: 700;" if bold else ""
    lb.setStyleSheet(f"color: {color}; font-size: {size}px; {font}{weight}"
                     "background: transparent; border: none;")
    return lb


class ReactionExamplesDialog(QDialog):
    """Category tree on the left, full explanation on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Reaction examples — {len(EXAMPLES)} worked schemes")
        self.resize(980, 620)
        self._selected: Optional[ReactionExample] = None

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(10)

        split = QSplitter(Qt.Horizontal)
        v.addWidget(split, 1)

        # ------------------------------------------------- left: browse
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(6)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search reactions… (e.g. epoxide, ENR)")
        self.search.setFixedHeight(T.H_CONTROL)
        self.search.textChanged.connect(self._refill)
        lv.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self._on_pick)
        lv.addWidget(self.tree, 1)
        split.addWidget(left)

        # ------------------------------------------------ right: detail
        self.detail_area = QScrollArea()
        self.detail_area.setWidgetResizable(True)
        self.detail_area.setStyleSheet(
            f"QScrollArea {{ background: {T.BG_SURFACE};"
            f" border: 1px solid {T.BORDER}; border-radius: {T.R_CARD}px; }}")
        split.addWidget(self.detail_area)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)

        # ------------------------------------------------------ buttons
        btns = QHBoxLayout()
        hint = _lb("Every example validates as-is — load it, run Validate, "
                   "then edit it into your own chemistry.",
                   T.FS_CAPTION, T.TEXT_MUTED)
        btns.addWidget(hint, 1)
        self.b_use = QPushButton("Use this example")
        self.b_use.setObjectName("primary")
        self.b_use.setFixedHeight(T.H_CONTROL + 4)
        self.b_use.setEnabled(False)
        self.b_use.clicked.connect(self.accept)
        btns.addWidget(self.b_use)
        b_close = QPushButton("Close")
        b_close.setFixedHeight(T.H_CONTROL + 4)
        b_close.clicked.connect(self.reject)
        btns.addWidget(b_close)
        v.addLayout(btns)

        self._refill()
        self._show_detail(None)

    # ---------------------------------------------------------- browse
    def _refill(self, *_a) -> None:
        needle = self.search.text().strip().lower()
        self.tree.clear()
        for cat, entries in by_category().items():
            hits = [e for e in entries
                    if not needle
                    or needle in e.name.lower()
                    or needle in e.description.lower()
                    or needle in e.category.lower()]
            if not hits:
                continue
            top = QTreeWidgetItem([f"{cat}  ({len(hits)})"])
            top.setFlags(top.flags() & ~Qt.ItemIsSelectable)
            self.tree.addTopLevelItem(top)
            for e in hits:
                child = QTreeWidgetItem([e.name])
                child.setData(0, Qt.UserRole, e.key)
                top.addChild(child)
            top.setExpanded(bool(needle) or self.tree.topLevelItemCount() <= 2)

    def _on_pick(self, item, _prev) -> None:
        key = item.data(0, Qt.UserRole) if item is not None else None
        ex = next((e for e in EXAMPLES if e.key == key), None)
        self._selected = ex
        self.b_use.setEnabled(ex is not None)
        self._show_detail(ex)

    # ---------------------------------------------------------- detail
    def _show_detail(self, ex: Optional[ReactionExample]) -> None:
        w = QWidget()
        w.setStyleSheet("background: transparent; border: none;")
        dv = QVBoxLayout(w)
        dv.setContentsMargins(16, 14, 16, 14)
        dv.setSpacing(8)

        if ex is None:
            dv.addWidget(_lb("Pick a reaction on the left.\n\nCategories "
                             "group the chemistry; the search box matches "
                             "names and descriptions.",
                             T.FS_BODY, T.TEXT_MUTED))
            dv.addStretch(1)
            self.detail_area.setWidget(w)
            return

        dv.addWidget(_lb(ex.name, T.FS_SECTION + 2, T.TEXT, bold=True))
        dv.addWidget(_lb(ex.category.upper(), T.FS_CAPTION, T.PRIMARY,
                         bold=True))
        dv.addWidget(_lb(ex.description, T.FS_BODY, T.TEXT_SECONDARY))

        def side(title: str, rows) -> None:
            dv.addSpacing(4)
            dv.addWidget(_lb(title, T.FS_CAPTION, T.TEXT_MUTED, bold=True))
            for label, smi in rows:
                line = QWidget()
                line.setStyleSheet(
                    f"background: {T.BG_SUNKEN}; border: 1px solid "
                    f"{T.BORDER}; border-radius: {T.R_CONTROL}px;")
                hl = QHBoxLayout(line)
                hl.setContentsMargins(8, 5, 8, 5)
                hl.setSpacing(10)
                name = _lb(label, T.FS_CAPTION, T.TEXT_MUTED)
                name.setFixedWidth(120)
                name.setWordWrap(False)
                hl.addWidget(name)
                hl.addWidget(_lb(smi, T.FS_BODY, T.TEXT, mono=True), 1)
                dv.addWidget(line)

        side("REACTANTS", ex.reactants)
        dv.addWidget(_lb("↓  reacts to form", T.FS_CAPTION, T.TEXT_MUTED))
        side("PRODUCTS", ex.products)

        dv.addSpacing(4)
        dv.addWidget(_lb("WHAT THE TAGS MEAN", T.FS_CAPTION, T.TEXT_MUTED,
                         bold=True))
        for tag, meaning in ex.tags:
            row = QHBoxLayout()
            row.setSpacing(8)
            t = _lb(tag, T.FS_BODY, T.MAP_TOKEN, bold=True, mono=True)
            t.setFixedWidth(44)
            row.addWidget(t, 0, Qt.AlignTop)
            row.addWidget(_lb(meaning, T.FS_BODY, T.TEXT_SECONDARY), 1)
            dv.addLayout(row)

        if ex.note:
            dv.addSpacing(4)
            dv.addWidget(_lb(ex.note, T.FS_CAPTION, T.TEXT_MUTED))
        dv.addStretch(1)
        self.detail_area.setWidget(w)

    # ------------------------------------------------------------ API
    def selected_example(self) -> Optional[ReactionExample]:
        return self._selected
