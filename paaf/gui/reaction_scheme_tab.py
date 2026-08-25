"""Reaction scheme tab — define a reaction as SMILES + atom-map numbers.

Implements screens 2-4 of the *PAAF UI redesign v1* spec (empty / validated /
error states). The user writes an equation:

    reactant 1  CC(=O)[OH:1]
    reactant 2  [C:2]CO
      product   CC(=O)O[C:2]

and PAAF reports which bonds form and break *before anything is written to
disk*. Validated reactions are appended to a reaction library JSON that the
crosslinking engine consumes.

Layout follows the spec's engineer rules: every row is a QHBoxLayout, every
column a QVBoxLayout; the equation divider is two QFrame.HLines around a
centred label; SMILES fields are one-line QPlainTextEdits carrying a
QSyntaxHighlighter for ``[X:n]`` tokens (QLineEdit cannot render rich text).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from PyQt5.QtCore import QObject, QRegExp, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import (
    QColor, QFont, QPainter, QSyntaxHighlighter, QTextCharFormat,
)
from PyQt5.QtWidgets import (
    QAbstractItemView, QFileDialog, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
    QCheckBox, QComboBox, QFormLayout, QScrollArea, QSizePolicy, QSpinBox,
    QStyle, QStyleOption, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import tokens as T
from .page import stat_row


# ===================================================================== worker
class _ExportWorker(QObject):
    """Runs the 3D build + optimisation + dl_field typing off the UI thread."""
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, jobs: list, out_dir: str, ff_key: str,
                 dl_field_dir: str, opt_ff: str, run_typing: bool,
                 optimize: bool = True):
        super().__init__()
        self.jobs = jobs
        self.out_dir = out_dir
        self.ff_key = ff_key
        self.dl_field_dir = dl_field_dir
        self.opt_ff = opt_ff
        self.run_typing = run_typing
        self.optimize = optimize

    @pyqtSlot()
    def run(self):
        try:
            from ..reaction_export import export_reaction
            done = []
            for job in self.jobs:
                self.progress.emit(f"Exporting '{job['name']}' …")
                done.append(export_reaction(
                    name=job["name"],
                    reactants=job["reactants"],
                    products=job["products"],
                    out_dir=self.out_dir,
                    ff_key=self.ff_key,
                    dl_field_dir=self.dl_field_dir or None,
                    opt_ff=self.opt_ff,
                    optimize=self.optimize,
                    run_typing=self.run_typing,
                    template=job.get("template"),
                    progress=self.progress.emit,
                ))
            self.finished.emit(done)
        except Exception as exc:
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


# ===================================================== atom-map highlighting
class _MapHighlighter(QSyntaxHighlighter):
    """Colour ``[C:1]``-style atom-map tokens inside a SMILES string."""

    def __init__(self, doc):
        super().__init__(doc)
        self._fmt = QTextCharFormat()
        self._build_format()

    def _build_format(self):
        self._fmt.setForeground(QColor(T.MAP_TOKEN))
        self._fmt.setBackground(QColor(T.MAP_TINT))
        self._fmt.setFontWeight(QFont.Bold)
        self._rx = QRegExp(r"\[[^\[\]:]+:\d+\]")

    def retheme(self):
        self._build_format()
        self.rehighlight()

    def highlightBlock(self, text: str) -> None:
        i = self._rx.indexIn(text)
        while i >= 0:
            ln = self._rx.matchedLength()
            self.setFormat(i, ln, self._fmt)
            i = self._rx.indexIn(text, i + ln)


class SmilesEdit(QPlainTextEdit):
    """A one-line SMILES input with atom-map highlighting.

    Styled to look identical to the other inputs, per the spec — QLineEdit
    cannot carry rich text, so a height-locked QPlainTextEdit is used instead.
    """
    editingFinished = pyqtSignal()

    def __init__(self, placeholder: str = "paste SMILES…", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedHeight(T.H_CONTROL)
        self.setTabChangesFocus(True)
        f = QFont()
        f.setFamily("Consolas")
        f.setStyleHint(QFont.Monospace)
        f.setPixelSize(T.FS_BODY)
        self.setFont(f)
        self._hl = _MapHighlighter(self.document())
        self._set_state("default")

    def text(self) -> str:
        return self.toPlainText().strip()

    def setText(self, s: str) -> None:
        self.setPlainText(s or "")

    def keyPressEvent(self, e):   # keep it a single line
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.editingFinished.emit()
            return
        super().keyPressEvent(e)

    def _set_state(self, state: str) -> None:
        """state: default | invalid — focus is handled by the stylesheet."""
        border = T.DANGER if state == "invalid" else T.BORDER_STRONG
        bg = T.DANGER_TINT if state == "invalid" else T.BG_SURFACE
        self.setStyleSheet(
            f"QPlainTextEdit {{ background: {bg}; border: 1px solid {border};"
            f" border-radius: {T.R_CONTROL}px; padding: 3px 6px;"
            f" selection-background-color: {T.PRIMARY_TINT}; }}"
            f"QPlainTextEdit:focus {{ border: 2px solid {T.PRIMARY};"
            f" padding: 2px 5px; }}"
        )

    def mark_invalid(self, invalid: bool) -> None:
        self._set_state("invalid" if invalid else "default")


# ============================================================ small helpers
def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color: {T.BORDER}; background: {T.BORDER}; max-height: 1px;")
    return f


def _label(text: str, size: int = T.FS_BODY, color: str = T.TEXT,
           bold: bool = False, mono: bool = False) -> QLabel:
    lb = QLabel(text)
    fam = f"font-family: {T.FONT_MONO};" if mono else ""
    weight = "font-weight: 600;" if bold else ""
    lb.setStyleSheet(f"color: {color}; font-size: {size}px; {weight} {fam}")
    return lb


def _overline(text: str) -> QLabel:
    lb = QLabel(text.upper())
    lb.setStyleSheet(
        f"color: {T.TEXT_MUTED}; font-size: {T.FS_OVERLINE}px; font-weight: 600;"
        f" letter-spacing: 1px;")
    return lb


def _card(title: str = "", subtitle: str = "") -> tuple:
    """Return ``(card_widget, body_layout, header_row_layout)``.

    A flat card: 1px border, 6px radius, sunken header bar. The title bar is a
    child widget rather than a native QGroupBox title (spec requirement).

    NOTE: every rule is scoped with ``#objectName``. An unscoped QSS rule set
    on a parent QWidget cascades to *every* descendant, which would draw a
    border around each label inside the card.
    """
    card = QWidget()
    card.setObjectName("cardRoot")
    card.setStyleSheet(
        f"QWidget#cardRoot {{ background: {T.BG_SURFACE};"
        f" border: 1px solid {T.BORDER}; border-radius: {T.R_CARD}px; }}")
    outer = QVBoxLayout(card)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)

    head_w = QWidget()
    head_w.setObjectName("cardHead")
    head_w.setStyleSheet(
        f"QWidget#cardHead {{ background: {T.BG_SUNKEN}; border: none;"
        f" border-bottom: 1px solid {T.BORDER};"
        f" border-top-left-radius: {T.R_CARD}px;"
        f" border-top-right-radius: {T.R_CARD}px; }}")
    head = QHBoxLayout(head_w)
    head.setContentsMargins(T.PAD_CARD, 8, T.PAD_CARD, 8)
    head.setSpacing(T.GAP_LABEL)
    if title:
        head.addWidget(_label(title, T.FS_SECTION, T.TEXT, bold=True))
    if subtitle:
        head.addWidget(_label(subtitle, T.FS_CAPTION, T.TEXT_MUTED))
    head.addStretch(1)
    outer.addWidget(head_w)

    body_w = QWidget()
    body_w.setObjectName("cardBody")
    body_w.setStyleSheet(
        "QWidget#cardBody { background: transparent; border: none; }")
    body = QVBoxLayout(body_w)
    body.setContentsMargins(T.PAD_CARD, T.PAD_CARD, T.PAD_CARD, T.PAD_CARD)
    body.setSpacing(T.GAP_FORM_ROW)
    outer.addWidget(body_w)
    return card, body, head


def _badge(text: str, level: str = "neutral") -> QLabel:
    fill, border, fg = T.badge_colors(level)
    lb = QLabel(text)
    lb.setStyleSheet(
        f"background: {fill}; border: 1px solid {border}; color: {fg};"
        f" border-radius: {T.R_PILL}px; padding: 2px 8px;"
        f" font-size: {T.FS_CAPTION}px; font-weight: 600;")
    lb.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
    return lb


def _button(text: str, kind: str = "secondary") -> QPushButton:
    b = QPushButton(text)
    b.setFixedHeight(T.H_CONTROL)
    b.setCursor(Qt.PointingHandCursor)
    if kind == "primary":
        css = (f"background: {T.PRIMARY}; color: #FFFFFF;"
               f" border: 1px solid {T.PRIMARY};")
        hover = f"background: {T.PRIMARY_HOVER}; border-color: {T.PRIMARY_HOVER};"
    elif kind == "danger":
        # Destructive actions are OUTLINED, never filled (spec rule).
        css = (f"background: {T.BG_SURFACE}; color: {T.DANGER_TEXT};"
               f" border: 1px solid {T.DANGER};")
        hover = f"background: {T.DANGER_TINT};"
    else:
        css = (f"background: {T.BG_SURFACE}; color: {T.TEXT};"
               f" border: 1px solid {T.BORDER_STRONG};")
        hover = f"background: {T.BG_SUNKEN};"
    b.setStyleSheet(
        f"QPushButton {{ {css} border-radius: {T.R_CONTROL}px;"
        f" padding: 0 12px; font-size: {T.FS_BODY}px; }}"
        f"QPushButton:hover {{ {hover} }}"
        f"QPushButton:disabled {{ background: {T.BG_SUNKEN};"
        f" color: {T.TEXT_DISABLED}; border-color: {T.BORDER}; }}")
    return b


def _plain_input(placeholder: str = "") -> QLineEdit:
    e = QLineEdit()
    e.setPlaceholderText(placeholder)
    e.setFixedHeight(T.H_CONTROL)
    e.setStyleSheet(
        f"QLineEdit {{ background: {T.BG_SURFACE};"
        f" border: 1px solid {T.BORDER_STRONG}; border-radius: {T.R_CONTROL}px;"
        f" padding: 0 6px; font-size: {T.FS_BODY}px; }}"
        f"QLineEdit:focus {{ border: 2px solid {T.PRIMARY}; padding: 0 5px; }}")
    return e


# ================================================================ reactant row
class MoleculeRow(QWidget):
    """One molecule in the equation: index · name · SMILES · maps · remove.

    Used for BOTH sides — a reaction consumes one or more molecules and
    produces one or more (esterification gives the ester *and* water).
    """
    changed = pyqtSignal()
    removeRequested = pyqtSignal(object)

    def __init__(self, number: int, placeholder: str = "e.g. acid", parent=None):
        super().__init__(parent)
        self.setObjectName("reactantRow")
        self.setStyleSheet("QWidget#reactantRow { border: none; }")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(T.GAP_LABEL)

        self.idx_lb = QLabel(str(number))
        self.idx_lb.setFixedWidth(18)
        self.idx_lb.setAlignment(Qt.AlignCenter)
        self._set_index_state("ok")
        row.addWidget(self.idx_lb)

        self.name = _plain_input(placeholder)
        self.name.setFixedWidth(T.W_NAME_COL)
        row.addWidget(self.name)

        self.smiles = SmilesEdit()
        self.smiles.textChanged.connect(self._on_changed)
        row.addWidget(self.smiles, 1)

        self.maps_lb = QLabel("—")
        self.maps_lb.setFixedWidth(T.W_MAPS_COL)
        self.maps_lb.setStyleSheet(
            f"color: {T.MAP_TOKEN}; font-family: {T.FONT_MONO};"
            f" font-size: {T.FS_CAPTION}px; font-weight: 600;")
        row.addWidget(self.maps_lb)

        rm = QPushButton("✕")
        rm.setFixedSize(T.W_REMOVE_COL, T.H_CONTROL)
        rm.setToolTip("Remove this molecule")
        rm.setCursor(Qt.PointingHandCursor)
        rm.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none;"
            f" color: {T.TEXT_MUTED}; font-size: {T.FS_BODY}px; }}"
            f"QPushButton:hover {{ color: {T.DANGER_TEXT};"
            f" background: {T.DANGER_TINT}; border-radius: {T.R_CONTROL}px; }}")
        rm.clicked.connect(lambda: self.removeRequested.emit(self))
        row.addWidget(rm)

    def _set_index_state(self, state: str) -> None:
        if state == "error":
            self.idx_lb.setText("✕")
            self.idx_lb.setStyleSheet(
                f"color: {T.DANGER_TEXT}; font-weight: 700;"
                f" font-size: {T.FS_BODY}px;")
        else:
            self.idx_lb.setStyleSheet(
                f"color: {T.TEXT_MUTED}; font-family: {T.FONT_MONO};"
                f" font-size: {T.FS_BODY}px;")

    def set_number(self, n: int) -> None:
        self.idx_lb.setText(str(n))
        self._set_index_state("ok")

    def mark_error(self, is_error: bool) -> None:
        self.smiles.mark_invalid(is_error)
        self._set_index_state("error" if is_error else "ok")

    def set_values(self, name: str, smiles: str) -> None:
        """Fill the row programmatically (used by the examples browser)."""
        self.name.setText(name)
        self.smiles.setText(smiles)
        self._on_changed()

    def _on_changed(self) -> None:
        from ..reaction_smiles import parse_atom_maps
        maps = sorted(parse_atom_maps(self.smiles.text()))
        self.maps_lb.setText(" ".join(str(m) for m in maps) if maps else "—")
        self.changed.emit()


    def paintEvent(self, ev):
        """Qt does NOT paint a stylesheet background on a QWidget *subclass*
        unless the subclass draws PE_Widget itself. Without this the widget is
        transparent and any `background:` rule is silently ignored."""
        opt = QStyleOption()
        opt.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, p, self)

    def values(self) -> tuple:
        return self.name.text().strip(), self.smiles.text()



# ============================================================ reaction block
class ReactionBlock(QWidget):
    """One complete reaction: N reactants → M products, with its own result.

    A scheme may hold several of these. Each block validates, exports and is
    added to the library independently, so "Reaction 1" and "Reaction 2" never
    share atom-map numbers or interfere with one another.
    """

    changed = pyqtSignal(object)          # emits self, so the tab can follow
    removeRequested = pyqtSignal(object)

    def __init__(self, number: int, parent=None):
        super().__init__(parent)
        self.number = number
        self._rows: List[MoleculeRow] = []
        self._prod_rows: List[MoleculeRow] = []
        self.result = None                # SchemeResult once validated
        self.reactant_smiles: List[str] = []
        self.product_smiles: List[str] = []

        self.setObjectName("reactionBlock")
        self.setStyleSheet(
            f"QWidget#reactionBlock {{ background: {T.BG_SURFACE};"
            f" border: 1px solid {T.BORDER_STRONG};"
            f" border-radius: {T.R_CARD}px; }}")

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # ---- block header: "Reaction N" + name + status + remove
        head_w = QWidget()
        head_w.setObjectName("reactionHead")
        head_w.setStyleSheet(
            f"QWidget#reactionHead {{ background: {T.BG_SUNKEN}; border: none;"
            f" border-bottom: 1px solid {T.BORDER};"
            f" border-top-left-radius: {T.R_CARD}px;"
            f" border-top-right-radius: {T.R_CARD}px; }}")
        head = QHBoxLayout(head_w)
        head.setContentsMargins(T.PAD_CARD, 8, T.PAD_CARD, 8)
        head.setSpacing(T.GAP_LABEL)

        self.title_lb = _label(f"Reaction {number}", T.FS_SECTION, T.TEXT, bold=True)
        head.addWidget(self.title_lb)
        self.name = _plain_input("name — e.g. esterification")
        self.name.setFixedWidth(220)
        head.addWidget(self.name)
        head.addStretch(1)

        self.status = _badge("Not run", "neutral")
        head.addWidget(self.status)
        self.b_remove = QPushButton("✕")
        self.b_remove.setFixedSize(T.W_REMOVE_COL, T.H_CONTROL)
        self.b_remove.setToolTip("Remove this reaction")
        self.b_remove.setCursor(Qt.PointingHandCursor)
        self.b_remove.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none;"
            f" color: {T.TEXT_MUTED}; font-size: {T.FS_BODY}px; }}"
            f"QPushButton:hover {{ color: {T.DANGER_TEXT};"
            f" background: {T.DANGER_TINT}; border-radius: {T.R_CONTROL}px; }}")
        self.b_remove.clicked.connect(lambda: self.removeRequested.emit(self))
        head.addWidget(self.b_remove)
        v.addWidget(head_w)

        body_w = QWidget()
        body_w.setObjectName("reactionBody")
        body_w.setStyleSheet(
            "QWidget#reactionBody { background: transparent; border: none; }")
        body = QVBoxLayout(body_w)
        body.setContentsMargins(T.PAD_CARD, T.PAD_CARD, T.PAD_CARD, T.PAD_CARD)
        body.setSpacing(T.GAP_FORM_ROW)
        v.addWidget(body_w)

        # ---- reactants
        rhead = QHBoxLayout(); rhead.setSpacing(T.GAP_LABEL)
        rhead.addWidget(_overline("Reactants"))
        self.reactant_count_lb = _label("", T.FS_CAPTION, T.TEXT_MUTED)
        rhead.addWidget(self.reactant_count_lb)
        rhead.addStretch(1)
        b_addr = _button("+ Add molecule")
        b_addr.clicked.connect(lambda: (self.add_reactant(), self._emit()))
        rhead.addWidget(b_addr)
        body.addLayout(rhead)
        body.addLayout(self._column_header())
        self.rows_box = QVBoxLayout(); self.rows_box.setSpacing(6)
        body.addLayout(self.rows_box)

        # ---- divider
        div = QHBoxLayout(); div.setSpacing(T.GAP_LABEL)
        div.addWidget(_hline(), 1)
        div.addWidget(_label("↓  reacts to form", T.FS_CAPTION, T.TEXT_MUTED))
        div.addWidget(_hline(), 1)
        body.addLayout(div)

        # ---- products
        phead = QHBoxLayout(); phead.setSpacing(T.GAP_LABEL)
        phead.addWidget(_overline("Products"))
        self.product_count_lb = _label("", T.FS_CAPTION, T.TEXT_MUTED)
        phead.addWidget(self.product_count_lb)
        phead.addStretch(1)
        b_addp = _button("+ Add molecule")
        b_addp.clicked.connect(lambda: (self.add_product(), self._emit()))
        phead.addWidget(b_addp)
        body.addLayout(phead)
        body.addLayout(self._column_header())
        self.prod_box = QVBoxLayout(); self.prod_box.setSpacing(6)
        body.addLayout(self.prod_box)

        # ---- per-reaction result
        self.result_box = QVBoxLayout(); self.result_box.setSpacing(4)
        body.addLayout(self.result_box)

        self.add_reactant(); self.add_reactant()
        self.add_product()
        self._renumber()

    # ------------------------------------------------------------ layout
    def _column_header(self) -> QHBoxLayout:
        h = QHBoxLayout(); h.setSpacing(T.GAP_LABEL)
        sp = QLabel(); sp.setFixedWidth(18); h.addWidget(sp)
        n = _overline("name"); n.setFixedWidth(T.W_NAME_COL); h.addWidget(n)
        h.addWidget(_overline("SMILES with atom maps"), 1)
        m = _overline("maps"); m.setFixedWidth(T.W_MAPS_COL); h.addWidget(m)
        sp2 = QLabel(); sp2.setFixedWidth(T.W_REMOVE_COL); h.addWidget(sp2)
        return h

    def paintEvent(self, ev):
        opt = QStyleOption(); opt.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, p, self)

    # ------------------------------------------------------------- rows
    def add_reactant(self) -> None:
        row = MoleculeRow(len(self._rows) + 1, "e.g. acid")
        row.removeRequested.connect(lambda r: self._remove(self._rows, r, "reactant"))
        row.changed.connect(self._emit)
        self._rows.append(row)
        self.rows_box.addWidget(row)
        self._renumber()

    def add_product(self) -> None:
        row = MoleculeRow(len(self._prod_rows) + 1, "e.g. ester")
        row.removeRequested.connect(lambda r: self._remove(self._prod_rows, r, "product"))
        row.changed.connect(self._emit)
        self._prod_rows.append(row)
        self.prod_box.addWidget(row)
        self._renumber()

    def _remove(self, rows, row, what: str) -> None:
        if len(rows) <= 1:
            QMessageBox.information(
                self, "Reaction scheme",
                f"Reaction {self.number} needs at least one {what}.")
            return
        rows.remove(row)
        row.setParent(None); row.deleteLater()
        self._renumber(); self._emit()

    def _renumber(self) -> None:
        for i, r in enumerate(self._rows, start=1):
            r.set_number(i)
        for i, r in enumerate(self._prod_rows, start=1):
            r.set_number(i)
        n, m = len(self._rows), len(self._prod_rows)
        self.reactant_count_lb.setText(f"{n} molecule{'s' if n != 1 else ''}")
        self.product_count_lb.setText(f"{m} molecule{'s' if m != 1 else ''}")

    def set_number(self, n: int) -> None:
        self.number = n
        self.title_lb.setText(f"Reaction {n}")

    def _emit(self) -> None:
        self.changed.emit(self)

    # ------------------------------------------------------------ values
    def reaction_name(self) -> str:
        return self.name.text().strip() or f"reaction_{self.number}"

    def values(self, expand: bool = True) -> tuple:
        """``(name, reactant_smiles, product_smiles)`` — blanks skipped."""

        def collect(rows):
            return [row.values()[1] for row in rows if row.values()[1]]

        return self.reaction_name(), collect(self._rows), collect(self._prod_rows)

    def load_example(self, example) -> None:
        """Fill this block from a :class:`paaf.reaction_examples.ReactionExample`."""
        self.name.setText(example.name)
        # Match the row counts to the example, reusing existing rows.
        while len(self._rows) > len(example.reactants):
            r = self._rows.pop()
            r.setParent(None); r.deleteLater()
        while len(self._rows) < len(example.reactants):
            self.add_reactant()
        while len(self._prod_rows) > len(example.products):
            r = self._prod_rows.pop()
            r.setParent(None); r.deleteLater()
        while len(self._prod_rows) < len(example.products):
            self.add_product()
        for row, (label, smi) in zip(self._rows, example.reactants):
            row.set_values(label, smi)
        for row, (label, smi) in zip(self._prod_rows, example.products):
            row.set_values(label, smi)
        self._renumber()
        self._emit()

    def is_filled(self) -> bool:
        _, r, p = self.values()
        return bool(r and p)

    def maps(self) -> tuple:
        """``({map: reactant_no}, {map: product_no})`` for the rail table."""
        from ..reaction_smiles import parse_atom_maps
        r_maps, p_maps = {}, {}
        for i, row in enumerate(self._rows, 1):
            for m in parse_atom_maps(row.smiles.text()):
                r_maps.setdefault(m, i)
        for i, row in enumerate(self._prod_rows, 1):
            for m in parse_atom_maps(row.smiles.text()):
                p_maps.setdefault(m, i)
        return r_maps, p_maps

    # -------------------------------------------------------- validation
    def clear_marks(self) -> None:
        for row in self._rows + self._prod_rows:
            row.mark_error(False)

    def validate(self):
        """Validate this reaction alone and render its result inline."""
        from ..reaction_smiles import validate_scheme
        name, reactants, products = self.values()
        self.clear_marks()
        res = validate_scheme(reactants, products, name=name)
        self.result = res
        self.reactant_smiles, self.product_smiles = reactants, products

        for iss in res.errors:
            rows = self._rows if iss.where == "reactant" else self._prod_rows
            if iss.index is not None and 0 <= iss.index < len(rows):
                rows[iss.index].mark_error(True)

        self._render_result(res)
        self._set_status(res)
        return res

    def _set_status(self, res) -> None:
        if res is None:
            text, level = "Not run", "neutral"
        elif res.ok:
            text, level = "✓ Validated", "ok"
        else:
            n = len(res.errors)
            text, level = f"✕ {n} error{'s' if n != 1 else ''}", "error"
        fill, border, fg = T.badge_colors(level)
        self.status.setText(text)
        self.status.setStyleSheet(
            f"background: {fill}; border: 1px solid {border}; color: {fg};"
            f" border-radius: {T.R_PILL}px; padding: 2px 8px;"
            f" font-size: {T.FS_CAPTION}px; font-weight: 600;")

    def _clear_layout(self, lay) -> None:
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None); w.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())

    def _render_result(self, res) -> None:
        self._clear_layout(self.result_box)
        if res is None:
            return
        if res.ok:
            self.result_box.addWidget(_label(
                f"✓  {res.n_bonds_formed} bond(s) formed · "
                f"{res.n_bonds_broken} broken · {res.n_atoms_deleted} atom(s) "
                f"deleted", T.FS_BODY, T.SUCCESS_TEXT, bold=True))
            for a, b in res.bonds_formed[:6]:
                self.result_box.addWidget(_label(
                    f"＋  atom {a} — atom {b}", T.FS_CAPTION,
                    T.SUCCESS_TEXT, mono=True))
        for iss in res.issues:
            level = "error" if iss.is_error else "warn"
            fill, border, fg = T.badge_colors(level)
            w = QWidget()
            w.setObjectName("issueBox")
            w.setStyleSheet(
                f"QWidget#issueBox {{ background: {fill};"
                f" border: 1px solid {border};"
                f" border-radius: {T.R_CONTROL}px; }}")
            h = QHBoxLayout(w)
            h.setContentsMargins(8, 6, 8, 6); h.setSpacing(T.GAP_LABEL)
            code = QLabel(iss.code)
            code.setFixedWidth(70)
            code.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            code.setStyleSheet(
                f"color: {fg}; font-family: {T.FONT_MONO}; font-weight: 700;"
                f" font-size: {T.FS_CAPTION}px; border: none;")
            h.addWidget(code)
            msg = QLabel(iss.message)
            msg.setWordWrap(True)
            msg.setStyleSheet(
                f"color: {fg}; font-size: {T.FS_CAPTION}px; border: none;")
            h.addWidget(msg, 1)
            self.result_box.addWidget(w)

    def load(self, name: str, reactants: list, products: list) -> None:
        """Fill this block from ``(label, smiles)`` pairs."""
        self.name.setText(name)
        while len(self._rows) < len(reactants):
            self.add_reactant()
        for row, (nm, smi) in zip(self._rows, reactants):
            row.name.setText(nm); row.smiles.setText(smi)
        while len(self._prod_rows) < len(products):
            self.add_product()
        for row, (nm, smi) in zip(self._prod_rows, products):
            row.name.setText(nm); row.smiles.setText(smi)
        self._emit()


# ==================================================================== the tab
class ReactionSchemeTab(QWidget):
    """Define one or more reactions as SMILES + atom-map numbers.

    Each reaction is an independent :class:`ReactionBlock` — "Reaction 1" may
    consume three molecules and give one product, while "Reaction 2" has its
    own molecules and its own product. Validation, the library and the LAMMPS
    export all operate per reaction, so each gets its own output folder.
    """

    log = pyqtSignal(str)

    EXAMPLE = {
        "name": "esterification",
        "reactants": [("acid", "CC(=O)[OH:1]"), ("alcohol", "[C:2]CO")],
        # Esterification really makes TWO products: the ester and water.
        "products": [("ester", "CC(=O)O[C:2]"), ("water", "O")],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._blocks: List[ReactionBlock] = []
        self._active: ReactionBlock = None
        self._library: List[dict] = []
        self._settings_provider = None
        self._build()
        self._add_reaction()
        self._refresh_maps_panel()

    # ------------------------------------------------------------- layout
    def _build(self) -> None:
        """Three steps of its own: Scheme → Force field → Export.

        The reaction scheme is a self-contained branch: it does not use the
        pipeline's Chain page (chain length lives on each reaction), and it has
        its OWN force field and export so a reaction reference set can be typed
        differently from the main system without disturbing it.
        """
        page = QVBoxLayout(self)
        page.setContentsMargins(T.PAD_PAGE, 8, T.PAD_PAGE, 12)
        page.setSpacing(T.GAP_FORM_ROW)

        self.steps = QTabWidget()
        self.steps.setDocumentMode(True)
        self.steps.addTab(self._scheme_page(), "1 · Scheme")
        self.steps.addTab(self._ff_page(), "2 · Force field")
        self.steps.addTab(self._export_page(), "3 · Export")
        self.steps.currentChanged.connect(self._on_step_changed)
        from .page import fit_tabs_to_current
        fit_tabs_to_current(self.steps)   # short steps must not inherit
                                          # the tallest step's height
        page.addWidget(self.steps, 1)
        page.addWidget(self._action_bar())

    def _scheme_page(self) -> QWidget:
        w = QWidget()
        body = QHBoxLayout(w)
        body.setContentsMargins(0, T.GAP_FORM_ROW, 0, 0)
        body.setSpacing(T.GAP_SECTION)

        col_host = QWidget()
        col_host.setStyleSheet("background: transparent; border: none;")
        col_host.setLayout(self._main_column())
        scroll = QScrollArea()
        scroll.setWidget(col_host)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        body.addWidget(scroll, 1)
        body.addWidget(self._right_rail())
        return w

    # ------------------------------------------------- step 2: force field
    def _ff_page(self) -> QWidget:
        from ..ff_registry import list_ffs

        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, T.GAP_FORM_ROW, 0, 0)
        outer.setSpacing(T.GAP_SECTION)

        row = QHBoxLayout(); row.setSpacing(T.GAP_SECTION)

        card, body, head = _card(
            "Force field for the reaction set",
            "independent of the pipeline's Force-field page")
        b_copy = _button("Copy from pipeline")
        b_copy.setToolTip(
            "Load whatever is currently selected on the main Force-field page")
        b_copy.clicked.connect(self._copy_ff_from_pipeline)
        head.addWidget(b_copy)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(T.GAP_FORM_ROW)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.rx_ff_combo = QComboBox()
        from .ff_combo import populate_force_field_combo
        from ..config import ForceFieldCfg as _FFCfg
        populate_force_field_combo(self.rx_ff_combo, _FFCfg().key)
        self.rx_ff_combo.currentIndexChanged.connect(self._on_rx_ff_changed)
        form.addRow("Force field", self.rx_ff_combo)

        self.rx_ff_notes = _label("", T.FS_CAPTION, T.TEXT_MUTED)
        self.rx_ff_notes.setWordWrap(True)
        form.addRow("", self.rx_ff_notes)

        self.rx_dl_lib = _plain_input("path to dl_f_4.13/lib (DL_FIELD FFs only)")
        b_browse = _button("Browse…")
        b_browse.clicked.connect(self._browse_rx_dl_lib)
        dl_row = QHBoxLayout(); dl_row.setSpacing(T.GAP_LABEL)
        dl_row.addWidget(self.rx_dl_lib, 1); dl_row.addWidget(b_browse)
        dl_wrap = QWidget(); dl_wrap.setLayout(dl_row)
        form.addRow("DL_FIELD lib dir", dl_wrap)

        self.rx_opt_ff = QComboBox()
        self.rx_opt_ff.addItems(["UFF", "MMFF94", "MMFF94s", "Ghemical", "GAFF"])
        self.rx_opt_ff.setToolTip(
            "OpenBabel force field used to minimise each 3D structure before "
            "typing")
        form.addRow("Geometry optimiser", self.rx_opt_ff)

        self.rx_optimize = QCheckBox("Energy-minimise each structure before typing")
        self.rx_optimize.setChecked(True)
        form.addRow("", self.rx_optimize)

        self.rx_run_typing = QCheckBox("Run dl_field to produce LAMMPS .data")
        self.rx_run_typing.setChecked(True)
        self.rx_run_typing.setToolTip(
            "Untick to export 3D structures only, without force-field typing")
        form.addRow("", self.rx_run_typing)
        body.addLayout(form)
        row.addWidget(card, 1)

        # summary
        card2, body2, _ = _card("What this applies to")
        card2.setFixedWidth(300)
        self.rx_stat_reactions = stat_row("Valid reactions", "0")
        self.rx_stat_ff = stat_row("Force field", "—")
        self.rx_stat_opt = stat_row("Optimiser", "UFF")
        for wdg in (self.rx_stat_reactions, self.rx_stat_ff, self.rx_stat_opt):
            body2.addWidget(wdg)
        body2.addWidget(_label(
            "Every reactant and product of every validated reaction is "
            "minimised and typed with these settings.",
            T.FS_CAPTION, T.TEXT_MUTED))
        body2.addStretch(1)
        row.addWidget(card2)
        outer.addLayout(row)
        outer.addStretch(1)
        self._on_rx_ff_changed(0)
        return w

    # ----------------------------------------------------- step 3: export
    def _export_page(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, T.GAP_FORM_ROW, 0, 0)
        outer.setSpacing(T.GAP_SECTION)

        card, body, head = _card(
            "Output", "one folder per reaction, ready for the reaction engine")
        b_browse = _button("Browse…")
        b_browse.clicked.connect(self._browse_rx_out_dir)
        head.addWidget(b_browse)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(T.GAP_FORM_ROW)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.rx_out_dir = _plain_input("folder that will hold the reaction folders")
        self.rx_out_dir.textChanged.connect(lambda _t: self._refresh_export_preview())
        form.addRow("Reactions folder", self.rx_out_dir)
        body.addLayout(form)

        self.rx_out_hint = _label("", T.FS_CAPTION, T.TEXT_MUTED)
        self.rx_out_hint.setWordWrap(True)
        body.addWidget(self.rx_out_hint)
        outer.addWidget(card)

        card2, body2, head2 = _card("Files that will be written")
        self.rx_preview_count = _label("", T.FS_CAPTION, T.TEXT_MUTED)
        head2.addWidget(self.rx_preview_count)
        self.rx_preview = QTableWidget(0, 4)
        self.rx_preview.setHorizontalHeaderLabels(
            ["reaction", "folder", "reactants → products", "files"])
        self.rx_preview.verticalHeader().setVisible(False)
        self.rx_preview.verticalHeader().setDefaultSectionSize(T.H_TABLE_ROW)
        self.rx_preview.horizontalHeader().setFixedHeight(T.H_TABLE_HEADER)
        self.rx_preview.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.rx_preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.rx_preview.setMinimumHeight(140)
        body2.addWidget(self.rx_preview)
        outer.addWidget(card2, 1)
        outer.addStretch(1)
        return w

    def _main_column(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(T.GAP_SECTION)

        # ---- scheme header: how many reactions, and add another
        top = QHBoxLayout(); top.setSpacing(T.GAP_LABEL)
        top.addWidget(_overline("Reactions in this scheme"))
        self.scheme_count_lb = _label("1 reaction", T.FS_CAPTION, T.TEXT_MUTED)
        top.addWidget(self.scheme_count_lb)
        top.addStretch(1)
        b_examples = _button("Examples…", "primary")
        b_examples.setToolTip(
            "Browse 50+ worked reactions — esterification, epoxide opening, "
            "ENR/maleic-acid bridges, polyurethanes, silicones and more — "
            "each with its mapped SMILES explained. One click loads an "
            "example into the scheme, ready to validate or edit.")
        b_examples.clicked.connect(self._open_examples)
        top.addWidget(b_examples)
        b_add = _button("+ Add reaction")
        b_add.setToolTip("Add another independent reaction to this scheme")
        b_add.clicked.connect(lambda: (self._add_reaction(), self._refresh_maps_panel()))
        top.addWidget(b_add)
        col.addLayout(top)

        self.blocks_box = QVBoxLayout()
        self.blocks_box.setSpacing(T.GAP_SECTION)
        col.addLayout(self.blocks_box)

        # ---- empty-state / guidance
        self.empty_box = QWidget()
        self.empty_box.setObjectName("emptyState")
        self.empty_box.setStyleSheet(
            f"QWidget#emptyState {{ background: {T.BG_SUNKEN};"
            f" border: 1px dashed {T.BORDER_STRONG};"
            f" border-radius: {T.R_CARD}px; }}")
        ev = QVBoxLayout(self.empty_box)
        ev.setContentsMargins(T.PAD_CARD, T.PAD_CARD, T.PAD_CARD, T.PAD_CARD)
        ev.setSpacing(6)
        ev.addWidget(_label("Nothing validated yet", T.FS_SECTION, T.TEXT, bold=True))
        d = _label(
            "Fill in each reaction's molecules with matching atom maps, then "
            "run Validate all. PAAF reports the bonds each reaction forms and "
            "breaks before anything is written to disk.",
            T.FS_BODY, T.TEXT_SECONDARY)
        d.setWordWrap(True)
        ev.addWidget(d)
        row = QHBoxLayout(); row.setSpacing(T.GAP_LABEL)
        b_ex = _button("Load esterification example")
        b_ex.clicked.connect(self._load_example)
        b_open = _button("Open library JSON…")
        b_open.clicked.connect(self._open_library)
        row.addWidget(b_ex); row.addWidget(b_open); row.addStretch(1)
        ev.addLayout(row)
        col.addWidget(self.empty_box)

        # ---- reaction library
        lcard, lbody, lhead = _card("Reaction library")
        self.lib_count_lb = _label("0 reactions", T.FS_CAPTION, T.TEXT_MUTED)
        lhead.addWidget(self.lib_count_lb)
        b_save = _button("Save library…")
        b_save.clicked.connect(self._save_library)
        lhead.addWidget(b_save)

        self.lib_table = QTableWidget(0, 6)
        self.lib_table.setHorizontalHeaderLabels(
            ["name", "reactants", "products", "formed", "broken", "status"])
        self.lib_table.verticalHeader().setVisible(False)
        self.lib_table.verticalHeader().setDefaultSectionSize(T.H_TABLE_ROW)
        self.lib_table.horizontalHeader().setFixedHeight(T.H_TABLE_HEADER)
        self.lib_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.lib_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.lib_table.setAlternatingRowColors(False)
        self.lib_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.lib_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.lib_table.setMinimumHeight(96)
        lbody.addWidget(self.lib_table)
        col.addWidget(lcard)

        col.addStretch(1)
        return col

    def _right_rail(self) -> QWidget:
        rail = QWidget()
        rail.setFixedWidth(T.W_RIGHT_RAIL)
        v = QVBoxLayout(rail)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(T.GAP_SECTION)

        help_card, help_body, _ = _card("How atom maps work")
        intro = _label(
            "A map number tags one atom so PAAF can follow it through the "
            "reaction. Write it inside brackets after the element: [C:1]. "
            "The same number on both sides means the same atom.",
            T.FS_CAPTION, T.TEXT_SECONDARY)
        intro.setWordWrap(True)
        help_body.addWidget(intro)
        help_body.addWidget(_overline("worked example · esterification"))
        for lab, smi in (("acid", "CC(=O)[OH:1]"),
                         ("alcohol", "[C:2]CO"),
                         ("product", "CC(=O)O[C:2]")):
            r = QHBoxLayout(); r.setSpacing(T.GAP_LABEL)
            k = _label(lab, T.FS_CAPTION, T.TEXT_MUTED); k.setFixedWidth(54)
            r.addWidget(k)
            r.addWidget(_label(smi, T.FS_BODY, T.TEXT, mono=True))
            r.addStretch(1)
            help_body.addLayout(r)
        rules = _label(
            "1 · Map only the atoms whose bonds change.\n"
            "2 · Every number must be unique per side.\n"
            "3 · A number missing from the products is a leaving group.\n"
            "4 · Numbers are per reaction — Reaction 2 may reuse :1.",
            T.FS_CAPTION, T.TEXT_MUTED)
        rules.setWordWrap(True)
        help_body.addWidget(rules)
        v.addWidget(help_card)

        map_card, map_body, map_head = _card("Map numbers in use")
        self.map_scope_lb = _label("Reaction 1", T.FS_CAPTION, T.TEXT_MUTED)
        map_head.addWidget(self.map_scope_lb)
        self.map_state_lb = _label("None yet", T.FS_CAPTION, T.TEXT_MUTED)
        map_head.addWidget(self.map_state_lb)
        self.map_table = QTableWidget(0, 4)
        self.map_table.setHorizontalHeaderLabels(["map", "reactant", "product", "state"])
        self.map_table.verticalHeader().setVisible(False)
        self.map_table.verticalHeader().setDefaultSectionSize(T.H_TABLE_ROW)
        self.map_table.horizontalHeader().setFixedHeight(T.H_TABLE_HEADER)
        self.map_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.map_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.map_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.map_table.setShowGrid(False)
        map_body.addWidget(self.map_table)
        self.map_empty_lb = _label(
            "Tokens appear here as you type, paired left to right.",
            T.FS_CAPTION, T.TEXT_MUTED)
        self.map_empty_lb.setWordWrap(True)
        map_body.addWidget(self.map_empty_lb)
        self.next_free_lb = _label("Next free number: 1", T.FS_CAPTION, T.TEXT_MUTED)
        map_body.addWidget(self.next_free_lb)
        v.addWidget(map_card)

        log_card, log_body, _ = _card("Run log")
        self.run_log = QPlainTextEdit()
        self.run_log.setReadOnly(True)
        self.run_log.setMinimumHeight(120)
        self.run_log.setStyleSheet(
            f"QPlainTextEdit {{ background: {T.BG_SUNKEN};"
            f" border: 1px solid {T.BORDER}; border-radius: {T.R_CONTROL}px;"
            f" color: {T.TEXT_SECONDARY}; font-family: {T.FONT_MONO};"
            f" font-size: {T.FS_CAPTION}px; padding: 6px; }}")
        log_body.addWidget(self.run_log)
        v.addWidget(log_card)

        v.addStretch(1)
        return rail

    def _action_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("actionBar")
        bar.setFixedHeight(T.H_ACTION_BAR)
        bar.setStyleSheet(
            f"QWidget#actionBar {{ background: {T.BG_SURFACE};"
            f" border: 1px solid {T.BORDER}; border-radius: {T.R_CARD}px; }}")
        h = QHBoxLayout(bar)
        h.setContentsMargins(T.PAD_CARD, 0, T.PAD_CARD, 0)
        h.setSpacing(T.GAP_LABEL)
        self.blocker_lb = _label(
            "Fill in a reaction to continue.", T.FS_BODY, T.TEXT_MUTED)
        h.addWidget(self.blocker_lb)
        h.addStretch(1)

        self.b_add_lib = _button("Add valid to library")
        self.b_add_lib.setEnabled(False)
        self.b_add_lib.clicked.connect(self._add_to_library)
        h.addWidget(self.b_add_lib)

        self.b_export = _button("Export .data…")
        self.b_export.setEnabled(False)
        self.b_export.setToolTip(
            "Build 3D structures, optimise, apply the force field from the "
            "Force-field page, and write reactant.data / product.data for "
            "every validated reaction")
        self.b_export.clicked.connect(self._export_data)
        h.addWidget(self.b_export)

        self.b_validate = _button("Validate all", "primary")
        self.b_validate.clicked.connect(self._validate_all)
        h.addWidget(self.b_validate)

        self.b_back = _button("‹ Back")
        self.b_back.clicked.connect(self._go_back)
        self.b_back.setVisible(False)
        h.insertWidget(1, self.b_back)

        self.b_next = _button("Next ›", "primary")
        self.b_next.clicked.connect(self._go_next)
        h.addWidget(self.b_next)
        return bar

    # ---------------------------------------------------------- reactions
    def _open_examples(self) -> None:
        """Browse the built-in examples; load the chosen one into a block.

        An untouched block (no SMILES typed yet) is reused; otherwise the
        example lands in a fresh reaction so nothing of the user's is
        overwritten.
        """
        from .reaction_examples_dialog import ReactionExamplesDialog
        dlg = ReactionExamplesDialog(self)
        if dlg.exec_() != dlg.Accepted:
            return
        ex = dlg.selected_example()
        if ex is None:
            return
        blk = self._active if (self._active is not None
                               and not self._active.is_filled()
                               and not any(r.smiles.text().strip()
                                           for r in self._active._rows
                                           + self._active._prod_rows)) \
            else self._add_reaction()
        blk.load_example(ex)
        self._active = blk
        self._refresh_maps_panel()
        self._rlog(f"loaded example: {ex.name} "
                   f"({len(ex.reactants)} reactant(s), "
                   f"{len(ex.products)} product(s))")

    def _add_reaction(self) -> ReactionBlock:
        blk = ReactionBlock(len(self._blocks) + 1)
        blk.changed.connect(self._on_block_changed)
        blk.removeRequested.connect(self._remove_reaction)
        self._blocks.append(blk)
        self.blocks_box.addWidget(blk)
        self._active = blk
        self._renumber_blocks()
        return blk

    def _remove_reaction(self, blk: ReactionBlock) -> None:
        if len(self._blocks) <= 1:
            self._toast("A scheme needs at least one reaction.")
            return
        self._blocks.remove(blk)
        blk.setParent(None); blk.deleteLater()
        if self._active is blk:
            self._active = self._blocks[0]
        self._renumber_blocks()
        self._refresh_maps_panel()

    def _renumber_blocks(self) -> None:
        for i, b in enumerate(self._blocks, start=1):
            b.set_number(i)
        n = len(self._blocks)
        self.scheme_count_lb.setText(f"{n} reaction{'s' if n != 1 else ''}")

    def _on_block_changed(self, blk: ReactionBlock) -> None:
        self._active = blk
        self._refresh_maps_panel()

    # -------------------------------------------------------- live panels
    def _refresh_maps_panel(self) -> None:
        """Show the atom-map pairing for the reaction being edited."""
        blk = self._active or (self._blocks[0] if self._blocks else None)
        self.map_table.setRowCount(0)
        if blk is None:
            return
        self.map_scope_lb.setText(f"Reaction {blk.number}")
        r_maps, p_maps = blk.maps()
        all_maps = sorted(set(r_maps) | set(p_maps))
        paired = 0
        for m in all_maps:
            in_r, in_p = m in r_maps, m in p_maps
            if in_r and in_p:
                state, level, paired = "✓ paired", "ok", paired + 1
            elif in_r:
                state, level = "leaves", "warn"
            else:
                state, level = "✕ unmatched", "error"
            r = self.map_table.rowCount()
            self.map_table.insertRow(r)
            self._set_map_cell(r, 0, f":{m}", mono=True)
            self._set_map_cell(r, 1, f"reactant {r_maps[m]}" if in_r else "absent")
            self._set_map_cell(r, 2, f"product {p_maps[m]}" if in_p else "absent")
            self._set_map_cell(r, 3, state, level=level)
        self.map_state_lb.setText(f"{paired} paired" if all_maps else "None yet")
        rows = max(1, self.map_table.rowCount())
        self.map_table.setFixedHeight(
            T.H_TABLE_HEADER + min(rows, 6) * T.H_TABLE_ROW + 2)
        self.map_empty_lb.setVisible(not all_maps)
        nxt = 1
        while nxt in set(all_maps):
            nxt += 1
        self.next_free_lb.setText(f"Next free number: {nxt}")

        ready = sum(1 for b in self._blocks if b.is_filled())
        self.blocker_lb.setText(
            "" if ready else "Fill in a reaction to continue.")

    def _set_map_cell(self, r: int, c: int, text: str, mono: bool = False,
                      level: str = "") -> None:
        it = QTableWidgetItem(text)
        if level == "error":
            it.setForeground(QColor(T.DANGER_TEXT))
        elif level == "ok":
            it.setForeground(QColor(T.SUCCESS_TEXT))
        elif level == "warn":
            it.setForeground(QColor(T.WARN_TEXT))
        if mono:
            f = it.font(); f.setFamily("Consolas"); it.setFont(f)
        self.map_table.setItem(r, c, it)

    # ------------------------------------------------------------ actions
    def _stamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _rlog(self, msg: str) -> None:
        self.run_log.appendPlainText(f"{self._stamp()}  {msg}")

    def valid_blocks(self) -> List[ReactionBlock]:
        return [b for b in self._blocks
                if b.result is not None and b.result.ok]

    def _validate_all(self) -> None:
        self.run_log.clear()
        filled = [b for b in self._blocks if b.is_filled()]
        if not filled:
            self._toast("Enter at least one reactant and one product.")
            return
        n_ok = 0
        for blk in self._blocks:
            if not blk.is_filled():
                blk.result = None
                blk._set_status(None)
                continue
            self._rlog(f"validating reaction {blk.number} ({blk.reaction_name()}) …")
            res = blk.validate()
            if res.ok:
                n_ok += 1
                self._rlog(f"  ok — {res.n_bonds_formed} formed, "
                           f"{res.n_bonds_broken} broken")
            else:
                for iss in res.errors:
                    self._rlog(f"  error {iss.code}")
        self.empty_box.setVisible(n_ok == 0)
        self.b_add_lib.setEnabled(n_ok > 0)
        self.b_export.setEnabled(n_ok > 0)
        total = len(filled)
        self.blocker_lb.setText(
            f"{n_ok} of {total} reaction(s) valid."
            if n_ok < total else "")
        self.log.emit(f"[reaction] validated {n_ok}/{total} reaction(s)")

    def _add_to_library(self) -> None:
        added = 0
        for blk in self.valid_blocks():
            t = blk.result.template
            if t is None:
                continue
            self._library.append(t.to_json())
            name, reactants, products = blk.values()
            r = self.lib_table.rowCount()
            self.lib_table.insertRow(r)
            vals = [name,
                    ", ".join(n or f"r{i}" for i, (n, s) in
                              enumerate(rw.values() for rw in blk._rows) if s),
                    ", ".join(n or f"p{i}" for i, (n, s) in
                              enumerate(rw.values() for rw in blk._prod_rows) if s),
                    str(blk.result.n_bonds_formed),
                    str(blk.result.n_bonds_broken), "✓ ok"]
            for c, val in enumerate(vals):
                it = QTableWidgetItem(str(val))
                if c in (3, 4):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if c == 5:
                    it.setForeground(QColor(T.SUCCESS_TEXT))
                self.lib_table.setItem(r, c, it)
            added += 1
        n = len(self._library)
        self.lib_count_lb.setText(f"{n} reaction{'s' if n != 1 else ''}")
        self._rlog(f"added {added} reaction(s) to the library")
        self.log.emit(f"[reaction] library now holds {n} reaction(s)")

    def _save_library(self) -> None:
        if not self._library:
            self._toast("Validate and add at least one reaction first.")
            return
        p, _ = QFileDialog.getSaveFileName(
            self, "Save reaction library", "reaction_library.json", "JSON (*.json)")
        if not p:
            return
        payload = {"format": "paaf-reaction-library-v1",
                   "n_reactions": len(self._library),
                   "reactions": self._library}
        Path(p).write_text(json.dumps(payload, indent=2) + "\n")
        self._rlog(f"saved library → {Path(p).name}")
        self.log.emit(f"[reaction] wrote library ({len(self._library)}) -> {p}")

    def _open_library(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Open reaction library", "", "JSON (*.json)")
        if not p:
            return
        try:
            data = json.loads(Path(p).read_text())
            reactions = data.get("reactions", [])
        except Exception as exc:
            self._toast(f"Could not read that library:\n{exc}")
            return
        for entry in reactions:
            self._library.append(entry)
            r = self.lib_table.rowCount()
            self.lib_table.insertRow(r)
            vals = [entry.get("name", "?"), entry.get("reactant", ""),
                    entry.get("product", ""),
                    str(len(entry.get("created_bonds", []))),
                    str(len(entry.get("deleted_bonds", []))), "✓ ok"]
            for c, val in enumerate(vals):
                self.lib_table.setItem(r, c, QTableWidgetItem(str(val)))
        n = len(self._library)
        self.lib_count_lb.setText(f"{n} reaction{'s' if n != 1 else ''}")
        self._rlog(f"opened library — {len(reactions)} reaction(s)")

    def _load_example(self) -> None:
        blk = self._blocks[0] if self._blocks else self._add_reaction()
        blk.load(self.EXAMPLE["name"], self.EXAMPLE["reactants"],
                 self.EXAMPLE["products"])
        self._refresh_maps_panel()
        self._rlog("loaded esterification example")

    def _toast(self, msg: str) -> None:
        QMessageBox.information(self, "Reaction scheme", msg)

    # ------------------------------------------------- steps 2 & 3 wiring
    def _on_step_changed(self, idx: int) -> None:
        """Keep the later steps in step with what step 1 produced, and show
        only the action that belongs to the step the user is on."""
        if idx == 1:
            self._refresh_ff_summary()
        elif idx == 2:
            self._refresh_export_preview()

        on_scheme, on_export = idx == 0, idx == 2
        self.b_validate.setVisible(on_scheme)
        self.b_add_lib.setVisible(on_scheme)
        self.b_export.setVisible(on_export)
        self.b_next.setVisible(not on_export)
        self.b_back.setVisible(idx > 0)
        if idx == 1:
            self.blocker_lb.setText(
                "These settings apply only to the reaction set — the "
                "pipeline's Force-field page is unaffected.")
        elif on_export:
            n = len(self.valid_blocks())
            self.blocker_lb.setText(
                "" if n else "Validate a reaction in step 1 first.")

    def _go_next(self) -> None:
        self.steps.setCurrentIndex(min(self.steps.currentIndex() + 1, 2))

    def _go_back(self) -> None:
        self.steps.setCurrentIndex(max(self.steps.currentIndex() - 1, 0))

    def _on_rx_ff_changed(self, _idx: int) -> None:
        key = self.rx_ff_combo.currentData()
        if not key:
            return
        try:
            from ..ff_registry import get_ff
            ff = get_ff(key)
            self.rx_ff_notes.setText(
                f"{ff.notes}  ·  typer: {ff.atom_typer}"
                + ("  ·  united atom" if ff.united_atom else ""))
        except Exception:
            self.rx_ff_notes.setText("")
        self._refresh_ff_summary()

    def _copy_ff_from_pipeline(self) -> None:
        """Load the main Force-field page's selection into this branch."""
        cfg = self._pipeline_settings()
        if not cfg.get("resolved"):
            self._toast(
                "Could not read the pipeline's Force-field page. Open the "
                "Reaction scheme from inside the main PAAF window.")
            return
        key = cfg.get("ff_key")
        for i in range(self.rx_ff_combo.count()):
            if self.rx_ff_combo.itemData(i) == key:
                self.rx_ff_combo.setCurrentIndex(i)
                break
        if cfg.get("dl_lib"):
            self.rx_dl_lib.setText(cfg["dl_lib"])
        opt = cfg.get("opt_ff")
        if opt:
            idx = self.rx_opt_ff.findText(opt)
            if idx >= 0:
                self.rx_opt_ff.setCurrentIndex(idx)
        self._rlog(f"copied force field from the pipeline: {key}")

    def _browse_rx_dl_lib(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select the DL_FIELD lib directory", self.rx_dl_lib.text())
        if d:
            self.rx_dl_lib.setText(d)

    def _browse_rx_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Choose the reactions folder", self.rx_out_dir.text())
        if d:
            self.rx_out_dir.setText(d)

    def _refresh_ff_summary(self) -> None:
        if not hasattr(self, "rx_stat_ff"):
            return
        self._set_stat(self.rx_stat_reactions, str(len(self.valid_blocks())))
        self._set_stat(self.rx_stat_ff, self.rx_ff_combo.currentData() or "—")
        self._set_stat(self.rx_stat_opt, self.rx_opt_ff.currentText())

    @staticmethod
    def _set_stat(row_widget, value: str) -> None:
        labels = row_widget.findChildren(QLabel)
        if labels:
            labels[-1].setText(str(value))

    def default_out_dir(self) -> str:
        """``<output>/<project>/reactions`` from the pipeline, if readable."""
        cfg = self._pipeline_settings()
        if cfg.get("resolved"):
            base = Path(cfg.get("out_dir") or "output")
            return str(base / (cfg.get("project") or "polymer") / "reactions")
        return str(Path("output") / "reactions")

    def _refresh_export_preview(self) -> None:
        """List exactly which folders and files the export will write."""
        if not hasattr(self, "rx_preview"):
            return
        if not self.rx_out_dir.text().strip():
            self.rx_out_dir.setText(self.default_out_dir())
        base = Path(self.rx_out_dir.text().strip() or "output")
        blocks = self.valid_blocks()

        self.rx_preview.setRowCount(0)
        for blk in blocks:
            name, reactants, products = blk.values()
            r = self.rx_preview.rowCount()
            self.rx_preview.insertRow(r)
            files = ("reactant.data, product.data"
                     if self.rx_run_typing.isChecked()
                     else "reactant.xyz, product.xyz")
            for c, val in enumerate([name, str(base / name),
                                     f"{len(reactants)} → {len(products)}",
                                     files]):
                self.rx_preview.setItem(r, c, QTableWidgetItem(str(val)))
        n = len(blocks)
        self.rx_preview_count.setText(
            f"{n} validated reaction{'s' if n != 1 else ''}")
        self.rx_out_hint.setText(
            f"Each reaction gets its own folder under this path. "
            f"Point reaction_engine_local.py at {base} to learn them all.")

    # ------------------------------------------------------------- export
    def set_settings_provider(self, fn) -> None:
        """Install a callable returning the force field / output settings.

        The main window supplies this explicitly rather than the tab reaching
        up through ``self.window()``. Walking the parent chain fails silently
        if the widget is reparented, and the failure mode — exporting with the
        DEFAULT force field instead of the chosen one — goes unnoticed.
        """
        self._settings_provider = fn
        # Seed the DL_FIELD dir from the pipeline immediately, so the field
        # is visibly filled rather than inherited silently at export time.
        # An empty field here cost a full export: "dl_field executable not
        # found" on an installation that worked on every other page.
        try:
            if not self.rx_dl_lib.text().strip():
                dl = self._pipeline_settings().get("dl_lib", "")
                if dl:
                    self.rx_dl_lib.setText(dl)
        except Exception:
            pass

    def _pipeline_settings(self) -> dict:
        """What the MAIN Force-field / project pages currently hold.

        Only used to seed defaults and by "Copy from pipeline" — the reaction
        branch keeps its own settings so a reference set can be typed
        differently from the system being built.
        """
        out = {"out_dir": "", "ff_key": "", "dl_lib": "",
               "opt_ff": "", "project": "", "resolved": False}
        fn = getattr(self, "_settings_provider", None)
        if fn is None:
            return out
        try:
            got = fn() or {}
            out.update({k: v for k, v in got.items() if v not in (None, "")})
            out["resolved"] = True
        except Exception as exc:
            self._rlog(f"could not read the pipeline settings: {exc}")
        return out

    def _export_settings(self) -> dict:
        """This branch's OWN force field / output settings (step 2 and 3)."""
        dl_lib = self.rx_dl_lib.text().strip()
        if not dl_lib:
            # Fall back to the pipeline's DL_FIELD dir rather than exporting
            # with none at all. An empty field here meant "dl_field
            # executable not found" for a user whose installation was set up
            # and working on every other page.
            dl_lib = self._pipeline_settings().get("dl_lib", "")
            if dl_lib:
                self.rx_dl_lib.setText(dl_lib)   # show what will be used
                self._rlog(f"DL_FIELD lib dir taken from the pipeline: "
                           f"{dl_lib}")
        return {
            "ff_key": self.rx_ff_combo.currentData() or "opls2005_dl",
            "dl_lib": dl_lib,
            "opt_ff": self.rx_opt_ff.currentText(),
            "optimize": self.rx_optimize.isChecked(),
            "run_typing": self.rx_run_typing.isChecked(),
            "out_dir": self.rx_out_dir.text().strip() or self.default_out_dir(),
            "resolved": True,
        }

    def _export_data(self) -> None:
        blocks = self.valid_blocks()
        if not blocks:
            self._toast("Validate at least one reaction before exporting.")
            return
        cfg = self._export_settings()
        base = Path(cfg["out_dir"])
        names = ", ".join(b.reaction_name() for b in blocks)
        what = ("LAMMPS .data files" if cfg["run_typing"]
                else "3D structures (no force-field typing)")
        ok = QMessageBox.question(
            self, "Export reaction data",
            f"Export {len(blocks)} reaction(s): {names}\n\n"
            f"Optimise with {cfg['opt_ff']}, apply {cfg['ff_key']} and write "
            f"{what} under:\n{base}\n\n"
            "Each reaction gets its own folder. This runs dl_field and can "
            "take a while.",
            QMessageBox.Ok | QMessageBox.Cancel)
        if ok != QMessageBox.Ok:
            return

        jobs = [{
            "name": b.reaction_name(),
            "reactants": list(b.reactant_smiles),
            "products": list(b.product_smiles),
            "template": b.result.template,
        } for b in blocks]

        self.b_export.setEnabled(False)
        self._rlog(f"exporting {len(jobs)} reaction(s) -> {base}")
        self._thread = QThread(self)
        self._worker = _ExportWorker(
            jobs, str(base), cfg["ff_key"], cfg["dl_lib"], cfg["opt_ff"],
            run_typing=cfg["run_typing"], optimize=cfg["optimize"])
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_export_progress)
        self._worker.finished.connect(self._on_export_done)
        self._worker.failed.connect(self._on_export_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(lambda: self.b_export.setEnabled(True))
        self._thread.start()

    @pyqtSlot(str)
    def _on_export_progress(self, msg: str) -> None:
        self._rlog(msg)
        self.log.emit(f"[reaction] {msg}")

    @pyqtSlot(object)
    def _on_export_done(self, exports) -> None:
        for e in exports:
            self._rlog(e.summary())
            self.log.emit(f"[reaction] {e.summary()}")
        untyped = [e for e in exports if not e.typed]
        if untyped:
            QMessageBox.warning(
                self, "Exported without force-field typing",
                f"{len(untyped)} of {len(exports)} reaction(s) produced 3D "
                "structures but no LAMMPS .data files.\n\n"
                + (untyped[0].messages[0] if untyped[0].messages else ""))
        else:
            QMessageBox.information(
                self, "Export complete",
                f"{len(exports)} reaction(s) written with reactant.data and "
                f"product.data under:\n{exports[0].folder.parent}")

    @pyqtSlot(str)
    def _on_export_failed(self, msg: str) -> None:
        self._rlog("export failed")
        self.log.emit(msg)
        QMessageBox.critical(self, "Export failed", msg[:2000])

    # ---------------------------------------------------------------- API
    def library(self) -> List[dict]:
        """Validated reaction templates collected so far."""
        return list(self._library)

    def reactions(self) -> List[ReactionBlock]:
        return list(self._blocks)
