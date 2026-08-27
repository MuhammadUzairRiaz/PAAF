"""Pick a polymer from the PAAF library.

The Builder page has a full library table; the Reaction-scheme tab needs the
same catalogue but only to fill in one SMILES field, so this is a compact modal
version: filter, pick, done.

It returns the *polymerization* SMILES (``[*]CC[*]``) rather than the
closed-shell one, because the reaction scheme may then expand it to an n-mer —
see :mod:`paaf.polymer_smiles`.
"""
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import tokens as T


class LibraryPicker(QDialog):
    """Modal polymer chooser. ``selected()`` gives the picked record."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Choose a polymer from the library")
        self.resize(900, 560)
        # Windows/WSLg renders the dialog transparent without an
        # explicit background; macOS painted a native one.
        self.setStyleSheet(
            f"QDialog {{ background: {T.BG_PAGE}; }}")
        self._records = []
        self._filtered = []

        v = QVBoxLayout(self)
        v.setContentsMargins(T.PAD_PAGE, T.PAD_PAGE, T.PAD_PAGE, T.PAD_PAGE)
        v.setSpacing(T.GAP_FORM_ROW)

        intro = QLabel(
            "Pick the monomer for this molecule. PAAF inserts its "
            "<b>polymerization SMILES</b> — the form with <tt>[*]</tt> "
            "connection points — so the chain length can be set afterwards.")
        intro.setWordWrap(True)
        v.addWidget(intro)

        bar = QHBoxLayout()
        bar.setSpacing(T.GAP_LABEL)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter by name / PID / SMILES…")
        self.filter.textChanged.connect(self._apply_filter)
        bar.addWidget(self.filter, 1)
        self.count_lb = QLabel("")
        self.count_lb.setStyleSheet(
            f"color: {T.TEXT_MUTED}; font-size: {T.FS_CAPTION}px;")
        bar.addWidget(self.count_lb)
        v.addLayout(bar)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["PID", "name", "SMILES", "Tg (K)"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(T.H_TABLE_ROW)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(False)
        self.table.doubleClicked.connect(lambda *_: self.accept())
        v.addWidget(self.table, 1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setObjectName("primary")
        bb.button(QDialogButtonBox.Ok).setText("Use this monomer")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

        self._load()

    # ------------------------------------------------------------ data
    def _load(self) -> None:
        try:
            from ..builder import list_library
            self._records = list(list_library("all"))
        except Exception:
            self._records = []
        self._apply_filter()

    def _apply_filter(self, *_args) -> None:
        q = self.filter.text().strip().lower()
        if q:
            self._filtered = [
                r for r in self._records
                if q in (r.name or "").lower()
                or q in (r.pid or "").lower()
                or q in (r.description or "").lower()
                or q in (r.smiles or "").lower()
            ]
        else:
            self._filtered = list(self._records)
        self._render()

    def _render(self) -> None:
        self.table.setRowCount(0)
        for rec in self._filtered:
            r = self.table.rowCount()
            self.table.insertRow(r)
            tg = "" if rec.tg_k is None else f"{rec.tg_k:.0f}"
            # Database rows carry the PID in `name` and the human
            # name in `description` — show the human one.
            shown = rec.name or ""
            if rec.pid and shown == rec.pid and (rec.description or "").strip():
                shown = rec.description.strip()
            for c, val in enumerate([rec.pid or "", shown,
                                     rec.smiles or "", tg]):
                it = QTableWidgetItem(str(val))
                if c == 2:
                    f = it.font(); f.setFamily("Consolas"); it.setFont(f)
                if c == 3:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, it)
        n = len(self._filtered)
        self.count_lb.setText(f"{n} of {len(self._records)} polymers")
        if n:
            self.table.selectRow(0)

    # ------------------------------------------------------------- API
    def selected(self):
        """The chosen record, or ``None``."""
        rows = {i.row() for i in self.table.selectedIndexes()}
        if not rows:
            return None
        r = min(rows)
        if 0 <= r < len(self._filtered):
            return self._filtered[r]
        return None
