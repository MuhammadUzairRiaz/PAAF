"""Define a component as a copolymer: monomers, fractions, arrangement.

A homopolymer needs one SMILES. A copolymer needs to say *which monomer sits
where*, because the ordering changes the material — a block copolymer and a
random one at the same composition behave differently. This dialog collects
that, and nothing else.

Fractions are **mole** fractions, not weight, because the sequence is built by
counting units. The dialog shows the weight percent each choice comes to, so
the two are never confused.
"""
from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import tokens as T
from .page import button, caption, label


class CopolymerDialog(QDialog):
    """Pick two or more monomers, their mole fractions, and the arrangement."""

    def __init__(self, monomers=None, arrangement: str = "random",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Copolymer")
        self.resize(720, 420)

        v = QVBoxLayout(self)
        v.setContentsMargins(T.PAD_PAGE, T.PAD_PAGE, T.PAD_PAGE, T.PAD_PAGE)
        v.setSpacing(T.GAP_FORM_ROW)

        intro = caption(
            "Fractions are MOLE fractions — they count repeat units, which is "
            "what the sequence is built from. The weight percent each one "
            "comes to is shown beside it. Every chain gets its own draw, so a "
            "short chain's composition scatters around what you ask for.")
        intro.setWordWrap(True)
        v.addWidget(intro)

        bar = QHBoxLayout()
        bar.setSpacing(T.GAP_LABEL)
        bar.addWidget(label("Arrangement", T.FS_BODY, T.TEXT_SECONDARY))
        self.arrangement_box = QComboBox()
        self.arrangement_box.setFixedHeight(T.H_CONTROL)
        for key, human in (("random", "Random — independent draw per unit"),
                           ("alternating", "Alternating — ABAB (two monomers)"),
                           ("block", "Block — all A, then all B")):
            self.arrangement_box.addItem(human, key)
        idx = max(0, self.arrangement_box.findData(arrangement or "random"))
        self.arrangement_box.setCurrentIndex(idx)
        bar.addWidget(self.arrangement_box, 1)

        b_add = button("+ Add monomer")
        b_add.clicked.connect(lambda: self._add_row("", 0.0))
        bar.addWidget(b_add)
        b_lib = button("Library…")
        b_lib.clicked.connect(self._from_library)
        bar.addWidget(b_lib)
        b_presets = button("Presets…")
        b_presets.setToolTip(
            "Load a saved copolymer. The same library backs the simple "
            "Builder, so a definition saved in either tool is available in "
            "both.")
        b_presets.clicked.connect(self._load_preset)
        bar.addWidget(b_presets)
        b_save = button("Save as preset")
        b_save.setToolTip("Store this definition in the shared copolymer "
                          "library so the Builder can use it too.")
        b_save.clicked.connect(self._save_preset)
        bar.addWidget(b_save)
        v.addLayout(bar)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["name", "SMILES", "mole %", "→ weight %"])
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(T.H_TABLE_ROW)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(lambda _i: self._refresh())
        v.addWidget(self.table, 1)

        self.summary = caption("")
        self.summary.setWordWrap(True)
        v.addWidget(self.summary)

        row = QHBoxLayout()
        b_del = button("Remove selected", "ghost")
        b_del.clicked.connect(self._remove_selected)
        row.addWidget(b_del)
        b_homo = button("Make homopolymer", "ghost")
        b_homo.setToolTip("Clear the copolymer and go back to one SMILES")
        b_homo.clicked.connect(self._make_homopolymer)
        row.addWidget(b_homo)
        row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        v.addLayout(row)

        self._cleared = False
        if monomers:
            for m in monomers:
                self._add_row(m.smiles, m.fraction, m.name)
        else:
            self._enr_preset()
        self._refresh()

    # ------------------------------------------------------------- rows
    def _add_row(self, smiles: str, fraction: float, name: str = "") -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(name))
        self.table.setItem(r, 1, QTableWidgetItem(smiles))
        self.table.setItem(r, 2, QTableWidgetItem(f"{100 * fraction:.1f}"))
        w = QTableWidgetItem("")
        w.setFlags(w.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(r, 3, w)

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()},
                      reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self._refresh()

    def _make_homopolymer(self) -> None:
        self._cleared = True
        self.accept()

    def _enr_preset(self) -> None:
        """The default when the dialog is opened on a fresh row."""
        self.table.setRowCount(0)
        self._add_row("[*]C/C=C(C)\\C[*]", 0.75, "isoprene")
        self._add_row("[*]CC1(C)OC1C[*]", 0.25, "epoxide")
        self._refresh()

    def _load_preset(self) -> None:
        """Pick a saved copolymer from the shared library."""
        from ..copolymer_library import list_copolymers

        recipes = list_copolymers()
        if not recipes:
            self.summary.setText(
                "The copolymer library is empty. Build a definition here and "
                "press Save as preset to add one.")
            return
        dlg = _PresetPicker(recipes, self)
        if dlg.exec_() != dlg.Accepted:
            return
        rec = dlg.selected()
        if rec is None:
            return
        self.table.setRowCount(0)
        self._add_row(rec.smiles_a, rec.fraction_a, f"{rec.pid}-A")
        self._add_row(rec.smiles_b, rec.fraction_b, f"{rec.pid}-B")
        idx = self.arrangement_box.findData(rec.sequence_mode or "random")
        if idx >= 0:
            self.arrangement_box.setCurrentIndex(idx)
        self._refresh()

    def _save_preset(self) -> None:
        """Store this definition where the simple Builder can also read it.

        The shared library's schema holds exactly two monomers, so a
        three-or-more-component definition cannot be saved there. Saying so is
        better than writing a file that silently loses a monomer.
        """
        from PyQt5.QtWidgets import QInputDialog
        from ..copolymer_library import CopolymerRecipe, save_copolymer

        mons = self.monomers()
        if not mons:
            self.summary.setText("Nothing to save — add at least two monomers.")
            return
        if len(mons) > 2:
            self.summary.setText(
                f"The shared library stores two monomers per recipe, and this "
                f"has {len(mons)}. It will still build here, but it cannot be "
                f"saved as a preset.")
            self.summary.setStyleSheet(
                f"color: {T.WARN_TEXT}; font-size: {T.FS_CAPTION}px;"
                f" background: transparent; border: none;")
            return

        pid, ok = QInputDialog.getText(self, "Save copolymer",
                                       "Name for this preset:")
        pid = (pid or "").strip()
        if not ok or not pid:
            return
        try:
            path = save_copolymer(CopolymerRecipe(
                pid=pid, smiles_a=mons[0].smiles, smiles_b=mons[1].smiles,
                fraction_b=mons[1].fraction,
                sequence_mode=self.arrangement(), random_seed=None))
        except Exception as exc:
            self.summary.setText(f"Could not save: {exc}")
            return
        self.summary.setText(f"Saved '{pid}' to {path.name}. It is now "
                             f"available in the Builder as well.")

    def _from_library(self) -> None:
        from .library_picker import LibraryPicker
        dlg = LibraryPicker(self)
        if dlg.exec_() != dlg.Accepted:
            return
        rec = dlg.selected()
        if rec is None:
            return
        self._add_row(rec.smiles or rec.monomer_smiles or "", 0.0,
                      (rec.name or "")[:20])
        self._refresh()

    # ---------------------------------------------------------- results
    def _rows(self):
        out = []
        for r in range(self.table.rowCount()):
            def cell(c):
                it = self.table.item(r, c)
                return it.text().strip() if it else ""
            try:
                frac = float(cell(2) or 0.0)
            except ValueError:
                frac = 0.0
            out.append((cell(0), cell(1), frac))
        return out

    def _refresh(self) -> None:
        from ..cell.sequence import Monomer
        from ..cell.grow import max_backbone_atoms_in_a_ring

        rows = [(n, s, f) for n, s, f in self._rows() if s]
        if not rows:
            self.summary.setText("Add at least two monomers.")
            return
        mons = []
        problems = []
        for name, smi, frac in rows:
            m = Monomer(smi, max(frac, 0.0), name)
            try:
                m.resolve()
            except Exception as exc:
                problems.append(f"{name or smi}: {exc}")
                continue
            if max_backbone_atoms_in_a_ring(smi) > 2:
                problems.append(
                    f"{name or smi}: its backbone runs through a ring that "
                    f"spans several bonds, so this monomer cannot be rebuilt "
                    f"as all-atom.")
            mons.append(m)

        total_mol = sum(m.fraction for m in mons) or 1.0
        total_mass = sum(m.mass_amu * m.fraction for m in mons) or 1.0
        self.table.blockSignals(True)
        for r, m in enumerate(mons):
            wt = 100.0 * m.mass_amu * m.fraction / total_mass
            it = self.table.item(r, 3)
            if it is not None:
                it.setText(f"{wt:.1f}")
        self.table.blockSignals(False)

        mean = sum(m.mass_amu * m.fraction for m in mons) / total_mol
        beads = sum(m.backbone_atoms * m.fraction for m in mons) / total_mol
        msg = (f"{len(mons)} monomers · mean repeat unit {mean:.1f} g/mol · "
               f"{beads:.1f} skeletal atoms per unit on average.")
        if abs(total_mol - 1.0) > 1e-6 and abs(total_mol - 100.0) > 1e-6:
            msg += "  Fractions are normalised, so any consistent ratio works."
        if problems:
            msg += "  " + "  ".join(problems)
        self.summary.setText(msg)
        self.summary.setStyleSheet(
            f"color: {T.WARN_TEXT if problems else T.TEXT_MUTED};"
            f" font-size: {T.FS_CAPTION}px; background: transparent;"
            f" border: none;")

    def monomers(self):
        """The monomers, or ``None`` if the user chose homopolymer.

        Fractions come back **normalised to sum to 1**, whatever was typed.
        The table shows mole *percent*, so without normalising, editing an
        existing copolymer would inflate it by 100x on every round trip:
        0.75 displays as 75.0 and would come back as 75.0.
        """
        if self._cleared:
            return None
        from ..cell.sequence import Monomer
        rows = [(name, smi, max(frac, 0.0))
                for name, smi, frac in self._rows() if smi]
        if len(rows) < 2:
            return None
        total = sum(f for _n, _s, f in rows)
        if total <= 0:                       # nothing entered: split evenly
            total, rows = len(rows), [(n, s, 1.0) for n, s, _f in rows]
        return [Monomer(smi, frac / total, name) for name, smi, frac in rows]

    def arrangement(self) -> str:
        """``random`` | ``alternating`` | ``block``."""
        return self.arrangement_box.currentData() or "random"


class _PresetPicker(QDialog):
    """Choose one saved copolymer recipe."""

    def __init__(self, recipes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Copolymer presets")
        self.resize(680, 320)
        self._recipes = list(recipes)

        v = QVBoxLayout(self)
        v.setContentsMargins(T.PAD_PAGE, T.PAD_PAGE, T.PAD_PAGE, T.PAD_PAGE)
        v.setSpacing(T.GAP_FORM_ROW)
        v.addWidget(caption(
            "These are shared with the simple Builder — the same file backs "
            "both tools."))

        self.table = QTableWidget(len(self._recipes), 4)
        self.table.setHorizontalHeaderLabels(
            ["name", "monomer A", "monomer B", "B fraction / mode"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(T.H_TABLE_ROW)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for r, rec in enumerate(self._recipes):
            for c, text in enumerate((
                    rec.pid, rec.smiles_a, rec.smiles_b,
                    f"{rec.fraction_b:.0%} {rec.sequence_mode}")):
                self.table.setItem(r, c, QTableWidgetItem(text))
        self.table.doubleClicked.connect(lambda _i: self.accept())
        if self._recipes:
            self.table.selectRow(0)
        v.addWidget(self.table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    def selected(self):
        rows = {i.row() for i in self.table.selectedIndexes()}
        if not rows:
            return None
        r = min(rows)
        return self._recipes[r] if 0 <= r < len(self._recipes) else None
