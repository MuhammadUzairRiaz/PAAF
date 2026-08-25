"""Reactions tab — upload reactant/product structures for many reactions,
learn a library, save it, and optionally apply it to a packed system.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)


# =============================================================== worker
class ReactionsWorker(QObject):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, specs, out_json, apply_to=None, apply_out=None,
                 max_events=100000, cutoff=5.0):
        super().__init__()
        self.specs = specs
        self.out_json = out_json
        self.apply_to = apply_to
        self.apply_out = apply_out
        self.max_events = max_events
        self.cutoff = cutoff

    @pyqtSlot()
    def run(self):
        try:
            from ..reaction import ReactionLibrary
            self.progress.emit(f"Learning {len(self.specs)} reaction(s)...")
            lib = ReactionLibrary.learn_many(self.specs)
            for t in lib.templates:
                self.progress.emit("  " + t.summary())
            lib.save(self.out_json)
            self.progress.emit(f"Reaction library saved -> {self.out_json}")

            result = {"library": str(self.out_json), "n_reactions": len(lib.templates)}
            if self.apply_to:
                from ..xlink_engine import apply_library
                stats = apply_library(
                    self.apply_to, lib, self.apply_out,
                    max_events=self.max_events, cutoff=self.cutoff,
                )
                result["xlink_stats"] = stats.as_dict()
                result["crosslinked_system"] = str(self.apply_out)
                self.progress.emit(f"Applied library -> {self.apply_out}")
                self.progress.emit(json.dumps(stats.as_dict(), indent=2))
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


# ============================================================ tab
class ReactionsTab(QWidget):
    log = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._thread = None
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(14)

        v.addWidget(QLabel(
            "<b>Reactions / crosslinking</b><br>"
            "Add one row per reaction. For each reaction, upload the "
            "<b>reactant</b> and <b>product</b> structures (xyz / pdb / mol2 / "
            "mol / sdf) — the tool learns the transformation by graph mapping "
            "and stores it in a reusable reaction library."
        ))

        # ---- Reactions table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Reaction name", "Reactant file", "Product file"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setMinimumWidth(0)
        v.addWidget(self.table, 1)

        row = QHBoxLayout()
        b_add = QPushButton("Add reaction row")
        b_add.clicked.connect(self._add_row)
        b_pick_r = QPushButton("Pick reactant...")
        b_pick_r.clicked.connect(lambda: self._pick_file(1))
        b_pick_p = QPushButton("Pick product...")
        b_pick_p.clicked.connect(lambda: self._pick_file(2))
        b_rem = QPushButton("Remove selected")
        b_rem.clicked.connect(self._remove_row)
        b_folder = QPushButton("Import folder...")
        b_folder.clicked.connect(self._import_folder)
        for b in (b_add, b_pick_r, b_pick_p, b_rem, b_folder):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)

        # ---- Optional: apply to a packed system
        gb = QGroupBox("Apply learned library to a packed system (optional)")
        f = QFormLayout(gb)
        self.apply_enabled = QCheckBox("Also apply learned templates to a packed system")
        self.apply_to = QLineEdit()
        self.apply_to.setPlaceholderText("Path to packed system (xyz / pdb / mol2 / lammps data)")
        b_pack = QPushButton("Browse..."); b_pack.clicked.connect(self._pick_packed)
        pack_row = QHBoxLayout(); pack_row.addWidget(self.apply_to); pack_row.addWidget(b_pack)
        pack_wrap = QWidget(); pack_wrap.setLayout(pack_row)
        self.apply_out = QLineEdit()
        self.apply_out.setPlaceholderText("Output path for crosslinked system")
        b_out = QPushButton("Save as..."); b_out.clicked.connect(self._pick_out)
        out_row = QHBoxLayout(); out_row.addWidget(self.apply_out); out_row.addWidget(b_out)
        out_wrap = QWidget(); out_wrap.setLayout(out_row)
        self.max_events = QSpinBox(); self.max_events.setRange(1, 10_000_000); self.max_events.setValue(100_000)
        self.cutoff = QDoubleSpinBox(); self.cutoff.setRange(0.5, 100.0); self.cutoff.setValue(5.0); self.cutoff.setSuffix(" Å")
        f.addRow(self.apply_enabled)
        f.addRow("Packed system", pack_wrap)
        f.addRow("Output", out_wrap)
        f.addRow("Max events", self.max_events)
        f.addRow("Reactive-pair cutoff", self.cutoff)
        v.addWidget(gb)

        # ---- Actions
        actions = QHBoxLayout()
        self.out_library = QLineEdit(str(Path.cwd() / "reaction_library.json"))
        b_libout = QPushButton("Library file..."); b_libout.clicked.connect(self._pick_lib_out)
        actions.addWidget(QLabel("Library JSON:")); actions.addWidget(self.out_library, 1); actions.addWidget(b_libout)
        v.addLayout(actions)

        row2 = QHBoxLayout()
        self.btn_learn = QPushButton("Learn library"); self.btn_learn.setObjectName("primary")
        self.btn_learn.clicked.connect(self._start)
        row2.addWidget(self.btn_learn)
        row2.addStretch(1)
        v.addLayout(row2)

    # --------------------------------------------------- table utils
    def _add_row(self, name: str = "", reactant: str = "", product: str = ""):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(name or f"reaction_{r + 1:03d}"))
        self.table.setItem(r, 1, QTableWidgetItem(reactant))
        self.table.setItem(r, 2, QTableWidgetItem(product))

    def _remove_row(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _pick_file(self, col: int):
        rows = {i.row() for i in self.table.selectedIndexes()}
        if not rows:
            self._add_row()
            rows = {self.table.rowCount() - 1}
        path, _ = QFileDialog.getOpenFileName(
            self, "Select structure",
            filter="Structures (*.xyz *.pdb *.mol2 *.mol *.sdf *.cml *.data);;All (*)",
        )
        if not path:
            return
        for r in rows:
            self.table.setItem(r, col, QTableWidgetItem(path))
            if col == 1 and not (self.table.item(r, 0) and self.table.item(r, 0).text()):
                self.table.setItem(r, 0, QTableWidgetItem(Path(path).parent.name or Path(path).stem))

    def _pick_packed(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Packed system",
            filter="Structures (*.xyz *.pdb *.mol2 *.data);;All (*)",
        )
        if p:
            self.apply_to.setText(p)

    def _pick_out(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "Save crosslinked system",
            filter="XYZ (*.xyz);;PDB (*.pdb);;MOL2 (*.mol2)",
        )
        if p:
            self.apply_out.setText(p)

    def _pick_lib_out(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save reaction library", filter="JSON (*.json)")
        if p:
            self.out_library.setText(p)

    def _import_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choose folder with reaction_XXX/ subfolders")
        if not d:
            return
        root = Path(d)
        exts = ("xyz", "pdb", "mol2", "mol", "sdf", "data")
        n = 0
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            r_path = p_path = None
            for e in exts:
                if not r_path and (sub / f"reactant.{e}").exists():
                    r_path = sub / f"reactant.{e}"
                if not p_path and (sub / f"product.{e}").exists():
                    p_path = sub / f"product.{e}"
            if r_path and p_path:
                self._add_row(sub.name, str(r_path), str(p_path))
                n += 1
        self.log.emit(f"Imported {n} reactions from {root}")

    # --------------------------------------------------- run
    def _start(self):
        specs = []
        for r in range(self.table.rowCount()):
            def _get(c: int) -> str:
                it = self.table.item(r, c)
                return it.text().strip() if it else ""
            name = _get(0)
            reactant = _get(1)
            product = _get(2)
            if reactant and product:
                specs.append({"name": name or f"reaction_{r + 1}", "reactant": reactant, "product": product})
        if not specs:
            QMessageBox.warning(self, "No reactions", "Add at least one reactant + product pair.")
            return

        apply_to = self.apply_to.text().strip() if self.apply_enabled.isChecked() else None
        apply_out = self.apply_out.text().strip() if self.apply_enabled.isChecked() else None
        if apply_to and not apply_out:
            QMessageBox.warning(self, "Missing output", "Choose an output path for the crosslinked system.")
            return

        self.btn_learn.setEnabled(False)
        self._thread = QThread(self)
        self._worker = ReactionsWorker(
            specs, self.out_library.text().strip(),
            apply_to=apply_to, apply_out=apply_out,
            max_events=self.max_events.value(), cutoff=self.cutoff.value(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.log.emit)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(lambda: self.btn_learn.setEnabled(True))
        self._thread.start()

    @pyqtSlot(dict)
    def _on_done(self, res):
        self.log.emit("=" * 60)
        self.log.emit("REACTIONS DONE")
        for k, v in res.items():
            self.log.emit(f"  {k}: {v}")

    @pyqtSlot(str)
    def _on_fail(self, msg):
        self.log.emit(msg)
        QMessageBox.critical(self, "Reactions failed", msg[:2000])
