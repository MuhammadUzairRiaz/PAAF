"""Builder tab (Step 1) — homopolymer or copolymer chooser.

Top of the page: a radio-button chooser for
  • Homopolymer   — one monomer, repeated N times.
  • Copolymer     — two monomers, mixed at a chosen fraction (random /
                    alternating / block) over N total units.

Below the chooser, one or two "monomer input" panels appear. Each input
panel offers two ways to add a monomer:
  1. SMILES + polymer library + fragment palette
  2. Periodic table

A third tab used to upload a monomer geometry built elsewhere (Avogadro and
the like). It was removed on request. Nothing in the pipeline depends on it:
a YAML config can still name a structure file through ``MonomerSpec.file``,
which is the path that was always the tested one.

After a monomer is built it lands in the shared table at the bottom of the
page which every downstream step (Optimizer / Chain / Force field / Box /
Export) reads from. For copolymer mode, the two monomers land in rows 0
and 1 (in that order) and the copolymer settings (fraction B, sequence
mode, total N, seed) are auto-forwarded to the Chain page.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPushButton, QRadioButton, QSizePolicy,
    QSpinBox, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from .page import wrap_tooltip

from ..config import MonomerSpec
from . import tokens as _tok
from .periodic_table import PeriodicTable


# =============================================================== worker
class BuilderWorker(QObject):
    finished = pyqtSignal(str, int)   # emit (path, slot) so we know WHICH monomer slot to put it in
    failed = pyqtSignal(str)

    def __init__(self, smiles: str, out_path: str, optimize: bool, opt_ff: str,
                 opt_steps: int, name: str, slot: int = 0):
        super().__init__()
        self.smiles = smiles; self.out_path = out_path
        self.optimize = optimize; self.opt_ff = opt_ff
        self.opt_steps = opt_steps; self.name = name
        self.slot = slot

    @pyqtSlot()
    def run(self):
        try:
            from ..builder import build_from_smiles
            out = build_from_smiles(
                self.smiles, self.out_path,
                optimize=self.optimize, opt_ff=self.opt_ff,
                opt_steps=self.opt_steps, name=self.name,
            )
            self.finished.emit(str(out), self.slot)
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


# =========================================================== single input
class MonomerInputPanel(QWidget):
    """One 'add a monomer' box with 3 sub-tabs (upload / library / periodic)."""

    file_ready = pyqtSignal(str, int)     # path, slot
    log = pyqtSignal(str)

    def __init__(self, slot: int, title: str,
                 all_polymers: list, all_fragments: list,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.slot = slot
        self._title = title
        self._all_polymers = all_polymers
        self._all_fragments = all_fragments
        self._worker = None; self._thread = None
        self._build()

    def _build(self):
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        # For copolymer mode (Monomer A / Monomer B) keep an inline title so
        # users can tell the two panels apart. For the solo homopolymer panel
        # ("Monomer") the tabs alone are sufficient — drop the redundant label.
        if self._title.strip().lower() != "monomer":
            title = QLabel(f"<b style='color:{_tok.TEXT};'>{self._title}</b>")
            title.setStyleSheet("padding: 2px 6px 0 6px;")
            v.addWidget(title)

        tabs = QTabWidget()
        # Preferred vertical — do NOT grow beyond the active tab's sizeHint.
        # This is what stops the empty-tab body from ballooning down.
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        tabs.setMinimumWidth(0)
        tabs.setDocumentMode(True)   # thinner tab frame
        # The "Custom / Avogadro (upload)" tab was removed at the user's
        # request. Loading a monomer from a file is still supported through a
        # YAML config's ``MonomerSpec.file`` — only the GUI entry point is
        # gone, so nothing in the pipeline changed.
        tabs.addTab(self._library_tab(),  "SMILES + library + fragments")
        tabs.addTab(self._periodic_tab(), "SMILES builder")
        v.addWidget(tabs)
        # CRUCIAL: absorb any leftover vertical space HERE so title+tabs
        # stay pinned to the top of the panel and never drift downward.
        v.addStretch(1)
        # The panel itself should not force its parent to grow unnecessarily.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    # ---------- 1: library / SMILES
    def _library_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        v.setContentsMargins(10, 6, 10, 6); v.setSpacing(6)

        bar = QHBoxLayout()
        self.filter = QLineEdit(); self.filter.setPlaceholderText("Filter by name / PID / SMILES…")
        self.filter.textChanged.connect(self._apply_filter)
        self.src_combo = QComboBox()
        self.src_combo.addItems(["All", "Curated", "Database", "My polymers"])
        self.src_combo.currentIndexChanged.connect(self._apply_filter)
        bar.addWidget(self.filter, 1); bar.addWidget(QLabel("Show:")); bar.addWidget(self.src_combo)
        v.addLayout(bar)

        self.lib_table = QTableWidget(0, 6)
        self.lib_table.setHorizontalHeaderLabels(
            ["PID", "Name", "Tg (K)", "ρ (kg/m³)", "SMILES", "src"])
        self.lib_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.lib_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.lib_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.lib_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.lib_table.setSortingEnabled(True)
        # Library is the primary content of this tab — give it real height.
        # 320 px shows ~10 rows without scrolling; still scrolls if the
        # library is bigger.
        self.lib_table.setMinimumHeight(320)
        self.lib_table.setMaximumHeight(420)
        self.lib_table.setMinimumWidth(0)
        self.lib_table.itemSelectionChanged.connect(self._on_pick_row)
        self.lib_table.doubleClicked.connect(lambda *_: self._start_build())
        v.addWidget(self.lib_table)

        # ------- Your own polymers: add / edit / remove -------
        # Kept beside the library rather than on a page of its own, because
        # the question "is my polymer in here?" and the answer "add it" belong
        # in the same place.
        own = QHBoxLayout(); own.setSpacing(6)
        own.addWidget(QLabel("My polymers:"))
        b_add = QPushButton("Add…")
        b_add.setToolTip(wrap_tooltip("Add your own polymer to the library. It is stored "
                         "in your home folder, so it survives updating PAAF."))
        b_add.clicked.connect(self._add_custom_polymer)
        own.addWidget(b_add)
        self.b_edit_custom = QPushButton("Edit…")
        self.b_edit_custom.setToolTip(wrap_tooltip("Edit the selected polymer. Only your "
                                      "own entries can be changed."))
        self.b_edit_custom.clicked.connect(self._edit_custom_polymer)
        own.addWidget(self.b_edit_custom)
        self.b_del_custom = QPushButton("Remove")
        self.b_del_custom.setToolTip(wrap_tooltip("Remove the selected polymer from your "
                                     "library."))
        self.b_del_custom.clicked.connect(self._delete_custom_polymer)
        own.addWidget(self.b_del_custom)
        own.addStretch(1)
        v.addLayout(own)
        self._refresh_custom_buttons()

        # ------- Single compact row: Name  |  SMILES  |  Append fragment -------
        # Was 2 rows (Name/SMILES + Fragment). Folding to 1 row saves ~30 px
        # and keeps the primary inputs on the same line as the fragment picker.
        one_row = QHBoxLayout(); one_row.setSpacing(6)
        one_row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit("MOL"); self.name_edit.setFixedWidth(90)
        one_row.addWidget(self.name_edit)
        one_row.addWidget(QLabel("SMILES:"))
        self.smiles_edit = QLineEdit()
        self.smiles_edit.setPlaceholderText("e.g. CCOCCOCCO")
        # If the user edits SMILES by hand, forget the picked library
        # polymerization SMILES so we don't attach a stale sidecar.
        self.smiles_edit.textEdited.connect(self._on_smiles_edited)
        one_row.addWidget(self.smiles_edit, 1)
        # Fragment picker inline — no separate row
        one_row.addSpacing(8)
        self.frag_cat = QComboBox(); self.frag_cat.setFixedWidth(90)
        self.frag_cat.currentIndexChanged.connect(self._reload_fragments)
        self.frag_list = QComboBox(); self.frag_list.setMinimumWidth(160)
        b_frag = QPushButton("Append"); b_frag.setFixedWidth(70)
        b_frag.clicked.connect(self._append_fragment)
        one_row.addWidget(QLabel("Frag:"))
        one_row.addWidget(self.frag_cat)
        one_row.addWidget(self.frag_list, 1)
        one_row.addWidget(b_frag)
        v.addLayout(one_row)
        cats = sorted({f.category for f in self._all_fragments})
        self.frag_cat.addItems(cats); self._reload_fragments()

        # NOTE: the per-monomer OpenBabel clean-up (MMFF94, 2000 steps) still
        # runs silently inside SMILES → 3D so the monomer has sensible geometry
        # before it's polymerized. It is NOT exposed here — the visible
        # 'Optimize' controls that minimize the *full chain* live on the
        # dedicated Optimize page, which is Step 3 of the pipeline (after
        # Chain, before Force field).
        self.opt_enabled = QCheckBox(); self.opt_enabled.setChecked(True); self.opt_enabled.hide()
        self.opt_ff = QComboBox(); self.opt_ff.addItems(["MMFF94", "MMFF94s", "UFF", "Ghemical"])
        self.opt_ff.hide()
        self.opt_steps = QSpinBox(); self.opt_steps.setRange(10, 500_000); self.opt_steps.setValue(2000)
        self.opt_steps.hide()

        # ------- Single row: Output path  |  Save as…  |  Build button -------
        # Was 2 rows (out + actions). Now everything sits on the same line
        # under the Name/SMILES/Fragment row.
        _scratch = Path.cwd() / "scratch" / "monomers"
        _scratch.mkdir(parents=True, exist_ok=True)
        self.out_path = QLineEdit(str(_scratch / f"monomer_slot{self.slot}.pdb"))
        b_out = QPushButton("Save as…"); b_out.setFixedWidth(90)
        b_out.clicked.connect(self._pick_out)
        self.btn_build = QPushButton("Build  →  add to Monomers")
        self.btn_build.setObjectName("primary")
        self.btn_build.clicked.connect(self._start_build)

        out_row = QHBoxLayout(); out_row.setSpacing(6)
        out_row.addWidget(QLabel("Output:"))
        out_row.addWidget(self.out_path, 1)
        out_row.addWidget(b_out)
        out_row.addWidget(self.btn_build)
        v.addLayout(out_row)

        self._render_library(self._all_polymers)
        return w

    def _render_library(self, rows: list):
        self.lib_table.setSortingEnabled(False); self.lib_table.setRowCount(0)
        for r in rows:
            row = self.lib_table.rowCount(); self.lib_table.insertRow(row)
            def _put(col, val):
                it = QTableWidgetItem("" if val is None else str(val))
                if isinstance(val, (int, float)): it.setData(Qt.EditRole, float(val))
                self.lib_table.setItem(row, col, it)
            _put(0, r.pid or r.name); _put(1, r.description)
            _put(2, r.tg_k); _put(3, r.density_kg_m3); _put(4, r.smiles)
            _put(5, "mine" if "custom" in r.tags
                 else ("curated" if "curated" in r.tags else "DB"))
        self.lib_table.setSortingEnabled(True)
        self.lib_table.resizeColumnsToContents()

    def _apply_filter(self):
        q = self.filter.text().strip().lower()
        source_idx = self.src_combo.currentIndex()
        rows = self._all_polymers
        if source_idx == 1:
            rows = [r for r in rows if "curated" in r.tags]
        elif source_idx == 2:
            rows = [r for r in rows if "database" in r.tags]
        elif source_idx == 3:
            rows = [r for r in rows if "custom" in r.tags]
        if q:
            rows = [r for r in rows
                    if q in (r.pid or "").lower() or q in r.description.lower()
                    or q in r.smiles.lower() or q in r.name.lower()]
        self._render_library(rows)

    def _on_smiles_edited(self, text: str) -> None:
        """Keep the polymerisation SMILES in step with what the user typed.

        Editing used to just FORGET the picked library SMILES, on the theory
        that a stale sidecar is worse than none. Both halves of that were
        wrong for a hand-typed SMILES: with no ``.polysmi`` written, the
        pipeline falls back to guessing the link atoms from the geometry —
        the exact guess that built poly(1-butene) as a straight chain. A
        user typing ``[*]CC(C)(C)[*]`` has SAID where the links are; throwing
        that away and guessing is strictly worse.

        So: typed text with two ``[*]`` marks IS the polymerisation SMILES
        and is carried through to the sidecar like a library pick. Text
        without them (a plain molecule, a half-finished edit) clears the
        stale one exactly as before.
        """
        text = (text or "").strip()
        self._picked_poly_smiles = text if text.count("[*]") == 2 else None

    # ------------------------------------------------ the user's own entries
    def _selected_pid(self) -> Optional[str]:
        rows = {i.row() for i in self.lib_table.selectedIndexes()}
        if not rows:
            return None
        item = self.lib_table.item(next(iter(rows)), 0)
        return item.text() if item else None

    def _selected_is_custom(self) -> bool:
        rows = {i.row() for i in self.lib_table.selectedIndexes()}
        if not rows:
            return False
        item = self.lib_table.item(next(iter(rows)), 5)
        return bool(item) and item.text() == "mine"

    def _refresh_custom_buttons(self) -> None:
        """Edit and Remove apply to your own entries only.

        The shipped library is read-only, and a disabled button says that more
        clearly than an error message after the fact.
        """
        own = self._selected_is_custom()
        for name in ("b_edit_custom", "b_del_custom"):
            b = getattr(self, name, None)
            if b is not None:
                b.setEnabled(own)

    def _reload_library_rows(self, select_pid: Optional[str] = None) -> None:
        from ..builder import list_library, reload_library

        reload_library()
        self._all_polymers = list_library("all")
        self._apply_filter()
        if select_pid:
            for row in range(self.lib_table.rowCount()):
                item = self.lib_table.item(row, 0)
                if item and item.text() == select_pid:
                    self.lib_table.selectRow(row)
                    break

    def _custom_dialog(self, title: str, pid="", smiles="", name=""):
        """Ask for id, name and SMILES. Returns the tuple or ``None``."""
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFormLayout

        d = QDialog(self); d.setWindowTitle(title)
        form = QFormLayout(d)
        e_pid = QLineEdit(pid); e_pid.setPlaceholderText("MY_PIB")
        e_name = QLineEdit(name); e_name.setPlaceholderText("Polyisobutylene")
        e_smiles = QLineEdit(smiles)
        e_smiles.setPlaceholderText("[*]CC(C)(C)[*]")
        form.addRow("ID:", e_pid)
        form.addRow("Name:", e_name)
        form.addRow("SMILES:", e_smiles)
        note = QLabel(
            "The SMILES needs two [*] marks showing where one unit joins the\n"
            "next — that is what tells PAAF which atoms to link, instead of\n"
            "guessing. Stored in your home folder, so updating PAAF keeps it.")
        note.setWordWrap(True)
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(d.accept); buttons.rejected.connect(d.reject)
        form.addRow(buttons)
        if d.exec_() != QDialog.Accepted:
            return None
        return e_pid.text(), e_smiles.text(), e_name.text()

    def _add_custom_polymer(self):
        from ..custom_library import CustomLibraryError, add_custom

        got = self._custom_dialog("Add a polymer to your library")
        while got is not None:
            pid, smiles, name = got
            try:
                entry = add_custom(pid, smiles, name)
            except CustomLibraryError as exc:
                QMessageBox.warning(self, "Cannot add this polymer", str(exc))
                # Reopened with what they typed, so a typo is a correction
                # rather than starting again.
                got = self._custom_dialog("Add a polymer to your library",
                                          pid, smiles, name)
                continue
            self._reload_library_rows(select_pid=entry.pid)
            self.log.emit(f"Added '{entry.name}' ({entry.pid}) to your "
                          f"library — it will be there next time too.")
            return

    def _edit_custom_polymer(self):
        from ..builder import get_recipe
        from ..custom_library import CustomLibraryError, update_custom

        pid = self._selected_pid()
        if not pid or not self._selected_is_custom():
            QMessageBox.information(
                self, "Nothing to edit",
                "Select one of your own polymers (marked 'mine' in the last "
                "column). The shipped library cannot be edited.")
            return
        try:
            rec = get_recipe(pid)
        except KeyError:
            return
        got = self._custom_dialog("Edit your polymer", pid, rec.smiles,
                                  rec.description)
        while got is not None:
            new_pid, smiles, name = got
            try:
                entry = update_custom(pid, new_pid, smiles, name)
            except CustomLibraryError as exc:
                QMessageBox.warning(self, "Cannot save this polymer", str(exc))
                got = self._custom_dialog("Edit your polymer", new_pid,
                                          smiles, name)
                continue
            self._reload_library_rows(select_pid=entry.pid)
            self.log.emit(f"Updated '{entry.name}' ({entry.pid}).")
            return

    def _delete_custom_polymer(self):
        from ..custom_library import delete_custom

        pid = self._selected_pid()
        if not pid or not self._selected_is_custom():
            QMessageBox.information(
                self, "Nothing to remove",
                "Select one of your own polymers (marked 'mine' in the last "
                "column). The shipped library cannot be changed.")
            return
        if QMessageBox.question(
                self, "Remove this polymer?",
                f"Remove '{pid}' from your library?\n\nAnything you have "
                f"already built from it is unaffected.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No) != QMessageBox.Yes:
            return
        if delete_custom(pid):
            self._reload_library_rows()
            self.log.emit(f"Removed '{pid}' from your library.")

    def _on_pick_row(self):
        self._refresh_custom_buttons()
        rows = {i.row() for i in self.lib_table.selectedIndexes()}
        if not rows: return
        r = next(iter(rows))
        pid = self.lib_table.item(r, 0).text()
        from ..builder import get_recipe
        try: rec = get_recipe(pid)
        except KeyError: return
        self.name_edit.setText(rec.pid or rec.name)
        # IMPORTANT: build from the POLYMERIZABLE closed-shell form, i.e. the
        # polymerization SMILES with [*] simply stripped (implicit H fills the
        # valence). This gives each connection atom a REMOVABLE hydrogen that
        # mBuild's Polymer can turn into the inter-monomer bond.
        #
        # Do NOT use rec.monomer_smiles for polyesters: some curated entries
        # (e.g. PBS) hand-cap the acid end as -C(=O)OH, which leaves the tail
        # carbon with no removable H → mBuild can't form the ester link and
        # produces a dangling/branched structure.
        build_smiles = rec.monomer_smiles
        if rec.smiles and "[*]" in rec.smiles:
            from ..builder import _closed_shell
            build_smiles = _closed_shell(rec.smiles)
        self.smiles_edit.setText(build_smiles)
        # Remember the POLYMERIZATION SMILES (with [*] wildcards) so
        # _start_build can drop it in a sidecar file next to the PDB.
        # This tells the pipeline WHICH atoms are the polymer head/tail,
        # bypassing the fragile geometry-based guess for polyesters like PBS.
        self._picked_poly_smiles = rec.smiles
        p = Path(self.out_path.text() or "monomer.pdb")
        stem = (rec.pid or rec.name).lower() + f"_slot{self.slot}"
        self.out_path.setText(str(p.with_stem(stem)))

    def _reload_fragments(self):
        cat = self.frag_cat.currentText()
        self.frag_list.clear()
        for f in self._all_fragments:
            if f.category == cat:
                self.frag_list.addItem(f"{f.name}  [{f.smiles}]", userData=f.smiles)

    def _append_fragment(self):
        smi = self.frag_list.currentData()
        if not smi: return
        cur = self.smiles_edit.text()
        self.smiles_edit.setText((cur + smi) if cur else smi)

    def _pick_out(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "Save structure",
            filter="PDB (*.pdb);;XYZ (*.xyz);;MOL2 (*.mol2);;MOL (*.mol);;SDF (*.sdf)")
        if p: self.out_path.setText(p)

    def _start_build(self):
        smiles = self.smiles_edit.text().strip()
        out = self.out_path.text().strip()
        name = self.name_edit.text().strip() or "MOL"
        if not smiles or not out:
            QMessageBox.warning(self, "Missing input", "Provide SMILES + output path.")
            return
        # A hand-typed polymerisation SMILES: the [*] marks say where the
        # chain joins, but the 3D builder needs the closed-shell molecule —
        # a wildcard is not an atom it can place. So the marks are stripped
        # for BUILDING and kept for the sidecar, which is exactly how a
        # library pick is handled.
        if smiles.count("[*]") == 2:
            self._picked_poly_smiles = smiles
            from ..builder import _closed_shell
            smiles = _closed_shell(smiles)
        self._thread = QThread(self)
        self._worker = BuilderWorker(smiles, out, self.opt_enabled.isChecked(),
                                     self.opt_ff.currentText(),
                                     self.opt_steps.value(), name, slot=self.slot)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    @pyqtSlot(str, int)
    def _on_done(self, path: str, slot: int):
        # If the user picked this monomer from the library (SMILES with [*]),
        # drop a sidecar so the pipeline can identify head/tail atoms by the
        # wildcard neighbours (no more guessing on polyester terminals).
        try:
            poly = getattr(self, "_picked_poly_smiles", None)
            if poly and "[*]" in poly:
                Path(str(path) + ".polysmi").write_text(poly.strip() + "\n")
                self.log.emit(f"Wrote polymerization SMILES sidecar: {path}.polysmi")
        except Exception as e:
            self.log.emit(f"[warn] failed to write polysmi sidecar: {e}")
        self._picked_poly_smiles = None
        self.log.emit(f"Built structure -> {path}  (slot {slot})")
        self.file_ready.emit(path, slot)

    @pyqtSlot(str)
    def _on_fail(self, msg: str):
        self.log.emit(msg)
        QMessageBox.critical(self, "Build failed", msg[:2000])

    # ---------- 2: periodic table
    def _periodic_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        v.setContentsMargins(8, 4, 8, 4); v.setSpacing(3)
        # No intro line: the sub-tab title says "SMILES builder".
        # SMILES field with an inline label.
        smi_row = QHBoxLayout(); smi_row.setSpacing(6)
        smi_row.addWidget(QLabel("<b>SMILES:</b>"))
        self.pt_smiles = QLineEdit(); self.pt_smiles.setFont(QFont("SF Mono", 12))
        smi_row.addWidget(self.pt_smiles, 1)
        # Utility buttons live on the same row as the field.
        _clear = QPushButton("Clear"); _clear.setFixedWidth(60)
        _clear.clicked.connect(lambda: self.pt_smiles.setText(""))
        _back = QPushButton("⌫"); _back.setFixedWidth(36)
        _back.setToolTip(wrap_tooltip("Backspace (delete last character)"))
        _back.clicked.connect(lambda: self.pt_smiles.setText(self.pt_smiles.text()[:-1]))
        smi_row.addWidget(_back); smi_row.addWidget(_clear)
        v.addLayout(smi_row)

        # ----- SMILES notation strip (flat rows, no groupbox) -----
        n_outer = QVBoxLayout()
        n_outer.setContentsMargins(0, 0, 0, 0); n_outer.setSpacing(2)

        def _mk_token_button(tok: str, tooltip: str) -> QPushButton:
            b = QPushButton(tok)
            b.setFixedSize(30, 24)
            b.setStyleSheet(
                "QPushButton { font-family: 'SF Mono', 'Menlo', monospace; "
                "font-size: 12px; font-weight: 600; padding: 0; }")
            b.setToolTip(wrap_tooltip(tooltip))
            b.clicked.connect(lambda _c=False, t=tok:
                              self.pt_smiles.setText(self.pt_smiles.text() + t))
            return b

        # Flat 3-row layout (was 9 rows in a groupbox). Groups are shown by
        # small inline captions so users can still tell tokens apart.
        _rows = [
            # (label, tokens list) — packed into three visual rows
            [("Bonds",  [("-", "single bond"), ("=", "double"),
                          ("#", "triple"),      (":", "aromatic"),
                          (".", "disconnected")]),
             ("Branch", [("(", "open branch"),  (")", "close branch")]),
             ("Rings",  [(str(i), f"ring {i}") for i in range(1, 10)])],
            [("Stereo", [("/", "E/Z up"),  ("\\", "E/Z down"),
                          ("@", "CCW chiral"), ("@@", "CW chiral")]),
             ("Arom",   [("c", "arom C"), ("n", "arom N"),
                          ("o", "arom O"), ("s", "arom S"), ("p", "arom P")])],
            [("Bracket", [("[", "open [ ] atom"), ("]", "close [ ] atom"),
                          ("[H]", "explicit H")]),
             ("Charge",  [("+", "positive"), ("-", "negative")]),
             ("Any",     [("*", "wildcard — polymer head/tail attachment")])],
        ]
        for groups_in_row in _rows:
            row = QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(4)
            for grp_label, tokens in groups_in_row:
                cap = QLabel(f"<span style='color:{_tok.TEXT_MUTED};font-size:{_tok.FS_CAPTION}px;'>"
                             f"{grp_label}</span>")
                row.addWidget(cap)
                for tok, tip in tokens:
                    row.addWidget(_mk_token_button(tok, tip))
                row.addSpacing(6)
            row.addStretch(1)
            n_outer.addLayout(row)
        v.addLayout(n_outer)

        # ----- Periodic table (below the notation strip) ------------------
        # Click any element to append its symbol to the SMILES field.
        pt = PeriodicTable(); pt.elementClicked.connect(
            lambda s: self.pt_smiles.setText(self.pt_smiles.text() + s))
        from PyQt5.QtWidgets import QScrollArea as _SA, QSizePolicy as _SP
        pt_scroll = _SA(); pt_scroll.setWidget(pt); pt_scroll.setWidgetResizable(False)
        pt_scroll.setFrameShape(_SA.NoFrame)
        pt_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        pt_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        pt_scroll.setMinimumWidth(0)
        # CRUCIAL: keep the scroll area from expanding into empty gray space.
        # Preferred+Maximum means "take your sizeHint, no more" so any leftover
        # vertical space in the tab gets absorbed by the stretch below, not
        # by an empty scroll area.
        pt_scroll.setSizePolicy(_SP.Preferred, _SP.Maximum)
        pt_scroll.setMaximumHeight(pt.sizeHint().height() + 20)
        v.addWidget(pt_scroll)

        row = QHBoxLayout()
        self.pt_name = QLineEdit("MOL"); self.pt_name.setFixedWidth(200)
        _pt_scratch = Path.cwd() / "scratch" / "monomers"
        _pt_scratch.mkdir(parents=True, exist_ok=True)
        self.pt_out = QLineEdit(str(_pt_scratch / f"pt_slot{self.slot}.pdb"))
        b_save = QPushButton("Save as..."); b_save.clicked.connect(
            lambda: self._pick_out_target(self.pt_out))
        b_build = QPushButton("Build structure  →  add to monomers")
        b_build.setObjectName("primary")
        b_build.clicked.connect(self._start_pt_build)
        row.addWidget(QLabel("Name:")); row.addWidget(self.pt_name)
        row.addWidget(QLabel("Output:")); row.addWidget(self.pt_out, 1); row.addWidget(b_save)
        row.addWidget(b_build)
        v.addLayout(row)
        # Absorb any leftover vertical space HERE (below the Build row),
        # not inside the periodic-table scroll area. This is what makes
        # everything snap to the top with no phantom padding.
        v.addStretch(1)
        return w

    def _pick_out_target(self, edit):
        p, _ = QFileDialog.getSaveFileName(self, "Save", filter="PDB (*.pdb);;XYZ (*.xyz)")
        if p: edit.setText(p)

    def _start_pt_build(self):
        smiles = self.pt_smiles.text().strip()
        out = self.pt_out.text().strip()
        name = self.pt_name.text().strip() or "MOL"
        if not smiles or not out:
            QMessageBox.warning(self, "Missing input", "Compose a SMILES + pick output path.")
            return
        self._thread = QThread(self)
        self._worker = BuilderWorker(smiles, out, True, "UFF", 2000, name, slot=self.slot)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()


# ============================================================ tab
class BuilderTab(QWidget):
    log = pyqtSignal(str)
    built_file = pyqtSignal(str)
    copolymer_settings_changed = pyqtSignal(dict)   # emitted so Chain page can auto-fill

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        # The Builder page has two top-level modes, presented as tabs:
        #   Monomer         — build the repeat unit (upload / SMILES / periodic)
        #   Reaction scheme — define a reaction as SMILES + atom-map numbers
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.top_tabs = QTabWidget()
        self.top_tabs.setDocumentMode(True)
        self.top_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        monomer_page = QWidget()
        v = QVBoxLayout(monomer_page)
        v.setContentsMargins(14, 6, 14, 6); v.setSpacing(4)

        # No intro line: the sidebar already says "Builder — Step 1", and
        # the mode radios below tell users what to pick.

        # Mode chooser
        row = QHBoxLayout()
        self.rb_homo = QRadioButton("Homopolymer (one monomer)")
        self.rb_cop  = QRadioButton("Copolymer (two monomers + %)")
        self.rb_homo.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.rb_homo)
        self._mode_group.addButton(self.rb_cop)
        self.rb_homo.toggled.connect(self._on_mode_changed)
        row.addWidget(self.rb_homo); row.addWidget(self.rb_cop); row.addStretch(1)
        v.addLayout(row)

        # Preload library and fragments once (shared across panels)
        from ..builder import list_library
        from ..cell.fragments import list_fragments
        polymers = list_library("all")
        fragments = list_fragments()

        # Two stacked pages: homo (1 panel) / copolymer (2 panels + settings)
        self.stack = QStackedWidget()

        # ---------- HOMOPOLYMER page
        homo_page = QWidget(); hv = QVBoxLayout(homo_page)
        hv.setContentsMargins(0, 0, 0, 0); hv.setSpacing(0)
        self.panel_solo = MonomerInputPanel(0, "Monomer", polymers, fragments)
        self.panel_solo.file_ready.connect(self._on_file_ready)
        self.panel_solo.log.connect(self.log.emit)
        hv.addWidget(self.panel_solo)
        # Absorb any extra vertical space so the panel stays at the top
        # of the stack page instead of getting pushed down.
        hv.addStretch(1)
        self.stack.addWidget(homo_page)

        # ---------- COPOLYMER page
        cop_page = QWidget(); cv = QVBoxLayout(cop_page)
        cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(10)

        self.panel_a = MonomerInputPanel(0, "Monomer A", polymers, fragments)
        self.panel_a.file_ready.connect(self._on_file_ready)
        self.panel_a.log.connect(self.log.emit)
        cv.addWidget(self.panel_a)

        self.panel_b = MonomerInputPanel(1, "Monomer B", polymers, fragments)
        self.panel_b.file_ready.connect(self._on_file_ready)
        self.panel_b.log.connect(self.log.emit)
        cv.addWidget(self.panel_b)

        gb_co = QGroupBox("Copolymer composition")
        fc = QFormLayout(gb_co)
        fc.setContentsMargins(6, 6, 6, 6); fc.setSpacing(4)

        # Preset fractions — editable comboboxes for BOTH A and B.
        # Presets cover common polymer-blend fractions; users can type any
        # custom value into the "other" slot.
        _fraction_presets = ["0.05", "0.10", "0.20", "0.25", "0.33",
                             "0.50", "0.67", "0.75", "0.80", "0.90", "0.95",
                             "other..."]

        def _make_fraction_combo(default: str) -> QComboBox:
            c = QComboBox(); c.setEditable(True)
            for p in _fraction_presets:
                c.addItem(p)
            c.setCurrentText(default)
            c.setToolTip(wrap_tooltip("Pick a preset or type a custom fraction between 0 and 1."))
            return c

        self.co_fraction_a = _make_fraction_combo("0.50")
        self.co_fraction_b = _make_fraction_combo("0.50")
        # When A or B changes, the other auto-updates so they sum to 1.0.
        # A guard flag stops the two signal handlers ping-ponging forever.
        self._suppress_fraction_sync = False
        self.co_fraction_a.currentTextChanged.connect(
            lambda _v: self._sync_fractions(edited="A"))
        self.co_fraction_b.currentTextChanged.connect(
            lambda _v: self._sync_fractions(edited="B"))

        self.co_total = QSpinBox(); self.co_total.setRange(2, 100000); self.co_total.setValue(20)
        self.co_mode = QComboBox(); self.co_mode.addItems(["random", "alternating", "block"])
        self.co_seed = QLineEdit(); self.co_seed.setPlaceholderText("int (optional)")

        for w in (self.co_total,):
            w.valueChanged.connect(self._push_copolymer_settings)
        self.co_mode.currentTextChanged.connect(self._push_copolymer_settings)
        self.co_seed.textChanged.connect(self._push_copolymer_settings)

        # Row: two fraction combos side by side so users see both at once.
        fr = QHBoxLayout()
        fr.addWidget(QLabel("Fraction A")); fr.addWidget(self.co_fraction_a, 1)
        fr.addSpacing(20)
        fr.addWidget(QLabel("Fraction B")); fr.addWidget(self.co_fraction_b, 1)
        fr_wrap = QWidget(); fr_wrap.setLayout(fr)
        self.co_fraction_note = QLabel(
            "A + B is normalised to 1.0 automatically. "
            "Type 'other…' if the preset you want isn't listed.")
        self.co_fraction_note.setWordWrap(True)
        self.co_fraction_note.setProperty("role", "hint")

        fc.addRow("Fractions", fr_wrap)
        fc.addRow("", self.co_fraction_note)
        fc.addRow("Total monomers", self.co_total)
        fc.addRow("Sequence mode", self.co_mode)
        fc.addRow("Random seed", self.co_seed)
        cv.addWidget(gb_co)
        cv.addStretch(1)
        self.stack.addWidget(cop_page)

        from .page import fit_stack_to_current
        fit_stack_to_current(self.stack)

        # No stretch factor: the stack should take only what its active page
        # actually needs, not gobble the whole viewport.
        v.addWidget(self.stack)

        # -------- Shared monomers table
        gb_tbl = QGroupBox("Monomers (shared with Chain / Optimizer / FF / Box / Export)")
        tv = QVBoxLayout(gb_tbl)
        tv.setContentsMargins(6, 8, 6, 6); tv.setSpacing(4)
        self.monomer_table = QTableWidget(0, 6)
        self.monomer_table.setHorizontalHeaderLabels(
            ["File", "Name", "Head", "Tail", "Head_H", "Tail_H"])
        self.monomer_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.monomer_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.monomer_table.setMinimumWidth(0)
        # Very tight: header + ~2 rows. Table scrolls internally if more.
        self.monomer_table.setMinimumHeight(72)
        self.monomer_table.setMaximumHeight(120)
        tv.addWidget(self.monomer_table)
        row = QHBoxLayout()
        b_rem = QPushButton("Remove selected"); b_rem.clicked.connect(self._remove_row)
        b_clr = QPushButton("Clear all"); b_clr.clicked.connect(lambda: self.monomer_table.setRowCount(0))
        row.addStretch(1); row.addWidget(b_rem); row.addWidget(b_clr)
        tv.addLayout(row)
        v.addWidget(gb_tbl)
        # Extra vertical space collects HERE — below the Monomers table —
        # instead of appearing as phantom padding between sections.
        v.addStretch(1)

        self.top_tabs.addTab(monomer_page, "Monomer")

        # ---- Reaction scheme (SMILES + atom-map numbers)
        from .reaction_scheme_tab import ReactionSchemeTab
        self.reaction_scheme = ReactionSchemeTab()
        self.reaction_scheme.log.connect(self.log.emit)
        self.top_tabs.addTab(self.reaction_scheme, "Reaction scheme")
        from .page import fit_tabs_to_current
        fit_tabs_to_current(self.top_tabs)

        outer.addWidget(self.top_tabs)

        self._on_mode_changed()

    # =========================================================== behavior
    def _on_mode_changed(self, *_):
        self.stack.setCurrentIndex(0 if self.rb_homo.isChecked() else 1)
        self._push_copolymer_settings()

    def _parse_frac(self, text: str, fallback: float = 0.5) -> float:
        """Parse a fraction from a combobox text; 'other...' or garbage → fallback."""
        s = (text or "").strip().lower().rstrip(".")
        if not s or s.startswith("other"):
            return fallback
        try:
            v = float(s)
        except ValueError:
            return fallback
        return max(0.0, min(1.0, v))

    def _sync_fractions(self, edited: str):
        """Keep A + B = 1.0. When user edits one, auto-update the other."""
        if self._suppress_fraction_sync:
            return
        self._suppress_fraction_sync = True
        try:
            if edited == "A":
                a = self._parse_frac(self.co_fraction_a.currentText())
                self.co_fraction_b.setCurrentText(f"{(1.0 - a):.2f}")
            else:
                b = self._parse_frac(self.co_fraction_b.currentText())
                self.co_fraction_a.setCurrentText(f"{(1.0 - b):.2f}")
        finally:
            self._suppress_fraction_sync = False
        self._push_copolymer_settings()

    def _push_copolymer_settings(self):
        """Broadcast copolymer settings so Chain page can auto-fill itself."""
        if not self.rb_cop.isChecked():
            self.copolymer_settings_changed.emit({"enabled": False})
            return
        seed_txt = self.co_seed.text().strip()
        fa = self._parse_frac(self.co_fraction_a.currentText(), 0.5)
        fb = self._parse_frac(self.co_fraction_b.currentText(), 1.0 - fa)
        # Normalise defensively in case both were edited to sum > 1
        total = fa + fb
        if total > 0:
            fa, fb = fa / total, fb / total
        self.copolymer_settings_changed.emit({
            "enabled": True,
            "fraction_a": fa,
            "fraction_b": fb,
            "total": int(self.co_total.value()),
            "mode": self.co_mode.currentText(),
            "seed": int(seed_txt) if seed_txt.isdigit() else None,
        })

    def _on_file_ready(self, path: str, slot: int):
        """Called when either MonomerInputPanel produces a monomer file.

        `slot` = 0 for solo/A, 1 for B. Ensures the row lands at the right
        table position so the copolymer helper's A/B ordering stays consistent.
        """
        # If a row for this slot already exists, replace it; otherwise append.
        target_row = slot
        if target_row < self.monomer_table.rowCount():
            self.monomer_table.setItem(target_row, 0, QTableWidgetItem(path))
            self.monomer_table.setItem(target_row, 1, QTableWidgetItem(Path(path).stem))
        else:
            while self.monomer_table.rowCount() <= target_row:
                r = self.monomer_table.rowCount(); self.monomer_table.insertRow(r)
                for c in range(6):
                    self.monomer_table.setItem(r, c, QTableWidgetItem(""))
            self.monomer_table.setItem(target_row, 0, QTableWidgetItem(path))
            self.monomer_table.setItem(target_row, 1, QTableWidgetItem(Path(path).stem))
        self.log.emit(f"Monomer landed at slot {slot} (table row {target_row}): {path}")
        self.built_file.emit(path)
        self._push_copolymer_settings()

    def _remove_row(self):
        rows = sorted({i.row() for i in self.monomer_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.monomer_table.removeRow(r)

    # =========================================================== public API
    def monomer_specs(self) -> List[MonomerSpec]:
        specs: List[MonomerSpec] = []
        for r in range(self.monomer_table.rowCount()):
            def _g(c: int, cast=str, default=None):
                it = self.monomer_table.item(r, c)
                if not it or not it.text().strip(): return default
                try: return cast(it.text().strip())
                except ValueError: return default
            file = _g(0)
            if not file: continue
            specs.append(MonomerSpec(
                file=file, name=_g(1),
                head=_g(2, int), tail=_g(3, int),
                head_h=_g(4, int), tail_h=_g(5, int),
                cap_mode="user" if _g(2, int) else "terminal_h",
            ))
        return specs

    def load_monomer_specs(self, specs: List[MonomerSpec]):
        self.monomer_table.setRowCount(0)
        for spec in specs:
            r = self.monomer_table.rowCount(); self.monomer_table.insertRow(r)
            self.monomer_table.setItem(r, 0, QTableWidgetItem(spec.file))
            self.monomer_table.setItem(r, 1, QTableWidgetItem(spec.name or ""))
            self.monomer_table.setItem(r, 2, QTableWidgetItem(str(spec.head or "")))
            self.monomer_table.setItem(r, 3, QTableWidgetItem(str(spec.tail or "")))
            self.monomer_table.setItem(r, 4, QTableWidgetItem(str(spec.head_h or "")))
            self.monomer_table.setItem(r, 5, QTableWidgetItem(str(spec.tail_h or "")))

    def is_copolymer(self) -> bool:
        return self.rb_cop.isChecked()

    def copolymer_settings(self) -> dict:
        if not self.is_copolymer():
            return {"enabled": False}
        seed_txt = self.co_seed.text().strip()
        fa = self._parse_frac(self.co_fraction_a.currentText(), 0.5)
        fb = self._parse_frac(self.co_fraction_b.currentText(), 1.0 - fa)
        total = fa + fb
        if total > 0:
            fa, fb = fa / total, fb / total
        return {
            "enabled": True,
            "fraction_a": fa,
            "fraction_b": fb,
            "total": int(self.co_total.value()),
            "mode": self.co_mode.currentText(),
            "seed": int(seed_txt) if seed_txt.isdigit() else None,
        }
