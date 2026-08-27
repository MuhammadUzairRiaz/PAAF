"""CG builder — a standalone coarse-grained (Kremer-Grest) cell tool.

Same growth engine as the Amorphous cell (Theodorou-Suter beads), but the
output is a runnable bead-spring system instead of an all-atom one: FENE
bonds, WCA pairs, optional angle stiffness fitted to the polymer's own
measured C_n. No DL_FIELD, no packmol, no back-mapping — the CG route
works on any machine that has LAMMPS.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QHeaderView, QLineEdit, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from . import tokens as T
from .page import Card, button as _btn, caption, page_header, wrap_tooltip

_COL_NAME, _COL_SMILES, _COL_DP, _COL_CHAINS, _COL_BTNS = 0, 1, 2, 3, 4


# ================================================================ worker
class _CGWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, options: dict):
        super().__init__()
        self.o = options
        from ..cell.packing import CancelToken
        self._cancel = CancelToken()

    def cancel(self) -> None:
        self._cancel.cancel()

    @pyqtSlot()
    def run(self):
        from ..cell.packing import PackCancelled
        try:
            o = self.o
            from ..cell.cg_model import CGSettings, build_cg_cell
            from ..cell.composition import Component, from_chain_counts
            from ..cell.grow import grow_amorphous_cell

            comps = [Component(name=r["name"], repeat_unit=r["smiles"],
                               degree_of_polymerisation=r["dp"],
                               n_chains=r["chains"], ris_key=r["name"])
                     for r in o["rows"]]
            comp = from_chain_counts(comps, o["density"],
                                     build_fraction=o["build_fraction"])
            self.progress.emit(
                f"growing {sum(c.n_chains for c in comps)} chain(s) in a "
                f"{comp.box_edge_a:.1f} A box …")
            res = grow_amorphous_cell(
                comp.grow_specs(), comp.box(),
                temperature=o["temperature"], seed=o["seed"],
                cancel=self._cancel,
                progress=self.progress.emit)
            cn = float(getattr(res, "mean_c_n", 0.0) or 0.0)
            if cn:
                self.progress.emit(f"measured mean C_n = {cn:.2f}")
            st = CGSettings(units=o["units"], mapping=o["mapping"],
                            angle_mode=o["angle_mode"],
                            angle_k=o["angle_k"],
                            target_cn=o["target_cn"],
                            temperature_k=o["temperature"])
            data, inp = build_cg_cell(
                res, comp.grow_specs(), o["out_dir"], name=o["name"],
                settings=st, measured_cn=cn, progress=self.progress.emit)
            self.finished.emit((data, inp))
        except PackCancelled:
            self.progress.emit("build cancelled")
            self.cancelled.emit()
        except Exception as exc:
            self.progress.emit(traceback.format_exc())
            self.failed.emit(str(exc) or exc.__class__.__name__)


# ================================================================== tab
class CGTab(QWidget):
    log = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._settings_provider = None
        self._thread = None
        self._worker = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, T.PAD_PAGE, T.PAD_PAGE)
        v.setSpacing(T.GAP_SECTION)

        v.addWidget(page_header(
            "CG builder",
            "Grow chains and write a runnable Kremer-Grest bead-spring "
            "cell — FENE bonds, WCA beads, stiffness fitted to the "
            "polymer's own C_n. Needs only LAMMPS.",
            breadcrumb="Tools › CG builder",
        ))

        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(T.PAD_PAGE, 0, 0, 0)
        iv.setSpacing(T.GAP_SECTION)

        # ---- polymers ---------------------------------------------------
        card = Card("Polymers", "one row per species; chains are exact")
        b_add = _btn("+ Add polymer"); b_add.clicked.connect(self._add_row)
        b_clr = _btn("Clear", "danger"); b_clr.clicked.connect(self._clear)
        card.header.addWidget(b_add); card.header.addWidget(b_clr)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["name", "polymerisation SMILES", "DP", "chains", ""])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(_COL_NAME, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(_COL_SMILES, QHeaderView.Stretch)
        hh.setSectionResizeMode(_COL_DP, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(_COL_CHAINS, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(_COL_BTNS, QHeaderView.ResizeToContents)
        hh.setFixedHeight(T.H_TABLE_HEADER)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(T.H_TABLE_ROW)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(120)
        card.body.addWidget(self.table)
        iv.addWidget(card)

        # ---- growth + model side by side --------------------------------
        row2 = QHBoxLayout(); row2.setSpacing(T.GAP_SECTION)

        card_g = Card("Growth", "same Theodorou-Suter engine as the "
                                "Amorphous cell")
        f = QFormLayout(); f.setContentsMargins(0, 0, 0, 0)
        f.setSpacing(T.GAP_FORM_ROW)
        f.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        card_g.body.addLayout(f)
        self.density = QDoubleSpinBox(); self.density.setRange(0.05, 3.0)
        self.density.setDecimals(3); self.density.setValue(0.85)
        self.density.setSuffix(" g/cm³")
        self.density.setToolTip(wrap_tooltip(
            "Target mass density. The box is sized from it; the KG bead "
            "diameter sigma is then derived so the bead packing matches "
            "the standard melt condition (0.85 beads per sigma³)."))
        f.addRow("Density", self.density)
        self.build_at = QDoubleSpinBox(); self.build_at.setRange(20.0, 100.0)
        self.build_at.setValue(100.0); self.build_at.setSuffix(" %")
        self.build_at.setDecimals(0)
        self.build_at.setToolTip(wrap_tooltip(
            "Grow loose and compress later, exactly as in the Amorphous "
            "cell. Beads are far more forgiving than atoms, so 100% "
            "usually just works — lower it only if growth keeps failing."))
        f.addRow("Build at", self.build_at)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(1.0, 2000.0); self.temperature.setValue(413.0)
        self.temperature.setSuffix(" K")
        f.addRow("Temperature", self.temperature)
        self.seed = QSpinBox(); self.seed.setRange(1, 2_000_000_000)
        self.seed.setValue(12345)
        f.addRow("Seed", self.seed)
        row2.addWidget(card_g, 1)

        card_m = Card("Kremer-Grest model")
        fm = QFormLayout(); fm.setContentsMargins(0, 0, 0, 0)
        fm.setSpacing(T.GAP_FORM_ROW)
        fm.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        card_m.body.addLayout(fm)
        self.mapping = QComboBox()
        self.mapping.addItem("1 bead per skeletal atom (fine)", "backbone")
        self.mapping.addItem("1 bead per repeat unit (classic CG)",
                             "monomer")
        self.mapping.setToolTip(wrap_tooltip(
            "How much chemistry one bead swallows. Fine keeps the "
            "grower's native resolution; per-repeat-unit is the classic "
            "CG picture (centre of mass of each unit)."))
        fm.addRow("Mapping", self.mapping)
        self.units = QComboBox()
        self.units.addItem("Real (Å, kcal/mol, fs)", "real")
        self.units.addItem("Reduced LJ (sigma = eps = m = 1)", "lj")
        self.units.setToolTip(wrap_tooltip(
            "Reduced units are the literature convention for KG; real "
            "units run alongside your other LAMMPS files without unit "
            "gymnastics."))
        fm.addRow("Units", self.units)
        self.angle_mode = QComboBox()
        self.angle_mode.addItem("Fit stiffness to measured C_n (auto)",
                                "auto")
        self.angle_mode.addItem("Manual stiffness", "manual")
        self.angle_mode.addItem("No angle term (fully flexible)", "off")
        self.angle_mode.currentIndexChanged.connect(self._on_angle_mode)
        fm.addRow("Chain stiffness", self.angle_mode)
        self.angle_k = QDoubleSpinBox(); self.angle_k.setRange(0.0, 100.0)
        self.angle_k.setValue(1.5); self.angle_k.setSuffix(" eps")
        fm.addRow("Manual K", self.angle_k)
        self.target_cn = QDoubleSpinBox(); self.target_cn.setRange(0.0, 50.0)
        self.target_cn.setValue(0.0); self.target_cn.setDecimals(2)
        self.target_cn.setToolTip(wrap_tooltip(
            "0 = use the C_n the grower measures for this build. Set a "
            "literature value (PE 6.9, PS 9.5, …) to target it instead."))
        fm.addRow("Target C_inf", self.target_cn)
        row2.addWidget(card_m, 1)
        iv.addLayout(row2)

        # ---- output -----------------------------------------------------
        card_o = Card("Output")
        fo = QFormLayout(); fo.setContentsMargins(0, 0, 0, 0)
        fo.setSpacing(T.GAP_FORM_ROW)
        fo.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        card_o.body.addLayout(fo)
        out_row = QHBoxLayout()
        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("defaults to the project output folder")
        b_out = _btn("Browse…"); b_out.clicked.connect(self._browse_out)
        out_row.addWidget(self.out_dir, 1); out_row.addWidget(b_out)
        wrap = QWidget(); wrap.setLayout(out_row)
        fo.addRow("Folder", wrap)
        self.cell_name = QLineEdit("cg_cell")
        fo.addRow("Cell name", self.cell_name)
        fo.addRow("", caption(
            "Writes cell_cg.data + cell_cg.in — run with "
            "`lmp -in cell_cg.in`. The script does its own push-off, so "
            "no other tool is needed."))
        iv.addWidget(card_o)

        # ---- action -----------------------------------------------------
        act = QHBoxLayout()
        self.stage_lb = caption("")
        self.stage_lb.setStyleSheet(
            f"color: {T.PRIMARY}; font-weight: 700;"
            f" font-size: {T.FS_CAPTION}px; background: transparent;"
            f" border: none;")
        self.stage_lb.hide()
        act.addWidget(self.stage_lb)
        act.addStretch(1)
        self.bar = QProgressBar(); self.bar.setFixedWidth(160)
        self.bar.setRange(0, 0); self.bar.hide()
        act.addWidget(self.bar)
        self.b_cancel = _btn("Cancel", "danger")
        self.b_cancel.clicked.connect(self._cancel)
        self.b_cancel.hide()
        act.addWidget(self.b_cancel)
        self.b_build = _btn("Build CG cell", "primary")
        self.b_build.clicked.connect(self._build)
        act.addWidget(self.b_build)
        iv.addLayout(act)
        iv.addStretch(1)
        v.addWidget(inner, 1)

        self._on_angle_mode()
        self._add_row()

    # ------------------------------------------------------------- rows
    def _add_row(self) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, _COL_NAME, QTableWidgetItem("PE"))
        self.table.setItem(r, _COL_SMILES, QTableWidgetItem("[*]CC[*]"))
        self.table.setItem(r, _COL_DP, QTableWidgetItem("50"))
        self.table.setItem(r, _COL_CHAINS, QTableWidgetItem("10"))
        btns = QWidget()
        hb = QHBoxLayout(btns); hb.setContentsMargins(2, 0, 2, 0)
        hb.setSpacing(4)
        b_lib = QPushButton("Library…")
        b_lib.clicked.connect(lambda _c, row=r: self._pick_library(row))
        b_rm = QPushButton("Remove")
        b_rm.clicked.connect(lambda _c, row=r: self._remove(row))
        for b in (b_lib, b_rm):
            b.setMinimumWidth(76); hb.addWidget(b)
        self.table.setCellWidget(r, _COL_BTNS, btns)

    def _pick_library(self, row: int) -> None:
        from .library_picker import LibraryPicker
        dlg = LibraryPicker(self)
        if dlg.exec_() != dlg.Accepted:
            return
        rec = dlg.selected()
        if rec is None:
            return
        self.table.setItem(row, _COL_SMILES,
                           QTableWidgetItem(rec.smiles or ""))
        shown = rec.name or ""
        if rec.pid and shown == rec.pid and (rec.description or "").strip():
            shown = rec.description.strip()
        self.table.setItem(row, _COL_NAME, QTableWidgetItem(shown))

    def _remove(self, row: int) -> None:
        if 0 <= row < self.table.rowCount():
            self.table.removeRow(row)
        for r in range(self.table.rowCount()):
            w = self.table.cellWidget(r, _COL_BTNS)
            if w is None:
                continue
            btns = w.findChildren(QPushButton)
            if len(btns) == 2:
                btns[0].clicked.disconnect()
                btns[1].clicked.disconnect()
                btns[0].clicked.connect(
                    lambda _c, row=r: self._pick_library(row))
                btns[1].clicked.connect(lambda _c, row=r: self._remove(row))

    def _clear(self) -> None:
        self.table.setRowCount(0)

    def _rows(self) -> List[dict]:
        out = []
        for r in range(self.table.rowCount()):
            def txt(col):
                it = self.table.item(r, col)
                return (it.text() if it else "").strip()
            smiles = txt(_COL_SMILES)
            if not smiles:
                continue
            try:
                dp = int(float(txt(_COL_DP) or "1"))
                chains = int(float(txt(_COL_CHAINS) or "1"))
            except ValueError:
                raise ValueError(f"Row {r + 1}: DP and chains must be "
                                 f"numbers.")
            out.append({"name": txt(_COL_NAME) or f"P{r + 1}",
                        "smiles": smiles, "dp": max(1, dp),
                        "chains": max(1, chains)})
        return out

    # ---------------------------------------------------------- settings
    def set_settings_provider(self, fn) -> None:
        self._settings_provider = fn

    def default_out_dir(self) -> str:
        fn = getattr(self, "_settings_provider", None)
        if fn is not None:
            try:
                cfg = fn() or {}
                if cfg.get("out_dir"):
                    return str(Path(cfg["out_dir"])
                               / (cfg.get("project") or "polymer") / "cg")
            except Exception:
                pass
        return str(Path.home() / "paaf_output" / "cg")

    def _browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Folder for CG cells",
            self.out_dir.text().strip() or self.default_out_dir())
        if d:
            self.out_dir.setText(d)

    def _on_angle_mode(self, *_a) -> None:
        mode = self.angle_mode.currentData()
        self.angle_k.setEnabled(mode == "manual")
        self.target_cn.setEnabled(mode == "auto")

    # ------------------------------------------------------------- build
    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.stage_lb.setText("Cancelling …")

    def _set_busy(self, busy: bool) -> None:
        self.b_build.setEnabled(not busy)
        self.b_cancel.setVisible(busy)
        self.bar.setVisible(busy)
        self.stage_lb.setVisible(busy)
        if busy:
            self.stage_lb.setText("Growing chains …")

    def _build(self) -> None:
        try:
            rows = self._rows()
        except ValueError as e:
            QMessageBox.warning(self, "CG builder", str(e)); return
        if not rows:
            QMessageBox.warning(self, "CG builder",
                                "Add at least one polymer with a SMILES.")
            return
        out_dir = Path(self.out_dir.text().strip() or self.default_out_dir())
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "CG builder",
                                 f"Cannot write to {out_dir}:\n{e}")
            return
        options = {
            "rows": rows,
            "density": float(self.density.value()),
            "build_fraction": float(self.build_at.value()) / 100.0,
            "temperature": float(self.temperature.value()),
            "seed": int(self.seed.value()),
            "units": self.units.currentData(),
            "mapping": self.mapping.currentData(),
            "angle_mode": self.angle_mode.currentData(),
            "angle_k": float(self.angle_k.value()),
            "target_cn": float(self.target_cn.value()),
            "out_dir": str(out_dir),
            "name": self.cell_name.text().strip() or "cg_cell",
        }
        self._set_busy(True)
        self._thread = QThread(self)
        self._worker = _CGWorker(options)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(
            lambda m: self.log.emit(f"[cg] {m}"))
        self._worker.finished.connect(self._done)
        self._worker.failed.connect(self._failed)
        self._worker.cancelled.connect(lambda: self._finish_thread())
        self._thread.start()

    def _finish_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = None
        self._worker = None
        self._set_busy(False)

    def _done(self, paths) -> None:
        data, inp = paths
        self._finish_thread()
        self.log.emit(f"[cg] wrote {data} and {inp}")
        QMessageBox.information(
            self, "CG builder",
            f"Wrote {Path(data).name} and {Path(inp).name}\n\n"
            f"in {Path(data).parent}\n\nRun with:  lmp -in {Path(inp).name}")

    def _failed(self, msg: str) -> None:
        self._finish_thread()
        QMessageBox.critical(self, "CG builder", f"Build failed:\n{msg}")
