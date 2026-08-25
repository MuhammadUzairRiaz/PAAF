"""Layering tab — stack typed components into their own regions.

Blend methodology (single-chain LAMMPS data + lammps.in per component,
merged types and coefficients, packmol placement), but every component is
packed into its OWN sub-box: layers stack along a chosen axis with a
configurable gap (default 5 Å), each layer has its own Lx x Ly x Lz, and a
layer can pin an explicit origin instead of auto-stacking. The total cell
is cubic or orthorhombic, sized automatically from the layers or fixed.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from . import tokens as T
from .page import (Card, button as _btn, caption, page_header, wrap_tooltip)

_COL_NAME, _COL_DATA, _COL_IN, _COL_COPIES = 0, 1, 2, 3
_COL_LX, _COL_LY, _COL_LZ, _COL_ORIGIN, _COL_BTNS = 4, 5, 6, 7, 8


class LayeringTab(QWidget):
    log = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._settings_provider = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, T.PAD_PAGE, T.PAD_PAGE)
        v.setSpacing(T.GAP_SECTION)

        v.addWidget(page_header(
            "Layering",
            "Stack two or more typed components into one cell, each packed "
            "into its own region — layer sizes, stacking axis, and the gap "
            "between layers are all yours to set.",
            breadcrumb="Tools › Layering",
        ))

        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(T.PAD_PAGE, 0, 0, 0)
        iv.setSpacing(T.GAP_SECTION)

        # ---- Layers ------------------------------------------------------
        card = Card("Layers",
                    "packed bottom-to-top in row order; single-chain "
                    ".data + lammps.in per layer")
        self.count_lb = caption("0 layers")
        card.header.addWidget(self.count_lb)
        b_add = _btn("+ Add layer"); b_add.clicked.connect(self._add_row)
        b_clr = _btn("Clear", "danger"); b_clr.clicked.connect(self._clear)
        card.header.addWidget(b_add); card.header.addWidget(b_clr)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["name", ".data file", "lammps.in", "copies",
             "Lx (Å)", "Ly (Å)", "Lz (Å)", "origin x,y,z (blank = stack)",
             "choose files"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(_COL_NAME, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(_COL_DATA, QHeaderView.Stretch)
        hh.setSectionResizeMode(_COL_IN, QHeaderView.Stretch)
        for c in (_COL_COPIES, _COL_LX, _COL_LY, _COL_LZ):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(_COL_ORIGIN, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(_COL_BTNS, QHeaderView.ResizeToContents)
        hh.setFixedHeight(T.H_TABLE_HEADER)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(T.H_TABLE_ROW)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(140)
        self.table.itemChanged.connect(lambda _i: self._refresh_plan())
        card.body.addWidget(self.table)
        iv.addWidget(card)

        # ---- Cell & stacking --------------------------------------------
        row2 = QHBoxLayout(); row2.setSpacing(T.GAP_SECTION)
        card_cell = Card("Cell & stacking")
        f = QFormLayout()
        f.setContentsMargins(0, 0, 0, 0)
        f.setSpacing(T.GAP_FORM_ROW)
        f.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        card_cell.body.addLayout(f)

        self.axis = QComboBox(); self.axis.addItems(["z", "x", "y"])
        self.axis.setToolTip(wrap_tooltip(
            "Which direction the layers stack along. Auto-placed layers are "
            "centred on the two other axes."))
        self.axis.currentIndexChanged.connect(self._refresh_plan)
        f.addRow("Stack along", self.axis)

        self.gap = QDoubleSpinBox(); self.gap.setRange(0.0, 100.0)
        self.gap.setValue(5.0); self.gap.setSuffix(" Å")
        self.gap.setToolTip(wrap_tooltip(
            "Empty spacing between consecutive auto-stacked layers. The "
            "next layer starts this far above the previous one."))
        self.gap.valueChanged.connect(self._refresh_plan)
        f.addRow("Gap between layers", self.gap)

        self.cell_mode = QComboBox()
        self.cell_mode.addItem("Auto — size the cell from the layers", "auto")
        self.cell_mode.addItem("Cubic (set edge)", "cubic")
        self.cell_mode.addItem("Orthorhombic (set a, b, c)", "ortho")
        self.cell_mode.currentIndexChanged.connect(self._on_cell_mode)
        f.addRow("Total cell", self.cell_mode)

        self.a = QDoubleSpinBox(); self.b = QDoubleSpinBox()
        self.c = QDoubleSpinBox()
        for sp in (self.a, self.b, self.c):
            sp.setRange(5, 5000); sp.setValue(80.0); sp.setSuffix(" Å")
            sp.valueChanged.connect(self._refresh_plan)
        f.addRow("a", self.a); f.addRow("b", self.b); f.addRow("c", self.c)

        self.tol = QDoubleSpinBox(); self.tol.setRange(0.5, 10.0)
        self.tol.setValue(2.0); self.tol.setSuffix(" Å")
        f.addRow("Packmol tolerance", self.tol)
        self.seed = QSpinBox(); self.seed.setRange(-1, 2_000_000_000)
        self.seed.setValue(-1)
        self.seed.setToolTip(wrap_tooltip(
            "-1 = fresh random seed each run"))
        f.addRow("Seed", self.seed)
        self.do_min = QCheckBox("Minimise each component before packing")
        f.addRow("", self.do_min)
        row2.addWidget(card_cell, 1)

        # ---- Planned regions --------------------------------------------
        card_plan = Card("Planned regions", "recomputed as you edit")
        self.plan_lb = caption("Add layers to see the plan.")
        self.plan_lb.setWordWrap(True)
        card_plan.body.addWidget(self.plan_lb)
        card_plan.body.addStretch(1)
        row2.addWidget(card_plan, 1)
        iv.addLayout(row2)

        # ---- Output ------------------------------------------------------
        card_out = Card("Output")
        fo = QFormLayout()
        fo.setContentsMargins(0, 0, 0, 0)
        fo.setSpacing(T.GAP_FORM_ROW)
        fo.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        card_out.body.addLayout(fo)
        out_row = QHBoxLayout()
        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("defaults to the project output folder")
        b_out = _btn("Browse…"); b_out.clicked.connect(self._browse_out_dir)
        out_row.addWidget(self.out_dir, 1); out_row.addWidget(b_out)
        wrap = QWidget(); wrap.setLayout(out_row)
        fo.addRow("Folder", wrap)
        self.out_data = QLineEdit("layered.data")
        fo.addRow("Data file", self.out_data)
        self.write_in = QCheckBox("Also write layered.in (styles + coeffs)")
        self.write_in.setChecked(True)
        fo.addRow("", self.write_in)
        iv.addWidget(card_out)

        # ---- Action ------------------------------------------------------
        act = QHBoxLayout()
        act.addStretch(1)
        b_pack = _btn("Build layered cell", "primary")
        b_pack.clicked.connect(self._pack)
        act.addWidget(b_pack)
        iv.addLayout(act)
        iv.addStretch(1)
        v.addWidget(inner, 1)

        self._on_cell_mode()
        self._add_row()
        self._add_row()

    # ------------------------------------------------------------- rows
    @staticmethod
    def _path_item(text: str = "") -> QTableWidgetItem:
        it = QTableWidgetItem(text)
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        if text:
            it.setToolTip(text)
        return it

    def _add_row(self) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, _COL_NAME, QTableWidgetItem(f"layer{r + 1}"))
        self.table.setItem(r, _COL_DATA, self._path_item(""))
        self.table.setItem(r, _COL_IN, self._path_item(""))
        self.table.setItem(r, _COL_COPIES, QTableWidgetItem("10"))
        for col, val in ((_COL_LX, "40"), (_COL_LY, "40"), (_COL_LZ, "20")):
            self.table.setItem(r, col, QTableWidgetItem(val))
        origin = QTableWidgetItem("")
        origin.setToolTip(wrap_tooltip(
            "Leave blank to stack this layer automatically after the "
            "previous one. Or type an explicit lower-corner origin as "
            "three numbers, e.g. 0, 0, 25 — the layer is then packed at "
            "exactly that position."))
        self.table.setItem(r, _COL_ORIGIN, origin)

        btns = QWidget()
        hb = QHBoxLayout(btns); hb.setContentsMargins(2, 0, 2, 0)
        hb.setSpacing(4)
        b_up = QPushButton("↑")
        b_up.setToolTip("Move this layer DOWN the stack (earlier row = "
                        "lower layer)")
        b_up.setFixedWidth(28)
        b_up.clicked.connect(lambda _c, row=r: self._move(row, -1))
        b_dn = QPushButton("↓")
        b_dn.setToolTip("Move this layer UP the stack")
        b_dn.setFixedWidth(28)
        b_dn.clicked.connect(lambda _c, row=r: self._move(row, +1))
        hb.addWidget(b_up); hb.addWidget(b_dn)
        b_data = QPushButton("Choose .data…")
        b_data.clicked.connect(lambda _c, row=r: self._browse_for(row))
        b_in = QPushButton("Choose .in…")
        b_in.clicked.connect(lambda _c, row=r: self._browse_in_for(row))
        b_rm = QPushButton("Remove")
        b_rm.clicked.connect(lambda _c, row=r: self._remove(row))
        for b in (b_data, b_in, b_rm):
            b.setMinimumWidth(92); hb.addWidget(b)
        self.table.setCellWidget(r, _COL_BTNS, btns)
        self._sync_count()

    def _move(self, row: int, delta: int) -> None:
        """Swap a row with its neighbour — row order IS the stack order."""
        other = row + delta
        if not (0 <= other < self.table.rowCount()) or other == row:
            return
        for col in range(_COL_BTNS):
            a = self.table.takeItem(row, col)
            b = self.table.takeItem(other, col)
            if b is not None:
                self.table.setItem(row, col, b)
            if a is not None:
                self.table.setItem(other, col, a)
        self._refresh_plan()

    def _remove(self, row: int) -> None:
        if 0 <= row < self.table.rowCount():
            self.table.removeRow(row)
        # Re-bind the button rows (their captured indices are now stale).
        for r in range(self.table.rowCount()):
            w = self.table.cellWidget(r, _COL_BTNS)
            if w is None:
                continue
            btns = w.findChildren(QPushButton)
            if len(btns) == 3:
                btns[0].clicked.disconnect()
                btns[1].clicked.disconnect()
                btns[2].clicked.disconnect()
                btns[0].clicked.connect(lambda _c, row=r: self._browse_for(row))
                btns[1].clicked.connect(lambda _c, row=r: self._browse_in_for(row))
                btns[2].clicked.connect(lambda _c, row=r: self._remove(row))
        self._sync_count()

    def _clear(self) -> None:
        self.table.setRowCount(0)
        self._sync_count()

    def _sync_count(self) -> None:
        n = self.table.rowCount()
        self.count_lb.setText(f"{n} layer{'s' if n != 1 else ''}")
        self._refresh_plan()

    def _browse_for(self, row: int) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Single-chain LAMMPS data file", "",
            "LAMMPS data (*.data);;All files (*)")
        if p:
            self.table.setItem(row, _COL_DATA, self._path_item(p))
            self._autofill_in(row, Path(p))
            self._refresh_plan()

    def _browse_in_for(self, row: int) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "LAMMPS input with styles / coefficients", "",
            "LAMMPS input (*.in);;All files (*)")
        if p:
            self.table.setItem(row, _COL_IN, self._path_item(p))

    def _autofill_in(self, row: int, data_path: Path) -> None:
        for cand in (data_path.with_suffix(".in"),
                     data_path.parent / "lammps.in"):
            if cand.exists():
                self.table.setItem(row, _COL_IN, self._path_item(str(cand)))
                return

    # ------------------------------------------------------------ plan
    def _cell_arg(self):
        mode = self.cell_mode.currentData()
        if mode == "auto":
            return None
        if mode == "cubic":
            return (self.a.value(),) * 3
        return (self.a.value(), self.b.value(), self.c.value())

    def _on_cell_mode(self, *_a) -> None:
        mode = self.cell_mode.currentData()
        self.a.setEnabled(mode != "auto")
        self.b.setEnabled(mode == "ortho")
        self.c.setEnabled(mode == "ortho")
        self._refresh_plan()

    def _layer_specs(self, need_files: bool = True) -> list:
        from ..layering import LayerSpec
        specs = []
        for r in range(self.table.rowCount()):
            def txt(col):
                it = self.table.item(r, col)
                return (it.text() if it else "").strip()
            data = txt(_COL_DATA)
            if need_files and not data:
                raise ValueError(f"Row {r + 1}: choose a .data file.")
            try:
                size = tuple(float(txt(c)) for c in (_COL_LX, _COL_LY,
                                                     _COL_LZ))
                count = int(float(txt(_COL_COPIES) or "1"))
            except ValueError:
                raise ValueError(
                    f"Row {r + 1}: copies and Lx/Ly/Lz must be numbers.")
            origin = None
            o_txt = txt(_COL_ORIGIN).replace(",", " ")
            if o_txt:
                parts = o_txt.split()
                if len(parts) != 3:
                    raise ValueError(
                        f"Row {r + 1}: origin needs three numbers "
                        f"(x, y, z) or blank for auto-stacking.")
                origin = tuple(float(p) for p in parts)
            specs.append(LayerSpec(
                name=txt(_COL_NAME) or f"layer{r + 1}",
                data_file=Path(data) if data else Path("unset.data"),
                count=max(1, count), size=size, origin=origin,
                forcefield_input=Path(txt(_COL_IN)) if txt(_COL_IN) else None))
        return specs

    def _refresh_plan(self, *_a) -> None:
        if not hasattr(self, "plan_lb"):
            return
        from ..layering import LayeringError, plan_regions
        try:
            specs = self._layer_specs(need_files=False)
            if not specs:
                self.plan_lb.setText("Add layers to see the plan.")
                return
            regions, cell = plan_regions(
                specs, axis=self.axis.currentText(),
                gap=float(self.gap.value()), total_box=self._cell_arg())
        except (LayeringError, ValueError) as exc:
            self.plan_lb.setText(str(exc))
            return
        lines = [f"Cell: {cell[0]:.1f} × {cell[1]:.1f} × {cell[2]:.1f} Å"]
        for s, reg in zip(specs, regions):
            lines.append(
                f"{s.name}: origin ({reg[0]:.1f}, {reg[1]:.1f}, "
                f"{reg[2]:.1f})"
                + ("" if s.origin is None else " pinned")
                + f" · x {reg[0]:.1f}–{reg[3]:.1f}, "
                f"y {reg[1]:.1f}–{reg[4]:.1f}, z {reg[2]:.1f}–{reg[5]:.1f}")
        self.plan_lb.setText("\n".join(lines))

    # ------------------------------------------------------------ output
    def set_settings_provider(self, fn) -> None:
        self._settings_provider = fn

    def default_out_dir(self) -> str:
        fn = getattr(self, "_settings_provider", None)
        if fn is not None:
            try:
                cfg = fn() or {}
                base = cfg.get("out_dir", "")
                proj = cfg.get("project", "polymer")
                if base:
                    return str(Path(base) / proj / "layered")
            except Exception:
                pass
        return str(Path.home() / "paaf_output" / "layered")

    def _browse_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Folder for the layered cell",
            self.out_dir.text().strip() or self.default_out_dir())
        if d:
            self.out_dir.setText(d)

    # ------------------------------------------------------------ build
    def _pack(self) -> None:
        from ..layering import LayeringError, build_layered_cell
        try:
            specs = self._layer_specs(need_files=True)
        except ValueError as e:
            QMessageBox.warning(self, "Layering", str(e)); return
        if len(specs) < 2:
            QMessageBox.warning(
                self, "Layering",
                "Add at least two layers (with .data files) to stack.")
            return
        out_dir = Path(self.out_dir.text().strip() or self.default_out_dir())
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Layering",
                                 f"Cannot write to {out_dir}:\n{e}")
            return
        out_data = out_dir / (self.out_data.text().strip() or "layered.data")
        out_in = (out_data.with_suffix(".in")
                  if self.write_in.isChecked() else None)
        from ..blend_minimise import MinimiseSettings
        self.log.emit(f"[layering] building {len(specs)} layers …")
        try:
            path = build_layered_cell(
                specs, out_data, axis=self.axis.currentText(),
                gap=float(self.gap.value()), total_box=self._cell_arg(),
                out_input_file=out_in, seed=int(self.seed.value()),
                tolerance=float(self.tol.value()),
                minimise=MinimiseSettings(enabled=self.do_min.isChecked()),
                progress=lambda m: self.log.emit(f"[layering] {m}"))
        except (LayeringError, Exception) as e:
            QMessageBox.critical(self, "Layering", f"Build failed:\n{e}")
            self.log.emit(f"[layering] ERROR: {e}")
            return
        self.log.emit(f"[layering] Wrote {path}")
        QMessageBox.information(
            self, "Layering",
            f"Wrote {path.name}"
            + (f" and {out_in.name}" if out_in else "")
            + f"\n\nin {out_dir}")
