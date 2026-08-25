"""System Builder tab — a Materials-Studio-style unified page with sub-modes:
Crystal, Surface/slab, Nanotube, Solvate, Layers.

Amorphous-cell packing used to live here too. It has moved to its own tool
(:mod:`paaf.gui.amorphous_tab`), which GROWS chains bond by bond rather than
inserting pre-built ones — the only approach that reaches melt density with
real chain lengths. Keeping a second, weaker amorphous builder here would
only be a way to get a worse answer by accident.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import List

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


# =============================================================== worker
class CellWorker(QObject):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, task: str, args: dict):
        super().__init__()
        self.task = task
        self.args = args

    @pyqtSlot()
    def run(self):
        try:
            if self.task == "crystal":
                self._crystal()
            elif self.task == "surface":
                self._surface()
            elif self.task == "nanotube":
                self._nanotube()
            elif self.task == "solvate":
                self._solvate()
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")

    def _crystal(self):
        from ..cell import build_preset
        from ..structure import write
        mol = build_preset(self.args["preset"], nx=self.args["nx"],
                          ny=self.args["ny"], nz=self.args["nz"])
        write(mol, self.args["out"])
        self.finished.emit({"task": "crystal", "out": self.args["out"],
                            "atoms": len(mol.atoms)})

    def _surface(self):
        from ..cell import cleave_surface
        from ..structure import write
        mol = cleave_surface(
            self.args["preset"], tuple(self.args["hkl"]),
            supercell=tuple(self.args["supercell"]),
            thickness_ang=self.args["thickness"], vacuum_ang=self.args["vacuum"],
        )
        write(mol, self.args["out"])
        self.finished.emit({"task": "surface", "out": self.args["out"],
                            "atoms": len(mol.atoms)})

    def _nanotube(self):
        from ..cell import build_nanotube
        from ..structure import write
        mol = build_nanotube(
            self.args["n"], self.args["m"],
            length_ang=self.args["length"], element=self.args["element"],
        )
        write(mol, self.args["out"])
        self.finished.emit({"task": "nanotube", "out": self.args["out"],
                            "atoms": len(mol.atoms)})

    def _solvate(self):
        from ..cell import solvate
        mol, box = solvate(
            self.args["solute"], solvent=self.args["solvent"],
            n_solvent=self.args["n_solvent"],
            density_kg_m3=self.args.get("density_kg_m3"),
            out_path=self.args["out"], seed=self.args["seed"],
        )
        self.finished.emit({"task": "solvate", "out": self.args["out"],
                            "atoms": len(mol.atoms), "box_ang": box})

    def _layers(self):
        from ..cell import build_layers, LayerSpec
        specs = []
        for L in self.args["layers"]:
            if L["kind"] == "preset":
                specs.append(LayerSpec(
                    preset=L["src"], supercell=tuple(L["supercell"]),
                    name=L["name"], lateral_offset=tuple(L["offset"]),
                    gap_before=L.get("gap_before"),
                ))
            else:
                specs.append(LayerSpec(
                    file=L["src"], name=L["name"],
                    lateral_offset=tuple(L["offset"]),
                    gap_before=L.get("gap_before"),
                ))
        mol, box = build_layers(specs, gap_ang=self.args["gap"],
                                center_xy=self.args["center"],
                                out_path=self.args["out"])
        self.finished.emit({"task": "layers", "out": self.args["out"],
                            "atoms": len(mol.atoms),
                            "box_ang": max(box)})


# =============================================================== tab
class CellTab(QWidget):
    log = pyqtSignal(str)
    built_file = pyqtSignal(str)

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
            "<b>System builder</b><br>"
            "Structure constructors: build crystal super-cells, "
            "cleave surfaces, build nanotubes, solvate a molecule, or stack "
            "layers."
        ))

        # Mode selector
        row = QHBoxLayout()
        row.addWidget(QLabel("Mode:"))
        self.mode = QComboBox()
        # Keyed, not positional. The dispatch below reads currentData(), so
        # adding or removing a mode cannot silently shift every branch — which
        # is exactly what would have happened when Amorphous cell was removed.
        for _text, _key in (("Crystal", "crystal"),
                            ("Surface / slab", "surface"),
                            ("Nanotube", "nanotube"),
                            ("Solvate", "solvate"),
                            ("Layers (bilayer / trilayer)", "layers")):
            self.mode.addItem(_text, _key)
        self.mode.currentIndexChanged.connect(self._on_mode)
        row.addWidget(self.mode, 1)
        v.addLayout(row)

        self.stack = QStackedWidget()
        for page in (self._page_crystal(),  self._page_surface(),
                     self._page_nanotube(), self._page_solvate(),
                     self._page_layers()):
            sa = QScrollArea()
            sa.setWidget(page)
            sa.setWidgetResizable(True)                                  # follow width
            sa.setFrameShape(QScrollArea.NoFrame)
            sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)       # vertical only
            sa.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            page.setMinimumWidth(0)
            self.stack.addWidget(sa)

        from .page import fit_stack_to_current
        fit_stack_to_current(self.stack)
        v.addWidget(self.stack, 1)

        # Actions
        row2 = QHBoxLayout()
        self.btn_build = QPushButton("Build system")
        self.btn_build.setObjectName("primary")
        self.btn_build.clicked.connect(self._start)
        row2.addWidget(self.btn_build); row2.addStretch(1)
        v.addLayout(row2)

    def _on_mode(self, idx: int):
        self.stack.setCurrentIndex(idx)

    # -------------------- pages --------------------
    def _page_crystal(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0)
        from ..cell import list_presets
        v.addWidget(QLabel("Pick a preset lattice or edit unit-cell parameters."))
        self.crys_list = QListWidget()
        for p in list_presets():
            it = QListWidgetItem(f"{p.name}  —  {p.description}"); it.setData(Qt.UserRole, p.name)
            self.crys_list.addItem(it)
        v.addWidget(self.crys_list, 1)

        gb = QGroupBox("Supercell")
        f = QFormLayout(gb)
        self.crys_nx = QSpinBox(); self.crys_nx.setRange(1, 200); self.crys_nx.setValue(4)
        self.crys_ny = QSpinBox(); self.crys_ny.setRange(1, 200); self.crys_ny.setValue(4)
        self.crys_nz = QSpinBox(); self.crys_nz.setRange(1, 200); self.crys_nz.setValue(4)
        f.addRow("nx", self.crys_nx); f.addRow("ny", self.crys_ny); f.addRow("nz", self.crys_nz)
        v.addWidget(gb)

        self.crys_out = QLineEdit(str(Path.cwd() / "crystal.xyz"))
        v.addLayout(self._out_row(self.crys_out, "XYZ (*.xyz);;PDB (*.pdb)"))
        return w

    def _page_surface(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0)
        from ..cell import list_presets
        v.addWidget(QLabel("Cleave a preset crystal along a Miller plane and add vacuum."))
        gb = QGroupBox("Slab parameters")
        f = QFormLayout(gb)
        self.surf_preset = QComboBox()
        for p in list_presets():
            self.surf_preset.addItem(f"{p.name} — {p.description}", userData=p.name)
        self.surf_h = QSpinBox(); self.surf_h.setRange(-9, 9); self.surf_h.setValue(1)
        self.surf_k = QSpinBox(); self.surf_k.setRange(-9, 9); self.surf_k.setValue(0)
        self.surf_l = QSpinBox(); self.surf_l.setRange(-9, 9); self.surf_l.setValue(0)
        self.surf_nx = QSpinBox(); self.surf_nx.setRange(1, 50); self.surf_nx.setValue(5)
        self.surf_ny = QSpinBox(); self.surf_ny.setRange(1, 50); self.surf_ny.setValue(5)
        self.surf_nz = QSpinBox(); self.surf_nz.setRange(1, 50); self.surf_nz.setValue(5)
        self.surf_thick = QDoubleSpinBox(); self.surf_thick.setRange(1, 200); self.surf_thick.setDecimals(1); self.surf_thick.setSingleStep(1.0); self.surf_thick.setGroupSeparatorShown(False); self.surf_thick.setValue(12)
        self.surf_thick.setSuffix(" Å")
        self.surf_vac = QDoubleSpinBox(); self.surf_vac.setRange(0, 200); self.surf_vac.setDecimals(1); self.surf_vac.setSingleStep(1.0); self.surf_vac.setGroupSeparatorShown(False); self.surf_vac.setValue(15)
        self.surf_vac.setSuffix(" Å")
        hkl = QHBoxLayout(); hkl.addWidget(self.surf_h); hkl.addWidget(self.surf_k); hkl.addWidget(self.surf_l)
        hkl_w = QWidget(); hkl_w.setLayout(hkl)
        sc = QHBoxLayout(); sc.addWidget(self.surf_nx); sc.addWidget(self.surf_ny); sc.addWidget(self.surf_nz)
        sc_w = QWidget(); sc_w.setLayout(sc)
        f.addRow("Preset crystal", self.surf_preset)
        f.addRow("Miller (h k l)", hkl_w)
        f.addRow("Supercell (nx ny nz)", sc_w)
        f.addRow("Slab thickness", self.surf_thick)
        f.addRow("Vacuum", self.surf_vac)
        v.addWidget(gb)

        self.surf_out = QLineEdit(str(Path.cwd() / "slab.xyz"))
        v.addLayout(self._out_row(self.surf_out, "XYZ (*.xyz);;PDB (*.pdb)"))
        v.addStretch(1)
        return w

    def _page_nanotube(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0)
        gb = QGroupBox("Nanotube (n, m)")
        f = QFormLayout(gb)
        self.cnt_n = QSpinBox(); self.cnt_n.setRange(1, 100); self.cnt_n.setValue(6)
        self.cnt_m = QSpinBox(); self.cnt_m.setRange(0, 100); self.cnt_m.setValue(6)
        self.cnt_len = QDoubleSpinBox(); self.cnt_len.setRange(3, 500); self.cnt_len.setDecimals(1); self.cnt_len.setSingleStep(1.0); self.cnt_len.setGroupSeparatorShown(False); self.cnt_len.setValue(20)
        self.cnt_len.setSuffix(" Å")
        self.cnt_element = QLineEdit("C")
        f.addRow("n", self.cnt_n); f.addRow("m", self.cnt_m)
        f.addRow("Length", self.cnt_len); f.addRow("Element", self.cnt_element)
        v.addWidget(gb)

        self.cnt_out = QLineEdit(str(Path.cwd() / "nanotube.xyz"))
        v.addLayout(self._out_row(self.cnt_out, "XYZ (*.xyz);;PDB (*.pdb)"))
        v.addStretch(1)
        return w

    def _page_layers(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(QLabel(
            "Stack N layers along Z with configurable gap. Each row = one "
            "layer (file OR crystal preset). Use for bilayers, trilayers, "
            "polymer-on-substrate systems, etc."
        ))
        self.lay_table = QTableWidget(0, 6)
        self.lay_table.setHorizontalHeaderLabels(
            ["File / preset", "Kind", "Name", "Supercell (for preset)", "Gap before (Å)", "xy offset"]
        )
        self.lay_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.lay_table.setMinimumWidth(0)
        v.addWidget(self.lay_table, 1)

        row = QHBoxLayout()
        b_add_f = QPushButton("Add layer from file..."); b_add_f.clicked.connect(self._lay_add_file)
        b_add_p = QPushButton("Add layer from crystal preset..."); b_add_p.clicked.connect(self._lay_add_preset)
        b_rem = QPushButton("Remove selected"); b_rem.clicked.connect(self._lay_remove)
        for b in (b_add_f, b_add_p, b_rem): row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)

        gb = QGroupBox("Stack parameters")
        f = QFormLayout(gb)
        self.lay_gap = QDoubleSpinBox(); self.lay_gap.setRange(0.0, 100.0); self.lay_gap.setDecimals(1); self.lay_gap.setSingleStep(0.5); self.lay_gap.setGroupSeparatorShown(False); self.lay_gap.setValue(3.0)
        self.lay_gap.setSuffix(" Å")
        self.lay_center = QCheckBox("Center every layer on largest xy footprint")
        self.lay_center.setChecked(True)
        f.addRow("Default gap between layers", self.lay_gap)
        f.addRow(self.lay_center)
        v.addWidget(gb)

        self.lay_out = QLineEdit(str(Path.cwd() / "layers.pdb"))
        v.addLayout(self._out_row(self.lay_out, "PDB (*.pdb);;XYZ (*.xyz)"))
        return w

    def _lay_add_file(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select structure",
            filter="Structures (*.xyz *.pdb *.mol2 *.mol *.sdf)")
        if not p: return
        r = self.lay_table.rowCount(); self.lay_table.insertRow(r)
        self.lay_table.setItem(r, 0, QTableWidgetItem(p))
        self.lay_table.setItem(r, 1, QTableWidgetItem("file"))
        self.lay_table.setItem(r, 2, QTableWidgetItem(Path(p).stem))
        self.lay_table.setItem(r, 3, QTableWidgetItem("1,1,1"))
        self.lay_table.setItem(r, 4, QTableWidgetItem(""))
        self.lay_table.setItem(r, 5, QTableWidgetItem("0,0"))

    def _lay_add_preset(self):
        from ..cell import list_presets
        # simple dialog: pick from a QInputDialog
        from PyQt5.QtWidgets import QInputDialog
        names = [f"{p.name}  —  {p.description}" for p in list_presets()]
        choice, ok = QInputDialog.getItem(self, "Crystal preset", "Preset:", names, 0, False)
        if not ok: return
        preset = choice.split(" ")[0]
        r = self.lay_table.rowCount(); self.lay_table.insertRow(r)
        self.lay_table.setItem(r, 0, QTableWidgetItem(preset))
        self.lay_table.setItem(r, 1, QTableWidgetItem("preset"))
        self.lay_table.setItem(r, 2, QTableWidgetItem(preset))
        self.lay_table.setItem(r, 3, QTableWidgetItem("4,4,1"))
        self.lay_table.setItem(r, 4, QTableWidgetItem(""))
        self.lay_table.setItem(r, 5, QTableWidgetItem("0,0"))

    def _lay_remove(self):
        rows = sorted({i.row() for i in self.lay_table.selectedIndexes()}, reverse=True)
        for r in rows: self.lay_table.removeRow(r)

    def _page_solvate(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0)
        gb = QGroupBox("Solvation")
        f = QFormLayout(gb)
        self.slv_solute = QLineEdit()
        self.slv_solute.setPlaceholderText("Path to solute (xyz/pdb/mol2)")
        b_pick = QPushButton("Browse..."); b_pick.clicked.connect(self._slv_pick_solute)
        row = QHBoxLayout(); row.addWidget(self.slv_solute); row.addWidget(b_pick)
        wrap = QWidget(); wrap.setLayout(row)
        self.slv_solvent = QComboBox()
        from ..cell.solvate import SOLVENTS
        for k in sorted(SOLVENTS.keys()):
            self.slv_solvent.addItem(k)
        self.slv_n = QSpinBox(); self.slv_n.setRange(1, 100000); self.slv_n.setValue(500)
        self.slv_density = QDoubleSpinBox()
        self.slv_density.setRange(0.0, 25.00)
        self.slv_density.setDecimals(2)
        self.slv_density.setSingleStep(0.05)
        self.slv_density.setGroupSeparatorShown(False)
        self.slv_density.setValue(0.0)
        self.slv_density.setSpecialValueText("solvent default")
        self.slv_density.setSuffix(" g/cm³")
        f.addRow("Solute", wrap)
        f.addRow("Solvent", self.slv_solvent)
        f.addRow("# solvent molecules", self.slv_n)
        f.addRow("Target density (0 = solvent default)", self.slv_density)
        v.addWidget(gb)
        self.slv_out = QLineEdit(str(Path.cwd() / "solvated.pdb"))
        v.addLayout(self._out_row(self.slv_out, "PDB (*.pdb);;XYZ (*.xyz)"))
        v.addStretch(1)
        return w

    # -------------------- helpers --------------------
    def _out_row(self, edit: QLineEdit, filt: str):
        h = QHBoxLayout()
        h.addWidget(QLabel("Output:")); h.addWidget(edit, 1)
        b = QPushButton("Save as...")
        b.clicked.connect(lambda: self._pick_out(edit, filt))
        h.addWidget(b)
        return h

    def _pick_out(self, edit: QLineEdit, filt: str):
        p, _ = QFileDialog.getSaveFileName(self, "Save", filter=filt)
        if p:
            edit.setText(p)

    def _slv_pick_solute(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Solute", filter="Structures (*.xyz *.pdb *.mol2 *.mol *.sdf)")
        if p: self.slv_solute.setText(p)

    # -------------------- run --------------------
    def _start(self):
        mode = self.mode.currentData()
        try:
            if mode == "crystal":
                it = self.crys_list.currentItem()
                if not it:
                    QMessageBox.warning(self, "No preset", "Pick a crystal preset."); return
                self._launch("crystal", {
                    "preset": it.data(Qt.UserRole),
                    "nx": self.crys_nx.value(), "ny": self.crys_ny.value(), "nz": self.crys_nz.value(),
                    "out": self.crys_out.text().strip(),
                })
            elif mode == "surface":
                self._launch("surface", {
                    "preset": self.surf_preset.currentData(),
                    "hkl": [self.surf_h.value(), self.surf_k.value(), self.surf_l.value()],
                    "supercell": [self.surf_nx.value(), self.surf_ny.value(), self.surf_nz.value()],
                    "thickness": self.surf_thick.value(),
                    "vacuum": self.surf_vac.value(),
                    "out": self.surf_out.text().strip(),
                })
            elif mode == "nanotube":
                self._launch("nanotube", {
                    "n": self.cnt_n.value(), "m": self.cnt_m.value(),
                    "length": self.cnt_len.value(),
                    "element": self.cnt_element.text() or "C",
                    "out": self.cnt_out.text().strip(),
                })
            elif mode == "solvate":
                if not self.slv_solute.text().strip():
                    QMessageBox.warning(self, "No solute", "Pick a solute file."); return
                args = {"solute": self.slv_solute.text().strip(),
                        "solvent": self.slv_solvent.currentText(),
                        "n_solvent": self.slv_n.value(),
                        "seed": 12345, "out": self.slv_out.text().strip()}
                if self.slv_density.value() > 0:
                    # UI is g/cm³; convert to kg/m³ for solvate()
                    args["density_kg_m3"] = self.slv_density.value() * 1000.0
                self._launch("solvate", args)
            elif mode == "layers":
                layers = []
                for r in range(self.lay_table.rowCount()):
                    src = self.lay_table.item(r, 0).text().strip()
                    kind = (self.lay_table.item(r, 1).text() or "file").strip()
                    name = self.lay_table.item(r, 2).text().strip() if self.lay_table.item(r, 2) else ""
                    sc_txt = self.lay_table.item(r, 3).text().strip() if self.lay_table.item(r, 3) else "1,1,1"
                    sc = [int(x) for x in sc_txt.split(",")] if sc_txt else [1, 1, 1]
                    gap_txt = self.lay_table.item(r, 4).text().strip() if self.lay_table.item(r, 4) else ""
                    gap = float(gap_txt) if gap_txt else None
                    off_txt = self.lay_table.item(r, 5).text().strip() if self.lay_table.item(r, 5) else "0,0"
                    off = [float(x) for x in off_txt.split(",")] if off_txt else [0.0, 0.0]
                    if not src: continue
                    layers.append({"src": src, "kind": kind, "name": name or f"layer_{r+1}",
                                   "supercell": (sc + [1, 1, 1])[:3],
                                   "offset": (off + [0.0, 0.0])[:2],
                                   "gap_before": gap})
                if not layers:
                    QMessageBox.warning(self, "No layers", "Add at least one layer."); return
                self._launch("layers", {
                    "layers": layers, "gap": self.lay_gap.value(),
                    "center": self.lay_center.isChecked(),
                    "out": self.lay_out.text().strip(),
                })
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _launch(self, task: str, args: dict):
        self.btn_build.setEnabled(False)
        self._thread = QThread(self)
        self._worker = CellWorker(task, args)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.log.emit)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(lambda: self.btn_build.setEnabled(True))
        self._thread.start()

    @pyqtSlot(dict)
    def _on_done(self, res):
        self.log.emit("=" * 60)
        self.log.emit(f"SYSTEM BUILT ({res['task']}): {res['atoms']} atoms -> {res['out']}")
        if "box_ang" in res:
            self.log.emit(f"  Box side: {res['box_ang']:.2f} Å")
        self.built_file.emit(res["out"])
        QMessageBox.information(self, "Done",
                               f"{res['task']} written to:\n{res['out']}")

    @pyqtSlot(str)
    def _on_fail(self, msg):
        self.log.emit(msg)
        QMessageBox.critical(self, "Failed", msg[:2000])
