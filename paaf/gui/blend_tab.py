"""Blend tab — multi-component polymer mixture packing.

Users add one row per species: a pre-built single-chain LAMMPS data file
(typically ``dlf_output1/lammps1.data`` from an earlier PAAF run) plus a
copy count. Clicking *Pack blend* invokes ``blend_replicator.replicate_blend``
and writes ``packed_blend.data`` + ``packed_blend.in`` into the chosen output
folder — the project's output folder by default, not wherever the first input
file happened to live.

This page is independent of the Step 1–6 pipeline flow; it consumes
already-typed single-chain data files and produces the merged system.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import tokens as T
from .page import Card, action_bar, button as _btn, caption, page_header, stat_row, wrap_tooltip


class BlendTab(QWidget):
    log = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._settings_provider = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, T.PAD_PAGE, T.PAD_PAGE)
        v.setSpacing(T.GAP_SECTION)

        v.addWidget(page_header(
            "Blend",
            "Pack two or more pre-built polymer chains into one amorphous cell. "
            "Type IDs are offset per component so they never collide, and all "
            "coefficient tables are merged.",
            breadcrumb="Tools › Blend",
        ))

        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(T.PAD_PAGE, 0, 0, 0)
        iv.setSpacing(T.GAP_SECTION)

        # ---- Components -------------------------------------------------
        card_comp = Card("Components", "single-chain exports, one row each")
        from PyQt5.QtWidgets import QComboBox
        self.engine = QComboBox()
        self.engine.addItems(["LAMMPS", "GROMACS"])
        self.engine.setToolTip(wrap_tooltip("Which MD engine the component files come from — and which the "
            "blend is written for. LAMMPS rows are .data + lammps.in; "
            "GROMACS rows are .gro + .top (with its .itp beside it)."))
        self.engine.currentIndexChanged.connect(self._on_engine_changed)
        card_comp.header.addWidget(QLabel("Engine:"))
        card_comp.header.addWidget(self.engine)
        self.comp_count_lb = caption("0 components")
        card_comp.header.addWidget(self.comp_count_lb)
        b_add = _btn("+ Add component"); b_add.clicked.connect(self._add_row)
        b_imp = _btn("Import from run…")
        b_imp.setToolTip(wrap_tooltip("Point at an output/<project> directory; every "
                         "lammps*.data (including in dlf_output1/) becomes a row."))
        b_imp.clicked.connect(self._import_from_run_dir)
        b_clr = _btn("Clear", "danger"); b_clr.clicked.connect(self._clear)
        for b in (b_add, b_imp, b_clr):
            card_comp.header.addWidget(b)

        # Column 2 is the lammps.in, shown explicitly rather than guessed.
        # The replicator has always NEEDED that file — the pair coefficients
        # and styles live in it, not in the .data — but it found it silently
        # beside the .data, so the user had no way to see which .in was used
        # or to point at a different one. Left empty it still auto-finds,
        # so nothing that worked before stops working.
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["name", "single-chain .data file", "lammps.in (styles/coeffs)",
             "chains", "choose files"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        # The buttons column takes what its buttons need. Left to the default
        # interactive width it squeezed three buttons into unlabelled stubs,
        # and the one row of controls the page depends on was unreadable.
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setFixedHeight(T.H_TABLE_HEADER)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(T.H_TABLE_ROW)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(140)
        card_comp.body.addWidget(self.table)
        iv.addWidget(card_comp)

        # ---- Packing + estimate side by side ---------------------------
        row2 = QHBoxLayout(); row2.setSpacing(T.GAP_SECTION)

        card_box = Card("Box & packing")
        f = QFormLayout()
        f.setContentsMargins(0, 0, 0, 0)
        f.setSpacing(T.GAP_FORM_ROW)
        f.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        card_box.body.addLayout(f)
        self.a = QDoubleSpinBox(); self.a.setRange(5, 5000); self.a.setValue(80.0)
        self.b = QDoubleSpinBox(); self.b.setRange(5, 5000); self.b.setValue(80.0)
        self.c = QDoubleSpinBox(); self.c.setRange(5, 5000); self.c.setValue(80.0)
        for sp in (self.a, self.b, self.c):
            sp.setSuffix(" Å")
            sp.valueChanged.connect(self._refresh_estimate)
        self.tol = QDoubleSpinBox(); self.tol.setRange(0.5, 10.0)
        self.tol.setValue(2.0); self.tol.setSuffix(" Å")
        self.seed = QSpinBox(); self.seed.setRange(-1, 2_000_000_000); self.seed.setValue(-1)
        self.seed.setToolTip(wrap_tooltip("-1 = fresh random seed each run (different packing each time)"))

        self.use_density = QCheckBox("Size the box from a target density")
        self.use_density.setChecked(True)
        self.use_density.setToolTip(wrap_tooltip("A box chosen by hand is almost always far too big. Packing then "
            "finishes instantly and leaves the chains floating in vacuum — "
            "the cell has the right contents but none of the contacts that "
            "make it a blend."))
        self.use_density.toggled.connect(self._refresh_estimate)
        card_box.body.addWidget(self.use_density)

        self.density = QDoubleSpinBox()
        self.density.setRange(0.05, 3.0); self.density.setValue(1.00)
        self.density.setSingleStep(0.05); self.density.setSuffix(" g/cm³")
        self.density.setToolTip(wrap_tooltip("Bulk density of the finished material."))
        self.density.valueChanged.connect(self._refresh_estimate)
        self.pack_fraction = QDoubleSpinBox()
        self.pack_fraction.setRange(10.0, 100.0); self.pack_fraction.setValue(40.0)
        self.pack_fraction.setSingleStep(5.0); self.pack_fraction.setSuffix(" %")
        self.pack_fraction.setToolTip(wrap_tooltip("Pack at this fraction of the target density, then compress the "
            "rest with NPT. Packing straight to bulk density usually fails: "
            "packmol cannot thread chains past one another with no room to "
            "move. 30–50 % is the usual compromise."))
        self.pack_fraction.valueChanged.connect(self._refresh_estimate)

        f.addRow("Target density", self.density)
        f.addRow("Pack at", self.pack_fraction)
        f.addRow("Edge a (x)", self.a)
        f.addRow("Edge b (y)", self.b)
        f.addRow("Edge c (z)", self.c)
        f.addRow("Tolerance", self.tol)
        f.addRow("Seed", self.seed)
        self._box_form = f
        row2.addWidget(card_box, 1)

        card_est = Card("Estimated system")
        card_est.setFixedWidth(300)
        self.est_components = stat_row("Components", "0")
        self.est_chains = stat_row("Total chains", "0")
        self.est_volume = stat_row("Cell volume", "512 000 Å³")
        self.est_density = stat_row("Packed density", "—")
        for wdg in (self.est_components, self.est_chains, self.est_volume,
                    self.est_density):
            card_est.body.addWidget(wdg)
        self.density_note = caption(
            "Add components to see the density this box implies.")
        self.density_note.setWordWrap(True)
        card_est.body.addWidget(self.density_note)
        card_est.body.addStretch(1)
        row2.addWidget(card_est)
        iv.addLayout(row2)

        # ---- Relax each component ---------------------------------------
        card_min = Card("Before packing",
                        "relax each chain once, not each copy")
        d_min = caption(
            "Packing only moves whole chains — it never changes their "
            "internal geometry. So whatever strain a chain arrives with is "
            "inherited by every copy of it. Minimising once per component "
            "fixes it for all of them, and costs one short LAMMPS run on a "
            "single chain.")
        d_min.setWordWrap(True)
        card_min.body.addWidget(d_min)

        self.do_min = QCheckBox("Energy-minimise each component first")
        self.do_min.setChecked(True)
        self.do_min.setToolTip(wrap_tooltip("Needs LAMMPS, and a .in file beside each component's .data so "
            "the real force field is used. Without either, the component is "
            "packed as-is and the log says so."))
        card_min.body.addWidget(self.do_min)

        self.min_found = caption("")
        self.min_found.setWordWrap(True)
        card_min.body.addWidget(self.min_found)
        iv.addWidget(card_min)

        # ---- Output -----------------------------------------------------
        card_out = Card("Output")
        f3 = QFormLayout()
        f3.setContentsMargins(0, 0, 0, 0)
        f3.setSpacing(T.GAP_FORM_ROW)
        f3.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        card_out.body.addLayout(f3)
        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText(
            "project output folder — leave blank to use it")
        self.out_dir.setToolTip(wrap_tooltip("Where the packed files are written. Blank uses the project's "
            "output folder (File > Project settings). Previously the files "
            "landed silently beside whichever component you added first."))
        self.out_dir.textChanged.connect(lambda _t: self._refresh_out_preview())
        b_browse = _btn("Browse…")
        b_browse.clicked.connect(self._browse_out_dir)
        dir_row = QHBoxLayout()
        dir_row.setSpacing(T.GAP_LABEL)
        dir_row.addWidget(self.out_dir, 1)
        dir_row.addWidget(b_browse)
        dir_wrap = QWidget()
        dir_wrap.setStyleSheet("background: transparent; border: none;")
        dir_wrap.setLayout(dir_row)
        f3.addRow("Output folder", dir_wrap)

        self.out_data = QLineEdit("packed_blend.data")
        self.out_in   = QLineEdit("packed_blend.in")
        self.out_data.textChanged.connect(lambda _t: self._refresh_out_preview())
        self.out_in.textChanged.connect(lambda _t: self._refresh_out_preview())
        self.write_in = QCheckBox("Also write a merged LAMMPS input (.in) with offset pair_coeff commands")
        self.write_in.setChecked(True)
        self.write_in.toggled.connect(lambda _b: self._refresh_out_preview())
        f3.addRow("Data file", self.out_data)
        f3.addRow("Input file", self.out_in)
        f3.addRow("", self.write_in)

        self.out_preview = caption("")
        self.out_preview.setWordWrap(True)
        card_out.body.addWidget(self.out_preview)
        iv.addWidget(card_out)

        self._refresh_out_preview()
        self._refresh_min_status()

        iv.addStretch(1)
        v.addWidget(inner, 1)

        # ---- Action bar: exactly one primary, right-most -----------------
        bar_wrap = QWidget()
        bw = QHBoxLayout(bar_wrap)
        bw.setContentsMargins(T.PAD_PAGE, 0, 0, 0)
        self.pack_btn = _btn("Pack blend", "primary")
        self.pack_btn.clicked.connect(self._pack)
        self._bar = action_bar("Add at least two components to pack.", [self.pack_btn])
        bw.addWidget(self._bar)
        v.addWidget(bar_wrap)

        # Start with two empty rows to make the shape obvious.
        self._add_row(); self._add_row()
        self._refresh_estimate()

    # -------------------------------------------------- row helpers
    def _refresh_estimate(self) -> None:
        """Keep the 'Estimated system' card and the action bar in step."""
        n_comp = 0
        n_chain = 0
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 1)
            if it and it.text().strip():
                n_comp += 1
                w = self.table.cellWidget(r, 3)
                n_chain += int(w.value()) if w else 0
        self._set_stat(self.est_components, str(n_comp))
        self._set_stat(self.est_chains, str(n_chain))

        on = self.use_density.isChecked()
        self.density.setEnabled(on)
        self.pack_fraction.setEnabled(on)
        for sp in (self.a, self.b, self.c):
            sp.setEnabled(not on)
        total_amu = self._total_mass_amu()
        if on and total_amu > 0:
            edge = self._edge_for(total_amu,
                                  self.density.value()
                                  * self.pack_fraction.value() / 100.0)
            for sp in (self.a, self.b, self.c):
                sp.blockSignals(True); sp.setValue(edge); sp.blockSignals(False)

        vol = self.a.value() * self.b.value() * self.c.value()
        self._set_stat(self.est_volume, f"{vol:,.0f} Å³".replace(",", " "))
        if total_amu > 0 and vol > 0:
            rho = total_amu / 6.02214076e23 / (vol * 1.0e-24)
            self._set_stat(self.est_density, f"{rho:.3f} g/cm³")
            self.density_note.setText(
                f"Packing at {rho:.2f} g/cm³; compress to "
                f"{self.density.value():.2f} g/cm³ with NPT."
                if on else
                f"This box packs at {rho:.4f} g/cm³. "
                + ("Far below any condensed phase — the chains will not "
                   "touch, so this is not yet a blend."
                   if rho < 0.10 else
                   "Compress with NPT before measuring anything."))
        else:
            self._set_stat(self.est_density, "—")
            self.density_note.setText(
                "Add components to see the density this box implies.")
        self.comp_count_lb.setText(
            f"{n_comp} component{'s' if n_comp != 1 else ''}")
        ready = n_comp >= 2
        self.pack_btn.setEnabled(ready)
        self._bar._blocker.setText(
            "" if ready else "Add at least two components to pack.")

    # ------------------------------------------------------------ density
    def _total_mass_amu(self) -> float:
        """Mass of everything that would be packed, in amu.

        Parsing a data file per keystroke would be wasteful, so each file's
        per-chain mass is cached against its path and modification time.
        """
        from ..blend_replicator import BlendComponent, component_mass_amu
        from ..lammps_replicator import parse_lammps_data

        if not hasattr(self, "_mass_cache"):
            self._mass_cache = {}
        total = 0.0
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 1)
            if not it or not it.text().strip():
                continue
            path = Path(it.text().strip())
            w = self.table.cellWidget(r, 3)
            count = int(w.value()) if w else 0
            try:
                key = (str(path), path.stat().st_mtime)
            except OSError:
                continue
            if key not in self._mass_cache:
                # In GROMACS mode the row is a .gro, which carries no masses;
                # they are in the topology's [atoms] table. Parsing the .gro
                # as a LAMMPS data file returned mass 0, so "size the box
                # from a target density" silently never fired and every
                # GROMACS blend packed into the 80 Å default — the same box
                # at 40 % and at 100 %, which is how the setting appeared to
                # do nothing.
                try:
                    if self._is_gromacs():
                        self._mass_cache[key] = self._gromacs_chain_mass(r, path)
                    else:
                        header, sections = parse_lammps_data(path)
                        comp = BlendComponent(name="", data_file=path, count=1)
                        comp.header, comp.sections = header, sections
                        self._mass_cache[key] = component_mass_amu(comp)
                except Exception:
                    self._mass_cache[key] = 0.0
            total += self._mass_cache[key] * count
        return total

    def _gromacs_chain_mass(self, row: int, gro: Path) -> float:
        """One chain's mass in amu, summed from the topology's [atoms] table.

        The mass is the 8th field of each atom line. The topology comes from
        the row's own column when filled, else the same lookup the auto-fill
        uses (gromacs.top beside the .gro, then dlf_output1/).
        """
        top = None
        it = self.table.item(row, 2)
        if it and it.text().strip():
            top = Path(it.text().strip())
        if top is None or not top.exists():
            for cand in (gro.parent / "gromacs.top",
                         gro.parent / "dlf_output1" / "gromacs.top"):
                if cand.exists():
                    top = cand
                    break
        if top is None or not top.exists():
            return 0.0

        from ..gromacs_blend import parse_top
        sections = parse_top(top)
        texts = [top.read_text(errors="replace")]
        for inc in sections.get("#include", []):
            inc_path = top.parent / inc
            if inc_path.exists():
                texts.append(inc_path.read_text(errors="replace"))
        mass = 0.0
        for text in texts:
            in_atoms = False
            for raw in text.splitlines():
                line = raw.strip()
                if line.startswith("["):
                    in_atoms = line.replace(" ", "") == "[atoms]"
                    continue
                if in_atoms and line and not line.startswith(";"):
                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            mass += float(parts[7])
                        except ValueError:
                            pass
        return mass

    @staticmethod
    def _edge_for(total_amu: float, density_g_cm3: float) -> float:
        """Cubic edge in Å holding ``total_amu`` at the given density."""
        if density_g_cm3 <= 0 or total_amu <= 0:
            return 0.0
        volume_cm3 = total_amu / (6.02214076e23 * density_g_cm3)
        return (volume_cm3 * 1.0e24) ** (1.0 / 3.0)

    @staticmethod
    def _set_stat(row_widget, value: str) -> None:
        labels = row_widget.findChildren(QLabel)
        if labels:
            labels[-1].setText(str(value))

    @staticmethod
    def _path_item(text: str) -> QTableWidgetItem:
        """A read-only cell for a file path.

        The default cell is editable, so double-clicking a path opened a text
        cursor and invited typing — but these paths are meant to come from
        the Choose buttons, where the file demonstrably exists. A hand-typed
        path fails later and further from the mistake. The full path stays
        readable in the tooltip when the column is narrower than the text.
        """
        it = QTableWidgetItem(text)
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        it.setToolTip(wrap_tooltip(text))
        return it

    def _add_row(self) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(f"component_{r + 1}"))
        self.table.setItem(r, 1, self._path_item(""))
        count = QSpinBox(); count.setRange(1, 100_000); count.setValue(10)
        self.table.setItem(r, 2, self._path_item(""))
        self.table.setCellWidget(r, 3, count)
        rowbtns = QWidget(); h = QHBoxLayout(rowbtns)
        h.setContentsMargins(4, 0, 4, 0); h.setSpacing(4)
        browse = QPushButton("Choose structure…")
        browse.setToolTip(wrap_tooltip("The single-chain coordinates: lammps1.data "
                          "(LAMMPS) or gromacs.gro (GROMACS), from that "
                          "component's export folder."))
        browse.setMinimumWidth(130)
        browse_in = QPushButton("Choose input/top…")
        browse_in.setToolTip(wrap_tooltip("The matching force-field file: lammps.in or "
                             ".top. Optional — left empty, the one beside "
                             "the structure file is used and shown here."))
        browse_in.setMinimumWidth(130)
        remove = QPushButton("Remove")
        browse.clicked.connect(lambda _=False, row=r: self._browse_for(row))
        browse_in.clicked.connect(lambda _=False, row=r: self._browse_in_for(row))
        remove.clicked.connect(lambda _=False, row=r: self._remove(row))
        h.addWidget(browse); h.addWidget(browse_in); h.addWidget(remove)
        self.table.setCellWidget(r, 4, rowbtns)
        count.valueChanged.connect(self._refresh_estimate)
        self._refresh_estimate()

    def _unique_name(self, base: str, skip_row: int) -> str:
        """``base``, or ``base_2`` … if another row already claims it."""
        taken = set()
        for r in range(self.table.rowCount()):
            if r == skip_row:
                continue
            it = self.table.item(r, 0)
            if it and it.text().strip():
                taken.add(it.text().strip())
        if base not in taken:
            return base
        n = 2
        while f"{base}_{n}" in taken:
            n += 1
        return f"{base}_{n}"

    def _autofill_in(self, row: int, data_path: Path) -> None:
        """Show the lammps.in that will be used, instead of using it silently.

        Only fills an EMPTY cell: a path the user chose by hand is theirs.
        """
        it = self.table.item(row, 2)
        if it is not None and it.text().strip():
            return
        found = None
        if self._is_gromacs():
            # The TYPED topology, not the skeleton. A PAAF project root holds
            # a template .top with placeholder types (atom "C", charge 0,
            # mass 0) and the force field left as a commented-out include —
            # picking that up produced "has no #include for its moleculetype"
            # on a component whose real topology sat one folder down.
            for cand in (data_path.parent / "gromacs.top",
                         data_path.parent / "dlf_output1" / "gromacs.top",
                         data_path.with_suffix(".top")):
                if cand.exists():
                    found = cand
                    break
        else:
            try:
                from ..blend_styles import find_forcefield_input
                found = find_forcefield_input(data_path)
            except Exception:
                found = None
        self.table.setItem(row, 2, self._path_item(str(found) if found else ""))

    def _is_gromacs(self) -> bool:
        return getattr(self, "engine", None) is not None and \
            self.engine.currentText() == "GROMACS"

    def _on_engine_changed(self, *_a) -> None:
        """Same table, same flow — only the file kinds change."""
        gmx = self._is_gromacs()
        self.table.setHorizontalHeaderLabels(
            ["name",
             "single-chain .gro file" if gmx else "single-chain .data file",
             ".top (topology)" if gmx else "lammps.in (styles/coeffs)",
             "chains", ""])
        self.comp_count_lb.setText(self.comp_count_lb.text())

    def _browse_in_for(self, row: int) -> None:
        if self._is_gromacs():
            p, _ = QFileDialog.getOpenFileName(
                self, "Select this component's topology", "",
                "GROMACS topology (*.top);;All files (*)")
        else:
            p, _ = QFileDialog.getOpenFileName(
                self, "Select this component's lammps.in", "",
                "LAMMPS input (*.in lammps*.in);;All files (*)")
        if p:
            self.table.setItem(row, 2, self._path_item(p))

    def _browse_for(self, row: int) -> None:
        if self._is_gromacs():
            p, _ = QFileDialog.getOpenFileName(
                self, "Select single-chain .gro file", "",
                "GROMACS coordinates (*.gro);;All files (*)")
        else:
            p, _ = QFileDialog.getOpenFileName(
                self, "Select single-chain LAMMPS data file", "",
                "LAMMPS data (*.data lammps*.data);;All files (*)")
        if p:
            self.table.setItem(row, 1, self._path_item(p))
            self._autofill_in(row, Path(p))
            # If the name column is still auto-generated, name it after the
            # file. DL_FIELD calls every one of them "lammps1", so a stem is
            # only useful when it distinguishes anything — otherwise the
            # folder does, and two components sharing a name would collide in
            # the log and in the generated group ids.
            item = self.table.item(row, 0)
            if item is None or item.text().startswith("component_"):
                self.table.setItem(row, 0, QTableWidgetItem(
                    self._unique_name(_suggest_name(Path(p)), row)))
            self._refresh_estimate()

    def _remove(self, row: int) -> None:
        # QTableWidget rows can shift after removal; find the current index by widget.
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 4) is self.sender().parent():
                self.table.removeRow(r); self._refresh_estimate(); return
        # Fallback: remove by supplied index if bounds are still valid.
        if 0 <= row < self.table.rowCount():
            self.table.removeRow(row)
        self._refresh_estimate()

    def _clear(self) -> None:
        self.table.setRowCount(0)
        self._refresh_estimate()

    def _import_from_run_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select a PAAF run directory (contains lammps*.data)")
        if not d:
            return
        root = Path(d)
        # Look one level down for dlf_output1/ style output too, and dedupe.
        candidates = list(root.glob("lammps*.data")) + list(root.glob("*/lammps*.data"))
        seen: List[Path] = []
        for p in candidates:
            if p not in seen:
                seen.append(p)
        if not seen:
            QMessageBox.information(self, "Import",
                f"No lammps*.data files found under:\n{root}")
            return
        for p in seen:
            self._add_row()
            r = self.table.rowCount() - 1
            self.table.setItem(r, 0, QTableWidgetItem(p.parent.name or p.stem))
            self.table.setItem(r, 1, self._path_item(str(p)))
            self._autofill_in(r, p)
        self._refresh_estimate()
        self.log.emit(f"[blend] Imported {len(seen)} single-chain data file(s) from {root}")

    # -------------------------------------------------- collect + pack
    def _collect_components(self) -> list:
        from ..blend_replicator import BlendComponent
        out: List[BlendComponent] = []
        for r in range(self.table.rowCount()):
            name_it = self.table.item(r, 0)
            path_it = self.table.item(r, 1)
            in_it = self.table.item(r, 2)
            count_w = self.table.cellWidget(r, 3)
            if not path_it or not path_it.text().strip():
                continue
            p = Path(path_it.text().strip())
            if not p.exists():
                raise FileNotFoundError(f"Row {r + 1}: file not found: {p}")
            name = (name_it.text().strip() if name_it else "") or p.stem
            n = int(count_w.value()) if count_w else 1
            comp = BlendComponent(name=name, data_file=p, count=n)
            # An explicitly chosen lammps.in wins over the auto-found one.
            in_text = (in_it.text().strip() if in_it else "")
            if in_text:
                in_path = Path(in_text)
                if not in_path.exists():
                    raise FileNotFoundError(
                        f"Row {r + 1}: lammps.in not found: {in_path}")
                comp.forcefield_input = in_path
            out.append(comp)
        return out



    def _refresh_min_status(self) -> None:
        """Say up front whether minimisation can actually run."""
        if not hasattr(self, "min_found"):
            return
        try:
            from ..cell.relax import find_lammps
            exe = find_lammps()
        except Exception:
            exe = None
        if exe is None:
            self.min_found.setText(
                "No LAMMPS executable found — components will be packed with "
                "their input geometry, and the log will say so.")
            self.min_found.setStyleSheet(
                f"color: {T.WARN_TEXT}; font-size: {T.FS_CAPTION}px;"
                f" background: transparent; border: none;")
        else:
            self.min_found.setText(f"LAMMPS: {exe}")
            self.min_found.setStyleSheet(
                f"color: {T.TEXT_MUTED}; font-size: {T.FS_CAPTION}px;"
                f" background: transparent; border: none;")

    # ------------------------------------------------------- output folder
    def set_settings_provider(self, fn) -> None:
        """Install a callable returning the project's output folder.

        Supplied explicitly by the main window rather than reached for via
        ``self.window()`` — walking the parent chain fails silently when the
        widget is reparented, and the failure mode is writing to the wrong
        place without saying so.
        """
        self._settings_provider = fn
        if not self.out_dir.text().strip():
            self.out_dir.setText(self.default_out_dir())
        self._refresh_out_preview()

    def default_out_dir(self) -> str:
        """``<project output>/<project>/blends``, or "" if unknown."""
        fn = getattr(self, "_settings_provider", None)
        if fn is None:
            return ""
        try:
            got = fn() or {}
        except Exception:
            return ""
        base = str(got.get("out_dir", "") or "")
        if not base:
            return ""
        project = str(got.get("project", "") or "").strip()
        return str(Path(base) / project / "blends" if project
                   else Path(base) / "blends")

    def resolved_out_dir(self, components) -> Path:
        """Where the packed files actually go.

        Order: what the user typed, then the project folder, then — only as a
        last resort — beside the first component. That last case used to be
        the ONLY behaviour, which is why outputs kept appearing in whichever
        directory the first input happened to live in.
        """
        typed = self.out_dir.text().strip()
        if typed:
            return Path(typed).expanduser()
        default = self.default_out_dir()
        if default:
            return Path(default)
        if not components:
            raise RuntimeError("Choose an output folder first — there is no "
                               "project folder to default to.")
        return Path(components[0].data_file).parent

    def _browse_out_dir(self) -> None:
        start = self.out_dir.text().strip() or str(Path.home())
        d = QFileDialog.getExistingDirectory(self, "Output folder", start)
        if d:
            self.out_dir.setText(d)

    def _refresh_out_preview(self) -> None:
        if not hasattr(self, "out_preview"):
            return
        typed = self.out_dir.text().strip()
        base = typed or self.default_out_dir()
        data = self.out_data.text().strip() or "packed_blend.data"
        inp = self.out_in.text().strip() or "packed_blend.in"
        files = data + (f" and {inp}" if self.write_in.isChecked() else "")
        if base:
            where = str(Path(base))
            src = "as typed" if typed else "the project output folder"
            self.out_preview.setText(f"Writes {files} to {where}  ({src}).")
        else:
            self.out_preview.setText(
                f"Writes {files} beside the first component's data file — no "
                f"output folder is set and no project folder is available. "
                f"Set one above, or via File > Project settings.")

    def _pack_gromacs(self, out_dir: Path) -> None:
        """The GROMACS route: .gro + .top per row, gmx does the packing."""
        from ..gromacs_blend import (GromacsBlendComponent,
                                     replicate_gromacs_blend)
        comps = []
        for r in range(self.table.rowCount()):
            path_it = self.table.item(r, 1)
            if not path_it or not path_it.text().strip():
                continue
            gro = Path(path_it.text().strip())
            if not gro.exists():
                QMessageBox.warning(self, "Blend",
                                    f"Row {r + 1}: file not found: {gro}")
                return
            name_it = self.table.item(r, 0)
            in_it = self.table.item(r, 2)
            count_w = self.table.cellWidget(r, 3)
            comp = GromacsBlendComponent(
                name=(name_it.text().strip() if name_it else "") or gro.stem,
                gro_file=gro,
                count=int(count_w.value()) if count_w else 1)
            if in_it and in_it.text().strip():
                comp.top_file = Path(in_it.text().strip())
            comps.append(comp)
        if len(comps) < 2:
            QMessageBox.warning(self, "Blend",
                "Add at least two components (with valid .gro paths).")
            return
        self.log.emit(f"[blend] GROMACS: packing {len(comps)} components "
                      f"into {self.a.value():.1f}×{self.b.value():.1f}"
                      f"×{self.c.value():.1f} Å")
        try:
            gro, top = replicate_gromacs_blend(
                comps, (self.a.value(), self.b.value(), self.c.value()),
                out_dir, seed=int(self.seed.value()))
        except Exception as e:
            QMessageBox.critical(self, "Blend", str(e))
            self.log.emit(f"[blend] FAILED: {e}")
            return
        self.log.emit(f"[blend] Wrote {gro} and {top}")
        QMessageBox.information(
            self, "Blend",
            f"Packed blend written:\n{gro}\n{top}\n\n"
            f"Run it with grompp/mdrun; equilibrate (NPT) before measuring.")

    def _pack(self) -> None:
        if self._is_gromacs():
            components = None
            try:
                out_dir = self.resolved_out_dir([])
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Blend", str(e)); return
            self._pack_gromacs(out_dir)
            return
        try:
            components = self._collect_components()
        except Exception as e:
            QMessageBox.warning(self, "Blend", str(e)); return
        if len(components) < 2:
            QMessageBox.warning(self, "Blend",
                "Add at least two components (with valid .data paths) before packing.")
            return
        try:
            from ..blend_replicator import replicate_blend
        except Exception as e:
            QMessageBox.critical(self, "Blend", f"Import error: {e}"); return

        out_dir = self.resolved_out_dir(components)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Blend",
                                 f"Cannot write to {out_dir}:\n{e}")
            return
        out_data = out_dir / (self.out_data.text().strip() or "packed_blend.data")
        out_in = (out_dir / (self.out_in.text().strip() or "packed_blend.in")
                  if self.write_in.isChecked() else None)
        self.log.emit(f"[blend] Packing {len(components)} components into "
                      f"{self.a.value():.1f}×{self.b.value():.1f}×{self.c.value():.1f} Å")
        from ..blend_minimise import MinimiseSettings
        try:
            path = replicate_blend(
                components,
                box_edges=(self.a.value(), self.b.value(), self.c.value()),
                out_data_file=out_data,
                out_input_file=out_in,
                seed=int(self.seed.value()),
                tolerance=float(self.tol.value()),
                minimise=MinimiseSettings(enabled=self.do_min.isChecked()),
                max_bond_length=3.0,
                progress=lambda m: self.log.emit(f"[blend] {m}"),
            )
        except Exception as e:
            QMessageBox.critical(self, "Blend", f"Pack failed:\n{e}")
            self.log.emit(f"[blend] ERROR: {e}")
            return
        self.log.emit(f"[blend] Wrote {path}")
        if out_in is not None:
            self.log.emit(f"[blend] Wrote {out_in}")
        QMessageBox.information(self, "Blend",
            f"Wrote {path.name}"
            + (f" and {out_in.name}" if out_in else "")
            + f"\n\nin {out_dir}")


# ---------------------------------------------------------------- naming
#: Stems that every DL_FIELD or moltemplate run produces, so they identify a
#: pipeline rather than a material. When one of these turns up, the folder is
#: the informative part of the path.
_GENERIC_STEMS = {"lammps", "lammps1", "system", "data", "in", "output",
                  "single_chain", "chain", "polymer"}


def _suggest_name(path: Path) -> str:
    """A human label for a component, from its path.

    ``mta_output/pbs/lammps1.data`` -> ``pbs``, because every component
    DL_FIELD writes is called ``lammps1`` and two rows sharing a name collide
    in the log and in the generated LAMMPS group ids.
    """
    stem = path.stem
    if stem.lower() in _GENERIC_STEMS or stem.lower().rstrip("0123456789") in _GENERIC_STEMS:
        parent = path.parent.name
        if parent.lower() in {"dlf_output1", "dlf_output", "output"}:
            parent = path.parent.parent.name
        if parent:
            return parent
    return stem
