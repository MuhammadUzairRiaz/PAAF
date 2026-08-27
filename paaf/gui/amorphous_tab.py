"""Amorphous builder — pack polymer chains into a periodic cell.

A self-contained branch with its own three steps, in the same shape as the
Reaction-scheme tab:

    1 · Composition   what to build, how much of each, at what density
    2 · Force field   how the cell should be typed
    3 · Relax         soft push-off, then LAMMPS minimisation
    4 · Export        where the files go, and what they will be

It is deliberately independent of the main pipeline. The pipeline builds one
chain and types it; this builds a many-chain periodic cell by
Theodorou–Suter growth and types that. Sharing the pipeline's Force-field page
would mean a user could not have a cell in OPLS-AA and a reference structure
in PCFF at the same time.

Composition, two ways
---------------------
The mode switch is not decoration. "Solve from weight %" is what a
formulation is actually specified as, but chain counts are integers, so the
requested composition is generally not attainable and the realised one is
shown next to it. "Enter chain counts" is for when you already know the cell
you want and would rather PAAF did not round anything.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QBrush, QColor, QPainter
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QRadioButton, QScrollArea, QSpinBox, QStyle,
    QStyleOption, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from . import tokens as T
from .page import Card, badge, button, caption, label, stat_row, wrap_tooltip


# ===================================================================== worker
class _BuildWorker(QObject):
    """Growth + back-mapping + typing, off the UI thread.

    Growth can take minutes on a large cell, and it must stay cancellable —
    a frozen window during a long build is the single most common reason a
    user kills the whole application and loses their settings.
    """
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)
    fraction = pyqtSignal(float)

    cancelled = pyqtSignal()

    def __init__(self, composition, options: dict):
        super().__init__()
        self.composition = composition
        self.options = options
        # Built HERE, on the GUI thread, not inside run(). Creating it in
        # run() leaves a window between thread.start() and the first line of
        # run() in which Cancel does nothing while telling the user it did.
        from ..cell.packing import CancelToken
        self._cancel = CancelToken()

    def cancel(self) -> None:
        self._cancel.cancel()

    @pyqtSlot()
    def run(self):
        from ..cell.packing import PackCancelled
        try:
            from ..cell.grow import grow_amorphous_cell
            from ..cell.cell_export import export_cell

            o = self.options
            specs = self.composition.grow_specs()

            def on_progress(p):
                msg = getattr(p, "message", "") or ""
                self.progress.emit(msg)
                frac = getattr(p, "fraction", None)
                if frac is not None:
                    self.fraction.emit(float(frac))

            self.progress.emit(
                f"growing {self.composition.total_chains} chains "
                f"({self.composition.total_beads} skeletal beads) …")

            # Grow, then CHECK FOR RING THREADING, and regrow on a fresh seed
            # while any is found.
            #
            # Growth rejects spearing at bead level, where there are no rings
            # yet; the rings only exist after back-mapping, so a bond through
            # a phenyl shows up later — measured, its carbons sit at aromatic
            # bond distance (1.41 A) from ring carbons, the typer bonds them,
            # and the cell cannot be typed. No push-off can fix interlocked
            # topology; a different seed can. The probe is a dress-only
            # back-map (no push-off), which costs well under a second.
            attempts = 8 if o.get("do_export") else 1
            result = None
            best = None                       # (n_speared, grown result)
            for attempt in range(attempts):
                grown = grow_amorphous_cell(
                    specs, self.composition.box(),
                    temperature=o["temperature"],
                    scan_depth=o["scan_depth"],
                    tolerance=o["tolerance"],
                    seed=o["seed"] + attempt,
                    check_spearing=o["check_spearing"],
                    progress=on_progress,
                    cancel=self._cancel,
                )
                result = grown
                if attempts == 1:
                    break
                try:
                    import numpy as _np
                    from ..cell.backmap import backmap_cell as _bm
                    self.progress.emit(
                        "checking the grown cell for ring threading "
                        "(probe back-map — silent but busy; minutes on "
                        "large cells) …")
                    probe = _bm(grown, specs, tacticity=o["tacticity"],
                                push_off=False)
                    n_speared = getattr(probe, "n_speared", None)
                    if n_speared is None:
                        n_speared = 0 if any(
                            "spearing check: none" in n
                            for n in probe.notes) else 1
                except Exception:
                    break                      # probe failed: use this cell
                if best is None or n_speared < best[0]:
                    best = (n_speared, grown)
                if n_speared == 0:
                    if attempt:
                        self.progress.emit(
                            f"seed {o['seed'] + attempt}: no ring threading "
                            f"({attempt} regrow(s) needed)")
                    break
                if attempt < attempts - 1:
                    self.progress.emit(
                        f"seed {o['seed'] + attempt}: {n_speared} bond(s) "
                        f"thread a ring — regrowing with seed "
                        f"{o['seed'] + attempt + 1}")
                else:
                    # Every seed threaded. Keep the LEAST-threaded cell, not
                    # whichever happened to come last, and say so.
                    result = best[1]
                    self.progress.emit(
                        f"all {attempts} seeds thread at least one ring; "
                        f"keeping the best ({best[0]} threaded bond(s)). "
                        f"Lower 'Build at' on the Composition step for a "
                        f"clean cell.")
            self.fraction.emit(1.0)

            export = None
            if o.get("do_export"):
                self.progress.emit("exporting …")
                rs = None
                if o.get("relax"):
                    from ..cell.relax import RelaxSettings
                    rs = RelaxSettings(
                        push_steps=o["push_steps"],
                        maxiter=o["min_iter"],
                        engine=o.get("engine", "auto"),
                        lammps_exe=o["lammps_exe"],
                        gmx_exe=o.get("gmx_exe", ""),
                        mpi_ranks=o["mpi_ranks"])
                def _emit_checked(msg: str) -> None:
                    # A progress tick is also a cancellation point: raising
                    # here aborts the export at the next stage boundary,
                    # and the external LAMMPS/dl_field runs are killed
                    # directly via the same token.
                    if self._cancel.is_cancelled():
                        raise PackCancelled("cancelled")
                    self.progress.emit(msg)

                export = export_cell(
                    result, specs, o["out_dir"], name=o["name"],
                    ff_key=o["ff_key"], dl_field_dir=o["dl_lib"] or None,
                    atomistic=o["atomistic"], tacticity=o["tacticity"],
                    run_typing=o["run_typing"], push_off=o["push_off"],
                    relax=bool(o.get("relax")), relax_settings=rs,
                    output_formats=o.get("output_formats", "lammps"),
                    progress=_emit_checked, cancel=self._cancel)
            self.finished.emit((result, export))
        except PackCancelled:
            # A deliberate cancel is not a failure and must not raise an
            # error dialog at someone who just pressed the Cancel button.
            self.progress.emit("build cancelled")
            self.cancelled.emit()
        except Exception as exc:
            # The message and the traceback go to different places. PackFailed
            # carries several paragraphs of specific, numeric advice, and it
            # is the only part worth showing the user; the traceback belongs
            # in the log, where it can be read if the failure is a real bug.
            self.progress.emit(traceback.format_exc())
            self.failed.emit(str(exc) or exc.__class__.__name__)


# ============================================================ component row
class ComponentRow(QWidget):
    """One polymer in the blend."""

    changed = pyqtSignal()
    weightEdited = pyqtSignal(object)
    removeRequested = pyqtSignal(object)

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self._name = f"componentRow{index}"
        self.setObjectName(self._name)
        self.setStyleSheet(
            f"QWidget#{self._name} {{ background: {T.BG_SUNKEN};"
            f" border: 1px solid {T.BORDER};"
            f" border-radius: {T.R_CONTROL}px; }}")

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(T.GAP_LABEL)

        self.tag = label(f"{index}", T.FS_CAPTION, T.TEXT_MUTED, bold=True)
        self.tag.setFixedWidth(14)
        row.addWidget(self.tag)

        self.name = QLineEdit()
        self.name.setPlaceholderText("name")
        self.name.setFixedWidth(90)
        self.name.setFixedHeight(T.H_CONTROL)
        self.name.textChanged.connect(lambda _t: self.changed.emit())
        row.addWidget(self.name)

        self.smiles = QLineEdit()
        self.smiles.setPlaceholderText("polymerisation SMILES, e.g. [*]CC[*]")
        self.smiles.setFixedHeight(T.H_CONTROL)
        self.smiles.textChanged.connect(self._on_smiles)
        row.addWidget(self.smiles, 1)

        self.b_lib = button("Library…")
        self.b_lib.setToolTip(wrap_tooltip("Pick a polymer from the PAAF library"))
        self.b_lib.clicked.connect(self._pick_library)
        row.addWidget(self.b_lib)

        row.addWidget(label("DP", T.FS_CAPTION, T.TEXT_MUTED))
        self.dp = QSpinBox()
        self.dp.setRange(2, 100_000)
        self.dp.setValue(50)
        self.dp.setFixedHeight(T.H_CONTROL)
        self.dp.setFixedWidth(72)
        self.dp.setToolTip(wrap_tooltip("Repeat units per chain"))
        self.dp.valueChanged.connect(lambda _v: self.changed.emit())
        row.addWidget(self.dp)

        self.wt_label = label("wt%", T.FS_CAPTION, T.TEXT_MUTED)
        row.addWidget(self.wt_label)
        self.wt = QDoubleSpinBox()
        self.wt.setRange(0.0, 100.0)
        self.wt.setDecimals(2)
        self.wt.setValue(50.0)
        self.wt.setFixedHeight(T.H_CONTROL)
        self.wt.setFixedWidth(78)
        self.wt.valueChanged.connect(self._on_wt)
        row.addWidget(self.wt)

        self.n_label = label("chains", T.FS_CAPTION, T.TEXT_MUTED)
        row.addWidget(self.n_label)
        self.n_chains = QSpinBox()
        self.n_chains.setRange(0, 100_000)
        self.n_chains.setValue(4)
        self.n_chains.setFixedHeight(T.H_CONTROL)
        self.n_chains.setFixedWidth(72)
        self.n_chains.valueChanged.connect(lambda _v: self.changed.emit())
        row.addWidget(self.n_chains)
        self.dp.setToolTip(wrap_tooltip("Repeat units per chain. Changing it changes this polymer's chain "
            "mass, and therefore the weight percent a given number of chains "
            "comes to."))

        self.info = label("", T.FS_CAPTION, T.TEXT_MUTED, mono=True)
        self.info.setFixedWidth(168)
        self.info.setToolTip(wrap_tooltip("Repeat-unit molar mass, and how many skeletal (backbone) atoms "
            "one repeat unit contributes. Polyethylene [*]CC[*] has 2; "
            "poly(butylene succinate) has 10. Growth places one bead per "
            "skeletal atom, so this sets the chain's contour length."))
        row.addWidget(self.info)

        # What the current settings actually produce. Updated on every edit,
        # so the consequence of a change is visible without pressing anything.
        self.outcome = label("", T.FS_CAPTION, T.PRIMARY, mono=True, bold=True)
        self.outcome.setFixedWidth(150)
        self.outcome.setToolTip(wrap_tooltip("Chains of this polymer in the cell, and the weight percent they "
            "actually come to once chain counts are whole numbers."))
        row.addWidget(self.outcome)

        self.b_copo = button("A/B…")
        self.b_copo.setToolTip(wrap_tooltip("Make this a copolymer: two or more monomers with fractions and "
            "an arrangement. ENR is isoprene plus epoxidised isoprene."))
        self.b_copo.clicked.connect(self._edit_copolymer)
        row.addWidget(self.b_copo)

        self.b_del = button("✕", "ghost")
        self.b_del.setFixedWidth(26)
        self.b_del.setToolTip(wrap_tooltip("Remove this component"))
        self.b_del.clicked.connect(lambda: self.removeRequested.emit(self))
        row.addWidget(self.b_del)

        self._ris_key = ""
        self._monomers = None            # List[sequence.Monomer] when copolymer
        self._arrangement = "random"

    def _on_wt(self, _v: float) -> None:
        self.weightEdited.emit(self)
        self.changed.emit()

    def set_outcome(self, chains: int, wt_percent: float) -> None:
        self.outcome.setText(f"{chains} ch · {wt_percent:.1f} wt%")

    def clear_outcome(self) -> None:
        self.outcome.setText("")

    def paintEvent(self, ev):
        """QWidget subclasses must draw PE_Widget or the stylesheet background
        is silently dropped. See tests/test_qt_painting.py."""
        opt = QStyleOption()
        opt.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, p, self)

    # ------------------------------------------------------------- library
    def _pick_library(self) -> None:
        from .library_picker import LibraryPicker
        dlg = LibraryPicker(self)
        if dlg.exec_() != dlg.Accepted:
            return
        rec = dlg.selected()
        # A frozen dataclass is always truthy, so `if not rec` would never be
        # the None guard it looks like. Test for None explicitly.
        if rec is None:
            return
        self.smiles.setText(rec.smiles or rec.monomer_smiles or "")
        if not self.name.text().strip():
            self.name.setText((rec.name or "")[:20])
        self.changed.emit()

    def _on_smiles(self, _t: str) -> None:
        self._refresh_info()
        self.changed.emit()

    def _refresh_info(self) -> None:
        """Show the repeat-unit mass and granularity as soon as it is known.

        Seeing "28.05 g/mol · 2 skeletal" the moment a SMILES is typed is what
        makes the weight-fraction arithmetic legible rather than magic.
        """
        if self.is_copolymer:
            try:
                from ..cell.grow import max_backbone_atoms_in_a_ring
                for mo in self._monomers:
                    if not mo.mass_amu:
                        mo.resolve()
                w = sum(m.fraction for m in self._monomers) or 1.0
                mean = sum(m.mass_amu * m.fraction
                           for m in self._monomers) / w
                worst = max(max_backbone_atoms_in_a_ring(m.smiles)
                            for m in self._monomers)
                pct = " / ".join(f"{100 * m.fraction / w:.0f}"
                                 for m in self._monomers)
                self.info.setText(f"{len(self._monomers)} monomers {pct}%"
                                  f" · {mean:.1f} g/mol")
                self.info.setToolTip(wrap_tooltip(f"Copolymer, {self._arrangement}. Mean repeat-unit mass "
                    f"shown; each chain gets its own sequence, so its exact "
                    f"composition scatters around what you asked for."
                    + ("" if worst <= 2 else
                       "  One monomer has a ring spanning several backbone "
                       "bonds and cannot be rebuilt as all-atom.")))
                self._style_info(T.TEXT_MUTED if worst <= 2 else T.WARN_TEXT)
            except Exception:
                self.info.setText("unreadable")
                self._style_info(T.DANGER)
            return

        smi = self.smiles.text().strip()
        if not smi:
            self.info.setText("")
            return
        try:
            from ..cell.grow import (
                backbone_atoms_per_unit, max_backbone_atoms_in_a_ring,
                repeat_unit_mass,
            )
            m = repeat_unit_mass(smi)
            b = backbone_atoms_per_unit(smi)
            # A ring spanning ONE backbone bond (an epoxide) is fine; only a
            # larger one blocks the all-atom rebuild.
            ring = max_backbone_atoms_in_a_ring(smi)
            ring = ring if ring > 2 else 0
            if ring:
                # Say this now, not after a build. The bead cell is still fine;
                # it is the all-atom reconstruction that cannot be done.
                self.info.setText(f"{m:.2f} g/mol · beads only")
                self.info.setToolTip(wrap_tooltip(f"The backbone of this repeat unit runs through a ring "
                    f"({ring} skeletal atoms are in it). The cell will still "
                    f"be grown and exported coarse-grained, but it cannot be "
                    f"rebuilt as all-atom, so no force field can be applied."))
                self._style_info(T.WARN_TEXT)
            else:
                self.info.setText(f"{m:.2f} g/mol · {b} skel")
                self.info.setToolTip(wrap_tooltip(""))
                self._style_info(T.TEXT_MUTED)
        except Exception:
            self.info.setText("unreadable")
            self.info.setToolTip(wrap_tooltip(""))
            self._style_info(T.DANGER)

    def _style_info(self, colour: str) -> None:
        """Restyle the readout without losing its monospace family.

        setStyleSheet replaces the whole sheet, so the font-family set by
        page.label(mono=True) has to be repeated or the figures stop lining
        up the moment the first character is typed.
        """
        self.info.setStyleSheet(
            f"color: {colour}; font-size: {T.FS_CAPTION}px;"
            f" font-family: {T.FONT_MONO}; background: transparent;"
            f" border: none;")

    def set_mode(self, weight_mode: bool) -> None:
        for w in (self.wt_label, self.wt):
            w.setVisible(weight_mode)
        for w in (self.n_label, self.n_chains):
            w.setVisible(not weight_mode)

    def to_component(self, weight_mode: bool):
        from ..cell.composition import Component
        smi = self.smiles.text().strip()
        name = self.name.text().strip() or (smi[:12] or "polymer")
        return Component(
            name=name, repeat_unit=smi,
            degree_of_polymerisation=int(self.dp.value()),
            weight_percent=float(self.wt.value()) if weight_mode else 100.0,
            n_chains=int(self.n_chains.value()),
            ris_key=self._guess_ris(smi),
            monomers=self._monomers,
            arrangement=self._arrangement,
        )

    # ------------------------------------------------------------ copolymer
    @property
    def is_copolymer(self) -> bool:
        """A property, deliberately — the same shape as GrowSpec's and
        Component's. It was briefly a method, and `if self.is_copolymer:` then
        tested a bound method object, which is always truthy: every row was
        treated as a copolymer and every mass read "unreadable"."""
        return bool(self._monomers) and len(self._monomers) > 1

    def set_copolymer(self, monomers, arrangement: str = "random") -> None:
        """Turn this row into a copolymer (or back into a homopolymer).

        The SMILES box shows the first monomer so the row still reads as a
        chemistry, and the readout switches to the blend description.
        """
        self._monomers = list(monomers) if monomers else None
        self._arrangement = arrangement or "random"
        if self.is_copolymer:
            self.smiles.setText(self._monomers[0].smiles)
            self.smiles.setEnabled(False)
            self.b_lib.setEnabled(False)
        else:
            self.smiles.setEnabled(True)
            self.b_lib.setEnabled(True)
        self._refresh_info()
        self.changed.emit()

    def _edit_copolymer(self) -> None:
        from .copolymer_dialog import CopolymerDialog
        dlg = CopolymerDialog(self._monomers, self._arrangement, self)
        if dlg.exec_() != dlg.Accepted:
            return
        self.set_copolymer(dlg.monomers(), dlg.arrangement())

    @staticmethod
    def _guess_ris(smiles: str) -> str:
        """Match the SMILES to a parameterised RIS model where one exists.

        An unmatched polymer falls back to the generic model, and the grower
        labels the result as qualitative — so guessing wrong here degrades
        gracefully and visibly rather than silently.
        """
        table = {
            "[*]CC[*]": "PE",
            "[*]CC([*])C": "PP",
            "[*]CC([*])c1ccccc1": "PS",
            "[*]CC([*])(C)C(=O)OC": "PMMA",
        }
        return table.get((smiles or "").strip(), "")


# ==================================================================== the tab
class AmorphousTab(QWidget):
    """Build a periodic amorphous cell from library polymers."""

    log = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: List[ComponentRow] = []
        self._composition = None
        self._result = None
        self._export = None
        self._settings_provider = None
        self._updating = False        # re-entry guard for linked weight boxes
        self._thread: Optional[QThread] = None
        self._worker: Optional[_BuildWorker] = None
        self._build()
        self._add_component("PE", "[*]CC[*]", 70.0)
        self._add_component("PS", "[*]CC([*])c1ccccc1", 30.0)
        self._on_mode_changed()

    # ------------------------------------------------------------- layout
    def _build(self) -> None:
        page = QVBoxLayout(self)
        page.setContentsMargins(T.PAD_PAGE, 8, T.PAD_PAGE, 12)
        page.setSpacing(T.GAP_FORM_ROW)

        self.steps = QTabWidget()
        self.steps.setDocumentMode(True)
        self.steps.addTab(self._composition_page(), "1 · Composition")
        self.steps.addTab(self._ff_page(), "2 · Force field")
        self.steps.addTab(self._relax_page(), "3 · Relax")
        self.steps.addTab(self._export_page(), "4 · Export")
        from .page import fit_tabs_to_current
        fit_tabs_to_current(self.steps)
        page.addWidget(self.steps, 1)
        page.addWidget(self._action_bar())

    # ------------------------------------------------ step 1: composition
    def _composition_page(self) -> QWidget:
        host = QWidget()
        host.setStyleSheet("background: transparent; border: none;")
        col = QVBoxLayout(host)
        col.setContentsMargins(0, T.GAP_FORM_ROW, 0, 0)
        col.setSpacing(T.GAP_SECTION)

        # ---- mode
        card_mode = Card("How is the composition specified?")
        self.rb_weight = QRadioButton("Solve chain counts from weight %")
        self.rb_weight.setChecked(True)
        self.rb_counts = QRadioButton("Enter chain counts myself")
        self.rb_weight.toggled.connect(self._on_mode_changed)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(T.GAP_SECTION)
        mode_row.addWidget(self.rb_weight)
        mode_row.addWidget(self.rb_counts)
        mode_row.addStretch(1)
        card_mode.body.addLayout(mode_row)
        self.mode_hint = caption("")
        self.mode_hint.setWordWrap(True)
        card_mode.body.addWidget(self.mode_hint)
        col.addWidget(card_mode)

        # ---- components
        card_c = Card("Polymers in this cell",
                      "pick from the library or paste a polymerisation SMILES")
        b_add = button("+ Add polymer")
        b_add.clicked.connect(lambda: (self._add_component(), self._invalidate()))
        card_c.header.addWidget(b_add)
        self.rows_box = QVBoxLayout()
        self.rows_box.setSpacing(6)
        card_c.body.addLayout(self.rows_box)
        col.addWidget(card_c)

        # ---- targets
        card_t = Card("Cell targets")
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(T.GAP_FORM_ROW)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.density = QDoubleSpinBox()
        self.density.setRange(0.05, 3.0)
        self.density.setDecimals(4)
        self.density.setSingleStep(0.01)
        self.density.setValue(0.95)
        self.density.setFixedHeight(T.H_CONTROL)
        self.density.setSuffix("  g/cm³")
        self.density.setToolTip(wrap_tooltip("Type a value or use the arrows. The box edge is computed to hit "
            "this exactly; masses come from the SMILES with hydrogens "
            "included."))
        self.density.valueChanged.connect(lambda _v: self._invalidate())
        form.addRow("Target density", self.density)

        # Build loose, then compress. Growth places BEADS at a hard-core
        # spacing; the side groups are bolted on afterwards and need room that
        # was never reserved for them. Measured on a PE/PS cell built straight
        # at 0.95 g/cm3: 14,528 non-bonded pairs under 2.0 A, which DL_FIELD
        # reads as bonds — carbons come out with 5-10 neighbours and it
        # refuses to type them. The Blend page has had this control from the
        # start; the cell builder needed it more.
        self.build_fraction = QDoubleSpinBox()
        self.build_fraction.setRange(20.0, 100.0)
        # 45, not 60: measured on the PE/PS cell, every one of five seeds at
        # 60% produced ring-threaded bonds (9, 7, 6...), and one threaded
        # bond is enough to stop DL_FIELD. At 45% the box has 2.2x the target
        # volume and threading becomes rare. The NPT compression afterwards
        # is the same either way.
        self.build_fraction.setValue(45.0)
        self.build_fraction.setSingleStep(5.0)
        self.build_fraction.setSuffix(" %")
        self.build_fraction.setDecimals(0)
        self.build_fraction.setFixedHeight(T.H_CONTROL)
        self.build_fraction.setToolTip(wrap_tooltip(
            "Build the cell at this fraction of the target density, then "
            "compress the rest with an NPT run. Constructing straight at bulk "
            "density leaves no room for side groups: they end up inside one "
            "another, and a typer that perceives bonds by distance sees rings "
            "and double bonds that are not there. 50-70% is the usual "
            "compromise; lower it if typing still fails."))
        self.build_fraction.valueChanged.connect(self._on_build_fraction)
        form.addRow("Build at", self.build_fraction)

        self.target_beads = QSpinBox()
        self.target_beads.setRange(50, 2_000_000)
        self.target_beads.setSingleStep(500)
        self.target_beads.setValue(3000)
        self.target_beads.setFixedHeight(T.H_CONTROL)
        self.target_beads.setToolTip(wrap_tooltip("Roughly how many skeletal (backbone) atoms the whole cell should "
            "hold — the size of the model, and what governs run time. It is "
            "also the resolution of the composition: weight fractions can "
            "only be as fine as one chain's mass, so a bigger cell expresses "
            "the requested split more exactly."))
        self.target_beads.valueChanged.connect(lambda _v: self._invalidate())
        self._targets_form = form
        self.beads_row = self._labelled(form, "Cell size", self.target_beads,
                                        "skeletal atoms")

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(1.0, 2000.0)
        self.temperature.setDecimals(1)
        self.temperature.setValue(413.0)
        self.temperature.setSuffix("  K")
        self.temperature.setFixedHeight(T.H_CONTROL)
        self.temperature.setToolTip(wrap_tooltip("Sets the trans/gauche balance through the RIS weights, so it "
            "changes how extended the chains are."))
        form.addRow("Temperature", self.temperature)
        card_t.body.addLayout(form)
        col.addWidget(card_t)

        # ---- growth settings
        card_g = Card("Growth", "Theodorou–Suter, one skeletal bond at a time")
        gform = QFormLayout()
        gform.setContentsMargins(0, 0, 0, 0)
        gform.setSpacing(T.GAP_FORM_ROW)
        gform.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0.5, 6.0)
        self.tolerance.setDecimals(2)
        self.tolerance.setSingleStep(0.1)
        self.tolerance.setValue(1.7)
        self.tolerance.setSuffix("  Å")
        self.tolerance.setFixedHeight(T.H_CONTROL)
        self.tolerance.setToolTip(wrap_tooltip("Hard minimum approach between non-bonded beads. Not the physical "
            "contact distance — the soft bias already handles that. Raising "
            "it makes dense cells fail to build."))
        gform.addRow("Overlap tolerance", self.tolerance)

        self.scan_depth = QSpinBox()
        self.scan_depth.setRange(0, 3)
        self.scan_depth.setValue(0)
        self.scan_depth.setFixedHeight(T.H_CONTROL)
        self.scan_depth.valueChanged.connect(self._on_scan_changed)
        gform.addRow("Look-ahead depth", self.scan_depth)

        self.scan_warn = caption("")
        self.scan_warn.setWordWrap(True)
        gform.addRow("", self.scan_warn)

        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setValue(12345)
        self.seed.setFixedHeight(T.H_CONTROL)
        self.seed.setToolTip(wrap_tooltip("Same seed, same cell — exactly."))
        gform.addRow("Random seed", self.seed)

        self.check_spearing = QCheckBox(
            "Reject bonds that thread through a ring")
        self.check_spearing.setChecked(True)
        self.check_spearing.setToolTip(wrap_tooltip("Speared rings survive minimisation happily and are a classic "
            "silent failure of constructed cells."))
        gform.addRow("", self.check_spearing)
        card_g.body.addLayout(gform)
        col.addWidget(card_g)
        # setValue(0) on a fresh spin box emits nothing, so the "off" guidance
        # would never appear until the user touched the control.
        self._on_scan_changed(self.scan_depth.value())

        # ---- preview
        card_p = Card("What will be built")
        self.solve_badge = badge("not solved", "neutral")
        card_p.header.addWidget(self.solve_badge)
        self.preview = QTableWidget(0, 6)
        self.preview.setHorizontalHeaderLabels(
            ["polymer", "chains", "DP", "chain mass", "wt% asked",
             "wt% realised"])
        self.preview.verticalHeader().setVisible(False)
        self.preview.verticalHeader().setDefaultSectionSize(T.H_TABLE_ROW)
        self.preview.horizontalHeader().setFixedHeight(T.H_TABLE_HEADER)
        self.preview.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview.setMinimumHeight(120)
        card_p.body.addWidget(self.preview)

        stats = QHBoxLayout()
        stats.setSpacing(T.GAP_SECTION)
        self.stat_box = stat_row("Box edge", "—")
        self.stat_beads = stat_row("Skeletal beads", "—")
        self.stat_mass = stat_row("Total mass", "—")
        for w in (self.stat_box, self.stat_beads, self.stat_mass):
            stats.addWidget(w, 1)
        card_p.body.addLayout(stats)

        self.solve_notes = caption("")
        self.solve_notes.setWordWrap(True)
        card_p.body.addWidget(self.solve_notes)
        col.addWidget(card_p)
        col.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(host)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }")
        return scroll

    @staticmethod
    def _labelled(form: QFormLayout, text: str, widget: QWidget,
                  suffix: str) -> QWidget:
        wrap = QWidget()
        wrap.setStyleSheet("background: transparent; border: none;")
        h = QHBoxLayout(wrap)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(T.GAP_LABEL)
        h.addWidget(widget)
        h.addWidget(label(suffix, T.FS_CAPTION, T.TEXT_MUTED))
        h.addStretch(1)
        form.addRow(text, wrap)
        return wrap

    # ------------------------------------------------- step 2: force field
    def _ff_page(self) -> QWidget:
        from ..ff_registry import list_ffs

        w = QWidget()
        w.setStyleSheet("background: transparent; border: none;")
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, T.GAP_FORM_ROW, 0, 0)
        outer.setSpacing(T.GAP_SECTION)

        row = QHBoxLayout()
        row.setSpacing(T.GAP_SECTION)

        card = Card("Force field for the cell",
                    "independent of the pipeline's Force-field page")
        b_copy = button("Copy from pipeline")
        b_copy.setToolTip(wrap_tooltip("Load whatever the main Force-field page has selected"))
        b_copy.clicked.connect(self._copy_ff_from_pipeline)
        card.header.addWidget(b_copy)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(T.GAP_FORM_ROW)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.ff_combo = QComboBox()
        self.ff_combo.setFixedHeight(T.H_CONTROL)
        from .ff_combo import populate_force_field_combo
        from ..config import ForceFieldCfg as _FFCfg
        populate_force_field_combo(self.ff_combo, _FFCfg().key)
        self.ff_combo.currentIndexChanged.connect(self._on_ff_changed)
        form.addRow("Force field", self.ff_combo)

        self.ff_route = caption("")
        self.ff_route.setWordWrap(True)
        form.addRow("", self.ff_route)

        self.dl_lib = QLineEdit()
        self.dl_lib.setPlaceholderText("path to dl_f_4.13/lib (DL_FIELD FFs only)")
        self.dl_lib.setFixedHeight(T.H_CONTROL)
        b_browse = button("Browse…")
        b_browse.clicked.connect(self._browse_dl_lib)
        dl_row = QHBoxLayout()
        dl_row.setSpacing(T.GAP_LABEL)
        dl_row.addWidget(self.dl_lib, 1)
        dl_row.addWidget(b_browse)
        dl_wrap = QWidget()
        dl_wrap.setStyleSheet("background: transparent; border: none;")
        dl_wrap.setLayout(dl_row)
        form.addRow("DL_FIELD lib dir", dl_wrap)
        card.body.addLayout(form)
        row.addWidget(card, 1)

        card2 = Card("Atoms from beads")
        card2.setFixedWidth(330)
        d = caption(
            "Growth builds the backbone one skeletal atom at a time. Before "
            "any force field can be applied, the side groups and hydrogens "
            "have to be put back — the backbone is never moved.")
        d.setWordWrap(True)
        card2.body.addWidget(d)

        aform = QFormLayout()
        aform.setContentsMargins(0, 0, 0, 0)
        aform.setSpacing(T.GAP_FORM_ROW)
        aform.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.atomistic = QCheckBox("Rebuild all-atom structure")
        self.atomistic.setChecked(True)
        self.atomistic.setToolTip(wrap_tooltip("Off leaves coarse-grained output only: one site per skeletal "
            "atom, and no force field can be applied."))
        self.atomistic.toggled.connect(self._on_atomistic_toggled)
        aform.addRow("", self.atomistic)

        self.tacticity = QComboBox()
        self.tacticity.addItems(["isotactic", "syndiotactic", "atactic"])
        self.tacticity.setFixedHeight(T.H_CONTROL)
        self.tacticity.setToolTip(wrap_tooltip("Which configuration each stereocentre takes. Real bulk vinyl "
            "polymers are usually atactic."))
        aform.addRow("Tacticity", self.tacticity)

        self.push_off = QCheckBox("Push overlapping side groups apart")
        self.push_off.setChecked(True)
        self.push_off.setToolTip(wrap_tooltip("Moves only side atoms, keeps bond lengths exact, leaves the "
            "backbone pinned. Without it the closest contact can be a few "
            "tenths of an Ångström and LAMMPS will not survive it."))
        aform.addRow("", self.push_off)

        self.run_typing = QCheckBox("Apply the force field and write MD files")
        self.run_typing.setChecked(True)
        aform.addRow("", self.run_typing)

        self.output_format = QComboBox()
        self.output_format.addItem("LAMMPS (cell.data + cell.in)", "lammps")
        self.output_format.addItem("GROMACS (cell.gro + cell.top)", "gromacs")
        self.output_format.addItem("Both", "both")
        self.output_format.setFixedHeight(T.H_CONTROL)
        self.output_format.setToolTip(wrap_tooltip(
            "Which MD engine the typed cell is written for. The typing is "
            "identical either way — DL_FIELD assigns the same force field to "
            "the same cell — only the file format differs. GROMACS output "
            "is a .gro/.top pair (plus .itp includes); 'Both' asks DL_FIELD "
            "twice, once per format."))
        self.output_format.currentIndexChanged.connect(
            lambda _i: self._refresh_files())
        aform.addRow("Write files for", self.output_format)
        card2.body.addLayout(aform)
        card2.body.addStretch(1)
        row.addWidget(card2)
        outer.addLayout(row)
        outer.addStretch(1)
        self._on_ff_changed(0)
        return w

    # ------------------------------------------------------ step 3: relax
    def _relax_page(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent; border: none;")
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, T.GAP_FORM_ROW, 0, 0)
        outer.setSpacing(T.GAP_SECTION)

        row = QHBoxLayout()
        row.setSpacing(T.GAP_SECTION)

        card = Card("Relax the constructed cell",
                    "soft push-off, then minimise with the real force field")
        d = caption(
            "Construction leaves strained contacts — around 1 Å after the "
            "geometric push-off. A Lennard-Jones potential at 1 Å is roughly "
            "three million times epsilon, so minimising straight away blows "
            "the run up. PAAF first ramps up a bounded soft repulsion with "
            "capped per-step displacement, then switches to the real force "
            "field and minimises.")
        card.body.addWidget(d)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(T.GAP_FORM_ROW)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.do_relax = QCheckBox("Run LAMMPS after export")
        self.do_relax.setChecked(False)
        self.do_relax.setToolTip(wrap_tooltip("Off by default because it needs a LAMMPS executable. Without it "
            "PAAF still writes a ready-to-run input deck."))
        self.do_relax.toggled.connect(self._on_relax_toggled)
        form.addRow("", self.do_relax)

        self.engine = QComboBox()
        self.engine.setFixedHeight(T.H_CONTROL)
        self.engine.addItem("Automatic (follows the force field)", "auto")
        self.engine.addItem("LAMMPS", "lammps")
        self.engine.addItem("GROMACS", "gromacs")
        self.engine.currentIndexChanged.connect(self._on_engine_changed)
        form.addRow("Engine", self.engine)

        self.engine_note = caption("")
        form.addRow("", self.engine_note)

        self.lammps_exe = QLineEdit()
        self.lammps_exe.setPlaceholderText(
            "auto-detect (PATH, then sibling conda environments)")
        self.lammps_exe.setFixedHeight(T.H_CONTROL)
        self.lammps_exe.setToolTip(wrap_tooltip("Leave blank to search PATH and every sibling conda environment. "
            "You may also give an environment name, e.g. polyrapid, or a full "
            "path to the binary."))
        b_detect = button("Detect")
        b_detect.clicked.connect(self._detect_lammps)
        b_browse = button("Browse…")
        b_browse.clicked.connect(self._browse_lammps)
        exe_row = QHBoxLayout()
        exe_row.setSpacing(T.GAP_LABEL)
        exe_row.addWidget(self.lammps_exe, 1)
        exe_row.addWidget(b_detect)
        exe_row.addWidget(b_browse)
        exe_wrap = QWidget()
        exe_wrap.setStyleSheet("background: transparent; border: none;")
        exe_wrap.setLayout(exe_row)
        form.addRow("LAMMPS", exe_wrap)

        self.lammps_found = caption("")
        form.addRow("", self.lammps_found)

        self.gmx_exe = QLineEdit()
        self.gmx_exe.setPlaceholderText("auto-detect (including SIMD subdirs)")
        self.gmx_exe.setFixedHeight(T.H_CONTROL)
        b_gdetect = button("Detect")
        b_gdetect.clicked.connect(self._detect_gromacs)
        g_row = QHBoxLayout()
        g_row.setSpacing(T.GAP_LABEL)
        g_row.addWidget(self.gmx_exe, 1)
        g_row.addWidget(b_gdetect)
        g_wrap = QWidget()
        g_wrap.setStyleSheet("background: transparent; border: none;")
        g_wrap.setLayout(g_row)
        form.addRow("GROMACS", g_wrap)

        self.gmx_found = caption("")
        form.addRow("", self.gmx_found)

        self.push_steps = QSpinBox()
        self.push_steps.setRange(0, 1_000_000)
        self.push_steps.setSingleStep(1000)
        self.push_steps.setValue(10_000)
        self.push_steps.setFixedHeight(T.H_CONTROL)
        self.push_steps.setToolTip(wrap_tooltip("Steps of ramped soft repulsion before the real potential is "
            "switched on. More steps for a denser or more strained cell."))
        form.addRow("Push-off steps", self.push_steps)

        self.min_iter = QSpinBox()
        self.min_iter.setRange(100, 1_000_000)
        self.min_iter.setSingleStep(1000)
        self.min_iter.setValue(10_000)
        self.min_iter.setFixedHeight(T.H_CONTROL)
        form.addRow("Minimiser iterations", self.min_iter)

        self.mpi_ranks = QSpinBox()
        self.mpi_ranks.setRange(1, 256)
        self.mpi_ranks.setValue(1)
        self.mpi_ranks.setFixedHeight(T.H_CONTROL)
        self.mpi_ranks.setToolTip(wrap_tooltip("Uses mpirun when greater than 1."))
        form.addRow("MPI ranks", self.mpi_ranks)
        card.body.addLayout(form)
        row.addWidget(card, 1)

        card2 = Card("What relaxation does not do")
        card2.setFixedWidth(330)
        d2 = caption(
            "Minimisation finds a nearby local minimum — effectively a 0 K "
            "glass. It removes strained contacts and makes the cell safe to "
            "run.\n\nIt does NOT equilibrate the melt. Chain dimensions relax "
            "on a diffusive timescale that only molecular dynamics reaches, "
            "so the construction's chain over-extension survives "
            "minimisation.\n\nRun NVT then NPT before quoting any property: "
            "construction, minimisation, and equilibration are three "
            "separate steps.")
        card2.body.addWidget(d2)
        card2.body.addStretch(1)
        row.addWidget(card2)
        outer.addLayout(row)

        card3 = Card("Before and after")
        self.relax_badge = badge("not run", "neutral")
        card3.header.addWidget(self.relax_badge)
        self.relax_table = QTableWidget(0, 6)
        self.relax_table.setHorizontalHeaderLabels(
            ["stage", "density", "closest contact", "R_g", "bond min",
             "bond max"])
        self.relax_table.verticalHeader().setVisible(False)
        self.relax_table.verticalHeader().setDefaultSectionSize(T.H_TABLE_ROW)
        self.relax_table.horizontalHeader().setFixedHeight(T.H_TABLE_HEADER)
        self.relax_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.relax_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.relax_table.setMinimumHeight(100)
        card3.body.addWidget(self.relax_table)
        outer.addWidget(card3)
        outer.addStretch(1)
        self._on_relax_toggled(False)
        self._detect_lammps(quiet=True)
        self._detect_gromacs()
        self._on_engine_changed()
        return w

    def _on_relax_toggled(self, on: bool) -> None:
        for name in ("engine", "lammps_exe", "gmx_exe", "push_steps",
                     "min_iter", "mpi_ranks"):
            wdg = getattr(self, name, None)
            if wdg is not None:
                wdg.setEnabled(on)


    def _on_engine_changed(self, _i: int = 0) -> None:
        """Explain what the force field actually permits.

        The engine is not a free choice: Moltemplate emits LAMMPS input only,
        while DL_FIELD can emit either. Saying so here prevents a user
        selecting GROMACS for OPLS-AA and being silently switched back.
        """
        if not hasattr(self, "engine_note"):
            return
        from ..ff_registry import REGISTRY
        key = self.ff_combo.currentData() if hasattr(self, "ff_combo") else ""
        ff = REGISTRY.get(key or "")
        want = self.engine.currentData() or "auto"
        if ff is None:
            self.engine_note.setText("")
            return
        dl = (ff.kind == "dlfield")
        if not dl:
            msg = (f"{ff.display_name} is typed by Moltemplate, which writes "
                   f"LAMMPS input only. GROMACS is not available for it.")
            if want == "gromacs":
                msg += " LAMMPS will be used instead."
        elif want == "lammps":
            msg = ("DL_FIELD's LAMMPS data file does not carry the "
                   "pair/bond/angle/dihedral styles, so PAAF supplies them "
                   "from a table and the soft push-off is replaced by a "
                   "displacement-capped minimisation. GROMACS avoids both "
                   "compromises.")
        else:
            msg = ("DL_FIELD writes a GROMACS .top that states every "
                   "functional form, so nothing has to be inferred. This is "
                   "the better-conditioned route for this force field.")
        self.engine_note.setText(msg)

    def _detect_gromacs(self) -> None:
        from ..cell.relax import find_gromacs
        exe = find_gromacs(self.gmx_exe.text().strip())
        if exe is None:
            self.gmx_found.setText(
                "Not found. Needed only for the GROMACS engine.")
            self.gmx_found.setStyleSheet(
                f"color: {T.WARN_TEXT}; font-size: {T.FS_CAPTION}px;"
                f" background: transparent; border: none;")
        else:
            self.gmx_found.setText(f"Found: {exe}")
            self.gmx_found.setStyleSheet(
                f"color: {T.TEXT_MUTED}; font-size: {T.FS_CAPTION}px;"
                f" background: transparent; border: none;")

    def _detect_lammps(self, quiet: bool = False) -> None:
        from ..cell.relax import find_lammps
        exe = find_lammps(self.lammps_exe.text().strip())
        if exe is None:
            self.lammps_found.setText(
                "Not found on PATH or in any sibling conda environment. "
                "PAAF will still write the input deck for you to run.")
            self.lammps_found.setStyleSheet(
                f"color: {T.WARN_TEXT}; font-size: {T.FS_CAPTION}px;"
                f" background: transparent; border: none;")
        else:
            self.lammps_found.setText(f"Found: {exe}")
            self.lammps_found.setStyleSheet(
                f"color: {T.TEXT_MUTED}; font-size: {T.FS_CAPTION}px;"
                f" background: transparent; border: none;")
            if not quiet:
                self._rlog(f"LAMMPS: {exe}")

    def _browse_lammps(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "LAMMPS executable", str(Path.home()))
        if path:
            self.lammps_exe.setText(path)
            self._detect_lammps()

    def _fill_relax_table(self, rr) -> None:
        rows = []
        if getattr(rr, "before", None):
            rows.append(("constructed", rr.before))
        if getattr(rr, "after", None):
            rows.append(("relaxed", rr.after))
        self.relax_table.setRowCount(len(rows))
        for i, (label, m) in enumerate(rows):
            vals = [label, f"{m.density_g_cm3:.4f}",
                    f"{m.closest_contact_a:.2f} Å", f"{m.r_gyration_a:.2f} Å",
                    f"{m.bond_min_a:.3f}", f"{m.bond_max_a:.3f}"]
            for j, v in enumerate(vals):
                self.relax_table.setItem(i, j, QTableWidgetItem(v))

    # ----------------------------------------------------- step 4: export
    def _export_page(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent; border: none;")
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, T.GAP_FORM_ROW, 0, 0)
        outer.setSpacing(T.GAP_SECTION)

        card = Card("Output", "one folder per cell")
        b_browse = button("Browse…")
        b_browse.clicked.connect(self._browse_out_dir)
        card.header.addWidget(b_browse)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(T.GAP_FORM_ROW)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("folder that will hold the cell folder")
        self.out_dir.setFixedHeight(T.H_CONTROL)
        self.out_dir.textChanged.connect(lambda _t: self._refresh_files())
        form.addRow("Cells folder", self.out_dir)

        self.cell_name = QLineEdit("cell")
        self.cell_name.setFixedHeight(T.H_CONTROL)
        self.cell_name.textChanged.connect(lambda _t: self._refresh_files())
        form.addRow("Cell name", self.cell_name)
        card.body.addLayout(form)
        self.out_hint = caption("")
        self.out_hint.setWordWrap(True)
        card.body.addWidget(self.out_hint)
        outer.addWidget(card)

        card2 = Card("Files that will be written")
        self.files = QTableWidget(0, 2)
        self.files.setHorizontalHeaderLabels(["file", "what it is"])
        self.files.verticalHeader().setVisible(False)
        self.files.verticalHeader().setDefaultSectionSize(T.H_TABLE_ROW)
        self.files.horizontalHeader().setFixedHeight(T.H_TABLE_HEADER)
        self.files.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.files.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.files.setMinimumHeight(150)
        card2.body.addWidget(self.files)
        outer.addWidget(card2, 1)

        card3 = Card("Result")
        self.result_badge = badge("nothing built yet", "neutral")
        card3.header.addWidget(self.result_badge)
        self.res_density = stat_row("Density", "—")
        self.res_cn = stat_row("Mean C_n", "—")
        self.res_rg = stat_row("Mean R_g", "—")
        self.res_atoms = stat_row("Atoms", "—")
        for wdg in (self.res_density, self.res_cn, self.res_rg, self.res_atoms):
            card3.body.addWidget(wdg)
        self.res_notes = caption("")
        self.res_notes.setWordWrap(True)
        card3.body.addWidget(self.res_notes)
        outer.addWidget(card3)
        outer.addStretch(1)
        self._refresh_files()
        return w

    # -------------------------------------------------------- action bar
    def _action_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("amorphActionBar")
        bar.setFixedHeight(48)
        bar.setStyleSheet(
            f"QWidget#amorphActionBar {{ background: {T.BG_SURFACE};"
            f" border-top: 1px solid {T.BORDER}; }}")
        h = QHBoxLayout(bar)
        h.setContentsMargins(T.PAD_CARD, 0, T.PAD_CARD, 0)
        h.setSpacing(T.GAP_LABEL)

        self.blocker = caption("Solve the composition first.")
        h.addWidget(self.blocker)
        h.addStretch(1)

        self.stage_lb = caption("")
        self.stage_lb.setStyleSheet(
            f"color: {T.PRIMARY}; font-weight: 700;"
            f" font-size: {T.FS_CAPTION}px; background: transparent;"
            f" border: none;")
        self.stage_lb.hide()
        h.addWidget(self.stage_lb)

        self.bar_progress = QProgressBar()
        self.bar_progress.setFixedWidth(180)
        self.bar_progress.setFixedHeight(T.H_CONTROL - 8)
        self.bar_progress.setRange(0, 100)
        self.bar_progress.setValue(0)
        self.bar_progress.setTextVisible(False)
        self.bar_progress.hide()
        h.addWidget(self.bar_progress)

        self.b_cancel = button("Cancel", "danger")
        self.b_cancel.clicked.connect(self._cancel_build)
        self.b_cancel.hide()
        h.addWidget(self.b_cancel)

        self.b_solve = button("Solve composition")
        self.b_solve.clicked.connect(lambda: self._solve(quiet=False))
        h.addWidget(self.b_solve)

        self.b_build = button("Build cell", "primary")
        self.b_build.clicked.connect(self._build_cell)
        self.b_build.setEnabled(False)
        h.addWidget(self.b_build)
        return bar

    # ==================================================== component list
    def _add_component(self, name: str = "", smiles: str = "",
                       wt: float = 50.0) -> ComponentRow:
        row = ComponentRow(len(self._rows) + 1, self)
        row.name.setText(name)
        row.smiles.setText(smiles)
        row.wt.setValue(wt)
        row.changed.connect(self._invalidate)
        row.weightEdited.connect(self._rebalance_weights)
        row.removeRequested.connect(self._remove_component)
        row.set_mode(self.rb_weight.isChecked()
                     if hasattr(self, "rb_weight") else True)
        self._rows.append(row)
        self.rows_box.addWidget(row)
        self._renumber()
        return row

    def _remove_component(self, row: ComponentRow) -> None:
        if len(self._rows) <= 1:
            self.blocker.setText("A cell needs at least one polymer.")
            return
        self._rows.remove(row)
        self.rows_box.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        self._renumber()
        self._invalidate()

    def _renumber(self) -> None:
        for i, r in enumerate(self._rows, start=1):
            r.tag.setText(str(i))
            r.b_del.setEnabled(len(self._rows) > 1)

    # ==================================================== mode / state
    def _on_mode_changed(self, *_a) -> None:
        weight = self.rb_weight.isChecked()
        for r in self._rows:
            r.set_mode(weight)
        # Qt5 has no QFormLayout.setRowVisible, so the label has to be hidden
        # alongside its field or "Cell size" is left labelling empty space.
        self.beads_row.setVisible(weight)
        lb = self._targets_form.labelForField(self.beads_row)
        if lb is not None:
            lb.setVisible(weight)
        self.mode_hint.setText(
            "Weight percentages stay linked: raising one lowers the others in "
            "proportion. PAAF picks integer chain counts closest to what you "
            "ask for and sizes the box for the density. Each row shows the "
            "chains it gets and the weight percent those whole chains "
            "actually come to — changing DP changes chain mass, so it moves "
            "the realised split even when the requested one is untouched. "
            "With three or more components the ones you are not editing keep "
            "their ratio to each other."
            if weight else
            "Chain counts are used exactly as entered. Each row shows the "
            "weight percent those counts produce, which changes when you "
            "change either the counts or the DP. The box is sized for the "
            "target density.")
        self._invalidate()

    def _rebalance_weights(self, edited) -> None:
        """Keep the weight percentages summing to 100.

        Editing one component's share has to come from somewhere. The others
        are rescaled in proportion to what they already had, which preserves
        their ratio to one another — the least surprising choice, and the one
        that makes a two-component blend behave like a single slider.
        """
        if (self._updating or len(self._rows) < 2
                or not self.rb_weight.isChecked()):
            return
        others = [r for r in self._rows if r is not edited]
        remaining = max(0.0, 100.0 - float(edited.wt.value()))
        current = sum(float(r.wt.value()) for r in others)
        self._updating = True
        try:
            if current <= 1e-9:
                share = remaining / len(others)
                for r in others:
                    r.wt.setValue(share)
            else:
                for r in others:
                    r.wt.setValue(float(r.wt.value()) * remaining / current)
        finally:
            self._updating = False

    def set_weights(self, weights) -> None:
        """Set every component's weight at once, normalised to 100.

        Needed because the live linking makes it impossible to *type* an exact
        multi-way split: each entry rescales the ones already entered, so
        50/30/20 typed left to right does not land on 50/30/20. Setting them
        together sidesteps the interaction entirely.
        """
        rows = self._rows[:len(weights)]
        total = sum(float(w) for w in weights) or 1.0
        self._updating = True
        try:
            for r, w in zip(rows, weights):
                r.wt.setValue(100.0 * float(w) / total)
        finally:
            self._updating = False
        self._invalidate()

    def _on_scan_changed(self, v: int) -> None:
        if v > 0:
            self.scan_warn.setText(
                "Look-ahead defeats attrition and lets denser cells be built, "
                "but choosing a torsion by its look-ahead weight is a biased "
                "estimator (Rosenbluth). It over-samples extended chains — "
                "measured C_n rose to ~9 against ~6.3 unbiased. Do not quote "
                "chain dimensions from a scanned cell.")
            self.scan_warn.setStyleSheet(
                f"color: {T.WARN_TEXT}; font-size: {T.FS_CAPTION}px;"
                f" background: transparent; border: none;")
        else:
            self.scan_warn.setText(
                "Off — chain dimensions are unbiased. Raise it only if the "
                "cell will not build at all.")
            self.scan_warn.setStyleSheet(
                f"color: {T.TEXT_MUTED}; font-size: {T.FS_CAPTION}px;"
                f" background: transparent; border: none;")

    def _on_atomistic_toggled(self, on: bool) -> None:
        if not hasattr(self, "run_typing"):
            return                        # still being constructed
        self.tacticity.setEnabled(on)
        self.push_off.setEnabled(on)
        self.run_typing.setEnabled(on)
        if hasattr(self, "output_format"):
            self.output_format.setEnabled(on)
        self._refresh_files()

    def _on_ff_changed(self, _i: int) -> None:
        from ..ff_registry import REGISTRY
        key = self.ff_combo.currentData()
        ff = REGISTRY.get(key)
        if ff is None:
            self.ff_route.setText("")
            return
        dl = (ff.kind == "dlfield" or ff.atom_typer == "dlfield_sf")
        route = ("typed by DL_FIELD, which writes the LAMMPS data file "
                 "directly" if dl else
                 "typed by PAAF, then compiled by moltemplate.sh")
        self.ff_route.setText(f"Route: {route}. {ff.notes}")
        self.dl_lib.setEnabled(dl)
        self._on_engine_changed()
        self._refresh_files()

    def _invalidate(self, *_a) -> None:
        """Any change to the inputs makes a previous solve stale.

        The stale numbers are wiped, not just disabled. Leaving the old chain
        counts and box edge on screen next to a "not solved" badge invites
        exactly the mistake this is meant to prevent — reading a figure that
        no longer corresponds to anything.
        """
        if not hasattr(self, "b_build") or self._updating:
            return                        # still being constructed, or looping
        # Re-solve immediately rather than blanking the screen. The solver is
        # pure arithmetic and takes microseconds, so there is no reason to
        # make the user press a button to find out what their edit did.
        self._solve(quiet=True)

    # ==================================================== solve
    def _solve(self, quiet: bool = False) -> None:
        """Turn the form into a composition, reporting problems inline.

        Validation failures are shown on the page — badge, blocker line,
        notes — rather than in a modal. A modal for something the user can see
        and fix in place is an interruption, and it makes the step untestable
        without a click.
        """
        from ..cell.composition import (
            from_chain_counts, solve_from_weight_fractions,
        )
        weight = self.rb_weight.isChecked()
        comps = []
        for r in self._rows:
            if not r.smiles.text().strip():
                continue
            comps.append(r.to_component(weight))
        if not comps:
            for r in self._rows:
                r.clear_outcome()
            self._solve_error(
                "No component has a SMILES.",
                "Use the Library button on a row, or paste a polymerisation "
                "SMILES such as [*]CC[*] — the [*] marks where the chain "
                "continues.")
            return
        try:
            if weight:
                comp = solve_from_weight_fractions(
                    comps, float(self.density.value()),
                    target_beads=int(self.target_beads.value()),
                    build_fraction=float(self.build_fraction.value()) / 100.0)
            else:
                comp = from_chain_counts(
                    comps, float(self.density.value()),
                    build_fraction=float(self.build_fraction.value()) / 100.0)
        except Exception as exc:
            self._solve_error("Could not solve the composition.", str(exc))
            return

        self._composition = comp
        self._fill_preview(comp)
        # Per-row outcome: the number of chains and the weight percent those
        # whole chains actually come to.
        by_name = {c.name: (n, w) for c, n, w in zip(
            comp.components, comp.chain_counts, comp.realised_weight_percent)}
        for r in self._rows:
            got = by_name.get(r.to_component(weight).name)
            if got is None:
                r.clear_outcome()
            else:
                r.set_outcome(got[0], got[1])
        self.b_build.setEnabled(True)
        self._set_badge(self.solve_badge, f"{comp.total_chains} chains", "ok")
        self.blocker.setText(
            f"{comp.total_beads:,} beads in a {comp.box_edge_a:.1f} Å box.")
        if not quiet:
            self._rlog(comp.summary())
        self._refresh_files()

    def _solve_error(self, headline: str, detail: str) -> None:
        self._composition = None
        self.b_build.setEnabled(False)
        self.preview.setRowCount(0)
        for w in (self.stat_box, self.stat_beads, self.stat_mass):
            self._set_stat(w, "—")
        self._set_badge(self.solve_badge, "cannot solve", "error")
        self.blocker.setText(headline)
        self.solve_notes.setText(detail)
        self.solve_notes.setStyleSheet(
            f"color: {T.DANGER}; font-size: {T.FS_CAPTION}px;"
            f" background: transparent; border: none;")
        self._rlog(f"{headline} {detail}")

    def _fill_preview(self, comp) -> None:
        self.preview.setRowCount(len(comp.components))
        for i, (c, n, req, real) in enumerate(zip(
                comp.components, comp.chain_counts,
                comp.requested_weight_percent,
                comp.realised_weight_percent)):
            vals = [c.name, str(n), str(c.degree_of_polymerisation),
                    f"{c.chain_mass:,.0f}", f"{req:.2f}", f"{real:.2f}"]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                # Colour the pair only when they actually disagree, so a
                # coloured cell always means something.
                if j >= 4 and abs(req - real) > 1.0:
                    item.setForeground(QBrush(QColor(T.WARN_TEXT)))
                self.preview.setItem(i, j, item)
        self._set_stat(self.stat_box, f"{comp.box_edge_a:.2f} Å")
        self._set_stat(self.stat_beads, f"{comp.total_beads:,}")
        self._set_stat(self.stat_mass, f"{comp.total_mass_amu:,.0f} g/mol")
        self.solve_notes.setText("  ".join(comp.notes))
        self.solve_notes.setStyleSheet(
            f"color: {T.WARN_TEXT if comp.notes else T.TEXT_MUTED};"
            f" font-size: {T.FS_CAPTION}px; background: transparent;"
            f" border: none;")

    @staticmethod
    def _set_stat(row: QWidget, value: str) -> None:
        labels = row.findChildren(QLabel)
        if len(labels) >= 2:
            labels[-1].setText(value)

    @staticmethod
    def _set_badge(lb: QLabel, text: str, level: str = "neutral") -> None:
        """Change a badge's word AND its colour.

        page.badge bakes the palette into an inline stylesheet, so setText
        alone leaves an error reading in neutral grey — colour that has
        stopped tracking meaning is worse than no colour.
        """
        fill, border, fg = T.badge_colors(level)
        lb.setText(text)
        lb.setStyleSheet(
            f"background: {fill}; border: 1px solid {border}; color: {fg};"
            f" border-radius: {T.R_PILL}px; padding: 2px 8px;"
            f" font-size: {T.FS_CAPTION}px; font-weight: 600;")

    # ==================================================== build
    def _needs_dlfield(self) -> bool:
        """True when the chosen force field is typed by DL_FIELD."""
        try:
            from ..ff_registry import get_ff
            return get_ff(self.ff_combo.currentData()).kind == "dlfield"
        except Exception:
            return False

    def _dlfield_missing_reason(self) -> str:
        """Why DL_FIELD cannot run, or "" if it can."""
        lib = self.dl_lib.text().strip()
        if not lib:
            return ("No DL_FIELD library folder is set on the Force field "
                    "step (it should point at dl_f_4.13/lib).")
        from pathlib import Path as _P
        if not _P(lib).exists():
            return f"The DL_FIELD library folder does not exist:\n    {lib}"
        try:
            from ..dlfield_runner import find_dl_field
            from ..cell.cell_export import _dlfield_root
            if find_dl_field(_dlfield_root(_P(lib))) is None:
                return (f"No dl_field executable was found near:\n    {lib}\n"
                        f"Set $DL_FIELD_EXE or put dl_field beside lib/.")
        except Exception:
            return ""
        return ""

    def _on_build_fraction(self, *_a) -> None:
        """A change of build density invalidates the solved composition."""
        self._composition = None

    def _build_cell(self) -> None:
        if self._composition is None:
            self._solve()
            if self._composition is None:
                return
        out_dir = self.out_dir.text().strip() or self.default_out_dir()
        if not out_dir:
            QMessageBox.warning(
                self, "Amorphous builder",
                "Choose an output folder on the Export step first.")
            self.steps.setCurrentIndex(3)
            return

        # Say NOW that the typed file will not appear.
        #
        # Growth takes minutes. Discovering afterwards, from a note at the end
        # of a 200-line log, that dl_field was never found — and that cell.data
        # therefore does not exist — wastes the whole run. The check is cheap
        # and the answer is known before anything starts.
        if self.run_typing.isChecked() and self._needs_dlfield():
            missing = self._dlfield_missing_reason()
            if missing:
                answer = QMessageBox.question(
                    self, "No force field — no cell.data",
                    f"{missing}\n\nWithout it PAAF will build and back-map the "
                    f"cell, but CANNOT type it: you will get "
                    f"cell_atomistic.xyz/.mol2 and no cell.data or cell.in.\n\n"
                    f"Build anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if answer != QMessageBox.Yes:
                    self.steps.setCurrentIndex(1)   # the Force-field step
                    return

        options = {
            "temperature": float(self.temperature.value()),
            "scan_depth": int(self.scan_depth.value()),
            "tolerance": float(self.tolerance.value()),
            "seed": int(self.seed.value()),
            "check_spearing": self.check_spearing.isChecked(),
            "do_export": True,
            "out_dir": out_dir,
            "name": self.cell_name.text().strip() or "cell",
            "ff_key": self.ff_combo.currentData() or "",
            "dl_lib": self.dl_lib.text().strip(),
            "atomistic": self.atomistic.isChecked(),
            "tacticity": self.tacticity.currentText(),
            "run_typing": self.run_typing.isChecked(),
            "output_formats": self.output_format.currentData() or "lammps",
            "push_off": self.push_off.isChecked(),
            "relax": self.do_relax.isChecked(),
            "push_steps": int(self.push_steps.value()),
            "min_iter": int(self.min_iter.value()),
            "lammps_exe": self.lammps_exe.text().strip(),
            "gmx_exe": self.gmx_exe.text().strip(),
            "engine": self.engine.currentData() or "auto",
            "mpi_ranks": int(self.mpi_ranks.value()),
        }

        self._set_busy(True)
        self._thread = QThread(self)
        self._worker = _BuildWorker(self._composition, options)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_build_progress)
        self._worker.fraction.connect(
            lambda f: self.bar_progress.setValue(int(f * 100)))
        self._worker.finished.connect(self._on_built)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)

        # Let Qt own the teardown. Destroying the worker from inside a slot
        # that the worker's own signal invoked is the classic PyQt crash, and
        # calling QThread.wait() on the GUI thread blocks the event loop.
        for sig in (self._worker.finished, self._worker.failed,
                    self._worker.cancelled):
            sig.connect(self._thread.quit)
            sig.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    #: substring -> stage shown while building (checked in order).
    _STAGES = (
        ("dl_field",       "4/5 · Force-field typing (DL_FIELD)"),
        ("push-off",       "3/5 · Overlap push-off (LAMMPS)"),
        ("back-mapping",   "2/5 · Back-mapping to atoms"),
        ("unthread",       "2/5 · Back-mapping to atoms"),
        ("side-group",     "2/5 · Back-mapping to atoms"),
        ("relaxing",       "5/5 · Relaxation"),
        ("wrote cell",     "5/5 · Writing outputs"),
        ("exporting",      "5/5 · Writing outputs"),
        ("bead cell",      "5/5 · Writing outputs"),
        ("probe back-map", "1/5 · Verifying growth (threading probe)"),
        ("chain ",         "1/5 · Growing chains"),
        ("template",       "1/5 · Growing chains"),
        ("regrow",         "1/5 · Growing chains"),
    )

    def _on_build_progress(self, msg: str) -> None:
        """Log the message and reflect the pipeline stage next to the bar."""
        low = msg.lower()
        for key, stage in self._STAGES:
            if key in low:
                if self.stage_lb.text() != stage:
                    self.stage_lb.setText(stage)
                break
        self._rlog(msg)

    def _cancel_build(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.stage_lb.setText("Cancelling …")
            self._rlog("cancelling — stopping the current stage "
                       "(external programs are killed immediately)")

    def _set_busy(self, busy: bool) -> None:
        self.b_build.setEnabled(not busy)
        self.b_solve.setEnabled(not busy)
        self.b_cancel.setVisible(busy)
        self.bar_progress.setVisible(busy)
        self.stage_lb.setVisible(busy)
        if busy:
            self.bar_progress.setValue(0)
            self.stage_lb.setText("Starting …")

    def _on_thread_finished(self) -> None:
        """Drop our references only once Qt has finished with the objects."""
        self._thread = None
        self._worker = None
        self._set_busy(False)

    @pyqtSlot()
    def _on_cancelled(self) -> None:
        self.blocker.setText("Build cancelled. Nothing was written.")
        self._set_badge(self.result_badge, "cancelled", "warn")
        self._rlog("build cancelled by the user")

    def _on_built(self, payload) -> None:
        result, export = payload
        self._result, self._export = result, export

        rg = ([c.radius_of_gyration for c in result.chains] or [0.0])
        self._set_stat(self.res_density,
                       f"{result.density_kg_m3 / 1000:.4f} g/cm³")
        self._set_stat(self.res_cn, f"{result.mean_c_n:.2f}")
        self._set_stat(self.res_rg, f"{sum(rg) / len(rg):.2f} Å")
        notes = list(result.notes)
        if export is not None:
            self._set_stat(
                self.res_atoms,
                f"{export.n_atoms:,} atoms / {export.n_beads:,} beads"
                if export.n_atoms else f"{export.n_beads:,} beads")
            notes.extend(export.messages)
            rr = getattr(export, "relax", None)
            if rr is not None:
                self._fill_relax_table(rr)
                eng = (getattr(rr, "engine", "") or "").upper()
                self._set_badge(
                    self.relax_badge,
                    f"minimised · {eng}" if rr.ran
                    else f"input written, not run · {eng}",
                    "ok" if rr.ran else "warn")
            if export.typed:
                self._set_badge(self.result_badge,
                                f"typed · {export.route}", "ok")
            else:
                self._set_badge(self.result_badge, "built, not typed", "warn")
        else:
            self._set_stat(self.res_atoms, f"{len(result.molecule.atoms):,} beads")
            self._set_badge(self.result_badge, "built", "ok")
        self.res_notes.setText("  ".join(notes))
        self._rlog(result.summary())
        if getattr(export, "relax", None) is not None:
            self._rlog(export.relax.summary())
        if export is not None:
            self._rlog(export.summary())
            for m in export.messages:
                self._rlog(f"  note: {m}")
        self.steps.setCurrentIndex(3)

    def _on_failed(self, message: str) -> None:
        message = message or "unknown error"
        self._rlog(f"build failed: {message}")
        headline = message.splitlines()[0]
        self.blocker.setText(headline)
        self._set_badge(self.result_badge, "build failed", "error")

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Amorphous builder")
        box.setText("The cell could not be built.")
        # The whole message, not just its first line — the guidance about
        # which knob is actually binding is in the paragraphs below it.
        box.setInformativeText(message)
        box.setStandardButtons(QMessageBox.Ok)
        box.exec_()

    # ==================================================== misc
    def _refresh_files(self) -> None:
        # Called from the force-field page's initial refresh, which runs
        # before the export page exists. Nothing to update yet.
        if not hasattr(self, "files"):
            return
        name = self.cell_name.text().strip() or "cell"
        rows = [
            (f"{name}/cell_beads.xyz",
             "the grown backbone, one site per skeletal atom"),
            (f"{name}/cell_beads.data",
             "coarse-grained LAMMPS data — masses, bonds, box; no pair coeffs"),
        ]
        if getattr(self, "atomistic", None) is not None and self.atomistic.isChecked():
            rows.append((f"{name}/cell_atomistic.xyz",
                         "all-atom cell, side groups and hydrogens restored"))
            rows.append((f"{name}/cell_atomistic.mol2",
                         "same, with bond orders (typers read this better)"))
            if self.run_typing.isChecked():
                fmt = (self.output_format.currentData() or "lammps"
                       if hasattr(self, "output_format") else "lammps")
                if fmt in ("lammps", "both"):
                    rows.append((f"{name}/cell.data",
                                 "typed LAMMPS data — written only if typing "
                                 "actually succeeds"))
                    rows.append((f"{name}/cell.in",
                                 "the LAMMPS input beside it: styles and pair "
                                 "coefficients. cell.data alone cannot be "
                                 "run"))
                if fmt in ("gromacs", "both"):
                    rows.append((f"{name}/cell.gro",
                                 "typed GROMACS coordinates — same typing, "
                                 "GROMACS format"))
                    rows.append((f"{name}/cell.top",
                                 "the topology beside it (+ .itp includes); "
                                 "run with gmx grompp/mdrun"))
        self.files.setRowCount(len(rows))
        for i, (f, what) in enumerate(rows):
            self.files.setItem(i, 0, QTableWidgetItem(f))
            self.files.setItem(i, 1, QTableWidgetItem(what))
        base = self.out_dir.text().strip() or self.default_out_dir()
        self.out_hint.setText(
            f"Everything lands in {base}/{name}/" if base else
            "Choose a folder — nothing can be written yet.")

    def _browse_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Folder for amorphous cells",
            self.out_dir.text().strip() or str(Path.home()))
        if d:
            self.out_dir.setText(d)

    def _browse_dl_lib(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "DL_FIELD lib directory",
            self.dl_lib.text().strip() or str(Path.home()))
        if d:
            self.dl_lib.setText(d)

    def _rlog(self, message: str) -> None:
        if message:
            self.log.emit(f"[amorphous] {message}")

    # ------------------------------------------------------- settings hook
    def set_settings_provider(self, fn) -> None:
        """Install a callable returning the pipeline's force field / paths.

        Supplied explicitly by the main window. Walking ``self.window()``
        instead fails silently when the widget is reparented, and the failure
        mode — exporting with the DEFAULT force field rather than the chosen
        one — is invisible until someone checks the atom types.
        """
        self._settings_provider = fn
        if not self.out_dir.text().strip():
            self.out_dir.setText(self.default_out_dir())
        # Inherit the pipeline's DL_FIELD library path.
        #
        # This page has its own field, and an empty one means dl_field is
        # never found — typing fails, and the cell exports its structure
        # files but no cell.data. The user had already set the path on the
        # Force-field page; there was no reason to make them find it twice,
        # and no sign on this page that it was needed at all. Only an EMPTY
        # field is filled, so a path chosen here still wins.
        try:
            inherited = (self._pipeline_settings().get("dl_lib") or "").strip()
            if inherited and not self.dl_lib.text().strip():
                self.dl_lib.setText(inherited)
        except Exception:
            pass
        self._refresh_files()

    def _pipeline_settings(self) -> dict:
        fn = getattr(self, "_settings_provider", None)
        if fn is None:
            return {}
        try:
            return fn() or {}
        except Exception as exc:
            self._rlog(f"could not read the pipeline settings: {exc}")
            return {}

    def default_out_dir(self) -> str:
        """``<pipeline output>/<project>/cells``.

        Nested rather than dropped in the pipeline output root, so a cell
        called "cell" cannot collide with the pipeline's own artefacts — the
        Reaction-scheme branch nests its folders for the same reason.
        """
        got = self._pipeline_settings()
        base = str(got.get("out_dir", "") or "")
        if not base:
            return ""
        project = str(got.get("project", "") or "").strip()
        return str(Path(base) / project / "cells") if project else str(
            Path(base) / "cells")

    def _copy_ff_from_pipeline(self) -> None:
        got = self._pipeline_settings()
        if not got:
            QMessageBox.information(
                self, "Amorphous builder",
                "The pipeline settings are not available. This happens when "
                "the tab is used outside the main PAAF window.")
            return
        key = got.get("ff_key")
        for i in range(self.ff_combo.count()):
            if self.ff_combo.itemData(i) == key:
                self.ff_combo.setCurrentIndex(i)
                break
        if got.get("dl_lib"):
            self.dl_lib.setText(str(got["dl_lib"]))
        self._rlog(f"copied force field '{key}' from the pipeline")
