"""Materials Studio-inspired main window for paaf.

Layout:

    +------------------------------------------------------------+
    | menubar                                                    |
    +------+-----------------------------------------------------+
    |      | Project header / subtitle                           |
    | side +-----------------------------------------------------+
    | bar  |                                                     |
    |      |   Active page (Monomers / Chain / FF / Optimizer /  |
    |      |   Box + LAMMPS / Reactions / Builder / Run)         |
    |      |                                                     |
    +------+-----------------------------------------------------+
    | log console (collapsible bottom pane)                      |
    +------------------------------------------------------------+
    | status bar                                                  |
    +------------------------------------------------------------+
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
    QStackedWidget, QStatusBar, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .page import wrap_tooltip

from .. import __full_name__, __product_name__
from ..__version__ import __version__
from ..config import (
    BoxCfg, ChainCfg, Config, ForceFieldCfg, LammpsCfg, MonomerSpec,
    OptimizerCfg,
)
from ..ff_registry import list_ffs
from ..logging_utils import QtLogHandler, get_logger
from ..pipeline import run_pipeline
from .builder_tab import BuilderTab
from .blend_tab import BlendTab
# System builder is parked for now:
# from .cell_tab import CellTab
# Reactions tool is parked for now:
# from .reactions_tab import ReactionsTab
from . import theme
from . import tokens as _tok
from . import tokens as T

log = get_logger("paaf.gui")


def _parse_int(text: str, default: int = 0) -> int:
    """Parse an integer from a combobox / lineedit value.

    Accepts human-friendly formats like ``"10 000"``, ``"10,000"``, or plain
    ``"10000"``. Falls back to `default` on error.
    """
    s = (text or "").replace(" ", "").replace(",", "").strip()
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return default


def _parse_float(text: str, default: float = 0.0) -> float:
    """Parse a float that may use scientific notation (1.0e-6)."""
    s = (text or "").replace(" ", "").strip().lower()
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


# =========================================================== worker thread
class Worker(QObject):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    @pyqtSlot()
    def run(self) -> None:
        try:
            res = run_pipeline(self.cfg, progress=self.progress.emit)
            self.finished.emit(res)
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


# =========================================================== main window
NAV_ITEMS = [
    ("Builder",        "Upload / SMILES / periodic table — Step 1"),
    ("Chain",          "Polymerize monomer into a chain — Step 2"),
    ("Optimize",       "OpenBabel MMFF94 / UFF / GAFF — Step 3 (minimizes the full chain)"),
    ("Force field",    "OPLS-AA, GAFF, PCFF, COMPASS, ... — Step 4"),
    ("Box",            "Pack chains: shape, dimensions, g/cm³ density — Step 5"),
    ("Export",         "Generate LAMMPS / GROMACS topology files — Step 6"),
    # Reactions tool is parked for now (crosslinker; code kept).
    # System builder is parked for now; Layering replaces it in the sidebar.
    ("Layering",       "Stack components into their own regions: sizes, axis, gap (independent)"),
    ("Blend",          "Multi-component polymer blend packing (independent)"),
    ("Amorphous cell", "Grow chains into a periodic cell: composition + density → simulation-ready files"),
]

# Pipeline step indices (used by Next/Back navigation, so 'Reactions' and
# 'System builder' are excluded from the linear flow).
PIPELINE_INDICES = [0, 1, 2, 3, 4, 5]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{__product_name__} — {__full_name__}  v{__version__}")
        try:
            from .branding import app_icon
            _icon = app_icon()
            if _icon is not None:
                self.setWindowIcon(_icon)
        except Exception:
            pass
        self.resize(1360, 860)
        self.setMinimumSize(900, 600)   # readable minimum; scroll areas do the rest
        self._worker: Optional[Worker] = None
        self._thread: Optional[QThread] = None
        self._build_menu()
        self._build_central()
        self._wire_logging()

    # --------------------------------------------------------- menu
    def _build_menu(self) -> None:
        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        for label, slot in [
            ("&New project", self._new_project),
            ("&Project settings...", self._edit_project),
            ("&Load config...", self._load_config),
            ("&Save config...", self._save_config),
        ]:
            a = QAction(label, self); a.triggered.connect(slot); m_file.addAction(a)
        m_file.addSeparator()
        a = QAction("&Quit", self); a.triggered.connect(self.close); m_file.addAction(a)

        m_view = mb.addMenu("&View")
        a = QAction("Toggle log console", self, checkable=True); a.setChecked(True)
        a.triggered.connect(self._toggle_console); m_view.addAction(a)

        m_tools = mb.addMenu("&Tools")
        a = QAction("Convert DL_FIELD .par...", self); a.triggered.connect(self._convert_par_dialog)
        m_tools.addAction(a)
        a = QAction("List force fields...", self); a.triggered.connect(self._show_ff_list)
        m_tools.addAction(a)

        m_help = mb.addMenu("&Help")
        a = QAction("About...", self); a.triggered.connect(self._about); m_help.addAction(a)

    # --------------------------------------------------------- central
    def _build_central(self) -> None:
        wrapper = QSplitter(Qt.Vertical, self)

        top = QSplitter(Qt.Horizontal, self)
        # ---- sidebar: pipeline stepper + independent tools (redesign v1).
        # API-compatible with the QListWidget it replaces (setCurrentRow /
        # currentRow / currentRowChanged), so navigation code is unchanged.
        from .sidebar import Sidebar
        self.sidebar = Sidebar(NAV_ITEMS, PIPELINE_INDICES)
        self.sidebar.currentRowChanged.connect(self._on_nav)
        self.sidebar.projectClicked.connect(self._edit_project)
        top.addWidget(self.sidebar)

        # ---- pages container
        right = QWidget(); rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)
        # Spec: every page reads  breadcrumb → 20px title → one-line
        # description → status badge.
        self.breadcrumb = QLabel("")
        self.breadcrumb.setObjectName("breadcrumb")
        self.header = QLabel("Builder"); self.header.setObjectName("project_header")
        self.subheader = QLabel("")
        self.subheader.setObjectName("project_sub")
        rl.addWidget(self.breadcrumb)
        rl.addWidget(self.header)
        rl.addWidget(self.subheader)

        self.pages = QStackedWidget()
        rl.addWidget(self.pages, 1)
        top.addWidget(right)
        top.setStretchFactor(0, 0)
        top.setStretchFactor(1, 1)
        wrapper.addWidget(top)

        # ---- log console (collapsible: drag the splitter handle to hide)
        self.console = QPlainTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(0)
        self.console.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # Wrap long log lines instead of scrolling horizontally.
        self.console.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.console.setWordWrapMode(3)   # WrapAnywhere
        wrapper.addWidget(self.console)
        wrapper.setStretchFactor(0, 5)
        wrapper.setStretchFactor(1, 1)
        wrapper.setCollapsible(1, True)   # allow the console to be collapsed to zero
        wrapper.setSizes([700, 160])

        # ---- status bar
        sb = QStatusBar(); self.setStatusBar(sb)
        sb.showMessage("Ready")

        # Helper: build page → shrinkify → wrap-with-nav → wrap-in-scroll.
        def _mk(page: QWidget, step: int) -> QWidget:
            self._shrinkify(page)
            return self._scroll(self._wrap_with_nav(page, step))

        # -------- Pipeline pages (in workflow order) --------
        # 0: Builder (uploads + SMILES + periodic table + shared monomers table)
        self.builder_tab = BuilderTab()
        self.builder_tab.log.connect(self._append_log)
        self.builder_tab.built_file.connect(self._on_builder_output)
        self.builder_tab.copolymer_settings_changed.connect(self._on_copolymer_settings_changed)
        self.pages.addWidget(_mk(self.builder_tab, 0))                   # 0

        # 1: Chain      (build polymer chain from monomer)
        self.pages.addWidget(_mk(self._build_chain_page(), 1))
        # 2: Optimize   (minimize the FULL chain — replaces the tiny per-
        #    monomer opt that used to hide under Builder)
        self.pages.addWidget(_mk(self._build_opt_page(),   2))
        # 3: Force field
        self.pages.addWidget(_mk(self._build_ff_page(),    3))
        # 4: Box (packing)
        self.pages.addWidget(_mk(self._build_box_page(),   4))
        # 5: Export (file generation)
        self.pages.addWidget(_mk(self._build_run_page(),   5))

        # -------- Independent tools (no Next/Back navigation) --------
        # Reactions tool parked — Layering is now page 6.

        # System builder (CellTab) is parked; Layering takes its slot so
        # every existing page index after it stays valid.
        from .layering_tab import LayeringTab
        self.layering_tab = LayeringTab()
        self.layering_tab.log.connect(self._append_log)
        self._shrinkify(self.layering_tab)
        self.pages.addWidget(self._scroll(self.layering_tab))            # 7

        self.blend_tab = BlendTab()
        self.blend_tab.log.connect(self._append_log)
        self._shrinkify(self.blend_tab)
        self.pages.addWidget(self._scroll(self.blend_tab))               # 8

        # The amorphous builder manages its own scrolling inside step 1 and
        # has a fixed action bar, so it is added unwrapped — putting it in
        # another scroll area would nest two scrollbars.
        from .amorphous_tab import AmorphousTab
        self.amorphous_tab = AmorphousTab()
        self.amorphous_tab.log.connect(self._append_log)
        self.pages.addWidget(self.amorphous_tab)                         # 9

        self.setCentralWidget(wrapper)
        # The reaction export must use the force field / output folder the
        # user chose, so hand it an explicit accessor rather than letting it
        # guess by walking the parent chain.
        try:
            self.builder_tab.reaction_scheme.set_settings_provider(
                self._reaction_export_settings)
        except Exception as exc:
            self._append_log(
                f"[warn] reaction export settings not wired: {exc}")
        try:
            self.amorphous_tab.set_settings_provider(
                self._reaction_export_settings)
        except Exception as exc:
            self._append_log(
                f"[warn] amorphous cell settings not wired: {exc}")
        try:
            self.blend_tab.set_settings_provider(
                self._reaction_export_settings)
        except Exception as exc:
            self._append_log(f"[warn] blend settings not wired: {exc}")
        try:
            self.layering_tab.set_settings_provider(
                self._reaction_export_settings)
        except Exception as exc:
            self._append_log(f"[warn] layering settings not wired: {exc}")

        self._sync_project_label()
        self.sidebar.setCurrentRow(0)

        theme.apply(QApplication.instance())

    def _shrinkify(self, container: QWidget) -> None:
        """Make a page and every widget inside it shrink-friendly.

        Only touches things that measurably force horizontal overflow:
          * intro QLabels get wordWrap so long descriptions wrap instead of
            pushing the page wide;
          * spin boxes / combos / line edits lose their intrinsic minimum
            width so they can shrink;
          * group boxes get Expanding horizontal size policy with min width 0.

        Deliberately does NOT touch QFormLayout field-growth policy — the
        default (FieldsStayAtSizeHint) actually renders group-box contents
        correctly; overriding it can hide the fields entirely on some
        macOS Qt builds.
        """
        from PyQt5.QtWidgets import (
            QAbstractSpinBox, QComboBox, QLabel, QLineEdit, QGroupBox,
        )
        for w in container.findChildren(QWidget):
            if isinstance(w, (QAbstractSpinBox, QComboBox, QLineEdit)):
                w.setMinimumWidth(0)
            elif isinstance(w, QGroupBox):
                w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                w.setMinimumWidth(0)
            elif isinstance(w, QLabel):
                # Only wrap descriptive labels (multi-word text). Field labels
                # like "a" / "b" / "α (b–c)" stay unwrapped.
                text = w.text() or ""
                if len(text) > 40:
                    w.setWordWrap(True)

    def _wrap_with_nav(self, page: QWidget, step: int) -> QWidget:
        """Add a persistent Back / Next footer bar to a pipeline page.

        `step` is the 0-based pipeline index (0=Builder, 5=Export). Independent
        tools (Reactions, System builder) skip this wrapper — they don't have
        Next/Back.
        """
        wrapper = QWidget()
        wrapper.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        wrapper.setMinimumWidth(0)
        v = QVBoxLayout(wrapper)
        v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        page.setMinimumWidth(0)
        v.addWidget(page, 1)

        # Divider + navigation footer
        footer = QWidget()
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(20, 10, 20, 14); fl.setSpacing(10)
        from . import tokens as _T
        step_label = QLabel(
            f"<span style='color:{_T.TEXT_MUTED};'>Step {step + 1} of "
            f"{len(PIPELINE_INDICES)}: "
            f"<b style='color:{_T.TEXT};'>{NAV_ITEMS[step][0]}</b></span>"
        )
        fl.addWidget(step_label); fl.addStretch(1)

        if step > 0:
            back = QPushButton("‹  Back")
            back.clicked.connect(lambda: self.sidebar.setCurrentRow(PIPELINE_INDICES[step - 1]))
            fl.addWidget(back)
        if step < len(PIPELINE_INDICES) - 1:
            nxt = QPushButton(f"Next: {NAV_ITEMS[PIPELINE_INDICES[step + 1]][0]}  ›")
            nxt.setObjectName("primary")
            # Advancing marks the step complete, which drives the sidebar
            # progress card and unlocks the following step.
            nxt.clicked.connect(
                lambda _=False, s=step: self._advance_step(s))
            fl.addWidget(nxt)
        v.addWidget(footer)

        # The pipeline footer belongs to pipeline pages. The Builder page
        # also hosts the Reaction scheme tab, which is its own 3-step tool —
        # "Next: Chain" under a reaction scheme was pure confusion, so the
        # footer follows the active top tab.
        tabs = getattr(page, "top_tabs", None)
        scheme = getattr(page, "reaction_scheme", None)
        if tabs is not None and scheme is not None:
            def _sync(_i=None):
                footer.setVisible(tabs.currentWidget() is not scheme)
            tabs.currentChanged.connect(_sync)
            _sync()
        return wrapper

    def _scroll(self, inner: QWidget) -> QScrollArea:
        """Wrap a page in a vertical-only scroll area.

        Design decision: vertical scrolling only. Users add or adjust options
        by scrolling down, never left-right. To achieve this the inner widget
        is always resized to the viewport width and its size policy expands
        horizontally so it fills the available space.
        """
        sa = QScrollArea()
        sa.setWidget(inner)
        sa.setWidgetResizable(True)                       # inner follows viewport width
        sa.setFrameShape(QScrollArea.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # NEVER scroll horizontally
        sa.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        inner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # Widgets set a "recommended minimum width"; here we clamp that so
        # nothing forces the scroll area to grow horizontally.
        inner.setMinimumWidth(0)
        return sa

    def _reaction_export_settings(self) -> dict:
        """Settings the Reaction-scheme export needs, read from their pages."""
        return {
            "ff_key": self.ff_combo.currentData() or "opls2005_dl",
            "dl_lib": self.ff_dl_lib.text().strip(),
            "out_dir": self.output_dir.text().strip(),
            "project": self.project_name.text().strip() or "polymer",
            "opt_ff": self.opt_ff.currentText(),
        }

    def _edit_project(self) -> None:
        """Open the project dialog (name + output folder)."""
        from .project_dialog import ProjectDialog
        dlg = ProjectDialog(self.project_name.text(),
                            self.output_dir.text(), parent=self)
        if dlg.exec_() == dlg.Accepted:
            name, out = dlg.values()
            self.project_name.setText(name)
            self.output_dir.setText(out)
            self._sync_project_label()
            self.statusBar().showMessage(
                f"Project '{name}' -> {Path(out) / name}", 6000)

    def _sync_project_label(self) -> None:
        """Keep the sidebar showing the real project name."""
        self.sidebar.set_project_name(
            self.project_name.text().strip() or "untitled project")

    def _advance_step(self, step: int) -> None:
        """Mark pipeline `step` complete and move to the next one.

        Spec: a done step shows its headline RESULT (``n=50``, ``MMFF94``)
        rather than the word "done", so the sidebar doubles as a run summary.
        """
        row = PIPELINE_INDICES[step]
        try:
            self.sidebar.mark_done(row, True)
            self.sidebar.set_step_result(row, self._step_result(row))
        except Exception:
            pass
        self.sidebar.setCurrentRow(PIPELINE_INDICES[step + 1])

    def _step_result(self, row: int) -> str:
        """One-token summary of a finished step, shown in the sidebar."""
        try:
            name = NAV_ITEMS[row][0]
            if name == "Builder":
                n = len(self.builder_tab.monomer_specs())
                return f"{n} monomer" + ("s" if n != 1 else "")
            if name == "Chain":
                return f"n={self.chain_n.value()}"
            if name == "Optimize":
                return self.opt_ff.currentText()
            if name == "Force field":
                return (self.ff_combo.currentData() or "").upper()[:10]
            if name == "Box":
                return f"{self.box_a.value():.0f} Å"
            if name == "Export":
                return self.engine_combo.currentText()
        except Exception:
            pass
        return ""

    def _on_nav(self, row: int) -> None:
        self.pages.setCurrentIndex(row)
        name, sub = NAV_ITEMS[row]
        self.header.setText(name)
        # One-line description, stripped of the old "— Step N" suffix (the
        # breadcrumb and the sidebar already state the step).
        desc = sub.split("—")[0].strip().rstrip(",")
        desc = desc.replace("(independent)", "").strip()
        self.subheader.setText(desc)
        # Breadcrumb: pipeline steps are numbered, tools are marked TOOL.
        if row in PIPELINE_INDICES:
            step = PIPELINE_INDICES.index(row) + 1
            self.breadcrumb.setText(f"Pipeline  ›  Step {step} · {name}")
        else:
            self.breadcrumb.setText(f"Tools  ›  {name}")
        # Chain page (row 2 in the new order) refreshes its copolymer picker
        # from the current Builder table so users don't have to click a
        # separate 'Refresh' button.
        if row == 2 and hasattr(self, "_refresh_copolymer_monomers"):
            try:
                self._refresh_copolymer_monomers()
            except Exception:
                pass

    def _toggle_console(self, checked: bool):
        self.console.setVisible(checked)

    # ================================================== pages (Monomers etc.)
    def _wrap_page(self, inner: QWidget) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        v.setContentsMargins(20, 12, 20, 20); v.setSpacing(14)
        v.addWidget(inner)
        return w

    # NOTE: the standalone Monomers page is gone — its table lives inside
    # BuilderTab.monomer_table now, and everything downstream reads via
    # self.builder_tab.monomer_specs() / .load_monomer_specs().

    def _build_chain_page(self) -> QWidget:
        page = QWidget(); v = QVBoxLayout(page)
        v.setContentsMargins(20, 12, 20, 20); v.setSpacing(14)

        _intro = QLabel(
            "<b>Chain — Step 3</b><br>"
            "Homopolymer: one monomer repeated N times. "
            "Copolymer: 2 monomers with a chosen fraction and sequence mode "
            "(random / alternating / block)."
        )
        _intro.setWordWrap(True)
        v.addWidget(_intro)

        # ---------------------------------------------- basic settings
        gb = QGroupBox("Chain assembly")
        f = QFormLayout(gb)
        self.chain_n = QSpinBox(); self.chain_n.setRange(1, 100000); self.chain_n.setValue(20)
        self.chain_mode = QComboBox()
        self.chain_mode.addItems(["homopolymer", "alternating", "block", "random"])
        self.chain_backend = QComboBox(); self.chain_backend.addItems(["auto", "mbuild", "simple"])
        self.chain_fractions = QLineEdit(); self.chain_fractions.setPlaceholderText("0.7,0.3 (for random)")
        self.chain_blocks = QLineEdit(); self.chain_blocks.setPlaceholderText("5,5 (for block)")
        self.chain_seed = QLineEdit(); self.chain_seed.setPlaceholderText("int (optional)")
        f.addRow("Total monomers per chain", self.chain_n)
        f.addRow("Sequence mode", self.chain_mode)
        f.addRow("Backend", self.chain_backend)
        f.addRow("Fractions (random)", self.chain_fractions)
        f.addRow("Block sizes (block)", self.chain_blocks)
        f.addRow("Random seed", self.chain_seed)
        # Auto-cap the polyester terminal (only takes effect for monomers
        # whose tail heavy atom is a carbonyl C, e.g. PBS/PET/PLA). Left on
        # by default; disable if you want the raw -C(=O)H aldehyde end.
        self.chain_cap_cooh = QCheckBox(
            "Auto-cap polyester chain end as -C(=O)OH  (safe for polyesters "
            "like PBS/PET/PLA; no-op for polyethylene, polystyrene, etc.)")
        self.chain_cap_cooh.setChecked(True)
        f.addRow("", self.chain_cap_cooh)
        v.addWidget(gb)

        # ---------------------------------------------- copolymer helper
        # This whole group box is hidden when the Builder is in Homopolymer
        # mode. Kept as `self._gb_copolymer_helper` so `_on_copolymer_settings_changed`
        # can toggle its visibility live.
        self._gb_copolymer_helper = QGroupBox("Copolymer helper (2 monomers)")
        gb_co = self._gb_copolymer_helper
        fc = QFormLayout(gb_co)
        _cop_intro = QLabel(
            "Pick two monomers (from Builder table), set the fraction of "
            "monomer B, and this fills the Fractions / Sequence-mode fields "
            "above. Fraction of A = 1 − fraction of B."
        )
        _cop_intro.setWordWrap(True)
        fc.addRow(_cop_intro)

        self.co_monomer_a = QComboBox()
        self.co_monomer_b = QComboBox()
        self.co_fraction_b = QDoubleSpinBox()
        self.co_fraction_b.setRange(0.0, 1.0); self.co_fraction_b.setDecimals(2)
        self.co_fraction_b.setSingleStep(0.05); self.co_fraction_b.setValue(0.50)
        self.co_fraction_b.setGroupSeparatorShown(False)
        self.co_fraction_b.setSuffix("  (A fraction = 1 − this)")
        self.co_total = QSpinBox()
        self.co_total.setRange(2, 100000); self.co_total.setValue(20)
        self.co_mode = QComboBox()
        self.co_mode.addItems(["random", "alternating", "block"])
        self.co_seed = QLineEdit(); self.co_seed.setPlaceholderText("int (optional)")

        b_apply_co = QPushButton("Apply to chain settings")
        b_apply_co.setObjectName("primary")
        b_apply_co.clicked.connect(self._apply_copolymer_helper)
        b_refresh_co = QPushButton("Refresh monomer list from Builder")
        b_refresh_co.clicked.connect(self._refresh_copolymer_monomers)

        fc.addRow("Monomer A", self.co_monomer_a)
        fc.addRow("Monomer B", self.co_monomer_b)
        fc.addRow("Fraction of B", self.co_fraction_b)
        fc.addRow("Total monomers", self.co_total)
        fc.addRow("Sequence mode", self.co_mode)
        fc.addRow("Random seed", self.co_seed)
        fc.addRow("", b_refresh_co)
        fc.addRow("", b_apply_co)
        v.addWidget(gb_co)
        # Hidden by default — Builder starts in Homopolymer mode. The
        # copolymer_settings_changed signal from BuilderTab will re-show it
        # if the user toggles to Copolymer.
        gb_co.setVisible(False)

        v.addStretch(1)
        return page

    def _refresh_copolymer_monomers(self):
        """Reload monomer choices from the shared Builder table."""
        specs = self.builder_tab.monomer_specs()
        labels = [f"{i}. {Path(s.file).stem}" for i, s in enumerate(specs)]
        for combo in (self.co_monomer_a, self.co_monomer_b):
            current = combo.currentIndex()
            combo.blockSignals(True); combo.clear()
            for lbl in labels:
                combo.addItem(lbl)
            if current >= 0 and current < len(labels):
                combo.setCurrentIndex(current)
            combo.blockSignals(False)
        # Default to picking A=first, B=second if available
        if self.co_monomer_a.count() > 0 and self.co_monomer_a.currentIndex() < 0:
            self.co_monomer_a.setCurrentIndex(0)
        if self.co_monomer_b.count() > 1 and self.co_monomer_b.currentIndex() <= 0:
            self.co_monomer_b.setCurrentIndex(1)

    def _apply_copolymer_helper(self):
        """Translate the Copolymer helper inputs into the chain-assembly fields."""
        specs = self.builder_tab.monomer_specs()
        ia = self.co_monomer_a.currentIndex()
        ib = self.co_monomer_b.currentIndex()
        if not (0 <= ia < len(specs)) or not (0 <= ib < len(specs)):
            QMessageBox.warning(
                self, "Pick 2 monomers",
                "Add at least two monomers on the Builder page, then click "
                "'Refresh monomer list from Builder' on this tab.")
            return
        if ia == ib:
            QMessageBox.warning(self, "Same monomer",
                                "Monomer A and Monomer B must be different rows.")
            return
        # The chain_builder consumes monomers in the same order as specs, so
        # the copolymer 'A/B' fractions map to the specs list at those indices.
        # For simplicity we require A and B to be indices 0 and 1 respectively;
        # if they aren't, we log a friendly note.
        if not (ia == 0 and ib == 1):
            self._append_log(
                "[copolymer] NOTE: the pipeline consumes monomers in Builder-"
                "table order. Reorder rows so A is row 0 and B is row 1, or "
                "the fractions will refer to the wrong monomers.")
        fb = self.co_fraction_b.value()
        fa = 1.0 - fb
        # Fill the chain-assembly fields
        self.chain_n.setValue(self.co_total.value())
        self.chain_mode.setCurrentText(self.co_mode.currentText())
        self.chain_fractions.setText(f"{fa:.2f},{fb:.2f}")
        if self.co_seed.text().strip():
            self.chain_seed.setText(self.co_seed.text().strip())
        self._append_log(
            f"[copolymer] applied: A(row {ia}, f={fa:.2f}) + "
            f"B(row {ib}, f={fb:.2f}), mode={self.co_mode.currentText()}, "
            f"N={self.co_total.value()}")

    def _build_ff_page(self) -> QWidget:
        # Redesign v1, screen 5: header ▸ [field selection | assignment
        # summary] ▸ DL_FIELD control ▸ action bar.
        from .page import Card, action_bar, button as _btn, page_header, stat_row, wrap_tooltip
        page = QWidget(); v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, T.PAD_PAGE, T.PAD_PAGE); v.setSpacing(T.GAP_SECTION)

        self.ff_status_badge_holder = page_header(
            "Force field",
            "Assign types and charges to every atom of the optimized chain.",
            breadcrumb="Pipeline › Step 4 · Force field",
        )
        v.addWidget(self.ff_status_badge_holder)

        body = QHBoxLayout(); body.setSpacing(T.GAP_SECTION)
        body.setContentsMargins(T.PAD_PAGE, 0, 0, 0)

        card_sel = Card("Field selection")
        f = QFormLayout()
        f.setContentsMargins(0, 0, 0, 0)
        f.setSpacing(T.GAP_FORM_ROW)
        f.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        card_sel.body.addLayout(f)

        self.ff_combo = QComboBox()
        # Grouped under the tool that produces the parameters; the Config
        # default is selected rather than index 0, which is now a heading.
        from .ff_combo import populate_force_field_combo
        from ..config import ForceFieldCfg as _FFCfg
        populate_force_field_combo(self.ff_combo, _FFCfg().key)
        self.ff_combo.currentIndexChanged.connect(self._on_ff_changed)
        self.ff_notes = QLabel(""); self.ff_notes.setWordWrap(True)
        self.ff_notes.setStyleSheet(f"color: {_tok.TEXT_MUTED};")
        self.ff_dl_lib = QLineEdit()
        self.ff_dl_lib.setPlaceholderText("path to dl_f_4.13/lib (only for DL_FIELD-based FFs)")
        # Pre-fill from QSettings / env vars / sensible default.
        default_dl = self._discover_default_dl_lib()
        if default_dl:
            self.ff_dl_lib.setText(default_dl)
        # Lock/unlock buttons — once a valid path is saved, the field goes
        # read-only. User must explicitly click "Change..." to edit it,
        # preventing accidental edits.
        self.b_browse_dl = QPushButton("Browse...")
        self.b_browse_dl.clicked.connect(self._browse_dl_lib)
        self.b_lock_dl = QPushButton("Change...")   # label toggles at runtime
        self.b_lock_dl.clicked.connect(self._toggle_dl_lib_lock)
        # Persist every edit
        self.ff_dl_lib.editingFinished.connect(self._save_dl_lib_setting)
        dl_row = QHBoxLayout()
        dl_row.addWidget(self.ff_dl_lib)
        dl_row.addWidget(self.b_browse_dl)
        dl_row.addWidget(self.b_lock_dl)
        dl_wrap = QWidget(); dl_wrap.setLayout(dl_row)
        # If a valid path was auto-filled at startup, lock it immediately.
        if default_dl:
            self._set_dl_lib_locked(True)
        else:
            self._set_dl_lib_locked(False)
        self.ff_manual_types = QPlainTextEdit()
        self.ff_manual_types.setPlaceholderText(
            "Optional manual atom types, one per line as 'atom_index:ff_type' (0-based)."
        )
        self.ff_manual_types.setMaximumHeight(120)
        b_pick_types = QPushButton("Pick atom types visually...")
        b_pick_types.clicked.connect(self._open_atom_typing_dialog)
        # ---- MD output format chooser -----------------------------------
        # For DL_FIELD-based FFs this is written into polymer.control's
        # 'output_engine' line, so dl_field emits LAMMPS or GROMACS
        # topology directly. For Moltemplate-native FFs the pipeline picks
        # the corresponding writer downstream. Always visible; the same
        # value is mirrored on the Export page so users can also change it
        # there.
        self.ff_engine_combo = QComboBox()
        self.ff_engine_combo.addItems(["lammps", "gromacs", "both"])
        self.ff_engine_combo.setToolTip(wrap_tooltip("Choose whether to generate LAMMPS (default) or GROMACS files "
            "for this run. For DL_FIELD force fields this rewrites the "
            "'output_engine' line in polymer.control."))
        self.ff_engine_combo.currentTextChanged.connect(
            self._on_ff_engine_changed)
        _engine_note = QLabel(
            f"<span style='color:{_tok.TEXT_MUTED};'>Choose the target MD engine. "
            "For DL_FIELD-based FFs (OPLS 2005 DL, COMPASS, PCFF, CVFF, "
            "CHARMM36, GROMOS 54A7, GAFF, TraPPE-EH, OPLS-UA) this is written "
            "straight into the <tt>polymer.control</tt> file so dl_field "
            "produces the right topology.</span>")
        _engine_note.setWordWrap(True)

        f.addRow("Force field",       self.ff_combo)
        f.addRow("",                   self.ff_notes)
        f.addRow("MD output format",   self.ff_engine_combo)
        f.addRow("",                   _engine_note)
        f.addRow("DL_FIELD lib dir",   dl_wrap)
        f.addRow("Manual atom types",  self.ff_manual_types)
        f.addRow("",                   b_pick_types)
        body.addWidget(card_sel, 1)

        # ---- Assignment summary (right column, fixed width per spec)
        card_sum = Card("Assignment summary")
        card_sum.setFixedWidth(300)
        self.ff_stat_monomers = stat_row("Monomers", "0")
        self.ff_stat_overrides = stat_row("Manual overrides", "0")
        self.ff_stat_engine = stat_row("MD output", "lammps")
        self.ff_stat_kind = stat_row("FF family", "—")
        for wdg in (self.ff_stat_monomers, self.ff_stat_overrides,
                    self.ff_stat_engine, self.ff_stat_kind):
            card_sum.body.addWidget(wdg)
        card_sum.body.addStretch(1)
        _hint = QLabel(
            "Typing runs when you generate files. Use "
            "<i>Pick atom types visually…</i> to inspect the chemical "
            "environment of every atom first.")
        _hint.setWordWrap(True)
        _hint.setProperty("role", "hint")
        card_sum.body.addWidget(_hint)
        body.addWidget(card_sum)
        v.addLayout(body)

        # ---- DL_FIELD control-file preview (only visible for DL_FIELD FFs)
        self.gb_dlctrl = QGroupBox(
            "DL_FIELD polymer.control (edit before generation)")
        gc = QVBoxLayout(self.gb_dlctrl)
        gc.addWidget(QLabel(
            "This is the control file dl_field will use. Edit any line and "
            "click 'Save changes' — the edits are preserved for the next "
            "Generate-files run. Click 'Reset to defaults' to regenerate."
        ))
        self.dlctrl_edit = QPlainTextEdit()
        self.dlctrl_edit.setPlaceholderText(
            "The polymer.control contents will appear here after Generate files "
            "runs at least once — or click 'Preview' below to render it now.")
        self.dlctrl_edit.setMinimumHeight(220)
        gc.addWidget(self.dlctrl_edit)
        row = QHBoxLayout()
        b_preview = QPushButton("Preview polymer.control now")
        b_preview.clicked.connect(self._preview_dlfield_control)
        b_save_ctrl = QPushButton("Save changes"); b_save_ctrl.setObjectName("primary")
        b_save_ctrl.clicked.connect(self._save_dlfield_control_edits)
        b_reset_ctrl = QPushButton("Reset to defaults")
        b_reset_ctrl.clicked.connect(self._reset_dlfield_control)
        row.addWidget(b_preview); row.addWidget(b_save_ctrl)
        row.addWidget(b_reset_ctrl); row.addStretch(1)
        gc.addLayout(row)
        _dl_wrap = QWidget()
        _dlv = QVBoxLayout(_dl_wrap)
        _dlv.setContentsMargins(T.PAD_PAGE, 0, 0, 0)
        _dlv.addWidget(self.gb_dlctrl)
        v.addWidget(_dl_wrap)
        # Visibility follows the FF selection.
        self._dlctrl_custom_text: Optional[str] = None
        self._update_dlctrl_visibility()

        v.addStretch(1)
        # ---- action bar: one primary, right-most (spec rule)
        _bar_wrap = QWidget()
        _bw = QHBoxLayout(_bar_wrap)
        _bw.setContentsMargins(T.PAD_PAGE, 0, 0, 0)
        b_types = _btn("Pick atom types visually…")
        b_types.clicked.connect(self._open_atom_typing_dialog)
        _bw.addWidget(action_bar(
            "Types are assigned when you generate files.", [b_types]))
        v.addWidget(_bar_wrap)

        self._on_ff_changed(0)
        return page

    def _build_opt_page(self) -> QWidget:
        page = QWidget(); v = QVBoxLayout(page)
        v.setContentsMargins(20, 12, 20, 20); v.setSpacing(14)

        _intro = QLabel(
            "<b>Optimizer — Step 2</b><br>"
            "Geometry minimization via OpenBabel. Pick a force field, a step "
            "count preset (or type a custom value), and a convergence tolerance "
            "in scientific notation (e.g. <code>1.0e-6</code>)."
        )
        _intro.setWordWrap(True)
        v.addWidget(_intro)

        gb = QGroupBox("OpenBabel optimizer")
        f = QFormLayout(gb)
        self.opt_enabled = QCheckBox("Optimize monomers + chain with OpenBabel")
        self.opt_enabled.setChecked(True)

        # Force field — combobox with the OpenBabel-supported list
        self.opt_ff = QComboBox()
        self.opt_ff.addItems(["MMFF94", "MMFF94s", "UFF", "Ghemical", "GAFF"])

        # ---------- Max steps as an editable combobox with presets
        self.opt_steps = QComboBox(); self.opt_steps.setEditable(True)
        for n in (500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000, 500_000):
            self.opt_steps.addItem(f"{n:,}".replace(",", " "))
        self.opt_steps.setCurrentText("10 000")
        self.opt_steps.setToolTip(wrap_tooltip("Pick a preset or type a custom integer."))

        # ---------- Convergence tolerance as an editable combobox in sci-notation
        self.opt_tol = QComboBox(); self.opt_tol.setEditable(True)
        for t in ("1.0e-3", "1.0e-4", "1.0e-5", "1.0e-6", "1.0e-8", "1.0e-10", "1.0e-12"):
            self.opt_tol.addItem(t)
        self.opt_tol.setCurrentText("1.0e-6")
        self.opt_tol.setToolTip(wrap_tooltip("Convergence tolerance in scientific notation.\n"
            "Pick a preset (1.0e-6 is a good default) or type a custom value."
        ))

        # ---------- Algorithm
        self.opt_alg = QComboBox()
        self.opt_alg.addItem("cg  — Conjugate Gradients (recommended)", userData="cg")
        self.opt_alg.addItem("sd  — Steepest Descent",                  userData="sd")

        # ---------- Energy reporting
        gb_energy = QGroupBox("Energy reporting")
        ef = QFormLayout(gb_energy)
        self.opt_report_energy = QCheckBox(
            "Log initial + final energy (ΔE shown after each optimization)")
        self.opt_report_energy.setChecked(True)
        self.opt_report_energy.setToolTip(wrap_tooltip("When on, the console shows lines like:\n"
            "  Optimization done (MMFF94): initial E = 234.7 → final E = 128.4 (ΔE = -106.3)"
        ))
        ef.addRow(self.opt_report_energy)

        f.addRow(self.opt_enabled)
        f.addRow("Force field", self.opt_ff)
        f.addRow("Max steps", self.opt_steps)
        f.addRow("Convergence tol", self.opt_tol)
        f.addRow("Algorithm", self.opt_alg)
        v.addWidget(gb)
        v.addWidget(gb_energy)
        v.addStretch(1)
        return page

    def _build_box_page(self) -> QWidget:
        """Box page: only packing options. No execution here."""
        page = QWidget(); v = QVBoxLayout(page)
        v.setContentsMargins(20, 12, 20, 20); v.setSpacing(14)
        _lbl = QLabel(
            "<b>Box — Step 5</b><br>"
            "Pack N copies of your chain into a periodic box. Choose the "
            "shape and either set edge lengths directly or target a density."
        )
        _lbl.setWordWrap(True)
        v.addWidget(_lbl)

        # ---------- Cell shape
        gb_shape = QGroupBox("Cell shape")
        fs = QFormLayout(gb_shape)
        self.box_shape = QComboBox()
        self.box_shape.addItems(["cubic", "orthorhombic", "triclinic"])
        self.box_shape.currentTextChanged.connect(self._on_box_shape_changed)
        self.n_chains = QSpinBox(); self.n_chains.setRange(1, 100000); self.n_chains.setValue(1)
        self.box_packmol = QCheckBox("Use packmol / mbuild if available (recommended)")
        self.box_packmol.setChecked(True)
        # ---- Packmol binary path (editable, auto-filled, persisted)
        self.box_packmol_path = QLineEdit()
        self.box_packmol_path.setPlaceholderText(
            "path to packmol binary (auto-detected if empty)")
        _default_packmol = self._discover_default_packmol()
        if _default_packmol:
            self.box_packmol_path.setText(_default_packmol)
        b_browse_pk = QPushButton("Browse...")
        b_browse_pk.clicked.connect(self._browse_packmol)
        self.box_packmol_path.editingFinished.connect(self._save_packmol_setting)
        pk_row = QHBoxLayout(); pk_row.addWidget(self.box_packmol_path); pk_row.addWidget(b_browse_pk)
        pk_wrap = QWidget(); pk_wrap.setLayout(pk_row)

        # Packmol seed — same semantics as the Blend page.
        self.box_packmol_seed = QSpinBox()
        self.box_packmol_seed.setRange(-1, 2_000_000_000)
        self.box_packmol_seed.setValue(-1)
        self.box_packmol_seed.setToolTip(wrap_tooltip("-1 → fresh random seed each run (different packing every time).\n"
            "Any value ≥ 0 → fixed seed; identical packing every run."))
        _seed_note = QLabel(
            f"<span style='color:{_tok.TEXT_MUTED};font-size:{_tok.FS_CAPTION}px;'>"
            "<b>-1</b> = fresh random seed each run → different packing every time. &nbsp;"
            "Any positive integer (e.g. 20) = fixed seed → identical packing every run."
            "</span>")
        _seed_note.setWordWrap(True)

        # Packmol tolerance — same knob the Blend page exposes.
        self.box_packmol_tol = QDoubleSpinBox()
        self.box_packmol_tol.setRange(0.5, 10.0); self.box_packmol_tol.setSingleStep(0.1)
        self.box_packmol_tol.setDecimals(2); self.box_packmol_tol.setValue(2.0)
        self.box_packmol_tol.setToolTip(wrap_tooltip("Minimum distance (Å) between atoms of different molecules. "
            "2.0 is safe for polymers; 1.5 packs tighter."))

        fs.addRow("Number of chains", self.n_chains)
        fs.addRow("Box shape", self.box_shape)
        fs.addRow(self.box_packmol)
        fs.addRow("Packmol binary", pk_wrap)
        fs.addRow("Packmol seed", self.box_packmol_seed)
        fs.addRow("", _seed_note)
        fs.addRow("Packmol tolerance (Å)", self.box_packmol_tol)
        v.addWidget(gb_shape)

        # ---------- GROMACS-native packing (visible when engine=gromacs) ---
        # Instead of packmol, use 'gmx insert-molecules' with either an
        # explicit chain count OR a target atom limit (matches user's
        # create_box.py workflow).
        self.gb_gmx = QGroupBox("GROMACS packing (gmx insert-molecules)")
        gg = QFormLayout(self.gb_gmx); gg.setContentsMargins(8, 8, 8, 8); gg.setSpacing(6)
        self.gmx_mode = QComboBox()
        self.gmx_mode.addItems(["Explicit chain count", "Target atom count"])
        self.gmx_mode.currentIndexChanged.connect(self._on_gmx_mode_changed)
        self.gmx_target_atoms = QSpinBox()
        self.gmx_target_atoms.setRange(0, 10_000_000); self.gmx_target_atoms.setValue(50000)
        self.gmx_target_atoms.setToolTip(wrap_tooltip("Pipeline computes n_chains = target_atoms // atoms_per_chain "
            "(exact same rule as your create_box.py script)."))
        self.gmx_pre_min = QCheckBox("Energy-minimise single chain before insertion")
        self.gmx_pre_min.setChecked(True)
        self.gmx_pre_min.setToolTip(wrap_tooltip("Runs `gmx grompp + mdrun` with a steepest-descent enmin.mdp on "
            "the single chain first, so chains aren't sterically strained "
            "before insert-molecules starts placing copies."))
        self.gmx_try = QSpinBox()
        self.gmx_try.setRange(100, 10_000_000); self.gmx_try.setValue(100000)
        self.gmx_try.setToolTip(wrap_tooltip("gmx insert-molecules -try N (default 100000)"))
        gg.addRow("Chains via", self.gmx_mode)
        gg.addRow("Target atoms", self.gmx_target_atoms)
        gg.addRow("", self.gmx_pre_min)
        gg.addRow("Try count", self.gmx_try)
        _gmx_note = QLabel(
            "Only used when <b>MD engine output</b> is set to <tt>gromacs</tt> "
            "on the Force-field or Export page. Requires <tt>gmx</tt> or "
            "<tt>gmx_mpi</tt> on your PATH.")
        _gmx_note.setWordWrap(True); _gmx_note.setProperty("role", "hint")
        gg.addRow("", _gmx_note)
        v.addWidget(self.gb_gmx)
        # Initialise visibility to match current engine choice.
        self._sync_gmx_visibility()
        self._on_gmx_mode_changed(0)

        # ---------- Dimensions
        gb_dim = QGroupBox("Cell edge lengths (Å)")
        fd = QFormLayout(gb_dim)
        def _edge():
            w = QDoubleSpinBox()
            w.setRange(1.0, 10000.0)
            w.setDecimals(1)              # 50.0 Å reads better than 50.000 Å
            w.setSingleStep(1.0)
            w.setGroupSeparatorShown(False)   # never insert thousand separators
            w.setValue(50.0)
            w.setSuffix(" Å"); return w
        self.box_a = _edge()
        self.box_b = _edge()
        self.box_c = _edge()
        fd.addRow("a", self.box_a)
        fd.addRow("b", self.box_b)
        fd.addRow("c", self.box_c)
        v.addWidget(gb_dim)
        self._gb_dim = gb_dim   # (kept so we could hide it in "density mode" later)

        # ---------- Angles (triclinic only)
        gb_ang = QGroupBox("Cell angles (degrees) — triclinic only")
        fa = QFormLayout(gb_ang)
        def _ang():
            w = QDoubleSpinBox()
            w.setRange(1.0, 179.0)
            w.setDecimals(1)
            w.setSingleStep(1.0)
            w.setGroupSeparatorShown(False)
            w.setValue(90.0)
            w.setSuffix(" °"); return w
        self.box_alpha = _ang()
        self.box_beta  = _ang()
        self.box_gamma = _ang()
        fa.addRow("α  (b–c)", self.box_alpha)
        fa.addRow("β  (a–c)", self.box_beta)
        fa.addRow("γ  (a–b)", self.box_gamma)
        v.addWidget(gb_ang)
        self._gb_angles = gb_ang

        # ---------- Density-based sizing
        gb_dens = QGroupBox("Target density (optional — overrides edge lengths)")
        fdn = QFormLayout(gb_dens)
        self.box_use_density = QCheckBox("Derive edge lengths from target density")
        self.box_density = QDoubleSpinBox()
        self.box_density.setRange(0.05, 25.00)
        self.box_density.setDecimals(2)                 # 1.00 g/cm³ style
        self.box_density.setSingleStep(0.05)
        self.box_density.setGroupSeparatorShown(False)
        self.box_density.setValue(1.00)
        self.box_density.setSuffix(" g/cm³")
        fdn.addRow(self.box_use_density)
        fdn.addRow("Density", self.box_density)
        v.addWidget(gb_dens)

        v.addStretch(1)
        self._on_box_shape_changed(self.box_shape.currentText())
        return page

    def _on_box_shape_changed(self, shape: str):
        """Enable/disable dimension and angle fields based on shape."""
        # b and c are only meaningful for orthorhombic and triclinic
        cubic = (shape == "cubic")
        self.box_b.setEnabled(not cubic)
        self.box_c.setEnabled(not cubic)
        # When cubic, sync b and c to a so they show the current value even
        # though they're disabled.
        if cubic:
            self.box_b.setValue(self.box_a.value())
            self.box_c.setValue(self.box_a.value())
        # And keep them synced live while cubic is selected.
        try:
            self.box_a.valueChanged.disconnect(self._sync_cubic_box)
        except (TypeError, RuntimeError):
            pass
        if cubic:
            self.box_a.valueChanged.connect(self._sync_cubic_box)
        # Angles are only meaningful for triclinic
        tri = (shape == "triclinic")
        for w in (self.box_alpha, self.box_beta, self.box_gamma):
            w.setEnabled(tri)
        self._gb_angles.setVisible(tri)

    def _sync_cubic_box(self, value: float):
        """Mirror the a-edge to b and c while cubic is selected."""
        self.box_b.setValue(value); self.box_c.setValue(value)

    def _build_run_page(self) -> QWidget:
        """Run page: ensemble, damping, multi-stage, engine, execute."""
        page = QWidget(); v = QVBoxLayout(page)
        v.setContentsMargins(20, 12, 20, 20); v.setSpacing(14)
        _lbl_e = QLabel(
            "<b>Export — Step 6</b><br>"
            "Generates LAMMPS or GROMACS topology files. MD execution is off "
            "by default — tick <i>Auto-run</i> below to also invoke the engine."
        )
        _lbl_e.setWordWrap(True)
        v.addWidget(_lbl_e)

        # Project + engine
        gb_proj = QGroupBox("Project")
        pf = QFormLayout(gb_proj)
        self.project_name = QLineEdit("polymer")
        # The page title shows the current STEP; the project name lives in the
        # sidebar brand block (redesign v1).
        self.project_name.textChanged.connect(
            lambda s: self.sidebar.set_project_name(s or "untitled project"))
        self.output_dir = QLineEdit(str(Path.cwd() / "output"))
        b_out = QPushButton("Browse..."); b_out.clicked.connect(self._browse_output_dir)
        out_row = QHBoxLayout(); out_row.addWidget(self.output_dir); out_row.addWidget(b_out)
        out_wrap = QWidget(); out_wrap.setLayout(out_row)
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["lammps", "gromacs", "both"])
        # Reverse-sync: if the user changes engine here, mirror it back to
        # the FF page's chooser (which drives polymer.control's output_engine).
        self.engine_combo.currentTextChanged.connect(
            lambda eng: self._sync_ff_engine_from_export(eng))
        # Explicit topology-builder choice — overrides the ff.kind auto-detect.
        self.topology_builder = QComboBox()
        self.topology_builder.addItem("Auto (based on chosen force field)", userData="auto")
        self.topology_builder.addItem("moltemplate.sh", userData="moltemplate")
        self.topology_builder.addItem("dl_field", userData="dl_field")
        self.gromacs_itp = QLineEdit()
        self.gromacs_itp.setPlaceholderText("optional path to a GROMACS FF .itp (e.g. oplsaa.ff/forcefield.itp)")
        self.run_moltemplate = QCheckBox(
            "Also run moltemplate.sh (Moltemplate-native FFs only) — "
            "DL_FIELD FFs always auto-run dl_field regardless of this checkbox."
        )
        self.run_moltemplate.setChecked(True)
        self.project_path_hint = QLabel("")
        self.project_path_hint.setProperty("role", "hint")
        self.project_path_hint.setWordWrap(True)

        def _hint():
            base = self.output_dir.text().strip() or "output"
            nm = self.project_name.text().strip() or "polymer"
            self.project_path_hint.setText(f"Files are written to:  {Path(base) / nm}")
        self.project_name.textChanged.connect(lambda _s: _hint())
        self.output_dir.textChanged.connect(lambda _s: _hint())
        _hint()

        pf.addRow("Project name", self.project_name)
        pf.addRow("Output directory", out_wrap)
        pf.addRow("", self.project_path_hint)
        pf.addRow("MD engine output", self.engine_combo)
        pf.addRow("Topology builder", self.topology_builder)
        pf.addRow("GROMACS #include .itp", self.gromacs_itp)
        pf.addRow("", self.run_moltemplate)
        v.addWidget(gb_proj)

        # Ensemble/damping controls belong to the run, not the build.
        # They are kept as hidden defaults on `self.lmp_*` so the rest of the
        # pipeline (which still constructs a LammpsCfg) is unchanged, but they
        # do NOT appear in the GUI. Users configure the ensemble in their own
        # LAMMPS/GROMACS input if they want to run MD.
        self.lmp_ensemble = QComboBox(); self.lmp_ensemble.addItems(["nvt", "npt", "nve"])
        self.lmp_ensemble.setCurrentText("npt")
        self.lmp_temp   = QDoubleSpinBox(); self.lmp_temp.setValue(300.0)
        self.lmp_press  = QDoubleSpinBox(); self.lmp_press.setValue(1.0)
        self.lmp_steps  = QSpinBox();  self.lmp_steps.setRange(100, 100_000_000); self.lmp_steps.setValue(500_000)
        self.lmp_ts     = QDoubleSpinBox(); self.lmp_ts.setValue(1.0)
        self.lmp_tdamp  = QDoubleSpinBox(); self.lmp_tdamp.setRange(1.0, 100000.0); self.lmp_tdamp.setValue(100.0)
        self.lmp_pdamp  = QDoubleSpinBox(); self.lmp_pdamp.setRange(1.0, 1_000_000.0); self.lmp_pdamp.setValue(1000.0)
        self.lmp_thermostat = QComboBox(); self.lmp_thermostat.addItems(["nose-hoover", "langevin", "berendsen", "csvr"])
        self.lmp_barostat   = QComboBox(); self.lmp_barostat.addItems(["nose-hoover", "berendsen", "parrinello"])
        self.lmp_coupling   = QComboBox(); self.lmp_coupling.addItems(["iso", "aniso", "tri", "x", "y", "z"])
        self.lmp_seed       = QSpinBox(); self.lmp_seed.setRange(1, 2_000_000_000); self.lmp_seed.setValue(4928459)
        self.lmp_multistage = QCheckBox(); self.lmp_multistage.setChecked(False)
        self.lmp_minimize   = QCheckBox(); self.lmp_minimize.setChecked(False)  # DEFAULT OFF now
        self.lmp_nvt_steps  = QSpinBox(); self.lmp_nvt_steps.setValue(0)
        self.lmp_nvt_temp   = QDoubleSpinBox(); self.lmp_nvt_temp.setValue(0.0)
        self.lmp_npt_steps  = QSpinBox(); self.lmp_npt_steps.setValue(0)
        self.lmp_thermo_every = QSpinBox(); self.lmp_thermo_every.setValue(1000)
        self.lmp_dump_every   = QSpinBox(); self.lmp_dump_every.setValue(0)

        # What actually appears on the Export page:
        gb_out = QGroupBox("What will be generated")
        vl = QVBoxLayout(gb_out)
        _out_lbl = QLabel(
            "<b>Moltemplate-native FFs</b> (OPLS-AA, GAFF, TraPPE, DREIDING, "
            "COMPASS-published, SDK, MARTINI …): "
            "<code>&lt;project&gt;.lt</code>, <code>system.lt</code>, plus "
            "<code>system.data</code>, <code>system.in.init</code>, "
            "<code>system.in.settings</code> via <code>moltemplate.sh</code>."
            "<br><br>"
            "<b>DL_FIELD-based FFs</b> (PCFF, COMPASS, CVFF, OPLS 2005, "
            "TraPPE-EH, CHARMM36, GROMOS 54A7, AMBER GAFF, OPLS-UA): "
            "LAMMPS <code>lammps.data</code> + <code>lammps.in</code>, or "
            "GROMACS <code>system.gro</code> + <code>system.top</code> + "
            "<code>*.mdp</code>, via <code>dl_field</code>. "
            "<b>Always auto-runs</b> — dl_field is invoked as soon as you "
            "click Generate files, regardless of the Auto-run checkbox."
            "<br><br>"
            "No ensemble / thermostat / damping is written by this tool — "
            "set those in your LAMMPS input or MDP when you run the sim."
        )
        _out_lbl.setWordWrap(True)
        vl.addWidget(_out_lbl)
        v.addWidget(gb_out)

        row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Generate files")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._start_run)
        b_save = QPushButton("Save config..."); b_save.clicked.connect(self._save_config)
        b_load = QPushButton("Load config..."); b_load.clicked.connect(self._load_config)
        row.addWidget(self.btn_run); row.addWidget(b_save); row.addWidget(b_load); row.addStretch(1)
        v.addLayout(row)
        v.addStretch(1)
        return page

    def _on_builder_output(self, path: str):
        """Whenever the Builder or System-builder tab produces a file, jump
        to the Builder page so the user sees it appear in the shared table."""
        self.sidebar.setCurrentRow(0)
        self.statusBar().showMessage(f"Added structure: {path}", 5000)

    def _on_copolymer_settings_changed(self, settings: dict):
        """Builder toggled homo/copolymer mode or changed the % / total / mode.
        Auto-populate the corresponding fields on the Chain page AND
        show/hide the Chain page's Copolymer helper accordingly."""
        enabled = bool(settings.get("enabled"))
        # Hide the Copolymer helper group when Builder is in Homopolymer mode.
        if hasattr(self, "_gb_copolymer_helper"):
            self._gb_copolymer_helper.setVisible(enabled)
        # Also nudge Chain-page fractions text to blank in homopolymer mode.
        if not enabled:
            if hasattr(self, "chain_mode"):
                self.chain_mode.setCurrentText("homopolymer")
            if hasattr(self, "chain_fractions"):
                self.chain_fractions.setText("")
            return
        fa = settings.get("fraction_a", 1.0 - settings.get("fraction_b", 0.5))
        fb = settings.get("fraction_b", 1.0 - fa)
        if hasattr(self, "chain_n"):
            self.chain_n.setValue(settings["total"])
        if hasattr(self, "chain_mode"):
            self.chain_mode.setCurrentText(settings["mode"])
        if hasattr(self, "chain_fractions"):
            self.chain_fractions.setText(f"{fa:.2f},{fb:.2f}")
        if settings.get("seed") is not None and hasattr(self, "chain_seed"):
            self.chain_seed.setText(str(settings["seed"]))

    # ------------------------------------------------ FF handlers
    # ---- GROMACS-packing visibility / mode helpers
    def _sync_gmx_visibility(self) -> None:
        """Show the GROMACS packing group only when engine is gromacs or both.
        Also hide the packmol-related rows when we're in gromacs-only mode."""
        if not hasattr(self, "gb_gmx"):
            return
        engine = self.engine_combo.currentText() if hasattr(self, "engine_combo") else "lammps"
        self.gb_gmx.setVisible(engine in ("gromacs", "both"))

    def _on_gmx_mode_changed(self, _idx: int) -> None:
        """Enable the target-atoms spinbox only when Target-atoms mode is picked."""
        if not hasattr(self, "gmx_target_atoms"):
            return
        target_mode = self.gmx_mode.currentIndex() == 1
        self.gmx_target_atoms.setEnabled(target_mode)
        if hasattr(self, "n_chains"):
            self.n_chains.setEnabled(not target_mode)

    def _sync_ff_engine_from_export(self, engine: str) -> None:
        """Called when the Export page changes its engine — mirror it on the
        FF page combobox so both stay in sync."""
        if hasattr(self, "ff_engine_combo") \
                and self.ff_engine_combo.currentText() != engine:
            self.ff_engine_combo.blockSignals(True)
            self.ff_engine_combo.setCurrentText(engine)
            self.ff_engine_combo.blockSignals(False)
        # Show/hide the GROMACS-packing group whenever engine changes.
        self._sync_gmx_visibility()

    def _on_ff_engine_changed(self, engine: str) -> None:
        """User picked MD engine on the FF page. Mirror the same choice on
        the Export page's engine combobox and re-render the polymer.control
        preview if it's currently visible."""
        if hasattr(self, "engine_combo") and self.engine_combo.currentText() != engine:
            # Block signals to avoid the reverse-sync loop.
            self.engine_combo.blockSignals(True)
            self.engine_combo.setCurrentText(engine)
            self.engine_combo.blockSignals(False)
        # Show/hide the Box-page GROMACS packer section based on new engine.
        self._sync_gmx_visibility()
        # If a DL_FIELD FF is active and the control-file editor holds a
        # preview, refresh it so the new engine shows up in the file.
        if hasattr(self, "gb_dlctrl") and self.gb_dlctrl.isVisible() \
                and self.dlctrl_edit.toPlainText().strip():
            try: self._preview_dlfield_control()
            except Exception: pass
        # Reflect in the status bar so the change is visible.
        self.statusBar().showMessage(
            f"MD output format set to {engine.upper()}. "
            + ("polymer.control will use it on next Generate."
               if engine != "both" else
               "PAAF will emit both LAMMPS and GROMACS files."),
            4000)

    def _on_ff_changed(self, _idx: int) -> None:
        key = self.ff_combo.currentData()
        if not key:
            return
        from ..ff_registry import get_ff
        ff = get_ff(key)
        self.ff_notes.setText(
            f"{ff.notes}\nUnited atom: {ff.united_atom} | Atom typer: {ff.atom_typer}"
        )
        self._update_dlctrl_visibility()
        self._refresh_ff_summary()

    def _set_stat(self, row_widget, value: str) -> None:
        """Update the value label of a page.stat_row widget."""
        try:
            from PyQt5.QtWidgets import QLabel as _QL
            labels = row_widget.findChildren(_QL)
            if labels:
                labels[-1].setText(str(value))
        except Exception:
            pass

    def _refresh_ff_summary(self) -> None:
        """Keep the Force-field page's Assignment summary card in step."""
        if not hasattr(self, "ff_stat_monomers"):
            return
        try:
            n_mon = len(self.builder_tab.monomer_specs())
        except Exception:
            n_mon = 0
        n_over = len([ln for ln in self.ff_manual_types.toPlainText().splitlines()
                      if ln.strip() and ":" in ln])
        self._set_stat(self.ff_stat_monomers, str(n_mon))
        self._set_stat(self.ff_stat_overrides, str(n_over))
        self._set_stat(self.ff_stat_engine, self.ff_engine_combo.currentText())
        try:
            from ..ff_registry import get_ff
            self._set_stat(self.ff_stat_kind, get_ff(self.ff_combo.currentData()).kind)
        except Exception:
            self._set_stat(self.ff_stat_kind, "—")

    def _update_dlctrl_visibility(self) -> None:
        """Show the polymer.control editor only when a DL_FIELD FF is picked."""
        if not hasattr(self, "gb_dlctrl"):
            return
        key = self.ff_combo.currentData()
        if not key:
            self.gb_dlctrl.setVisible(False); return
        from ..ff_registry import get_ff
        ff = get_ff(key)
        self.gb_dlctrl.setVisible(ff.kind == "dlfield")

    def _preview_dlfield_control(self) -> None:
        """Render the polymer.control the pipeline would use — right now."""
        from ..dlfield_runner import write_dlfield_control
        from ..ff_registry import get_ff
        import tempfile
        key = self.ff_combo.currentData()
        if not key: return
        ff = get_ff(key)
        # Use a placeholder structure path since we may not have built yet.
        struct = Path(self.output_dir.text() or "./output") / (
            self.project_name.text() or "polymer") / (
            (self.project_name.text() or "polymer") + "_opt.xyz")
        tmp = Path(tempfile.mkdtemp(prefix="paaf_ctrl_")) / "polymer.control"
        # Use whatever the user chose on the FF page — dl_field respects
        # "gromacs" as an alternative secondary output.
        eng_choice = self.ff_engine_combo.currentText() if hasattr(self, "ff_engine_combo") else "lammps"
        dlf_engine = "gromacs" if eng_choice == "gromacs" else "lammps"
        write_dlfield_control(
            tmp, struct, ff_key=ff.key,
            output_engine=dlf_engine,
            box_ang=(self.box_a.value(), self.box_b.value(), self.box_c.value()),
        )
        self.dlctrl_edit.setPlainText(tmp.read_text())

    def _save_dlfield_control_edits(self) -> None:
        """Remember the user's edits so the pipeline uses them next run."""
        self._dlctrl_custom_text = self.dlctrl_edit.toPlainText()
        self.statusBar().showMessage(
            "polymer.control edits saved (will be applied on next Generate files run).",
            5000,
        )

    def _reset_dlfield_control(self) -> None:
        self._dlctrl_custom_text = None
        self.dlctrl_edit.setPlainText("")
        self.statusBar().showMessage("polymer.control reset to defaults.", 4000)

    def _open_atom_typing_dialog(self) -> None:
        """Open the manual atom-typing dialog for the current FF + monomers."""
        from ..ff_registry import get_ff
        from .atom_type_dialog import AtomTypingDialog
        specs = self.builder_tab.monomer_specs()
        if not specs:
            QMessageBox.warning(
                self, "No monomers",
                "Add at least one monomer on the Builder page first."
            )
            return
        key = self.ff_combo.currentData()
        if not key:
            return
        ff = get_ff(key)
        lt_path = ff.bundled_path() if ff.bundled_lt else None
        if lt_path is None or not lt_path.exists():
            QMessageBox.information(
                self, "No FF library",
                f"{ff.display_name} has no bundled .lt file to browse. "
                "Manual typing is only available for Moltemplate-native FFs "
                "(OPLS-AA, GAFF, TraPPE, DREIDING, COMPASS-published, SDK, MARTINI)."
            )
            return
        # Pre-fill with the current manual overrides
        initial: dict[int, str] = {}
        for line in self.ff_manual_types.toPlainText().splitlines():
            line = line.strip()
            if line and ":" in line:
                k, v = line.split(":", 1)
                try:
                    initial[int(k.strip())] = v.strip()
                except ValueError:
                    pass
        dlg = AtomTypingDialog(
            monomer_files=[s.file for s in specs],
            ff_lt_path=lt_path,
            initial_types=initial,
            parent=self,
            ff_key=ff.key,
            ff_inherit=ff.inherit,
            monomer_specs=specs,
        )
        if dlg.exec_() == dlg.Accepted:
            overrides = dlg.overrides()
            self.ff_manual_types.setPlainText(
                "\n".join(f"{k}:{v}" for k, v in sorted(overrides.items()))
            )
            self.statusBar().showMessage(
                f"Applied {len(overrides)} manual atom-type overrides.", 5000)

    def _browse_dl_lib(self) -> None:
        # Browsing always temporarily unlocks the field so we can write to it.
        d = QFileDialog.getExistingDirectory(self, "Select DL_FIELD lib directory")
        if d:
            self.ff_dl_lib.setReadOnly(False)
            self.ff_dl_lib.setText(d)
            self._save_dl_lib_setting()
            # Auto-lock after a successful pick.
            self._set_dl_lib_locked(True)

    def _set_dl_lib_locked(self, locked: bool) -> None:
        """Toggle read-only state of the DL_FIELD lib path and re-label buttons."""
        self.ff_dl_lib.setReadOnly(locked)
        # Grey out the field visually when locked so the state is obvious.
        if locked:
            self.ff_dl_lib.setStyleSheet(
                f"QLineEdit {{ background: {_tok.BG_SUNKEN};"
                f" color: {_tok.TEXT_SECONDARY}; }}")
            self.ff_dl_lib.setToolTip(wrap_tooltip("Locked. Click 'Change...' to edit, or 'Browse...' to pick a new folder."))
            self.b_lock_dl.setText("Change...")
            self.b_browse_dl.setEnabled(True)   # browse always works
        else:
            self.ff_dl_lib.setStyleSheet("")
            self.ff_dl_lib.setToolTip(wrap_tooltip("Editable. Type the new path and click 'Save & Lock' when done."))
            self.b_lock_dl.setText("Save && Lock")

    def _toggle_dl_lib_lock(self) -> None:
        """Handler for the Change / Save-&-Lock button."""
        if self.ff_dl_lib.isReadOnly():
            # Currently locked -> unlock for editing
            self._set_dl_lib_locked(False)
            self.ff_dl_lib.setFocus()
        else:
            # Currently editable -> validate + persist + relock
            p = self.ff_dl_lib.text().strip()
            if p and not Path(p).exists():
                QMessageBox.warning(
                    self, "Path not found",
                    f"'{p}' does not exist. Fix the path or click Cancel-style "
                    "and pick another folder.")
                return
            self._save_dl_lib_setting()
            self._set_dl_lib_locked(True)

    def _discover_default_dl_lib(self) -> Optional[str]:
        """Try to auto-fill the DL_FIELD lib path so users don't retype it.

        Order of precedence:
          1. QSettings — last path used in a previous session.
          2. $DL_FIELD_LIB or $DL_FIELD_DIR env vars.
          3. Common developer default ~/project/dl_f_4.13/lib.
        Only returns a path that actually exists on disk.
        """
        import os
        from PyQt5.QtCore import QSettings
        settings = QSettings("PAAF", "PAAF")
        saved = settings.value("dl_field/lib_dir", "", type=str)
        for candidate in (saved,
                          os.environ.get("DL_FIELD_LIB", ""),
                          os.environ.get("DL_FIELD_DIR", ""),
                          str(Path.home() / "project" / "dl_f_4.13" / "lib")):
            if candidate and Path(candidate).exists():
                return candidate
        return None

    # ---------------- packmol path (mirrors the DL_FIELD lib pattern)
    def _discover_default_packmol(self) -> Optional[str]:
        import os
        from PyQt5.QtCore import QSettings
        saved = QSettings("PAAF", "PAAF").value("packmol/path", "", type=str)
        for candidate in (
            saved,
            os.environ.get("PACKMOL_EXE", ""),
            "/opt/homebrew/bin/packmol",
            "/usr/local/bin/packmol",
            "/opt/homebrew/Caskroom/miniconda/base/envs/mta/bin/packmol",
            str(Path.home() / "opt/miniconda3/envs/mta/bin/packmol"),
        ):
            if candidate and Path(candidate).exists():
                return candidate
        # Fall back to a straight PATH lookup
        import shutil as _sh
        p = _sh.which("packmol")
        return p if p else None

    def _browse_packmol(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Select packmol binary",
                                          filter="All files (*)")
        if p:
            self.box_packmol_path.setText(p); self._save_packmol_setting()

    def _save_packmol_setting(self) -> None:
        from PyQt5.QtCore import QSettings
        p = self.box_packmol_path.text().strip()
        if p and Path(p).exists():
            QSettings("PAAF", "PAAF").setValue("packmol/path", p)

    def _save_dl_lib_setting(self) -> None:
        """Remember the current DL_FIELD lib path for the next session."""
        from PyQt5.QtCore import QSettings
        p = self.ff_dl_lib.text().strip()
        if p and Path(p).exists():
            QSettings("PAAF", "PAAF").setValue("dl_field/lib_dir", p)

    def _browse_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self.output_dir.setText(d)

    def _show_ff_list(self):
        from ..ff_registry import list_ffs
        text = "\n".join(f"{ff.key:<20}  {ff.display_name}" for ff in list_ffs())
        QMessageBox.information(self, "Supported force fields", text)

    def _convert_par_dialog(self):
        par, _ = QFileDialog.getOpenFileName(self, "DL_FIELD .par", filter="*.par")
        if not par:
            return
        out, _ = QFileDialog.getSaveFileName(self, "Save Moltemplate .lt", filter="*.lt")
        if not out:
            return
        try:
            from ..dlfield_converter import convert_par
            convert_par(par, out)
            QMessageBox.information(self, "Converted", f"Wrote {out}")
        except Exception as exc:
            QMessageBox.critical(self, "Convert failed", str(exc))

    # ================================================== config <-> UI
    def _build_config(self) -> Config:
        # Monomers live inside the Builder tab (upload / SMILES / periodic)
        monomers: List[MonomerSpec] = self.builder_tab.monomer_specs()
        manual_types = {}
        for line in self.ff_manual_types.toPlainText().splitlines():
            line = line.strip()
            if line and ":" in line:
                k, v = line.split(":", 1)
                try:
                    manual_types[int(k.strip())] = v.strip()
                except ValueError:
                    pass
        fractions = [float(x) for x in self.chain_fractions.text().split(",") if x.strip()] or None
        blocks = [int(x) for x in self.chain_blocks.text().split(",") if x.strip()] or None
        seed_txt = self.chain_seed.text().strip()
        return Config(
            project_name=self.project_name.text() or "polymer",
            output_dir=self.output_dir.text() or "output",
            monomers=monomers,
            optimizer=OptimizerCfg(
                enabled=self.opt_enabled.isChecked(),
                ff=self.opt_ff.currentText(),
                steps=_parse_int(self.opt_steps.currentText(), default=10000),
                tol=_parse_float(self.opt_tol.currentText(), default=1.0e-6),
                algorithm=(self.opt_alg.currentData()
                           or self.opt_alg.currentText().split()[0]),
            ),
            chain=ChainCfg(
                n_monomers=self.chain_n.value(),
                mode=self.chain_mode.currentText(),
                fractions=fractions, block_sizes=blocks,
                seed=int(seed_txt) if seed_txt else None,
                backend=self.chain_backend.currentText(),
                cap_carboxyl_end=bool(getattr(self, "chain_cap_cooh",
                                              None) and self.chain_cap_cooh.isChecked()),
            ),
            box=BoxCfg(
                n_chains=self.n_chains.value(),
                shape=self.box_shape.currentText(),
                a=self.box_a.value(),
                b=self.box_b.value(),
                c=self.box_c.value(),
                alpha=self.box_alpha.value(),
                beta=self.box_beta.value(),
                gamma=self.box_gamma.value(),
                use_density=self.box_use_density.isChecked(),
                density_g_cm3=self.box_density.value(),
                packmol=self.box_packmol.isChecked(),
                packmol_path=self.box_packmol_path.text().strip(),
                packmol_seed=int(self.box_packmol_seed.value()),
                packmol_tolerance=float(self.box_packmol_tol.value()),
                # GROMACS-native packing (only used when engine=gromacs/both)
                gmx_target_atoms=(int(self.gmx_target_atoms.value())
                                  if hasattr(self, "gmx_mode")
                                     and self.gmx_mode.currentIndex() == 1
                                  else 0),
                gmx_pre_minimise=bool(getattr(self, "gmx_pre_min",
                                              None) and self.gmx_pre_min.isChecked()),
                gmx_try_count=(int(self.gmx_try.value())
                               if hasattr(self, "gmx_try") else 100000),
                # legacy compat: also carry the same lengths in `size`
                size=[self.box_a.value(), self.box_b.value(), self.box_c.value()],
            ),
            force_field=ForceFieldCfg(
                key=self.ff_combo.currentData(),
                dl_lib_dir=self.ff_dl_lib.text() or None,
                manual_types=manual_types,
            ),
            lammps=LammpsCfg(
                ensemble=self.lmp_ensemble.currentText(),
                temperature=self.lmp_temp.value(),
                pressure=self.lmp_press.value(),
                steps=self.lmp_steps.value(),
                timestep=self.lmp_ts.value(),
                tdamp_fs=self.lmp_tdamp.value(),
                pdamp_fs=self.lmp_pdamp.value(),
                thermostat=self.lmp_thermostat.currentText(),
                barostat=self.lmp_barostat.currentText(),
                pressure_coupling=self.lmp_coupling.currentText(),
                multistage=self.lmp_multistage.isChecked(),
                minimize=self.lmp_minimize.isChecked(),
                nvt_steps=self.lmp_nvt_steps.value(),
                nvt_temperature=(self.lmp_nvt_temp.value() or None),
                npt_steps=self.lmp_npt_steps.value(),
                thermo_every=self.lmp_thermo_every.value(),
                dump_every=self.lmp_dump_every.value(),
                seed=self.lmp_seed.value(),
            ),
            run_moltemplate=self.run_moltemplate.isChecked(),
            engine=self.engine_combo.currentText(),
            topology_builder=(self.topology_builder.currentData() or "auto"),
            gromacs_include_itp=self.gromacs_itp.text().strip() or None,
        )

    def _apply_config(self, cfg: Config) -> None:
        self.project_name.setText(cfg.project_name)
        self.output_dir.setText(cfg.output_dir)
        self.builder_tab.load_monomer_specs(cfg.monomers)
        self.chain_n.setValue(cfg.chain.n_monomers)
        self.chain_mode.setCurrentText(cfg.chain.mode)
        self.chain_backend.setCurrentText(cfg.chain.backend)
        if hasattr(self, "chain_cap_cooh"):
            self.chain_cap_cooh.setChecked(getattr(cfg.chain, "cap_carboxyl_end", True))
        self.chain_fractions.setText(",".join(str(x) for x in cfg.chain.fractions or []))
        self.chain_blocks.setText(",".join(str(x) for x in cfg.chain.block_sizes or []))
        self.chain_seed.setText(str(cfg.chain.seed) if cfg.chain.seed else "")
        for i in range(self.ff_combo.count()):
            if self.ff_combo.itemData(i) == cfg.force_field.key:
                self.ff_combo.setCurrentIndex(i); break
        self.ff_dl_lib.setReadOnly(False)
        self.ff_dl_lib.setText(cfg.force_field.dl_lib_dir or "")
        # Re-lock after applying config if the loaded path is valid.
        p = self.ff_dl_lib.text().strip()
        self._set_dl_lib_locked(bool(p) and Path(p).exists())
        self.ff_manual_types.setPlainText(
            "\n".join(f"{k}:{v}" for k, v in cfg.force_field.manual_types.items())
        )
        self.opt_enabled.setChecked(cfg.optimizer.enabled)
        self.opt_ff.setCurrentText(cfg.optimizer.ff)
        # Comboboxes: setEditText for editable ones
        self.opt_steps.setCurrentText(
            f"{int(cfg.optimizer.steps):,}".replace(",", " "))
        self.opt_tol.setCurrentText(f"{cfg.optimizer.tol:.1e}")
        # Match algorithm by data or by prefix
        for i in range(self.opt_alg.count()):
            if (self.opt_alg.itemData(i) == cfg.optimizer.algorithm
                    or self.opt_alg.itemText(i).startswith(cfg.optimizer.algorithm)):
                self.opt_alg.setCurrentIndex(i); break
        self.n_chains.setValue(cfg.box.n_chains)
        self.box_shape.setCurrentText(getattr(cfg.box, "shape", "cubic"))
        self.box_a.setValue(getattr(cfg.box, "a", cfg.box.size[0] if cfg.box.size else 50.0))
        self.box_b.setValue(getattr(cfg.box, "b", cfg.box.size[1] if len(cfg.box.size) > 1 else 50.0))
        self.box_c.setValue(getattr(cfg.box, "c", cfg.box.size[2] if len(cfg.box.size) > 2 else 50.0))
        self.box_alpha.setValue(getattr(cfg.box, "alpha", 90.0))
        self.box_beta.setValue(getattr(cfg.box, "beta", 90.0))
        self.box_gamma.setValue(getattr(cfg.box, "gamma", 90.0))
        self.box_use_density.setChecked(getattr(cfg.box, "use_density", False))
        self.box_density.setValue(getattr(cfg.box, "density_g_cm3", 1.0))
        self.box_packmol.setChecked(getattr(cfg.box, "packmol", True))
        _cfg_pk = getattr(cfg.box, "packmol_path", "")
        if _cfg_pk:
            self.box_packmol_path.setText(_cfg_pk)
        self.lmp_ensemble.setCurrentText(cfg.lammps.ensemble)
        self.lmp_temp.setValue(cfg.lammps.temperature)
        self.lmp_press.setValue(cfg.lammps.pressure)
        self.lmp_steps.setValue(cfg.lammps.steps)
        self.lmp_ts.setValue(cfg.lammps.timestep)
        self.lmp_tdamp.setValue(cfg.lammps.tdamp_fs)
        self.lmp_pdamp.setValue(cfg.lammps.pdamp_fs)
        self.lmp_thermostat.setCurrentText(cfg.lammps.thermostat)
        self.lmp_barostat.setCurrentText(cfg.lammps.barostat)
        self.lmp_coupling.setCurrentText(cfg.lammps.pressure_coupling)
        self.lmp_multistage.setChecked(cfg.lammps.multistage)
        self.lmp_minimize.setChecked(cfg.lammps.minimize)
        self.lmp_nvt_steps.setValue(cfg.lammps.nvt_steps)
        self.lmp_nvt_temp.setValue(cfg.lammps.nvt_temperature or 0.0)
        self.lmp_npt_steps.setValue(cfg.lammps.npt_steps)
        self.lmp_thermo_every.setValue(cfg.lammps.thermo_every)
        self.lmp_dump_every.setValue(cfg.lammps.dump_every)
        self.lmp_seed.setValue(cfg.lammps.seed)
        self.run_moltemplate.setChecked(cfg.run_moltemplate)
        self.engine_combo.setCurrentText(cfg.engine)
        # Keep FF-page chooser in sync when a config is loaded from disk.
        if hasattr(self, "ff_engine_combo"):
            self.ff_engine_combo.setCurrentText(cfg.engine)
        _tb = getattr(cfg, "topology_builder", "auto")
        for i in range(self.topology_builder.count()):
            if self.topology_builder.itemData(i) == _tb:
                self.topology_builder.setCurrentIndex(i)
                break
        self.gromacs_itp.setText(cfg.gromacs_include_itp or "")
        self._sync_project_label()

    # ================================================== config IO / run
    def _new_project(self):
        self._apply_config(Config())
        self._sync_project_label()

    def _save_config(self) -> None:
        cfg = self._build_config()
        p, _ = QFileDialog.getSaveFileName(self, "Save config", filter="YAML (*.yaml);;JSON (*.json)")
        if not p:
            return
        cfg.save(p); self._append_log(f"Config saved to {p}")

    def _load_config(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Load config", filter="YAML/JSON (*.yaml *.yml *.json)")
        if not p:
            return
        cfg = Config.load(p); self._apply_config(cfg)
        self._append_log(f"Config loaded from {p}")

    def _start_run(self) -> None:
        cfg = self._build_config()
        if not cfg.monomers:
            # Jump to Builder tab and highlight what's needed.
            self.sidebar.setCurrentRow(0)
            QMessageBox.warning(
                self, "No monomers yet",
                "The shared monomers table (bottom of the Builder tab) is empty.\n\n"
                "Add a monomer by:\n"
                "  • SMILES + library tab → pick a row and click 'Build structure'\n"
                "  • Periodic table tab   → compose SMILES and click 'Build structure'\n\n"
                "After a build finishes you should see a log line: "
                "'Monomer added to shared table (row 1): ...'"
            )
            return
        self.btn_run.setEnabled(False)
        self._thread = QThread(self)
        self._worker = Worker(cfg)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(lambda: self.btn_run.setEnabled(True))
        self._thread.start()

    @pyqtSlot(dict)
    def _on_finished(self, result: dict) -> None:
        self._append_log("=" * 60)
        self._append_log("PIPELINE FINISHED")
        for k, v in result.items():
            if k == "config":
                continue
            self._append_log(f"  {k}: {v}")
        self.statusBar().showMessage("Finished", 5000)

    @pyqtSlot(str)
    def _on_failed(self, err: str) -> None:
        self._append_log("PIPELINE FAILED")
        self._append_log(err)
        QMessageBox.critical(self, "Pipeline failed", err[:2000])

    # ================================================== log console
    def _wire_logging(self) -> None:
        handler = QtLogHandler(self._append_log)
        handler.setLevel(logging.INFO)
        logging.getLogger("paaf").addHandler(handler)

    def _append_log(self, msg: str) -> None:
        self.console.appendPlainText(msg)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def _about(self) -> None:
        QMessageBox.information(
            self, f"About {__product_name__}",
            f"<h3>{__product_name__} — {__full_name__}</h3>"
            f"<p>Version {__version__}</p>"
            "<p>Automated polymer builder, reactive-MD engine and "
            "Moltemplate / DL_FIELD / LAMMPS / GROMACS system generator.</p>"
            "<p>Supports OPLS-AA/UA, GAFF/GAFF2, TraPPE-UA, PCFF, COMPASS, "
            "CVFF, DREIDING, CHARMM, GROMOS, MARTINI, SDK, and any "
            "DL_FIELD .par file.</p>"
            "<p>Open source. Please cite <b>PAAF: Polymer Auto-Assembly "
            "Framework</b> if used in publication.</p>"
        )
