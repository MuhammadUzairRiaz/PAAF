"""Project settings — name and output folder.

These two values decide **where every generated file lands**::

    <output directory>/<project name>/
        <name>_opt.xyz          optimized monomer / chain
        system.lt               moltemplate input
        polymer.control         DL_FIELD control file
        dlf_output1/            dl_field results (lammps.data, lammps.in, ...)
        packed_blend.data       blend packing output

Before the redesign these lived in a group box at the bottom of Step 6, which
is the last place a user looks when starting a project. They are now reachable
from the sidebar (click the project name) and from *File → Project settings…*.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from . import tokens as T


class ProjectDialog(QDialog):
    """Edit the project name and output directory, with a live path preview."""

    def __init__(self, name: str, output_dir: str,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Project settings")
        self.setMinimumWidth(560)

        v = QVBoxLayout(self)
        v.setContentsMargins(T.PAD_PAGE, T.PAD_PAGE, T.PAD_PAGE, T.PAD_PAGE)
        v.setSpacing(T.GAP_SECTION)

        intro = QLabel(
            "<b>Every file PAAF generates is written to "
            "<i>output directory / project name</i>.</b><br>"
            "Change the project name to start a separate run without "
            "overwriting the previous one.")
        intro.setWordWrap(True)
        v.addWidget(intro)

        form = QFormLayout()
        form.setSpacing(T.GAP_FORM_ROW)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.name_edit = QLineEdit(name or "polymer")
        self.name_edit.setPlaceholderText("e.g. PBS-20 or PLA-crosslink-study")
        self.name_edit.textChanged.connect(self._update_preview)
        form.addRow("Project name", self.name_edit)

        self.dir_edit = QLineEdit(output_dir or str(Path.cwd() / "output"))
        self.dir_edit.textChanged.connect(self._update_preview)
        b_browse = QPushButton("Browse…")
        b_browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.setSpacing(T.GAP_LABEL)
        row.addWidget(self.dir_edit, 1)
        row.addWidget(b_browse)
        wrap = QWidget()
        wrap.setLayout(row)
        form.addRow("Output directory", wrap)
        v.addLayout(form)

        # Live preview of the folder that will actually be created.
        self.preview = QLabel("")
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet(
            f"background: {T.BG_SUNKEN}; border: 1px solid {T.BORDER};"
            f" border-radius: {T.R_CONTROL}px; padding: 8px 10px;"
            f" color: {T.TEXT_SECONDARY}; font-family: {T.FONT_MONO};"
            f" font-size: {T.FS_CAPTION}px;")
        v.addWidget(self.preview)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setObjectName("primary")
        bb.button(QDialogButtonBox.Ok).setText("Save")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

        self._update_preview()

    # ------------------------------------------------------------ helpers
    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Choose the output directory", self.dir_edit.text().strip())
        if d:
            self.dir_edit.setText(d)

    def _update_preview(self, *_args) -> None:
        """Recompute the destination path. Accepts the text argument that
        ``QLineEdit.textChanged`` emits, so it can be connected directly."""
        name = self.name_edit.text().strip() or "polymer"
        base = self.dir_edit.text().strip() or "output"
        target = Path(base) / name
        self.preview.setText(
            f"Files will be written to:\n{target}\n\n"
            "e.g.  system.lt · polymer.control · dlf_output1/lammps.data")

    # --------------------------------------------------------------- API
    def values(self) -> tuple:
        """Return ``(project_name, output_dir)`` as entered."""
        return (self.name_edit.text().strip() or "polymer",
                self.dir_edit.text().strip() or str(Path.cwd() / "output"))
