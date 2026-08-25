"""Interactive periodic table widget for the molecule builder.

Data lives in :mod:`paaf.periodic_data` (PyQt-free) so tests
and CLI utilities can query element info without a display.

Compact layout: 32×26 px buttons, no title bar, tiny inline legend at the
bottom. Whole widget is ~260 px tall — small enough to sit under the
SMILES notation strip without pushing anything else off-screen.
"""
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from . import tokens as _tok
from PyQt5.QtWidgets import (
    QGridLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from ..periodic_data import ELEMENTS, BY_SYMBOL, CATEGORY_COLORS, Element  # noqa: F401


class PeriodicTable(QWidget):
    """Interactive 118-element table (compact).

    Signals
    -------
    elementClicked(str)     — the element symbol (e.g. "C", "Cl")
    """
    elementClicked = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None,
                 button_size: tuple = (32, 26),
                 show_legend: bool = True):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0); v.setSpacing(2)

        bw, bh = button_size
        grid = QGridLayout(); grid.setSpacing(1)
        for e in ELEMENTS:
            btn = QPushButton(e.symbol)
            btn.setToolTip(f"{e.symbol} — {e.name}  (Z={e.Z}, mass={e.mass:.3f})")
            btn.setFixedSize(bw, bh)
            btn.setFont(QFont("SF Pro Text", 9, QFont.Bold))
            fg, bg = CATEGORY_COLORS.get(e.category, CATEGORY_COLORS["unknown"])
            btn.setStyleSheet(
                f"QPushButton {{ color: {fg}; background: {bg}; border: 1px solid "
                f"rgba(0,0,0,0.08); border-radius: 3px; padding: 0; }} "
                f"QPushButton:hover {{ border: 2px solid {_tok.PRIMARY}; }}"
            )
            btn.clicked.connect(lambda _c, s=e.symbol: self.elementClicked.emit(s))
            grid.addWidget(btn, e.row - 1, e.col - 1)
        grid.setRowMinimumHeight(6, 6)   # small gap before f-block
        v.addLayout(grid)

        if show_legend:
            legend = QLabel(
                "  ".join(
                    f'<span style="background:{bg};color:{fg};padding:1px 5px;'
                    f'border-radius:2px;font-size:10px;">{cat}</span>'
                    for cat, (fg, bg) in CATEGORY_COLORS.items()
                )
            )
            legend.setStyleSheet("padding-top: 2px;")
            legend.setWordWrap(True)
            v.addWidget(legend)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
