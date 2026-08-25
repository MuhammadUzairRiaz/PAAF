"""PAAF design system — global stylesheet.

Implements the *PAAF UI redesign v1* spec (``docs/design/PAAF_Redesign_v1.dc.html``).
Every value is pulled from :mod:`paaf.gui.tokens` so the tokens module stays the
single source of truth; nothing here hard-codes a colour.

Core ideas from the spec:

* **No shadows.** Depth comes from three background planes plus border weight::

      page #F1F5F9  →  card #FFFFFF  →  sunken #F8FAFC

* **Fixed type scale** — 10 / 11 / 12 / 13 / 20 only. Everything that used to
  sit between 12 and 16px collapses into the 13px section header.
* **Fixed control metrics** — control 26px, table row 28px, header row 26px,
  sidebar item 30px.
* **Focus is a 2px border**, with padding reduced by 1px so the control does not
  change size when focused. The focus indicator is never removed.
* **Radii**: 3 controls · 6 cards/panels · 10 pills. Nothing else.

Role hooks (via ``setObjectName``): ``primary``, ``danger``, ``ghost``,
``console``, ``sidebar``, ``project_header``, ``project_sub``. Property hooks:
``role="hint"``, ``role="sectionTitle"``.
"""
from __future__ import annotations

from . import tokens as T


def _light_qss() -> str:
    return f"""
* {{
    font-family: {T.FONT_UI};
    font-size: {T.FS_BODY}px;
    color: {T.TEXT};
}}
/* A small white card, not a stripe across the window.
   Explanatory tooltips here run to several sentences, and Qt lays plain
   text out on ONE line however long it is — the "Cell size" help rendered
   as a full-width grey band over the page, unreadable and covering the
   controls it described. Wrapping is a property of the TEXT, not the
   stylesheet (see wrap_tooltip), so this handles the look and that
   handles the width. */
QToolTip {{
    background: #FFFFFF; color: {T.TEXT};
    border: 1px solid {T.BORDER_STRONG}; border-radius: {T.R_CARD}px;
    padding: 8px 10px; font-size: {T.FS_CAPTION}px;
}}

/* -------- Frame & page (plane 1) -------- */
QMainWindow, QDialog {{ background: {T.BG_PAGE}; }}
QDialog {{ border: 1px solid {T.BORDER_STRONG}; }}
QStatusBar {{
    background: {T.BG_SUNKEN}; border-top: 1px solid {T.BORDER};
    color: {T.TEXT_MUTED}; padding: 2px 8px; font-size: {T.FS_CAPTION}px;
}}
QStatusBar::item {{ border: none; }}

/* -------- Menus -------- */
QMenuBar {{ background: {T.BG_SURFACE}; border-bottom: 1px solid {T.BORDER}; }}
QMenuBar::item {{ padding: 6px 10px; color: {T.TEXT_SECONDARY}; }}
QMenuBar::item:selected {{ background: {T.PRIMARY_TINT}; color: {T.PRIMARY_HOVER}; }}
QMenu {{
    background: {T.BG_SURFACE}; border: 1px solid {T.BORDER_STRONG};
    border-radius: {T.R_CARD}px; padding: 4px;
}}
QMenu::item {{ padding: 6px 14px; border-radius: {T.R_CONTROL}px; }}
QMenu::item:selected {{ background: {T.PRIMARY_TINT}; color: {T.PRIMARY_HOVER}; }}

/* -------- Sidebar (chrome plane) -------- */
QListWidget#sidebar {{
    background: {T.BG_CHROME}; color: #CBD5E1;
    border: none; outline: 0; padding-top: 8px;
}}
QListWidget#sidebar::item {{
    min-height: {T.H_SIDEBAR_ITEM}px;
    padding: 4px 16px;
    border-left: 3px solid transparent;
    font-size: {T.FS_BODY}px;
}}
QListWidget#sidebar::item:selected {{
    background: {T.BG_CHROME_RAISED}; color: #FFFFFF;
    border-left: 3px solid {T.PRIMARY}; font-weight: 600;
}}
QListWidget#sidebar::item:hover:!selected {{ background: {T.BG_CHROME_RAISED}; }}
QListWidget#sidebar::item:disabled {{ color: #64748B; }}

/* Sidebar section captions and the progress card */
QLabel#navSection {{
    color: {T.TEXT_DISABLED}; font-size: {T.FS_OVERLINE}px; font-weight: 600;
    letter-spacing: 1px; padding: 10px 16px 4px 16px; background: {T.BG_CHROME};
}}
QWidget#navProgress {{
    background: {T.BG_CHROME_RAISED}; border: none; border-radius: {T.R_CARD}px;
}}

/* -------- Page headers -------- */
QLabel#project_header {{
    font-size: {T.FS_DISPLAY}px; font-weight: 600; color: {T.TEXT};
    padding: 14px 20px 2px 20px; letter-spacing: -0.2px;
}}
QLabel#project_sub {{
    color: {T.TEXT_MUTED}; font-size: {T.FS_BODY}px; padding: 0 20px 10px 20px;
}}
QLabel#breadcrumb {{
    color: {T.TEXT_MUTED}; font-size: {T.FS_CAPTION}px; padding: 8px 20px 0 20px;
}}
QLabel[role="hint"], QLabel#hint {{ color: {T.TEXT_MUTED}; font-size: {T.FS_CAPTION}px; }}
QLabel[role="sectionTitle"] {{
    font-size: {T.FS_SECTION}px; font-weight: 600; color: {T.TEXT};
    padding-bottom: 2px;
}}

/* -------- Cards (QGroupBox) --------
   Title sits entirely above the border; no background pill, no shadow. */
QGroupBox {{
    border: 1px solid {T.BORDER};
    border-radius: {T.R_CARD}px;
    margin-top: 20px;
    padding: {T.PAD_CARD}px;
    background: {T.BG_SURFACE};
    font-weight: 600;
    color: {T.TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 2px; top: 0;
    padding: 0 4px 4px 0;
    background: transparent;
    color: {T.TEXT};
    font-size: {T.FS_SECTION}px;
    font-weight: 600;
}}

/* -------- Buttons (26px, radius 3) -------- */
QPushButton {{
    background: {T.BG_SURFACE}; color: {T.TEXT};
    border: 1px solid {T.BORDER_STRONG};
    padding: 4px 12px; border-radius: {T.R_CONTROL}px;
    font-size: {T.FS_BODY}px; min-height: {T.H_CONTROL - 10}px;
}}
QPushButton:hover  {{ background: {T.BG_SUNKEN}; }}
QPushButton:pressed{{ background: {T.BORDER}; }}
QPushButton:disabled {{
    background: {T.BG_SUNKEN}; color: {T.TEXT_DISABLED}; border-color: {T.BORDER};
}}
QPushButton:focus {{ border: 2px solid {T.PRIMARY}; padding: 3px 11px; outline: none; }}

QPushButton#primary {{
    background: {T.PRIMARY}; color: #FFFFFF; border: 1px solid {T.PRIMARY};
    font-weight: 600;
}}
QPushButton#primary:hover   {{ background: {T.PRIMARY_HOVER}; border-color: {T.PRIMARY_HOVER}; }}
QPushButton#primary:pressed {{ background: {T.PRIMARY_HOVER}; }}
QPushButton#primary:disabled{{
    background: {T.BG_SUNKEN}; color: {T.TEXT_DISABLED}; border-color: {T.BORDER};
}}

/* Destructive actions are OUTLINED, never filled — a filled red next to a
   filled blue primary is a misclick risk in a dense action bar. */
QPushButton#danger {{
    background: {T.BG_SURFACE}; color: {T.DANGER_TEXT};
    border: 1px solid {T.DANGER}; font-weight: 600;
}}
QPushButton#danger:hover {{ background: {T.DANGER_TINT}; }}

QPushButton#ghost {{
    background: transparent; color: {T.PRIMARY_HOVER};
    border: 1px solid transparent; padding: 4px 8px;
}}
QPushButton#ghost:hover {{ background: {T.PRIMARY_TINT}; }}

QToolButton {{
    background: transparent; border: 1px solid transparent;
    padding: 4px 10px; border-radius: {T.R_CONTROL}px; color: {T.TEXT_SECONDARY};
}}
QToolButton:hover {{ background: {T.BG_SUNKEN}; }}
QToolButton:checked {{
    background: {T.PRIMARY_TINT}; border-color: #BFDBFE;
    color: {T.PRIMARY_HOVER}; font-weight: 600;
}}

/* -------- Inputs (26px, radius 3, 2px focus border) -------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {T.BG_SURFACE};
    border: 1px solid {T.BORDER_STRONG};
    border-radius: {T.R_CONTROL}px;
    padding: 3px 6px;
    selection-background-color: {T.PRIMARY_TINT};
    selection-color: {T.TEXT};
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ min-height: {T.H_CONTROL - 8}px; }}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    border-color: {T.TEXT_DISABLED};
}}
/* Focus: 2px border with padding reduced 1px so the control does not resize. */
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 2px solid {T.PRIMARY}; padding: 2px 5px;
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    background: {T.BG_SUNKEN}; color: {T.TEXT_DISABLED};
}}
QLineEdit[readOnly="true"] {{ background: {T.BG_SUNKEN}; color: {T.TEXT_SECONDARY}; }}
/* Combo arrows. Same lesson as the spin boxes below: with the arrow hidden
   a combo reads as plain text — users were clicking past "Write files for"
   and "Tacticity" without realising they were choices. The chevron sits in
   its own sunken button zone, exactly like the spin arrows. */
QComboBox {{ padding-right: 20px; }}
QComboBox::drop-down {{
    subcontrol-origin: border;
    subcontrol-position: center right;
    width: 18px;
    background: {T.BG_SUNKEN};
    border-left: 1px solid {T.BORDER};
    border-top-right-radius: {T.R_CONTROL}px;
    border-bottom-right-radius: {T.R_CONTROL}px;
}}
QComboBox::drop-down:hover {{ background: {T.PRIMARY_TINT}; }}
QComboBox::down-arrow {{
    image: none; width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {T.TEXT_SECONDARY};
}}
QComboBox::down-arrow:disabled {{ border-top-color: {T.TEXT_DISABLED}; }}

/* Spin arrows. Without them a spin box reads as a read-only label — users
   were not realising density, cell size and temperature could be typed in. */
QSpinBox, QDoubleSpinBox {{ padding-right: 16px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    width: 14px;
    background: {T.BG_SUNKEN};
    border-left: 1px solid {T.BORDER};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-position: top right;
    border-top-right-radius: {T.R_CONTROL}px;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-position: bottom right;
    border-bottom-right-radius: {T.R_CONTROL}px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {T.PRIMARY_TINT};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: none; width: 0; height: 0;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-bottom: 4px solid {T.TEXT_SECONDARY};
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: none; width: 0; height: 0;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 4px solid {T.TEXT_SECONDARY};
}}
QComboBox QAbstractItemView {{
    background: {T.BG_SURFACE}; border: 1px solid {T.BORDER_STRONG};
    border-radius: {T.R_CONTROL}px;
    selection-background-color: {T.PRIMARY_TINT}; selection-color: {T.TEXT};
    padding: 2px;
}}

/* -------- Tabs (32px; active = blue text + 2px bottom rule, no box) -------- */
QTabWidget::pane {{
    border: none; border-top: 1px solid {T.BORDER}; top: -1px;
    background: transparent;
}}
QTabBar {{ qproperty-drawBase: 0; background: transparent; }}
QTabBar::tab {{
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 7px 14px;
    margin: 0 2px 0 0;
    min-height: 18px;
    color: {T.TEXT_MUTED};
    font-size: {T.FS_BODY}px;
}}
QTabBar::tab:hover:!selected {{ color: {T.TEXT}; }}
/* NOTE: do NOT change font-weight here. Qt sizes each tab with the tab bar's
   normal-weight font, so making the *selected* tab bold pushes the text wider
   than its allocated width and clips it ("Custom" renders as "ustom"). The
   spec asks for blue text + a 2px underline, which needs no weight change. */
QTabBar::tab:selected {{
    color: {T.PRIMARY_HOVER};
    border-bottom: 2px solid {T.PRIMARY};
    background: transparent;
}}

/* -------- Tables (row 28, header 26, alternating OFF) -------- */
QTableWidget, QTableView {{
    background: {T.BG_SURFACE}; gridline-color: {T.BG_SUNKEN};
    border: 1px solid {T.BORDER_STRONG}; border-radius: {T.R_CONTROL}px;
    selection-background-color: {T.PRIMARY_TINT}; selection-color: {T.TEXT};
    font-size: {T.FS_BODY}px;
}}
QTableWidget::item, QTableView::item {{ padding: 3px 6px; }}
QHeaderView::section {{
    background: {T.BG_SUNKEN}; padding: 4px 6px; border: none;
    border-bottom: 1px solid {T.BORDER};
    font-weight: 600; color: {T.TEXT_MUTED}; font-size: {T.FS_OVERLINE}px;
}}
QHeaderView::section:horizontal {{ height: {T.H_TABLE_HEADER - 8}px; }}

/* -------- Console -------- */
QPlainTextEdit#console {{
    background: {T.BG_CHROME}; color: #CBD5E1;
    border: none; border-top: 1px solid {T.BORDER}; border-radius: 0;
    font-family: {T.FONT_MONO}; font-size: {T.FS_CAPTION}px;
    padding: 8px;
    selection-background-color: {T.PRIMARY_HOVER}; selection-color: #FFFFFF;
}}

/* -------- Scrollbars -------- */
QScrollBar:vertical   {{ background: transparent; width: 10px; margin: 0; border: none; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; border: none; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {T.BORDER_STRONG}; border-radius: 5px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background: {T.TEXT_DISABLED};
}}
QScrollBar::add-line, QScrollBar::sub-line {{ background: none; height: 0; width: 0; }}
QScrollArea {{ background: transparent; border: none; }}

/* -------- Misc -------- */
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border: 1px solid {T.BORDER_STRONG};
    border-radius: {T.R_CONTROL}px; background: {T.BG_SURFACE};
}}
QCheckBox::indicator:hover   {{ border-color: {T.TEXT_DISABLED}; }}
QCheckBox::indicator:checked {{ background: {T.PRIMARY}; border-color: {T.PRIMARY}; }}
QRadioButton {{ spacing: 6px; }}
QRadioButton::indicator {{
    width: 13px; height: 13px;
    border: 1px solid {T.BORDER_STRONG};
    border-radius: 8px;             /* > half the 15px box => a true circle */
    background: {T.BG_SURFACE};
}}
QRadioButton::indicator:hover {{ border-color: {T.TEXT_DISABLED}; }}
QRadioButton::indicator:checked {{
    border: 4px solid {T.PRIMARY};
    background: {T.BG_SURFACE};
}}
QCheckBox::indicator:checked {{
    background: {T.PRIMARY}; border: 1px solid {T.PRIMARY};
}}

QSplitter::handle {{ background: {T.BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical   {{ height: 1px; }}
"""


def build_qss(theme: str = "light") -> str:
    """Render the application stylesheet from the token palette."""
    T.set_theme("light")
    return _light_qss()


LIGHT_QSS = build_qss()


def apply(app, dark: bool = False) -> None:
    """Apply the design system to the whole application.

    ``dark`` is accepted and ignored: PAAF ships a single light theme.
    """
    from PyQt5.QtGui import QPalette
    app.setStyle("Fusion")
    app.setPalette(QPalette())   # let the QSS drive colors
    app.setStyleSheet(build_qss())
