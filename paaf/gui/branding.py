"""Application icon and brand mark.

The icon is a polymer backbone — a zig-zag of bonded nodes with a pendant
group on the middle one — which is literally what PAAF builds. Regenerate the
PNGs with ``python scripts/make_logo.py``.

Everything here degrades gracefully: if the asset files are missing (a partial
checkout, say) the app still starts, just without an icon.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

ASSETS = Path(__file__).resolve().parent / "assets"
SIZES = (16, 24, 32, 48, 64, 128, 256, 512)


def icon_path(size: int = 256) -> Optional[Path]:
    """Path to the PNG nearest ``size``, or ``None`` if none are installed."""
    exact = ASSETS / f"paaf_{size}.png"
    if exact.exists():
        return exact
    for s in sorted(SIZES, key=lambda x: abs(x - size)):
        p = ASSETS / f"paaf_{s}.png"
        if p.exists():
            return p
    return None


def app_icon():
    """A multi-resolution ``QIcon``, or ``None`` when the assets are absent.

    All sizes are added to one QIcon so Qt picks the right one per context —
    title bar, dock, task switcher — instead of scaling a single bitmap.
    """
    try:
        from PyQt5.QtGui import QIcon
    except Exception:
        return None

    icon = QIcon()
    added = 0
    for s in SIZES:
        p = ASSETS / f"paaf_{s}.png"
        if p.exists():
            icon.addFile(str(p))
            added += 1
    return icon if added else None


def logo_pixmap(size: int = 24):
    """A ``QPixmap`` of the mark for in-window use (e.g. the sidebar brand)."""
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QPixmap
    except Exception:
        return None
    p = icon_path(size)
    if p is None:
        return None
    pm = QPixmap(str(p))
    if pm.isNull():
        return None
    if pm.width() != size:
        pm = pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pm
