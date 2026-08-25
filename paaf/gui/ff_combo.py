"""One force-field chooser, grouped by the tool that produces the parameters.

The list had grown to 46 entries in one flat run, which is more than anyone
can scan. The natural split is by *typing engine*, because that is what
actually differs: Moltemplate reads bundled ``.lt`` libraries and writes
``system.in.init``/``system.in.settings``, while DL_FIELD types atoms from its
own ``.sf``/``.par`` files and writes a self-contained data file. That choice
also decides which relaxation engine is available, so it is worth seeing.

Headings are inserted as disabled rows. Qt has no group concept in a
``QComboBox``, and the alternative — a tree view in a popup — breaks keyboard
selection. A disabled row is skipped by the arrow keys and cannot be chosen,
which is the behaviour wanted.

Force fields that cannot actually be run are shown greyed out rather than
hidden, and say why in their tooltip. ``OPLS2020.par`` ships in DL_FIELD's
``lib`` but has no control-file key in 4.13, so selecting it aborts the run
with no output at all; silently omitting it would leave a user hunting for a
force field they can see on disk.
"""
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QStandardItem
from PyQt5.QtWidgets import QComboBox

from . import tokens as T

__all__ = ["populate_force_field_combo", "GROUPS"]

#: Heading text per ``ForceField.kind``, in the order they are shown.
GROUPS = (
    ("moltemplate_native", "MOLTEMPLATE",
     "Bundled .lt libraries; relaxed with LAMMPS."),
    ("gaff", "ANTECHAMBER",
     "Typed by antechamber, then written through Moltemplate."),
    ("dlfield", "DL_FIELD",
     "Typed from DL_FIELD's .sf/.par; LAMMPS or GROMACS."),
)


def _heading(combo: QComboBox, text: str, subtitle: str = "") -> None:
    """A non-selectable row that separates one group from the next."""
    combo.addItem(text)
    index = combo.count() - 1
    model = combo.model()
    item = model.item(index) if hasattr(model, "item") else None
    if item is None:                                   # pragma: no cover
        return
    # No flags at all: not enabled, not selectable. Both are needed — the
    # first stops the arrow keys landing on it, the second stops a mouse
    # click choosing it.
    item.setFlags(Qt.NoItemFlags)
    font = QFont(combo.font())
    font.setBold(True)
    item.setFont(font)
    item.setData(text, Qt.DisplayRole)
    if subtitle:
        item.setData(subtitle, Qt.ToolTipRole)


def _unavailable_reason(ff) -> str:
    """Why this force field cannot be run, or ``""`` if it can."""
    if ff.kind != "dlfield":
        return ""
    from ..dlfield_styles import dl_field_accepts, scheme_for

    scheme = scheme_for(ff.key)
    if scheme is None:
        return ""
    if not scheme.supported or not dl_field_accepts(scheme.dl_key):
        return (f"{scheme.par} ships with DL_FIELD but version 4.13 has no "
                f"control-file key for it, so a run would abort immediately. "
                f"{scheme.note}".strip())
    return ""


def populate_force_field_combo(combo: QComboBox,
                               default_key: Optional[str] = None) -> None:
    """Fill ``combo`` with every force field, grouped under its typing engine.

    ``userData`` is the registry key on selectable rows and ``None`` on
    headings, so callers keep using ``currentData()`` unchanged.
    """
    from ..ff_registry import list_ffs

    combo.clear()
    by_kind = {}
    for ff in list_ffs():
        by_kind.setdefault(ff.kind, []).append(ff)

    seen = set()
    for kind, title, subtitle in GROUPS:
        entries = by_kind.get(kind, [])
        if not entries:
            continue
        seen.add(kind)
        _heading(combo, f"— {title} —", subtitle)
        for ff in entries:
            _add_force_field(combo, ff)

    # Anything with a kind nobody listed still has to appear, or adding a new
    # engine would silently make its force fields unreachable.
    leftovers = [ff for k, v in by_kind.items() if k not in seen for ff in v]
    if leftovers:
        _heading(combo, "— OTHER —")
        for ff in leftovers:
            _add_force_field(combo, ff)

    # Index 0 is now a heading, so a combo left at its default index would
    # display one and return None from currentData(). Land on something real.
    if not (default_key and select_force_field(combo, default_key)):
        for i in range(combo.count()):
            if combo.itemData(i) is not None:
                combo.setCurrentIndex(i)
                break


def _add_force_field(combo: QComboBox, ff) -> None:
    combo.addItem(f"    {ff.display_name}  [{ff.key}]", userData=ff.key)
    index = combo.count() - 1
    model = combo.model()
    item = model.item(index) if hasattr(model, "item") else None
    if item is None:                                   # pragma: no cover
        return

    reason = _unavailable_reason(ff)
    if reason:
        item.setEnabled(False)
        item.setData(f"    {ff.display_name}  [{ff.key}] — unavailable",
                     Qt.DisplayRole)
        item.setData(reason, Qt.ToolTipRole)
    elif ff.notes:
        item.setData(ff.notes, Qt.ToolTipRole)


def select_force_field(combo: QComboBox, key: str) -> bool:
    """Select a force field by registry key. ``False`` if it is not there."""
    for i in range(combo.count()):
        if combo.itemData(i) == key:
            combo.setCurrentIndex(i)
            return True
    return False
