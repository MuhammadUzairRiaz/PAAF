"""Advanced manual typing: mix a united-atom library into an all-atom chain.

A thin extension of :class:`AtomTypingDialog` (which is left untouched).
On top of the primary force field's type table it loads a second,
united-atom library (TraPPE-UA or OPLS-UA). Its bead types appear in the
same table tagged ``UA:<id>`` and can be assigned to carbons like any other
type. The dialog then *learns* the consequence from the type itself: a
``CH2`` bead absorbs two hydrogens, ``CH3`` three, ``CH`` one — those
hydrogen rows are shown as "absorbed" and are removed at export
(:mod:`paaf.ua_hybrid`). Assign an all-atom type to the same carbon again
and its hydrogens come back.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel,
                             QPushButton, QTableWidgetItem)

from ..ua_hybrid import UA_PREFIX, bead_table
from .atom_type_dialog import AtomTypingDialog

ABSORBED = "(absorbed into UA bead)"


class AdvancedTypingDialog(AtomTypingDialog):
    """Manual typing with an optional second (united-atom) library."""

    def __init__(self, *args, ua_key: Optional[str] = None, **kwargs):
        self._ua_key: Optional[str] = ua_key
        self._ua_beads: Dict[str, Tuple[int, float, str]] = {}
        self._ua_types = []
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Advanced atom typing — mix united-atom and all-atom")
        self._install_ua_bar()
        if ua_key:
            idx = self.ua_combo.findData(ua_key)
            if idx >= 0:
                self.ua_combo.setCurrentIndex(idx)
            self._load_ua_library()

    # ------------------------------------------------------------ UI
    def _install_ua_bar(self) -> None:
        from ..ff_registry import list_ffs
        gb = QGroupBox("United-atom library to mix in (Advanced)")
        h = QHBoxLayout(gb)
        h.addWidget(QLabel("Secondary force field:"))
        self.ua_combo = QComboBox()
        self.ua_combo.addItem("— none —", userData=None)
        for ff in list_ffs():
            if ff.united_atom and ff.kind == "moltemplate_native" and ff.bundled_lt:
                self.ua_combo.addItem(ff.display_name, userData=ff.key)
        h.addWidget(self.ua_combo, 1)
        b = QPushButton("Load its bead types")
        b.clicked.connect(self._load_ua_library)
        h.addWidget(b)
        h.addWidget(QLabel("Show types from:"))
        self.lib_filter = QComboBox()
        for label, key in (("both libraries", "all"), ("all-atom only", "aa"),
                           ("united-atom only (UA:)", "ua")):
            self.lib_filter.addItem(label, userData=key)
        self.lib_filter.currentIndexChanged.connect(self._apply_type_filter)
        h.addWidget(self.lib_filter)
        self.ua_note = QLabel("")
        self.ua_note.setWordWrap(True)
        self.ua_note.setStyleSheet("color:#64748B;")
        # Put the bar at the top of the dialog, under the intro text.
        lay = self.layout()
        lay.insertWidget(1, gb)
        lay.insertWidget(2, self.ua_note)

    def _library_filter(self) -> str:
        return self.lib_filter.currentData() if hasattr(self, "lib_filter") else "all"

    def ua_key(self) -> Optional[str]:
        """The secondary library actually used (None if no UA type assigned)."""
        if any(t.startswith(UA_PREFIX) for t in self._types_by_atom.values()):
            return self._ua_key
        return None

    # ------------------------------------------------------------ library
    def _load_ua_library(self) -> None:
        from ..ff_registry import get_ff
        from ..lt_parser import parse_atom_types
        key = self.ua_combo.currentData()
        # drop previously appended UA rows
        self._ff_types = [t for t in self._ff_types if not t.ff_id.startswith(UA_PREFIX)]
        self._ua_key = key
        self._ua_beads = {}
        if not key:
            self._render_types(self._ff_types)
            self.ua_note.setText("")
            self._refresh_current_column()
            return
        ff = get_ff(key)
        lt = ff.bundled_path()
        self._ua_beads = bead_table(lt)
        ua_types = parse_atom_types(str(lt))
        tagged = []
        for t in ua_types:
            n = self._ua_beads.get(t.ff_id)
            desc = f"[UA {ff.key}] " + (t.description or t.key or "")
            if n is not None:
                desc += f"  — bead absorbs {n[0]} H (mass {n[1]:.3f})"
            tagged.append(type(t)(ff_id=UA_PREFIX + t.ff_id, element=t.element,
                                  key=t.key, charge=t.charge, mass=(n[1] if n else t.mass),
                                  params=t.params, description=desc))
        self._ff_types = list(self._ff_types) + tagged
        self._render_types(self._ff_types)
        n_beads = len(self._ua_beads)
        self.ua_note.setText(
            f"{ff.display_name}: {len(tagged)} types loaded (tagged UA:), "
            f"{n_beads} of them carbon beads. Assign a bead to a carbon and its "
            f"hydrogens are absorbed — they will not be written. "
            + ("OPLS-UA beads live inside OPLS-AA 2024, so mixed bonded terms "
               "are Jorgensen's own." if key == "oplsua_2024" else
               "A bridge library (paaf_ua_bridge.lt) is generated at export: "
               "UA Lennard-Jones and mass, bonded terms borrowed from the "
               "all-atom sp3 carbon class."))

    # ------------------------------------------------------------ assignment
    def _entry_for_key(self, key: int) -> Optional[dict]:
        for entries in self._rows_by_path.values():
            for e in entries:
                if e["key"] == key:
                    return e
        return None

    def _key_for(self, molecule, source_index: int) -> Optional[int]:
        for entries in self._rows_by_path.values():
            for e in entries:
                if e["molecule"] is molecule and e["source_index"] == source_index:
                    return e["key"]
        return None

    def _absorbed_h_keys(self) -> Dict[int, int]:
        """``{H row key: bead row key}`` for every hydrogen a UA bead absorbs."""
        out: Dict[int, int] = {}
        for key, tid in self._types_by_atom.items():
            if not tid.startswith(UA_PREFIX):
                continue
            spec = self._ua_beads.get(tid[len(UA_PREFIX):])
            if not spec:
                continue
            e = self._entry_for_key(key)
            if e is None:
                continue
            mol = e["molecule"]; i = e["source_index"]
            hs = sorted(j for j in mol.neighbors(i) if mol.atoms[j].element == "H")
            for j in hs[:spec[0]]:
                hk = self._key_for(mol, j)
                if hk is not None:
                    out[hk] = key
        return out

    def _assign(self, quiet: bool = False):
        tid = self._selected_type_id()
        if tid and tid.startswith(UA_PREFIX):
            atoms = self._selected_atom_indices()
            if not atoms:
                if not quiet:
                    self._alert("Select one or more atoms on the left.")
                return
            spec = self._ua_beads.get(tid[len(UA_PREFIX):])
            blocked = []
            for idx in atoms:
                e = self._entry_for_key(idx)
                if e is None:
                    continue
                mol = e["molecule"]; i = e["source_index"]
                el = mol.atoms[i].element
                if spec is not None:
                    if el != "C":
                        blocked.append(f"atom {idx}: {tid} is a carbon bead, this atom is {el}")
                        continue
                    n_h = sum(1 for j in mol.neighbors(i) if mol.atoms[j].element == "H")
                    if n_h < spec[0]:
                        blocked.append(f"atom {idx}: has {n_h} H but bead {tid} absorbs {spec[0]}")
                        continue
                elif el == "H":
                    blocked.append(f"atom {idx}: united-atom libraries have no hydrogen types")
                    continue
            if blocked:
                self._alert("\n".join(blocked))
                return
            for idx in atoms:
                self._types_by_atom[idx] = tid
            self._refresh_current_column()
            self._refresh_apply_button()
            self._sync_viewer_types()
            return
        super()._assign(quiet=quiet)

    def _refresh_current_column(self):
        super()._refresh_current_column()
        absorbed = self._absorbed_h_keys()
        if not absorbed:
            return
        for r in range(self.atom_table.rowCount()):
            key = self._row_key(r)
            if key in absorbed:
                it = QTableWidgetItem(ABSORBED)
                it.setForeground(QColor("#94A3B8"))
                self.atom_table.setItem(r, 6, it)

    def overrides(self) -> Dict[int, str]:
        """Manual assignments; hydrogens absorbed by a bead are dropped."""
        absorbed = self._absorbed_h_keys()
        return {k: v for k, v in self._types_by_atom.items()
                if k not in absorbed and v != ABSORBED}

    def _apply_all(self) -> None:
        # Absorbed hydrogens are not "untyped": mark them so the base class
        # does not ask about them; overrides() drops the marker again.
        for hk in self._absorbed_h_keys():
            self._types_by_atom.setdefault(hk, ABSORBED)
        super()._apply_all()
