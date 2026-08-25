"""Manual atom-typing dialog.

Opened from the Force-field page after the user picks a Moltemplate-native
FF. Presents:

- LEFT PANEL: table of every atom in the loaded monomer(s) with columns
  Index / Element / Neighbors / SMARTS guess / Current type.
- RIGHT PANEL: searchable list of every atom type in the imported .lt file
  (id, element, key, charge, description) with a filter by element.

Workflow:
1. Click an atom row on the left (or Ctrl-click multiple).
2. Click a type row on the right.
3. Hit "Assign" (or double-click the type) to write it into the atom row.
4. Repeat until every atom shows a specific FF type (or use "Auto-type all"
   to seed the whole thing from the SMARTS typer, then hand-edit).
5. "OK" writes the manual overrides back to the caller as a dict of
   ``{atom_index: ff_type_id}``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .page import wrap_tooltip


class AtomTypingDialog(QDialog):
    """Modal dialog for manually assigning FF atom types.

    Parameters
    ----------
    monomer_files : list[str]
        Structure files (xyz/pdb/mol2) whose atoms will be shown.
    ff_lt_path : Path
        Path to a Moltemplate .lt file to read atom types from.
    initial_types : dict[int, str]
        Pre-filled overrides (e.g. the SMARTS typer result).

    On accept, use :meth:`overrides` to fetch the resulting mapping.
    """

    def __init__(
        self,
        monomer_files: List[str],
        ff_lt_path: Path,
        initial_types: Optional[Dict[int, str]] = None,
        parent: Optional[QWidget] = None,
        ff_key: Optional[str] = None,
        ff_inherit: Optional[str] = None,
        monomer_specs: Optional[list] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Manual atom typing")
        self.resize(1560, 880)
        self.setSizeGripEnabled(True)
        self._monomer_files = monomer_files
        # Trimer built per monomer file, so the 3D view can show the
        # junction the types were actually decided in.
        self._trimer_by_path = {}
        # Per file, the assignable-atom entries backing the table rows — one
        # per atom of each of the three units.
        self._rows_by_path = {}
        self._ff_lt_path = Path(ff_lt_path) if ff_lt_path else None
        self._ff_key = (ff_key or "").lower()
        self._ff_inherit = (ff_inherit or "").upper()
        # Map file -> MonomerSpec so we can honour manual head/tail atoms and
        # relabel polymer connection carbons as in-chain backbone atoms.
        self._specs_by_file = {}
        for s in (monomer_specs or []):
            try:
                self._specs_by_file[s.file] = s
            except Exception:
                pass
        self._types_by_atom: Dict[int, str] = dict(initial_types or {})
        self._notes: List[str] = []
        self._loaded_atoms = []      # (index, element, neighbors_summary)
        self._ff_types = []          # list of AtomTypeInfo
        self._build()
        # Load the FF-types table FIRST so that _load_atoms can annotate its
        # SMARTS-guess column with human-readable descriptions from the FF library.
        self._load_ff_types()
        self._load_atoms()

    # ============================================================ layout
    def _build(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 14, 14, 14); v.setSpacing(10)

        intro = QLabel(
            "<b>Assign each atom a force-field type.</b><br>"
            "Left: every atom in your monomer with its <i>authentic chemical "
            "environment</i> (e.g. epoxide CH, aromatic CH, ester carbonyl C) — "
            "and for each hydrogen, the environment of the carbon it sits on "
            "(\"H on epoxide CH\" vs \"H on CH3\"), because most force fields "
            "type those hydrogens differently. Right: the atom types available "
            "in this FF's library. Click an atom, then a type, then "
            "<i>Assign</i> — or hit <i>Auto-type all</i> to seed the OPLS-AA "
            "family from the SMARTS typer and fine-tune.<br>"
            "The <b>Unit</b> column shows which part of the chain each row "
            "belongs to. <b>Head</b> and <b>Tail</b> are the two chain ends: "
            "they keep the cap hydrogen the junction would have consumed, so "
            "their link atom is a terminal CH3 where the <b>repeat unit</b>'s "
            "is a backbone CH2 — a different force-field type. All three are "
            "yours to assign, and all three are clickable in the 3D view."
        )
        intro.setWordWrap(True)
        v.addWidget(intro)

        split = QSplitter(Qt.Horizontal, self)
        split.setChildrenCollapsible(False)

        # ---------------------------- LEFT: atoms table
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        gb_atoms = QGroupBox("Atoms")
        gl = QVBoxLayout(gb_atoms)
        self.atom_table = QTableWidget(0, 7)
        self.atom_table.setHorizontalHeaderLabels(
            ["Idx", "Unit", "Element", "Chemical environment", "Neighbors",
             "Suggested type", "Current type"]
        )
        # Don't stretch the Neighbors column — Stretch was actually shrinking it
        # to leftover width and wrapping the text ("C-H-\nH-H"). Give it a fixed,
        # generous width so entries like "C-H-H-H" always fit on one line.
        self.atom_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Interactive)
        self.atom_table.setColumnWidth(4, 180)
        # Turn off word wrap so cells never split into two rows.
        self.atom_table.setWordWrap(False)
        self.atom_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.atom_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.atom_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.atom_table.setMinimumWidth(0)
        # Picking a row highlights that atom in 3D. With labels set to
        # "selected + neighbours" this is what makes the table and the picture
        # one thing rather than two: you can drive it from either end.
        self.atom_table.itemSelectionChanged.connect(self._on_row_selected)
        gl.addWidget(self.atom_table)
        ll.addWidget(gb_atoms, 1)

        # Failures used to be sent to a `log` attribute this dialog does not
        # have, so `getattr(self, "log", None)` was always None and every
        # reason a suggestion was missing went straight to the bin. The result
        # was a molecule drawn in plain element colours with no explanation --
        # untyped atoms get no palette colour, so "black" MEANS "untyped", and
        # nothing said why.
        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet("color:#8a6d3b;")
        self.notes.setVisible(False)
        ll.addWidget(self.notes)

        b_auto = QPushButton("Auto-type all (from SMARTS)")
        b_auto.clicked.connect(self._auto_type_all)
        ll.addWidget(b_auto)
        split.addWidget(left)

        # ---------------------------- RIGHT: FF types
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        gb_ff = QGroupBox("Force-field atom types")
        rg = QVBoxLayout(gb_ff)
        bar = QHBoxLayout()
        self.type_filter = QLineEdit()
        self.type_filter.setPlaceholderText("Filter by id / element / key / description…")
        self.type_filter.textChanged.connect(self._apply_type_filter)
        self.elem_filter = QComboBox()
        self.elem_filter.addItem("all elements")
        self.elem_filter.currentTextChanged.connect(self._apply_type_filter)
        bar.addWidget(self.type_filter, 1); bar.addWidget(self.elem_filter)
        rg.addLayout(bar)

        self.type_table = QTableWidget(0, 5)
        self.type_table.setHorizontalHeaderLabels(
            ["FF id", "Element", "Key", "Charge", "Description"]
        )
        self.type_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.type_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.type_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.type_table.setMinimumWidth(0)
        self.type_table.doubleClicked.connect(self._assign)
        rg.addWidget(self.type_table)
        rl.addWidget(gb_ff, 1)

        # Click an atom, click a type, done.
        #
        # Pressing a third button for every atom is 25 extra clicks on a PBS
        # monomer and 75 across the three units, and every one of them is a
        # chance to have moved the selection since. With this on, picking a
        # type IS the assignment, and the only button left to press is the one
        # at the end.
        self.quick_assign = QCheckBox("Assign as I click — picking a type "
                                      "applies it to the selected atoms")
        self.quick_assign.setChecked(True)
        self.quick_assign.setToolTip(wrap_tooltip("On: click an atom, then click a type, and it is assigned.\n"
            "Off: use the Assign button below, as before."))
        rl.addWidget(self.quick_assign)
        # itemClicked, NOT itemSelectionChanged.
        #
        # The selection signal also fires when the table is filtered or
        # re-sorted, so typing in the filter box could assign whatever row
        # happened to land under the selection to whatever atoms were
        # selected -- silently, and to the wrong element. Only a real click
        # counts as "I chose this type".
        self.type_table.itemClicked.connect(self._on_type_picked)

        b_assign = QPushButton("Assign selected type ▶  to selected atoms")
        b_assign.clicked.connect(self._assign)
        rl.addWidget(b_assign)

        # Typing one row at a time is where a mis-keyed row hides among a dozen
        # near-identical ones. Assigning by equivalence class removes the
        # opportunity entirely.
        b_equiv = QPushButton("Assign to ALL chemically equivalent atoms")
        b_equiv.setToolTip(wrap_tooltip("Click one CH2 hydrogen and every symmetry-equivalent hydrogen "
            "gets the same type. Element mismatches are skipped."))
        b_equiv.clicked.connect(self._assign_equivalent)
        rl.addWidget(b_equiv)
        split.addWidget(right)

        # ---------------------------- 3D view of the actual molecule
        self.viewer3d = None
        try:
            from .mol3d_view import Mol3DView
            if Mol3DView is not None:
                self.viewer3d = Mol3DView(self)
                self.viewer3d.atom_clicked.connect(self._on_viewer_atom_clicked)
                self.viewer3d.popout_requested.connect(self._popout_viewer)
                gb_3d = QGroupBox("Structure")
                g3 = QVBoxLayout(gb_3d)
                g3.setContentsMargins(6, 6, 6, 6)
                g3.addWidget(self.viewer3d)
                self._viewer_box = gb_3d
                split.addWidget(gb_3d)
        except Exception as exc:                        # pragma: no cover
            self.viewer3d = None
            _log = getattr(self, "log", None)
            if _log is not None:
                _log.emit(f"3D view unavailable: {exc}")

        # The 3D view gets the largest share: it is the surface you work on,
        # and the previous split left it too narrow for the molecule to be
        # read at all. The tables stay fully usable, and the splitter is still
        # yours to drag.
        split.setSizes([460, 430, 760] if self.viewer3d is not None
                       else [550, 650])
        v.addWidget(split, 1)

        # ---------------------------- bottom bar
        row = QHBoxLayout()
        b_save = QPushButton("Save profile...");  b_save.clicked.connect(self._save_profile)
        b_load = QPushButton("Load profile...");  b_load.clicked.connect(self._load_profile)
        row.addWidget(b_save); row.addWidget(b_load); row.addStretch(1)
        # The single button at the end. It says how many atoms it is about to
        # commit, because "OK" on a typing dialog tells you nothing about
        # whether you finished.
        self.apply_button = QPushButton("Apply all assignments")
        self.apply_button.setObjectName("primary")
        self.apply_button.clicked.connect(self._apply_all)
        row.addWidget(self.apply_button)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        row.addWidget(bb)
        v.addLayout(row)

    # ============================================================ data
    # The chemistry classifier lives in the force-field-independent
    # ``paaf.chem_env`` module so the SAME authentic mapping is available for
    # every Moltemplate-native force field. These thin delegators are kept for
    # backwards compatibility with older callers/tests.
    @staticmethod
    def _group_label(element: str, neighbor_elements: list, **ctx) -> str:
        from ..chem_env import group_label
        return group_label(element, neighbor_elements, **ctx)

    @staticmethod
    def _detect_context(mol, atom_index: int):
        from ..chem_env import detect_context
        return detect_context(mol, atom_index)

    def _load_monomer(self, path: str):
        """Return (mol, link_carbons) for a monomer file.

        ``link_carbons`` maps each polymer connection carbon index to the cap
        hydrogen removed on polymerisation (or None).  Uses PAAF's monomer
        head/tail detection (explicit table indices or the ``.polysmi``
        sidecar).  Falls back to a plain structure load with no link info.
        """
        from ..structure import load_structure
        spec = self._specs_by_file.get(path)
        try:
            from ..monomer import monomer_from_file
            kw = {}
            if spec is not None:
                kw = dict(head=spec.head, tail=spec.tail,
                          head_h=spec.head_h, tail_h=spec.tail_h,
                          name=spec.name)
            mono = monomer_from_file(path, **kw)
            mol = mono.molecule
            link = {}
            link[mono.head_index] = (mono.head_removes[0]
                                     if mono.head_removes else None)
            link[mono.tail_index] = (mono.tail_removes[0]
                                     if mono.tail_removes else None)
            # Only keep carbon connection atoms (that's what the user asked about).
            link = {ci: h for ci, h in link.items()
                    if 0 <= ci < len(mol.atoms) and mol.atoms[ci].element == "C"}
            return mol, link
        except Exception:
            return load_structure(path), {}

    def _is_opls_family(self) -> bool:
        """True when the chosen FF uses the OPLS-AA numeric type IDs.

        All OPLS-AA variants (2024, 2008, L-OPLS-AA 2024/2008) inherit the same
        ``OPLSAA`` base and share one atom-type numbering, so the authentic
        SMARTS typer in ``paaf.typers.oplsaa`` applies to every one of them.
        United-atom OPLS (OPLS-UA) uses a different numbering, so it is excluded.
        """
        if self._ff_inherit == "OPLSAA":
            return True
        name = (self._ff_key or "")
        if "ua" in name:            # oplsua_2024, opls_ua -> different numbering
            return False
        return name.startswith("oplsaa") or name.startswith("loplsaa")

    def _load_atoms(self):
        """Load the atoms from the monomer files into the left table.

        Each atom row includes:
          * Element
          * Group        — human-readable (CH3 methyl / epoxide O / =CH- / ar-CH ...)
          * Neighbors    — raw neighbor elements
          * SMARTS guess — the SMARTS typer's authentic OPLS numeric type + a
                           short chemistry description read out of oplsaa2024.lt
          * Current type — whatever the user (or Auto-type all) has chosen
        """
        try:
            from ..structure import load_structure
        except Exception:
            QMessageBox.critical(self, "Load failed",
                                 "Structure loader not available."); return

        # Load FF-type descriptions once so we can annotate the SMARTS guess.
        type_desc: Dict[str, str] = {}
        for t in self._ff_types:
            # e.g. "135" -> "CT / sp3 CH3 alkane"
            type_desc[t.ff_id] = (
                f"{t.key} / {t.description}" if t.key else t.description
            )

        from ..chem_env import all_groups_with_links, suggest_ff_type

        # The authentic SMARTS typer only knows the OPLS-AA numbering, so it is
        # used only for the OPLS family. For every other Moltemplate FF we fall
        # back to a generic environment-based ranking of that FF's OWN atom-type
        # library (so the suggested id always exists in the imported .lt).
        use_opls = self._is_opls_family()
        smarts_guess_by_index: Dict[int, str] = {}
        _typer_ok = False
        if use_opls:
            try:
                from ..typers.oplsaa import type_oplsaa
                _typer_ok = True
            except Exception:
                _typer_ok = False

        rows = []
        for path in self._monomer_files:
            try:
                mol, link_carbons = self._load_monomer(path)
            except Exception as exc:
                self._alert(f"Could not load {path}: {exc}"); continue

            # ---- every unit of the chain, not only the repeat unit --------
            #
            # A chain is head - repeat... - tail, and the two ends are a
            # different molecule from the middle: they keep the cap hydrogen
            # the junction would have consumed, so their link carbon is a
            # terminal CH3 (135) where the interior is a backbone CH2 (136).
            # Typing only the middle leaves both ends to the automatic typer
            # working on a chain whose bond orders may already be wrong.
            #
            # So all three units of the trimer are listed and assignable. The
            # trimer is the chain in miniature, which means every row here is
            # an atom in the environment it will really have.
            unit_rows = self._unit_rows(mol, path)

            if _typer_ok:
                try:
                    smarts_guess_by_index.update(
                        self._suggestions_in_chain_context(mol, path))
                except Exception as exc:
                    self._note(
                        f"No suggested types: the SMARTS typer failed on "
                        f"{Path(path).name} ({exc}). Assign types by hand, or "
                        f"pick a force field whose library PAAF can rank "
                        f"against.")

            # Connection-aware chemistry for the fallback (no-trimer) case:
            # backbone carbons that link monomers are shown as their in-chain
            # CH2 environment, not the capped CH3.
            fallback_groups = all_groups_with_links(mol, link_carbons)

            for entry in unit_rows:
                source = entry["molecule"]
                index = entry["source_index"]
                a = source.atoms[index]
                nbr_elems = [source.atoms[j].element
                             for j in source.neighbors(index)]
                group = entry.get("group") or fallback_groups.get(index, "")

                guess_id = entry.get("guess") or ""
                if not guess_id and entry["role"] == "middle":
                    guess_id = smarts_guess_by_index.get(
                        entry["monomer_index"], "")
                if not guess_id and self._ff_types:
                    # Generic per-FF suggestion from the imported library.
                    cand = suggest_ff_type(group, a.element, self._ff_types)
                    if cand:
                        guess_id = cand[0][0]

                if guess_id and guess_id in type_desc:
                    guess_col = f"{guess_id}  ({type_desc[guess_id]})"
                else:
                    guess_col = guess_id
                rows.append((
                    entry["key"], entry["monomer_index"], entry["role"],
                    a.element, group,
                    "-".join(sorted(nbr_elems)) or "(none)",
                    guess_col,
                ))

        role_label = {"head": "Head (chain start)",
                      "middle": "Repeat unit",
                      "tail": "Tail (chain end)",
                      # Not a unit: the -OH the builder adds after assembly to
                      # turn a polyester's terminal -C(=O)H into -C(=O)OH.
                      "cap": "End cap (-COOH)"}
        self.atom_table.setRowCount(0)
        for key, monomer_index, role, elem, group, nbrs, guess in rows:
            r = self.atom_table.rowCount(); self.atom_table.insertRow(r)
            # The row shows the MONOMER atom index, because that is what the
            # user sees in their file. The role-encoded key that actually
            # identifies the row rides along as item data, so a head atom 4 and
            # a tail atom 4 stay distinct without showing the user an
            # unrecognisable number.
            id_item = QTableWidgetItem(str(monomer_index))
            id_item.setData(Qt.UserRole, int(key))
            self.atom_table.setItem(r, 0, id_item)
            self.atom_table.setItem(r, 1, QTableWidgetItem(role_label[role]))
            self.atom_table.setItem(r, 2, QTableWidgetItem(elem))
            self.atom_table.setItem(r, 3, QTableWidgetItem(group))
            self.atom_table.setItem(r, 4, QTableWidgetItem(nbrs))
            self.atom_table.setItem(r, 5, QTableWidgetItem(guess))
            # If the user hasn't already assigned a type, seed with the guess.
            current = self._types_by_atom.get(key)
            if not current and guess:
                # strip the descriptive suffix "(CT / ...)" to keep just the id
                current = guess.split()[0]
                self._types_by_atom[key] = current
            self.atom_table.setItem(r, 6, QTableWidgetItem(current or ""))
        self.atom_table.resizeColumnsToContents()
        # Re-enforce the Neighbors column width after resize-to-contents
        self.atom_table.setColumnWidth(4, max(160, self.atom_table.columnWidth(4)))
        self._loaded_atoms = rows

        # Drop assignments that belong to a monomer no longer on the table.
        #
        # ``_types_by_atom`` outlives a change of monomer — it is seeded from
        # the project and kept as the user moves around — so switching from a
        # 26-atom PBS to a 12-atom butadiene left 13 keys pointing at atoms
        # that do not exist here. Apply then refused everything with "cannot be
        # assigned to a ? atom" for each of them, and no amount of correcting
        # the visible rows could clear it, because the offending rows were not
        # visible. Said out loud rather than dropped in silence: they were the
        # user's assignments and they are being discarded.
        from ..type_guard import drop_unknown

        kept, stale = drop_unknown(self._types_by_atom,
                                   [row[0] for row in rows])
        if stale:
            self._types_by_atom = kept
            from ..logging_utils import get_logger
            self._note(f"{len(stale)} atom type(s) assigned to a different "
                       f"monomer were cleared — they do not correspond to any "
                       f"atom in this one.")
            get_logger(__name__).info(
                "Cleared %d stale atom type(s) not present in this monomer: %s",
                len(stale), stale[:12])

        self._refresh_apply_button()

        # Hand the same molecules to the 3D view, with the table's own atom
        # indices, so a click there selects the right row here.
        viewer = getattr(self, "viewer3d", None)
        if viewer is not None:
            # Show the TRIMER wherever one could be built, not the isolated
            # monomer. The junction is the thing being reasoned about: seeing
            # two disconnected fragments still leaves you inferring what
            # happens when they join, and that inference is what put 135 on a
            # backbone CH2.
            #
            # Every atom of all three units is clickable, because every one of
            # them needs a type — the head and tail are the real chain ends,
            # not decoration.
            mols, index_map = [], []
            for path in self._monomer_files:
                ctx = self._trimer_by_path.get(str(path))
                if ctx is not None:
                    mols.append(ctx.molecule)
                    keys = {e["source_index"]: e["key"]
                            for e in self._rows_by_path.get(str(path), [])}
                    index_map.extend(keys.get(i, -1)
                                     for i in range(len(ctx.molecule.atoms)))
                    continue
                try:
                    mol, _links = self._load_monomer(path)
                except Exception:
                    continue
                mols.append(mol)
                index_map.extend(a.index for a in mol.atoms)
            if mols:
                try:
                    viewer.show_molecules(mols, index_map=index_map)
                    # The viewer addresses atoms by the role-encoded key, but
                    # nobody should have to read "1000000" off a picture. Label
                    # each atom with its monomer index and which unit it is in:
                    # h0 / 0 / t0.
                    viewer.set_display_names(self._viewer_labels())
                    # "Atom name" labels: element plus the number shown in the
                    # table, so the picture and the rows say the same thing.
                    if hasattr(viewer, "set_names"):
                        viewer.set_names(self._viewer_names())
                    # Let the user isolate one unit. On a trimer this is the
                    # difference between 13 atoms and 41; on anything larger it
                    # is the difference between usable and not.
                    if hasattr(viewer, "set_groups"):
                        viewer.set_groups(self._viewer_groups(),
                                          order=["Head unit", "Repeat unit",
                                                 "Tail unit"])
                    viewer.set_types(self._types_by_atom)
                except Exception as exc:               # pragma: no cover
                    _log = getattr(self, "log", None)
                    if _log is not None:
                        _log.emit(f"3D view could not render: {exc}")

    # ---------------------------------------------------------- unit rows
    def _unit_rows(self, mol, path) -> List[dict]:
        """One entry per assignable atom, across all three units of the chain.

        Each entry carries where the atom came from (``role`` and
        ``monomer_index``), where to read its chemistry from (``molecule`` and
        ``source_index``) and the role-encoded ``key`` that identifies its row.

        When no trimer can be built — a monomer with no cap atoms, so no
        junction exists — this degrades to the old single-unit listing rather
        than showing three identical copies of a molecule that never links.
        """
        from ..chem_env import atom_group

        ctx = self._trimer_context(path, mol)
        if ctx is None:
            fallback = [{"key": a.index, "monomer_index": a.index,
                         "role": "middle", "molecule": mol,
                         "source_index": a.index, "group": "", "guess": ""}
                        for a in mol.atoms]
            self._rows_by_path[str(path)] = fallback
            return fallback

        from ..typing_context import role_key

        guesses: Dict[int, str] = {}
        if self._is_opls_family():
            try:
                from ..typers.oplsaa import type_oplsaa
                # Typed on the trimer, so each unit's suggestion is already
                # right for its position: the head's link carbon still has its
                # cap and reads CH3, the middle's does not and reads CH2. The
                # distinction that had to be explained is simply true here.
                guesses = type_oplsaa(ctx.molecule)
            except Exception:
                guesses = {}

        from ..typing_context import CAP_UNIT

        role_of = {0: "head", 1: "middle", 2: "tail", CAP_UNIT: "cap"}
        out: List[dict] = []
        for trimer_index in sorted(ctx.provenance):
            unit, monomer_index = ctx.provenance[trimer_index]
            role = role_of[unit]
            out.append({
                "key": role_key(role, monomer_index),
                "monomer_index": monomer_index,
                "role": role,
                "molecule": ctx.molecule,
                "source_index": trimer_index,
                "group": atom_group(ctx.molecule, trimer_index),
                "guess": guesses.get(trimer_index, ""),
            })
        self._rows_by_path[str(path)] = out
        return out

    def _trimer_context(self, path, mol):
        """The cached trimer for a monomer file, building it if needed."""
        ctx = self._trimer_by_path.get(str(path))
        if ctx is not None:
            return ctx
        monomer = self._monomer_for(path, mol)
        if monomer is None:
            self._note(
                f"{Path(path).name} has no link atoms PAAF can identify, so "
                f"the three chain units cannot be shown. A .polysmi sidecar "
                f"with the [*] polymerisation SMILES fixes this.")
            return None
        try:
            from ..typing_context import build_trimer
            ctx = build_trimer([monomer])
        except Exception as exc:
            self._note(
                f"Could not build the head/repeat/tail context for "
                f"{Path(path).name} ({exc}), so only one unit is listed and "
                f"the chain ends cannot be typed separately.")
            return None
        self._trimer_by_path[str(path)] = ctx
        return ctx

    def _viewer_labels(self) -> Dict[int, str]:
        """``{role key: short label}`` for the 3D view.

        ``h0`` is monomer atom 0 in the head unit, ``0`` the same atom in the
        repeat unit, ``t0`` in the tail. Same atom of the same monomer, three
        different places in the chain, three different types.
        """
        # ``.get`` with a default, not ``[]``.
        #
        # This was a plain lookup, and when the "cap" role was added it raised
        # KeyError here. That call sits inside the try block that also does
        # set_groups and set_types, so ONE missing key silently disabled the
        # unit filter, the type colours AND the display names — which is why
        # atoms came out black, the Show dropdown came out empty, and labels
        # showed raw internal keys like 2000010 instead of t10.
        prefix = {"head": "h", "middle": "", "tail": "t", "cap": "cap"}
        out: Dict[int, str] = {}
        for entries in self._rows_by_path.values():
            for e in entries:
                out[e["key"]] = (f"{prefix.get(e['role'], '?')}"
                                 f"{e['monomer_index']}")
        return out

    def _viewer_names(self) -> Dict[int, str]:
        """``{role key: element + number}``, e.g. ``C7`` — the atom's name."""
        out: Dict[int, str] = {}
        for entries in self._rows_by_path.values():
            for e in entries:
                element = e["molecule"].atoms[e["source_index"]].element
                out[e["key"]] = f"{element}{e['monomer_index']}"
        return out

    def _viewer_groups(self) -> Dict[str, List[int]]:
        """``{unit name: [role keys]}`` for the viewer's unit filter."""
        label = {"head": "Head unit", "middle": "Repeat unit",
                 "tail": "Tail unit", "cap": "Tail unit"}
        groups: Dict[str, List[int]] = {}
        for entries in self._rows_by_path.values():
            for e in entries:
                groups.setdefault(label[e["role"]], []).append(e["key"])
        return groups

    def _viewer_names(self) -> Dict[int, str]:
        """``{role key: "C11"}`` — element and the index the table shows."""
        out: Dict[int, str] = {}
        for entries in self._rows_by_path.values():
            for e in entries:
                element = e["molecule"].atoms[e["source_index"]].element
                out[e["key"]] = f"{element}{e['monomer_index']}"
        return out

    def _popout_viewer(self) -> None:
        """Open the 3D view in its own window, and remember it."""
        from .mol3d_view import Mol3DWindow

        existing = getattr(self, "_popout", None)
        if existing is not None and existing.isVisible():
            existing.raise_(); existing.activateWindow(); return
        if self.viewer3d is None or Mol3DWindow is None:
            return
        self._popout = Mol3DWindow(self.viewer3d, self,
                                   title="Structure — atom typing")
        self._popout.show()

    # The outer units used to be greyed out as context, back when only the
    # middle unit could be typed. They are assignable now — they are the real
    # chain ends, and dimming them would say the opposite of what is true.

    def _load_ff_types(self):
        from ..lt_parser import parse_atom_types
        if not self._ff_lt_path or not self._ff_lt_path.exists():
            self._alert(f"Force-field .lt not found at {self._ff_lt_path}"); return
        types = parse_atom_types(str(self._ff_lt_path))
        self._ff_types = types
        # Populate element filter
        elems = sorted({t.element for t in types if t.element}) or []
        self.elem_filter.blockSignals(True); self.elem_filter.clear()
        self.elem_filter.addItem("all elements")
        for e in elems:
            self.elem_filter.addItem(e)
        self.elem_filter.blockSignals(False)
        self._render_types(types)

    def _render_types(self, types):
        self.type_table.setSortingEnabled(False); self.type_table.setRowCount(0)
        for t in types:
            r = self.type_table.rowCount(); self.type_table.insertRow(r)
            id_item = QTableWidgetItem(t.ff_id)
            try:
                id_item.setData(Qt.EditRole, int(t.ff_id))
            except ValueError:
                pass
            self.type_table.setItem(r, 0, id_item)
            self.type_table.setItem(r, 1, QTableWidgetItem(t.element))
            self.type_table.setItem(r, 2, QTableWidgetItem(t.key))
            q = QTableWidgetItem(f"{t.charge:+.4f}")
            q.setData(Qt.EditRole, t.charge)
            self.type_table.setItem(r, 3, q)
            self.type_table.setItem(r, 4, QTableWidgetItem(t.description))
        self.type_table.setSortingEnabled(True)
        # Sort ascending by FF id so users see @atom:1, 2, 3, ... first
        # (common polymer types 135/140/145 land near the middle of a long
        # list, not buried under the water-model 999x codes).
        self.type_table.sortByColumn(0, Qt.AscendingOrder)
        self.type_table.resizeColumnsToContents()

    def _apply_type_filter(self):
        q = self.type_filter.text().strip().lower()
        elem = self.elem_filter.currentText()
        rows = self._ff_types
        if elem != "all elements":
            rows = [t for t in rows if t.element == elem]
        if q:
            rows = [t for t in rows
                    if q in t.ff_id.lower() or q in t.element.lower()
                    or q in t.key.lower() or q in t.description.lower()]
        self._render_types(rows)

    # ============================================================ actions
    def _selected_atom_indices(self) -> List[int]:
        """The role-encoded keys of the selected rows.

        The key, not the displayed index: the head unit's atom 4 and the tail
        unit's atom 4 are different atoms in different environments and must
        not collapse onto one another.
        """
        rows = sorted({i.row() for i in self.atom_table.selectedIndexes()})
        out = []
        for r in rows:
            it = self.atom_table.item(r, 0)
            if it is None:
                continue
            key = it.data(Qt.UserRole)
            if key is not None:
                out.append(int(key))
            elif it.text().isdigit():
                out.append(int(it.text()))
        return out

    def _selected_type_id(self) -> Optional[str]:
        rows = {i.row() for i in self.type_table.selectedIndexes()}
        if not rows:
            return None
        r = next(iter(rows))
        it = self.type_table.item(r, 0)
        return it.text() if it else None

    def _assign(self, quiet: bool = False):
        atoms = self._selected_atom_indices()
        tid = self._selected_type_id()
        if not atoms:
            if not quiet:
                self._alert("Select one or more atoms on the left.")
            return
        if not tid:
            if not quiet:
                self._alert("Select an atom type on the right.")
            return

        # Refuse an element mismatch rather than exporting it. A carbon type on
        # a hydrogen writes that hydrogen with carbon's mass, and nothing
        # downstream complains — the data file is well formed and LAMMPS reads
        # it happily.
        from ..type_guard import check_assignment

        blocked = []
        for idx in atoms:
            message = check_assignment(self._element_of(idx), tid)
            if message:
                blocked.append(f"atom {idx}: {message}")
        if blocked:
            self._alert("\n\n".join(blocked))
            return

        for idx in atoms:
            self._types_by_atom[idx] = tid
        self._refresh_current_column()
        self._refresh_apply_button()
        self._sync_viewer_types()

    # ------------------------------------------------- in-chain suggestions
    def _suggestions_in_chain_context(self, mol, path) -> Dict[int, str]:
        """SMARTS types for the monomer, read off a trimer's middle unit.

        Falls back to typing the bare monomer if a trimer cannot be built —
        a worse answer, but better than none, and the caller logs it.
        """
        from ..typers.oplsaa import type_oplsaa

        # The cached context, not a fresh one: the table rows and the 3D view
        # both index into this exact molecule, and handing them a second,
        # separately built trimer would leave them pointing at different
        # objects that only happen to agree.
        ctx = self._trimer_context(path, mol)
        if ctx is None:
            return type_oplsaa(mol)

        trimer_types = type_oplsaa(ctx.molecule)
        folded = ctx.types_by_monomer_index(trimer_types)

        # Cap atoms exist in the monomer but not in the middle unit, so they
        # get no in-chain answer. They are removed on linking, so what they
        # are typed as never reaches the chain — but the table shows them, so
        # give them the monomer's own answer rather than a blank.
        if len(folded) < len(mol.atoms):
            for index, value in type_oplsaa(mol).items():
                folded.setdefault(index, value)
        return folded

    def _monomer_for(self, path, mol):
        """A Monomer for ``mol``, with its link atoms, or ``None``.

        Uses the same head/tail detection the rest of the dialog does, so the
        trimer is joined at exactly the atoms the chain builder will join.
        """
        spec = self._specs_by_file.get(path)
        try:
            from ..monomer import monomer_from_file
            kw = {}
            if spec is not None:
                kw = dict(head=spec.head, tail=spec.tail,
                          head_h=spec.head_h, tail_h=spec.tail_h,
                          name=spec.name)
            mono = monomer_from_file(path, **kw)
        except Exception as exc:
            log = getattr(self, "log", None)
            if log is not None:
                log.emit(f"No link information for {path} ({exc}); typing the "
                         f"monomer as-is, so chain-link carbons will read as "
                         f"CH3 (135) rather than CH2 (136).")
            return None
        if not mono.head_removes or not mono.tail_removes:
            # Without caps to consume there is no junction to build, and a
            # trimer would just be three separate molecules.
            return None
        return mono

    def _element_of(self, key: int) -> str:
        """The element of an atom, from the loaded table, by role-encoded key."""
        for row in getattr(self, "_loaded_atoms", []):
            if row[0] == key:
                return row[3]
        return ""

    def _assign_equivalent(self):
        """Assign the selected type to every chemically equivalent atom.

        Clicking one CH2 hydrogen types all of them. This is what removes the
        row-by-row hand editing where a single mis-keyed row goes unnoticed
        among fifteen near-identical ones.
        """
        atoms = self._selected_atom_indices()
        tid = self._selected_type_id()
        if not atoms or not tid:
            self._alert("Select an atom and a type first."); return

        from ..equivalence import equivalent_atoms

        # Equivalence is computed on the trimer and stays WITHIN a unit. A
        # head-unit hydrogen and a middle-unit hydrogen can be symmetry-related
        # in the trimer and still need different types, because one sits on a
        # terminal CH3 and the other on a backbone CH2 — spreading across units
        # would undo the very distinction the three units exist to make.
        targets = set(atoms)
        selected = set(atoms)
        for path in self._monomer_files:
            entries = self._rows_by_path.get(str(path))
            if entries:
                key_of = {e["source_index"]: e["key"] for e in entries}
                role_of = {e["key"]: e["role"] for e in entries}
                groups = equivalent_atoms(entries[0]["molecule"])
                for entry in entries:
                    if entry["key"] not in selected:
                        continue
                    for other in groups.get(entry["source_index"], []):
                        key = key_of.get(other)
                        if key is not None and role_of.get(key) == entry["role"]:
                            targets.add(key)
                continue
            try:
                mol, _links = self._load_monomer(path)
            except Exception:
                continue
            groups = equivalent_atoms(mol)
            for idx in list(atoms):
                targets.update(groups.get(idx, []))

        from ..type_guard import check_assignment

        ok = [i for i in sorted(targets)
              if check_assignment(self._element_of(i), tid) is None]
        for idx in ok:
            self._types_by_atom[idx] = tid
        self._refresh_current_column()
        self._refresh_apply_button()
        self._sync_viewer_types()
        from ..typing_context import split_role_key
        shown = ", ".join(f"{split_role_key(k)[1]} ({split_role_key(k)[0]})"
                          for k in ok)
        self._alert(f"Assigned {tid} to {len(ok)} equivalent atoms: {shown}")

    # ------------------------------------------------------------- 3D view
    def _sync_viewer_types(self) -> None:
        viewer = getattr(self, "viewer3d", None)
        if viewer is not None:
            try:
                viewer.set_types(self._types_by_atom)
            except Exception:                          # pragma: no cover
                pass

    def _on_type_picked(self, _item=None) -> None:
        """A type was clicked. In quick-assign mode that IS the assignment."""
        box = getattr(self, "quick_assign", None)
        if box is None or not box.isChecked():
            return
        if not self._selected_atom_indices() or not self._selected_type_id():
            return
        self._assign(quiet=True)

    def _apply_all(self) -> None:
        """Commit everything assigned so far, after one last element check."""
        from ..type_guard import check_all

        elements = {row[0]: row[3] for row in getattr(self, "_loaded_atoms", [])}
        problems = check_all(elements, self._types_by_atom)
        if problems:
            # check_all returns (atom index, message) pairs, not strings.
            # Joining them directly raised TypeError and took the whole
            # application down at the moment the user pressed Apply — an error
            # path that crashes is worse than the error it was reporting.
            lines = [f"atom {index}: {message}"
                     for index, message in problems[:8]]
            if len(problems) > 8:
                lines.append(f"...and {len(problems) - 8} more.")
            self._alert("These assignments would give an atom the mass and "
                        "charge of a different element, so nothing has been "
                        "applied:\n\n" + "\n\n".join(lines))
            return
        untyped = [k for k in elements if not self._types_by_atom.get(k)]
        if untyped:
            from PyQt5.QtWidgets import QMessageBox as _QMB
            answer = _QMB.question(
                self, "Some atoms have no type",
                f"{len(untyped)} of {len(elements)} atoms have no type yet. "
                f"They will be typed automatically.\n\nApply anyway?",
                _QMB.Yes | _QMB.No, _QMB.No)
            if answer != _QMB.Yes:
                return
        self.accept()

    def _refresh_apply_button(self) -> None:
        button = getattr(self, "apply_button", None)
        if button is None:
            return
        total = len(getattr(self, "_loaded_atoms", []))
        done = sum(1 for row in getattr(self, "_loaded_atoms", [])
                   if self._types_by_atom.get(row[0]))
        button.setText(f"Apply all assignments  ({done} of {total} atoms)")

    def _on_row_selected(self) -> None:
        """Highlight the selected row's atom in the 3D view."""
        viewer = getattr(self, "viewer3d", None)
        if viewer is None or not hasattr(viewer, "select_index"):
            return
        keys = self._selected_atom_indices()
        if len(keys) != 1:
            return              # a multi-row selection has no single subject
        try:
            viewer.select_index(keys[0])
        except Exception:                              # pragma: no cover
            pass

    def _on_viewer_atom_clicked(self, index: int) -> None:
        """A click in 3D selects the matching table row, and vice versa.

        ``index`` is the role-encoded key the viewer was given, so clicking the
        tail unit's link carbon selects the tail row, not the head's.
        """
        for r in range(self.atom_table.rowCount()):
            if self._row_key(r) == int(index):
                self.atom_table.selectRow(r)
                item = self.atom_table.item(r, 0)
                if item is not None:
                    self.atom_table.scrollToItem(item)
                break

    def _row_key(self, row: int) -> Optional[int]:
        item = self.atom_table.item(row, 0)
        if item is None:
            return None
        key = item.data(Qt.UserRole)
        if key is not None:
            return int(key)
        return int(item.text()) if item.text().isdigit() else None

    def _refresh_current_column(self):
        for r in range(self.atom_table.rowCount()):
            key = self._row_key(r)
            if key is None:
                continue
            self.atom_table.setItem(r, 6, QTableWidgetItem(
                self._types_by_atom.get(key, "")))

    def _auto_type_all(self):
        try:
            from ..structure import load_structure
            from ..chem_env import atom_group, suggest_ff_type
        except Exception as exc:
            self._alert(f"Typer unavailable: {exc}"); return

        use_opls = self._is_opls_family()
        type_oplsaa = None
        if use_opls:
            try:
                from ..typers.oplsaa import type_oplsaa as _to
                type_oplsaa = _to
            except Exception:
                use_opls = False

        for path in self._monomer_files:
            try:
                mol = load_structure(path)
            except Exception:
                continue
            # Type the trimer, so the head and tail get end types and the
            # middle gets repeat types — one pass, all three units, each in
            # its own environment.
            ctx = self._trimer_context(path, mol)
            entries = (self._rows_by_path.get(str(path))
                       or self._unit_rows(mol, path))
            source = ctx.molecule if ctx is not None else mol
            opls = (type_oplsaa(source)
                    if (use_opls and type_oplsaa is not None) else {})
            for entry in entries:
                where = entry["source_index"]
                value = opls.get(where)
                if not value:
                    # Generic environment-based ranking against THIS FF's library.
                    group = atom_group(entry["molecule"], where)
                    cand = suggest_ff_type(group, entry["molecule"].atoms[where].element,
                                           self._ff_types)
                    value = cand[0][0] if cand else None
                if value:
                    self._types_by_atom[entry["key"]] = value

        for r in range(self.atom_table.rowCount()):
            key = self._row_key(r)
            if key is None:
                continue
            self.atom_table.setItem(r, 5, QTableWidgetItem(
                self._types_by_atom.get(key, "")))
        self._refresh_current_column()
        self._refresh_apply_button()
        self._sync_viewer_types()

    def _save_profile(self):
        import json
        p, _ = QFileDialog.getSaveFileName(self, "Save typing profile",
                                          filter="JSON (*.json)")
        if p:
            Path(p).write_text(json.dumps(
                {"types": {str(k): v for k, v in self._types_by_atom.items()}},
                indent=2,
            ))

    def _load_profile(self):
        import json
        p, _ = QFileDialog.getOpenFileName(self, "Load typing profile",
                                          filter="JSON (*.json)")
        if not p:
            return
        try:
            data = json.loads(Path(p).read_text())
            for k, v in data.get("types", {}).items():
                self._types_by_atom[int(k)] = str(v)
        except Exception as exc:
            self._alert(f"Load failed: {exc}"); return
        self._refresh_current_column()

    def _note(self, message: str) -> None:
        """Show the user something that went wrong but is not fatal."""
        self._notes.append(message)
        widget = getattr(self, "notes", None)
        if widget is not None:
            widget.setText("  ".join(self._notes))
            widget.setVisible(True)

    def _alert(self, msg: str):
        QMessageBox.warning(self, "Atom typing", msg)

    # ============================================================ result
    def overrides(self) -> Dict[int, str]:
        """The user's manual assignments as ``{key: type_id}``.

        Repeat-unit atoms keep their plain monomer index, so anything that read
        this before the chain ends became assignable still reads what it
        expects. Head and tail atoms carry a role offset (see
        :func:`paaf.typing_context.role_key`), which
        :func:`paaf.ff_assigner.expand_manual_types` unfolds again when it
        places the types on the chain.
        """
        return dict(self._types_by_atom)

    def types_by_role(self) -> Dict[str, Dict[int, str]]:
        """The same assignments split into ``head`` / ``middle`` / ``tail``."""
        from ..typing_context import split_by_role
        return split_by_role(self._types_by_atom)
