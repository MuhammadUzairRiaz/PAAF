"""An interactive 3D molecule view for assigning atom types by clicking.

Why a web view rather than OpenGL
---------------------------------
Typing atoms from a table means reading a text description of an atom's
environment and trusting that it refers to the atom you think it does. That is
how a carbon type ended up on a hydrogen: one row of fifteen, indistinguishable
from its neighbours except by an index.

Seeing the molecule removes that class of error, but only if the picture is
good enough to be trusted — correct double bonds, real perspective, smooth
rotation. Writing that from scratch means spheres, cylinders, lighting, depth
sorting, trackball maths and colour-buffer picking; a first attempt looks
worse than the tools people already use, and every bug in it is a bug in what
the chemist sees.

So the rendering is delegated to 3Dmol.js, which chemists already use, and
this module is the bridge: it converts the molecule to a molfile (bond orders
included), hands it over, and turns clicks back into PAAF atom indices.

The bridge, not the renderer
----------------------------
Everything decision-making lives on the Python side and is unit-tested:
molfile generation, the serial-to-atom mapping, the element guard. The
JavaScript only draws and reports clicks.

Offline use
-----------
3Dmol.js is loaded from ``ff_libraries/viewer/3Dmol-min.js`` when present and
from the CDN otherwise. :func:`vendor_hint` returns the one command that makes
the viewer work without a network.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["Mol3DView", "Mol3DWindow", "webengine_available", "vendor_hint",
           "VIEWER_ASSET", "build_page", "build_payload", "LABEL_MODES",
           "LABEL_CONTENTS", "LABEL_COLOURS", "COLOUR_MODES",
           "DEFAULT_COLOUR_BY"]

#: Where a downloaded copy of the renderer is looked for.
VIEWER_ASSET = (Path(__file__).resolve().parent.parent.parent
                / "ff_libraries" / "viewer" / "3Dmol-min.js")

_CDN = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"


def webengine_error() -> Optional[str]:
    """Why the web engine is unusable, or ``None`` if it is fine.

    Two very different failures produce the same ImportError, and telling
    them apart is the difference between a two-second fix and an afternoon:

    * the package genuinely is not installed;
    * it is installed, but something already created a ``QCoreApplication``.
      Qt refuses the import in that case, and the message says so — but a
      caller that catches ``Exception`` and reports "not installed" hides it.

    :func:`paaf.gui.app.launch` imports the module before the QApplication
    exists precisely to avoid the second case; this reports what actually
    happened rather than assuming the first.
    """
    import os

    recorded = os.environ.get("PAAF_WEBENGINE_ERROR")
    try:
        from PyQt5 import QtWebEngineWidgets  # noqa: F401
        return None
    except Exception as exc:
        message = str(exc) or exc.__class__.__name__
        if recorded and recorded != message:
            message = f"{message} (at startup: {recorded})"
        return message


def webengine_available() -> bool:
    """True if Qt's web engine can be imported."""
    return webengine_error() is None


def vendor_hint() -> str:
    return (f"curl -L -o {VIEWER_ASSET} --create-dirs {_CDN}")


# =====================================================================
#: How many atoms may be labelled at once before the picture is a pile of
#: overlapping boxes rather than information. Well under a trimer of anything.
LABEL_MODES = ("clicked", "selected", "all", "none")

#: What a label says. Three different questions get asked of a structure —
#: "what type did I give this?", "which atom is this?", "where is it?" — and
#: one label cannot answer all three without becoming unreadable.
LABEL_CONTENTS = ("type", "name", "xyz")

#: Offered label colours. Plain black is first and darker than the #111 used
#: before: with the label background now transparent, a mid-grey vanishes
#: against a pale atom, which is the opposite of what a label is for.
LABEL_COLOURS = {"black": "#000000", "dark grey": "#3a3a3a",
                 "blue": "#0b3ea8", "red": "#a80b1e", "white": "#ffffff"}
DEFAULT_LABEL_COLOUR = "#000000"

#: How atoms are coloured. ``element`` is standard CPK — grey carbon, white
#: hydrogen, red oxygen — and is the default because that is what everyone
#: reads a molecule as without being told.
#:
#: ``type`` colours by assigned force-field type instead, which makes an
#: untyped atom obvious at a glance. It was the default, and it was a mistake:
#: the palette's first colour is red, so a terminal CH3 came out red and was
#: read as oxygen — reported three times on three different polymers, once as
#: "why is there also oxygen at both ends" on a polyethylene whose formula is
#: C6H14. The information was useful; taking over the element colours to show
#: it was not.
COLOUR_MODES = ("element", "type")
DEFAULT_COLOUR_BY = "element"


def build_payload(molblock: str, *, labels: Optional[Dict[int, str]] = None,
                  colors: Optional[Dict[int, str]] = None,
                  context: Optional[Sequence[int]] = None,
                  style: str = "stick+sphere",
                  label_mode: str = "clicked",
                  label_content: str = "type",
                  label_colour: str = DEFAULT_LABEL_COLOUR,
                  names: Optional[Dict[int, str]] = None,
                  neighbours: Optional[Dict[int, Sequence[int]]] = None,
                  hidden: Optional[Sequence[int]] = None,
                  colour_by: str = DEFAULT_COLOUR_BY,
                  pan_mode: bool = False) -> dict:
    """Everything the page needs, as a plain dict.

    Separated from :func:`build_page` because once the page is up, state
    changes are pushed into it as JSON rather than by rebuilding and reloading
    the whole document — reloading throws away the camera, so every label
    toggle would have snapped the molecule back to its starting angle.
    """
    if label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {LABEL_MODES}")
    if label_content not in LABEL_CONTENTS:
        raise ValueError(f"label_content must be one of {LABEL_CONTENTS}")
    if colour_by not in COLOUR_MODES:
        raise ValueError(f"colour_by must be one of {COLOUR_MODES}")
    return {
        "molblock": molblock,
        "labels": {str(k): v for k, v in (labels or {}).items()},
        "colors": {str(k): v for k, v in (colors or {}).items()},
        "context": sorted(int(s) for s in (context or [])),
        "style": style,
        "labelMode": label_mode,
        "labelContent": label_content,
        "labelColour": label_colour,
        "names": {str(k): v for k, v in (names or {}).items()},
        "neighbours": {str(k): [int(v) for v in vs]
                       for k, vs in (neighbours or {}).items()},
        "hidden": sorted(int(s) for s in (hidden or [])),
        "colourBy": colour_by,
        "panMode": bool(pan_mode),
    }


def build_page(molblock: str, *, labels: Optional[Dict[int, str]] = None,
               colors: Optional[Dict[int, str]] = None,
               context: Optional[Sequence[int]] = None,
               script_src: Optional[str] = None,
               style: str = "stick+sphere",
               label_mode: str = "clicked",
               label_content: str = "type",
               label_colour: str = DEFAULT_LABEL_COLOUR,
               names: Optional[Dict[int, str]] = None,
               neighbours: Optional[Dict[int, Sequence[int]]] = None,
               hidden: Optional[Sequence[int]] = None,
               colour_by: str = DEFAULT_COLOUR_BY,
               pan_mode: bool = False) -> str:
    """The HTML shown in the view.

    Kept as a pure function of its inputs so the generated page can be
    inspected and tested without a browser, a display or Qt.

    ``labels`` and ``colors`` are keyed by atom serial (0-based, matching the
    molblock's atom order). Colouring by assigned type is what makes an
    untyped atom visible at a glance instead of requiring a table scan.
    """
    src = script_src or (VIEWER_ASSET.as_uri() if VIEWER_ASSET.is_file() else _CDN)
    payload = json.dumps(build_payload(
        molblock, labels=labels, colors=colors, context=context, style=style,
        label_mode=label_mode, label_content=label_content,
        label_colour=label_colour, names=names,
        neighbours=neighbours, hidden=hidden, colour_by=colour_by,
        pan_mode=pan_mode))

    return """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html, body { margin:0; padding:0; height:100%; background:#ffffff; }
  #view { position:relative; width:100%; height:100%; }
  #hud { position:absolute; left:8px; bottom:8px; z-index:10;
         font:11px -apple-system,Helvetica,Arial,sans-serif; color:#5b6472;
         background:rgba(255,255,255,.85); padding:4px 8px; border-radius:4px; }
</style>
<script src="__SRC__"></script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
</head><body>
<div id="view"></div>
<div id="hud">drag rotate &middot; <b>shift</b>-drag pan &middot; scroll zoom &middot; click an atom</div>
<script>
var DATA = __PAYLOAD__;
var bridge = null, viewer = null, selected = null;

function elementColors() {
  // Standard CPK, which is what every chemist reads without a legend.
  return {C:0x303030, H:0xf0f0f0, O:0xd93025, N:0x1a73e8, S:0xf9ab00,
          F:0x34a853, Cl:0x34a853, Br:0x9c27b0, P:0xff7043, Si:0x9e9e9e};
}

function hiddenSet() {
  var h = {};
  for (var i = 0; i < (DATA.hidden || []).length; i++) h[DATA.hidden[i]] = true;
  return h;
}

function labelledSerials() {
  // Labelling every atom is what made this unreadable: 41 boxes overlapping
  // into a pile. "selected" labels the picked atom and whatever it is bonded
  // to, which is the neighbourhood you actually need to judge a type by.
  var mode = DATA.labelMode || 'clicked';
  if (mode === 'none') return [];
  var out = [], k;
  if (mode === 'all') {
    for (k in DATA.labels) out.push(parseInt(k));
    for (k in DATA.names) { k = parseInt(k); if (out.indexOf(k) < 0) out.push(k); }
    return out;
  }
  if (selected === null) return [];
  out.push(selected);
  // 'clicked' is the atom you picked and nothing else. 'selected' adds its
  // bonded neighbours, which is useful for judging an environment but is
  // three or four extra boxes when all you wanted was to read one type.
  if (mode === 'selected') {
    var nb = (DATA.neighbours || {})[String(selected)] || [];
    for (var i = 0; i < nb.length; i++) out.push(nb[i]);
  }
  return out;
}

function labelText(serial, atom) {
  var content = DATA.labelContent || 'type';
  if (content === 'name') return (DATA.names || {})[String(serial)] || '';
  if (content === 'xyz') {
    if (!atom) return '';
    return atom.x.toFixed(2) + ', ' + atom.y.toFixed(2) + ', '
         + atom.z.toFixed(2);
  }
  return (DATA.labels || {})[String(serial)] || '';
}

function render() {
  var hide = hiddenSet();
  viewer.setStyle({}, {});
  // Ball and stick. Double bonds come from the molfile's order column, so
  // they are DRAWN as double rather than inferred from distance.
  viewer.setStyle({}, {stick: {radius: 0.13, singleBonds: false,
                               colorscheme: {prop:'elem', map: elementColors()}},
                       sphere: {scale: 0.24,
                                colorscheme: {prop:'elem', map: elementColors()}}});
  // Per-atom overrides for atoms that carry an assigned type — only when the
  // user has asked to see types. Otherwise the CPK colours above stand, and a
  // carbon looks like a carbon.
  if ((DATA.colourBy || 'element') === 'type') {
    for (var s in DATA.colors) {
      if (hide[parseInt(s)]) continue;
      viewer.setStyle({serial: parseInt(s)},
        {stick: {radius: 0.13, singleBonds: false},
         sphere: {scale: 0.30, color: DATA.colors[s]}});
    }
  }
  // Units filtered out are removed from the picture entirely rather than
  // faded: the point of the filter is to get the other 28 atoms out of the
  // way, and a faint atom is still something to click by accident.
  if (DATA.hidden && DATA.hidden.length) {
    viewer.setStyle({serial: DATA.hidden}, {});
  }
  // Context atoms (the outer units of the trimer) are drawn faint: they are
  // the real chain ends and worth seeing, but they are NOT what is being
  // typed, and a user who cannot tell the two apart will type the wrong one.
  for (var ci = 0; ci < (DATA.context || []).length; ci++) {
    viewer.setStyle({serial: DATA.context[ci]},
      {stick: {radius: 0.07, singleBonds: false, opacity: 0.35, color: 0xb0b6c0},
       sphere: {scale: 0.13, opacity: 0.35, color: 0xb0b6c0}});
  }
  // The selected atom gets a translucent halo AROUND it — never a solid ball
  // on top of it.
  //
  // A solid core was tried, on the reasoning that "bigger and brighter is
  // easier to find". It was: it was also mistaken for an extra atom bonded
  // into the chain, because at scale 0.34 in a signal colour, sitting over an
  // atom drawn at 0.24, that is exactly what it looks like. The halo has to
  // read as a marker, which means you must be able to see the atom through it.
  // Small enough to read as "this one", not as an extra atom. A large sphere
  // was tried twice: opaque it looked like an atom bonded into the chain, and
  // even translucent at 0.52 it swamped its neighbours.
  if (selected !== null && !hide[selected]) {
    viewer.addStyle({serial: selected},
      {sphere: {scale: 0.36, color: 0x00e5ff, opacity: 0.45}});
  }
  viewer.removeAllLabels();
  var show = labelledSerials();
  for (var li = 0; li < show.length; li++) {
    var serial = show[li];
    if (hide[serial]) continue;
    var a = viewer.getModel().selectedAtoms({serial: serial})[0];
    if (!a) continue;
    var text = labelText(serial, a);
    if (!text) continue;
    var isSel = (serial === selected);
    // No box behind the text. A white panel behind every label is what turned
    // a labelled structure into a wall of rectangles; plain dark text sits on
    // the molecule and stays readable.
    viewer.addLabel(text, {position: {x:a.x, y:a.y, z:a.z},
      fontSize: isSel ? 15 : 13,
      fontColor: DATA.labelColour || '#000000',
      backgroundColor: '#ffffff', backgroundOpacity: 0.0,
      borderThickness: 0.0, inFront: true, fontOpacity: 1.0});
  }
  viewer.render();
}

// ---------------------------------------------------------------- panning
//
// 3Dmol puts translate on middle-drag, which is undiscoverable and awkward on
// a trackpad, so a user who wants to move along a chain has no obvious way to.
// These handlers run in the capture phase, ahead of 3Dmol's own, and turn
// left-drag into a pan whenever pan mode is on or shift is held.
var armed = false, dragging = false, panX = 0, panY = 0;

//: How far the mouse must move before a press counts as a drag rather than a
//: click. Without this, pan mode swallowed every click and no atom could be
//: selected while it was on.
var DRAG_SLOP = 4;

function wantsPan(ev) {
  return (DATA.panMode || ev.shiftKey) && ev.button === 0;
}

function installPan(el) {
  el.addEventListener('mousedown', function (ev) {
    if (!wantsPan(ev)) return;
    // Arm, but do NOT swallow the event: a press that never moves is a click,
    // and clicking is how atoms get selected. Only once it moves does it
    // become a pan.
    armed = true; dragging = false;
    panX = ev.clientX; panY = ev.clientY;
  }, true);
  el.addEventListener('mousemove', function (ev) {
    if (!armed) return;
    var dx = ev.clientX - panX, dy = ev.clientY - panY;
    if (!dragging &&
        Math.abs(dx) + Math.abs(dy) < DRAG_SLOP) return;
    dragging = true;
    panX = ev.clientX; panY = ev.clientY;
    viewer.translate(dx, dy, 0, true);
    // Swallowed so the renderer does not also rotate on the same gesture.
    ev.preventDefault(); ev.stopPropagation();
  }, true);
  var stop = function (ev) {
    if (!armed) return;
    var wasDragging = dragging;
    armed = false; dragging = false;
    // Swallow the release only if this really was a drag, so a pan does not
    // end by selecting whatever atom happens to be under the cursor.
    if (wasDragging) { ev.preventDefault(); ev.stopPropagation(); }
  };
  el.addEventListener('mouseup', stop, true);
  el.addEventListener('mouseleave', stop, true);
}

function setPanMode(on) {
  DATA.panMode = !!on;
  document.getElementById('view').style.cursor = on ? 'move' : 'default';
  hud();
}

function hud() {
  var el = document.getElementById('hud');
  if (!el) return;
  el.innerHTML = DATA.panMode
    ? '<b>Pan mode</b> &middot; drag to move &middot; scroll zoom &middot; click an atom'
    : 'drag rotate &middot; <b>shift</b>-drag pan &middot; scroll zoom &middot; click an atom';
}

function focusVisible() {
  // Fill the window with whatever is actually shown, so switching to one unit
  // does not leave it as a speck framed for a molecule three times the size.
  var shown = [];
  var hide = hiddenSet();
  var atoms = viewer.getModel().selectedAtoms({});
  for (var i = 0; i < atoms.length; i++) {
    if (!hide[atoms[i].serial]) shown.push(atoms[i].serial);
  }
  if (shown.length) viewer.zoomTo({serial: shown}); else viewer.zoomTo();
  viewer.render();
}

function bindClicks() {
  // MUST be called after every addModel.
  //
  // 3Dmol binds clickability to the atoms of the model that exists when
  // setClickable runs. apply() replaces the model, so a single call in boot()
  // leaves every atom in every later model unclickable — and because the old
  // code reloaded the whole page on every change, boot() re-ran each time and
  // hid that completely. The moment updates became cheap, clicking died.
  viewer.setClickable({}, true, function(atom) {
    // A filtered-out atom is not there as far as the user is concerned, so it
    // must not be selectable by a stray click on where it used to be.
    if (hiddenSet()[atom.serial]) return;
    selected = atom.serial;
    render();
    if (bridge) bridge.atom_clicked(atom.serial);
  });
}

function boot() {
  var el = document.getElementById('view');
  viewer = $3Dmol.createViewer(el, {backgroundColor: 'white', antialias: true});
  viewer.addModel(DATA.molblock, 'mol');
  bindClicks();
  installPan(el);
  setPanMode(DATA.panMode);
  focusVisible();
  render();
}

function apply(payload) {           // called from Python on every update
  var wasHidden = JSON.stringify(DATA.hidden || []);
  DATA = JSON.parse(payload);
  var keep = viewer ? viewer.getView() : null;
  var sameStructure = false;
  try { sameStructure = !!viewer.getModel(); } catch (e) { sameStructure = false; }
  viewer.removeAllModels();
  viewer.addModel(DATA.molblock, 'mol');
  bindClicks();                       // the new model has none of its own
  if (selected !== null && hiddenSet()[selected]) selected = null;
  render();
  // Keep the camera unless the set of visible atoms changed, in which case
  // re-frame: that IS the change the user asked to see.
  if (JSON.stringify(DATA.hidden || []) !== wasHidden) {
    focusVisible();
  } else if (keep && sameStructure) {
    viewer.setView(keep);
  }
  setPanMode(DATA.panMode);
}

function select(serial) { selected = serial; render(); }
function resetView() { selected = selected; focusVisible(); }

new QWebChannel(qt.webChannelTransport, function(channel) {
  bridge = channel.objects.bridge;
  boot();
});
</script></body></html>
""".replace("__SRC__", src).replace("__PAYLOAD__", payload)


# =====================================================================
#: Distinct, high-contrast colours cycled per assigned atom type.
_TYPE_PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                 "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
                 "#dcbeff", "#9a6324", "#800000", "#aaffc3", "#808000"]


def _neighbour_serials(molecules: Sequence) -> Dict[int, List[int]]:
    """``{serial: [bonded serials]}`` across a whole scene.

    Used to decide what "selected + neighbours" means when labelling. It is the
    bonded neighbourhood — the thing that actually determines an atom's type —
    rather than whatever happens to be nearby on screen.
    """
    out: Dict[int, List[int]] = {}
    offset = 0
    for mol in molecules:
        for bond in getattr(mol, "bonds", []):
            i, j = int(bond[0]) + offset, int(bond[1]) + offset
            out.setdefault(i, []).append(j)
            out.setdefault(j, []).append(i)
        offset += len(getattr(mol, "atoms", []))
    return {k: sorted(set(v)) for k, v in out.items()}


def colors_for_types(types_by_serial: Dict[int, str]) -> Dict[int, str]:
    """A stable colour per distinct assigned type.

    Stable matters: the same type keeps its colour as you work, so the picture
    does not reshuffle every time one atom is assigned.
    """
    order: List[str] = []
    for _serial, t in sorted(types_by_serial.items()):
        if t and t not in order:
            order.append(t)
    lookup = {t: _TYPE_PALETTE[i % len(_TYPE_PALETTE)]
              for i, t in enumerate(order)}
    return {s: lookup[t] for s, t in types_by_serial.items() if t}


try:                                    # pragma: no cover - GUI import
    from PyQt5.QtCore import QObject, Qt, pyqtSignal, pyqtSlot
    from PyQt5.QtWidgets import (
        QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
        QWidget,
    )
    from .page import wrap_tooltip

    class _Bridge(QObject):
        """Receives clicks from the page."""
        clicked = pyqtSignal(int)

        @pyqtSlot(int)
        def atom_clicked(self, serial: int) -> None:
            self.clicked.emit(int(serial))

    class Mol3DView(QWidget):
        """Ball-and-stick view; emits ``atom_clicked`` with a PAAF atom index.

        The widget owns the serial-to-index mapping, so callers never see the
        viewer's own numbering. For a copolymer scene the signal carries the
        global index the typing table uses.
        """
        atom_clicked = pyqtSignal(int)
        #: Emitted when the user asks for the view in its own window.
        popout_requested = pyqtSignal()

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            self._serial_to_index: Dict[int, int] = {}
            self._index_to_serial: Dict[int, int] = {}
            self._display_names: Dict[int, str] = {}
            self._neighbours: Dict[int, List[int]] = {}
            self._groups: Dict[str, List[int]] = {}
            self._hidden: List[int] = []
            self._label_mode = "clicked"
            self._label_content = "type"
            self._label_colour = DEFAULT_LABEL_COLOUR
            self._names: Dict[int, str] = {}
            self._pan_mode = False
            self._loaded = False
            self._molblock = ""
            self._view = None
            self._toolbar = None

            reason = webengine_error()
            if reason is not None:
                looks_like_order = "QCoreApplication" in reason or "before" in reason
                if looks_like_order:
                    detail = (
                        "Qt requires QtWebEngineWidgets to be imported before "
                        "the application starts. PAAF now does that in "
                        "paaf/gui/app.py — if you are seeing this, PAAF was "
                        "most likely started some other way.\n\n"
                        "    python run_paaf.py")
                else:
                    detail = "    pip install PyQtWebEngine"
                msg = QLabel(f"3D view unavailable.\n\n{reason}\n\n{detail}\n\n"
                             f"The typing table on the left works without it.")
                msg.setWordWrap(True)
                msg.setTextInteractionFlags(Qt.TextSelectableByMouse)
                msg.setStyleSheet("color:#8a6d3b; padding:16px;")
                layout.addWidget(msg)
                return

            from PyQt5.QtWebChannel import QWebChannel
            from PyQt5.QtWebEngineWidgets import QWebEngineView

            self._view = QWebEngineView(self)
            self._bridge = _Bridge()
            self._bridge.clicked.connect(self._on_serial_clicked)
            self._channel = QWebChannel(self._view.page())
            self._channel.registerObject("bridge", self._bridge)
            self._view.page().setWebChannel(self._channel)
            self._view.loadFinished.connect(self._on_load_finished)
            layout.addWidget(self._build_toolbar())
            layout.addWidget(self._view, 1)

        # ---------------------------------------------------------- toolbar
        def _build_toolbar(self) -> QWidget:
            """The controls belong to the viewer, not to any one dialog.

            That way the pop-out window and the inline pane are the same thing
            with the same controls, rather than two views that drift apart.
            """
            bar = QWidget(self)
            row = QHBoxLayout(bar)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)

            self._unit_label = QLabel("Show")
            row.addWidget(self._unit_label)
            self.unit_combo = QComboBox()
            self.unit_combo.setToolTip(wrap_tooltip("Show one unit of the chain at a time, so the other atoms are "
                "out of the way while you type this one."))
            self.unit_combo.currentIndexChanged.connect(self._on_unit_changed)
            row.addWidget(self.unit_combo)

            row.addWidget(QLabel("Labels"))
            self.label_combo = QComboBox()
            for text, mode in (("Clicked atom only", "clicked"),
                               ("Clicked + neighbours", "selected"),
                               ("All", "all"), ("None", "none")):
                self.label_combo.addItem(text, mode)
            self.label_combo.setToolTip(wrap_tooltip("Labelling every atom at once overlaps into an unreadable "
                "pile. The default labels only the atom you clicked."))
            self.label_combo.currentIndexChanged.connect(self._on_label_changed)
            row.addWidget(self.label_combo)

            # What the label says. One label cannot answer "what type is it",
            # "which atom is it" and "where is it" at once without becoming a
            # paragraph, so it answers whichever you asked for.
            self.content_combo = QComboBox()
            for text, content in (("Atom type", "type"),
                                  ("Atom name", "name"),
                                  ("XYZ", "xyz")):
                self.content_combo.addItem(text, content)
            self.content_combo.setToolTip(wrap_tooltip("Show the assigned force-field type, the atom's name and "
                "number, or its coordinates."))
            self.content_combo.currentIndexChanged.connect(
                self._on_content_changed)
            row.addWidget(self.content_combo)

            self.colour_combo = QComboBox()
            for name, value in LABEL_COLOURS.items():
                self.colour_combo.addItem(name.capitalize(), value)
            self.colour_combo.setToolTip(wrap_tooltip("Label text colour. The label has no background panel, so on "
                "a dark atom you may want white."))
            self.colour_combo.currentIndexChanged.connect(
                self._on_colour_changed)
            row.addWidget(self.colour_combo)

            self.pan_button = QPushButton("Pan")
            self.pan_button.setCheckable(True)
            self.pan_button.setToolTip(wrap_tooltip("Drag to move the molecule around instead of rotating it. "
                "Holding shift does the same without switching mode."))
            self.pan_button.toggled.connect(self.set_pan_mode)
            row.addWidget(self.pan_button)

            b_fit = QPushButton("Fit")
            b_fit.setToolTip(wrap_tooltip("Re-frame whatever is currently shown."))
            b_fit.clicked.connect(self.fit_view)
            row.addWidget(b_fit)

            row.addStretch(1)
            self.popout_button = QPushButton("Pop out ⧉")
            self.popout_button.setToolTip(wrap_tooltip("Open the 3D view in its own resizable window."))
            self.popout_button.clicked.connect(self.popout_requested.emit)
            row.addWidget(self.popout_button)

            self._toolbar = bar
            return bar

        # ---------------------------------------------------------- content
        def show_molecules(self, molecules: Sequence,
                           index_map: Optional[Sequence[int]] = None) -> None:
            """Display one or more molecules as a single scene.

            ``index_map`` gives the PAAF atom index for each atom, in scene
            order. Without it the scene order is taken to be the index.
            """
            from ..molblock import combined_molblock

            molblock, mapping = combined_molblock(list(molecules))
            self._molblock = molblock
            if index_map is not None:
                self._serial_to_index = {s: int(i)
                                         for s, i in enumerate(index_map)}
            else:
                self._serial_to_index = {s: m[1] for s, m in enumerate(mapping)}
            self._index_to_serial = {v: k
                                     for k, v in self._serial_to_index.items()}
            self._neighbours = _neighbour_serials(list(molecules))
            self._reload()

        def set_types(self, types_by_index: Dict[int, str]) -> None:
            """Colour and label atoms by their assigned type.

            Context atoms — index below zero — get neither, so the picture
            says which unit is being typed without anyone reading a caption.
            """
            by_serial = {s: types_by_index.get(i, "")
                         for s, i in self._serial_to_index.items() if i >= 0}
            self._colors = colors_for_types(by_serial)
            self._labels = {
                s: (f"{self._name_for(i)}:{t}" if t else self._name_for(i))
                for s, i in self._serial_to_index.items() if i >= 0
                for t in [types_by_index.get(i, "")]}
            self._reload()

        def _name_for(self, index: int) -> str:
            """What to print beside an atom.

            The caller may address atoms by a key of its own — the typing
            dialog distinguishes the same monomer atom in three different chain
            units that way — and that key is an implementation detail. Where it
            supplied a display name, use it; otherwise the index is the name.
            """
            return self._display_names.get(int(index), str(index))

        def set_display_names(self, names: Dict[int, str]) -> None:
            """Override the text shown for given atom indices."""
            self._display_names = {int(k): str(v) for k, v in (names or {}).items()}
            self._reload()

        # ------------------------------------------------------- view state
        def set_groups(self, groups: Dict[str, Sequence[int]],
                       order: Optional[Sequence[str]] = None) -> None:
            """Named subsets of atoms the user can isolate, by atom index.

            The viewer stays ignorant of what a "unit" is — it is handed named
            groups and shows one at a time. A caller with something other than
            head/repeat/tail to isolate gets the same behaviour for free.
            """
            self._groups = {str(k): [int(v) for v in vs]
                            for k, vs in (groups or {}).items()}
            names = list(order) if order else list(self._groups)
            combo = getattr(self, "unit_combo", None)
            if combo is None:
                return
            blocked = combo.blockSignals(True)
            combo.clear()
            combo.addItem("All", None)
            for name in names:
                if name in self._groups:
                    combo.addItem(name, name)
            combo.blockSignals(blocked)
            # Nothing to isolate means no control: an empty dropdown is a
            # promise the viewer cannot keep.
            combo.setVisible(bool(self._groups))
            label = getattr(self, "_unit_label", None)
            if label is not None:
                label.setVisible(bool(self._groups))
            self.show_group(None)

        def show_group(self, name: Optional[str]) -> None:
            """Show only the named group; ``None`` shows everything."""
            if not name or name not in self._groups:
                self._hidden = []
            else:
                keep = set(self._groups[name])
                self._hidden = sorted(
                    s for s, i in self._serial_to_index.items()
                    if i not in keep)
            self._reload()

        def set_label_mode(self, mode: str) -> None:
            if mode not in LABEL_MODES:
                raise ValueError(f"label_mode must be one of {LABEL_MODES}")
            self._label_mode = mode
            self._reload()

        def set_pan_mode(self, on: bool) -> None:
            self._pan_mode = bool(on)
            button = getattr(self, "pan_button", None)
            if button is not None and button.isChecked() != self._pan_mode:
                button.setChecked(self._pan_mode)
            if self._loaded and self._view is not None:
                self._view.page().runJavaScript(
                    f"setPanMode({'true' if self._pan_mode else 'false'});")
            else:
                self._reload()

        def fit_view(self) -> None:
            if self._loaded and self._view is not None:
                self._view.page().runJavaScript("focusVisible();")

        def _on_unit_changed(self, _index: int) -> None:
            self.show_group(self.unit_combo.currentData())

        def _on_label_changed(self, _index: int) -> None:
            self.set_label_mode(self.label_combo.currentData() or "clicked")

        def set_label_content(self, content: str) -> None:
            if content not in LABEL_CONTENTS:
                raise ValueError(
                    f"label_content must be one of {LABEL_CONTENTS}")
            self._label_content = content
            self._reload()

        def _on_content_changed(self, _index: int) -> None:
            self.set_label_content(self.content_combo.currentData() or "type")

        def set_label_colour(self, colour: str) -> None:
            self._label_colour = str(colour)
            self._reload()

        def _on_colour_changed(self, _index: int) -> None:
            self.set_label_colour(
                self.colour_combo.currentData() or DEFAULT_LABEL_COLOUR)

        def set_names(self, names: Dict[int, str]) -> None:
            """Per-atom names for the "Atom name" label content.

            Keyed by the caller's atom index, like everything else it hands
            over; converted to the renderer's serials on the way out.
            """
            self._names = {int(k): str(v) for k, v in (names or {}).items()}
            self._reload()

        def _names_by_serial(self) -> Dict[int, str]:
            """``_names`` translated from caller indices to scene serials.

            Falls back to the display name, so "Atom name" says something
            useful even for a caller that never supplied names.
            """
            out: Dict[int, str] = {}
            for serial, index in self._serial_to_index.items():
                if index < 0:
                    continue
                value = self._names.get(index) or self._display_names.get(index)
                if value:
                    out[serial] = value
            return out

        def _on_load_finished(self, ok: bool) -> None:
            self._loaded = bool(ok)

        def set_context(self, serials: Sequence[int]) -> None:
            """Draw these atoms faint: shown for context, not for typing."""
            self._context = sorted(int(s) for s in serials)
            self._reload()

        def select_index(self, index: int) -> None:
            """Highlight one atom, addressed by PAAF index."""
            serial = self._index_to_serial.get(int(index))
            if serial is None or self._view is None:
                return
            self._view.page().runJavaScript(f"select({serial});")

        # ------------------------------------------------------------ inner
        def _payload(self) -> dict:
            return build_payload(
                self._molblock,
                labels=getattr(self, "_labels", None),
                colors=getattr(self, "_colors", None),
                context=getattr(self, "_context", None),
                label_mode=self._label_mode,
                label_content=self._label_content,
                label_colour=self._label_colour,
                names=self._names_by_serial(),
                neighbours=self._neighbours,
                hidden=self._hidden,
                pan_mode=self._pan_mode)

        def _reload(self) -> None:
            """Push the current state to the page.

            Once the page is up this is a JSON hand-off, not a reload. Calling
            ``setHtml`` again would drop the camera, so every label toggle or
            unit switch would snap the molecule back to its opening angle and
            you would lose your place — the opposite of what the controls are
            for.
            """
            if self._view is None or not self._molblock:
                return
            if self._loaded:
                self._view.page().runJavaScript(
                    "apply(" + json.dumps(json.dumps(self._payload())) + ");")
                return
            page = build_page(self._molblock,
                              labels=getattr(self, "_labels", None),
                              colors=getattr(self, "_colors", None),
                              context=getattr(self, "_context", None),
                              label_mode=self._label_mode,
                              label_content=self._label_content,
                              label_colour=self._label_colour,
                              names=self._names_by_serial(),
                              neighbours=self._neighbours,
                              hidden=self._hidden,
                              pan_mode=self._pan_mode)
            self._loaded = False
            self._view.setHtml(page, baseUrl=_base_url())

        def _on_serial_clicked(self, serial: int) -> None:
            index = self._serial_to_index.get(int(serial))
            if index is not None and index >= 0:
                self.atom_clicked.emit(int(index))

    class Mol3DWindow(QDialog):
        """The 3D view in a window of its own.

        The widget is MOVED here, not duplicated. A second viewer would mean a
        second WebEngine process, a second copy of the structure and two states
        to keep in step — and the first time they disagreed, the user would be
        typing against a picture that was no longer true. On close it goes back
        where it came from.
        """

        def __init__(self, view: "Mol3DView", parent: Optional[QWidget] = None,
                     title: str = "Structure"):
            super().__init__(parent)
            self.setWindowTitle(title)
            self.setModal(False)
            self.setSizeGripEnabled(True)
            self.resize(1000, 760)
            self._view = view
            self._home = view.parentWidget()
            self._home_layout = (self._home.layout()
                                 if self._home is not None else None)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(8, 8, 8, 8)
            view.setParent(self)
            layout.addWidget(view)
            view.show()
            button = getattr(view, "popout_button", None)
            if button is not None:
                button.setEnabled(False)

        def closeEvent(self, event):            # noqa: N802 - Qt naming
            """Hand the view back to the dialog it came from."""
            button = getattr(self._view, "popout_button", None)
            if button is not None:
                button.setEnabled(True)
            if self._home_layout is not None:
                self._view.setParent(self._home)
                self._home_layout.addWidget(self._view)
                self._view.show()
            super().closeEvent(event)

    def _base_url():
        from PyQt5.QtCore import QUrl
        return QUrl.fromLocalFile(str(VIEWER_ASSET.parent) + "/")

except Exception:                       # pragma: no cover - no PyQt5 present
    Mol3DView = None                    # type: ignore[assignment]
    Mol3DWindow = None                  # type: ignore[assignment]
