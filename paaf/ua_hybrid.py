"""United-atom / all-atom hybrid typing: absorb hydrogens into UA beads.

What "united atom" means for the structure
------------------------------------------
A united-atom type such as TraPPE ``CH2`` or OPLS-UA ``71`` (CH2) is one
particle standing for a carbon *and* its hydrogens. PAAF builds every chain
all-atom, so when an atom is given a UA type its hydrogens must be removed
before export, and the bead must carry the combined mass. How many hydrogens
a bead absorbs is read from the library itself — the type name or
description (``CH3``, ``CH2 (SP3) ALKANES``, ``CH (SP3) ISOBUTANE``,
``C (SP3) NEOPENTANE``) — so nothing here is hard-wired per force field.

Three situations are handled with the same code:

1. **UA-only** (TraPPE-UA or OPLS-UA chosen as the force field): every atom
   typed with a bead type loses its hydrogens; hydrogens the library cannot
   model are reported before anything is written.
2. **Hybrid, same family** (OPLS-AA 2024 + OPLS-UA 2024): the UA block
   (types 66–134) lives *inside* ``oplsaa2024.lt`` with its own bonded
   classes (C2, C3, CH …) and Jorgensen's mixed terms (``CT_C2``,
   ``HC_CT_C2`` …), so a chain may carry both — no bridging needed.
3. **Hybrid, different family** (e.g. OPLS-AA + TraPPE-UA): a small bridge
   library ``paaf_ua_bridge.lt`` is generated. It inherits the all-atom
   force field, declares the beads as new atom types with the UA library's
   own Lennard-Jones parameters and masses, zero charge, and maps their
   bonded behaviour onto the all-atom sp3-carbon classes (``replace{}``
   equivalence), so every bond/angle/dihedral across the UA/AA boundary
   resolves. Cross non-bonded terms follow the all-atom force field's mixing
   rule, exactly as DL_FIELD does between force fields.

Types coming from the secondary (UA) library are tagged ``UA:<id>`` in the
manual-type map so they can be told apart from the primary library's ids.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .logging_utils import get_logger
from .structure import Atom, Molecule

log = get_logger(__name__)

UA_PREFIX = "UA:"
BRIDGE_NAME = "PAAF_UA"          # moltemplate object name of the bridge
BRIDGE_FILE = "paaf_ua_bridge.lt"
_H_MASS = 1.008
_C_MASS = 12.011

_OPLS_AA_KEYS = {"oplsaa", "oplsaa2008", "loplsaa", "loplsaa2008"}


# ------------------------------------------------------------ bead spec
def bead_h_count(type_key: str, description: str = "", element: str = "") -> Optional[int]:
    """Hydrogens absorbed by a united-atom type, read from its name/description.

    ``CH3`` -> 3, ``CH2 (SP3) ALKANES`` -> 2, ``CH (AROM)`` -> 1,
    ``C (SP3) NEOPENTANE`` -> 0, ``CH4`` -> 4. ``None`` when the text does not
    describe a carbon bead (e.g. ``O ALCOHOLS``): such a type absorbs nothing.
    """
    for text in (description or "", type_key or ""):
        text = text.strip()
        if not text:
            continue
        m = re.match(r"^(?:@atom:)?CH(\d)?(?![A-Za-z0-9])", text)
        if m:
            return int(m.group(1)) if m.group(1) else 1
        m = re.match(r"^(?:@atom:)?C(?![A-Za-z0-9])", text)
        if m and element in ("", "C"):
            # bare "C (SP3) NEOPENTANE" — a carbon bead with no hydrogens
            return 0
    return None


def bead_table(lt_path: Path) -> Dict[str, Tuple[int, float, str]]:
    """``{type_id: (n_H, mass, description)}`` for every carbon bead in ``lt_path``."""
    from .lt_parser import parse_atom_types
    out: Dict[str, Tuple[int, float, str]] = {}
    for t in parse_atom_types(str(lt_path)):
        if t.element and t.element != "C":
            continue
        n = bead_h_count(t.key or t.ff_id, t.description or "", t.element or "")
        if n is None:
            continue
        mass = float(t.mass) if t.mass and t.mass > _C_MASS + 0.5 else _C_MASS + n * _H_MASS
        out[t.ff_id] = (n, mass, t.description or t.key or t.ff_id)
    return out


# ------------------------------------------------------------ absorption
@dataclass
class AbsorptionPlan:
    remove: List[int] = field(default_factory=list)          # H indices to delete
    beads: Dict[int, Tuple[str, int, float]] = field(default_factory=dict)  # heavy -> (type, nH, mass)
    problems: List[str] = field(default_factory=list)


def plan_absorption(mol: Molecule, types: Dict[int, str],
                    beads: Dict[str, Tuple[int, float, str]],
                    strip_prefix: bool = True) -> AbsorptionPlan:
    """Decide which hydrogens disappear into which beads.

    ``types`` maps atom index -> type id (possibly ``UA:``-prefixed);
    ``beads`` is a :func:`bead_table`. A bead type on a carbon that has
    *fewer* hydrogens than the bead absorbs is an error (the user asked for
    CH3 on a CH2), reported in ``problems``. Extra hydrogens are kept — a CH
    bead on a CH2 carbon leaves one explicit hydrogen, which the caller must
    then type with the all-atom library.
    """
    plan = AbsorptionPlan()
    el = mol.elements()
    for i, t in types.items():
        tid = t[len(UA_PREFIX):] if (strip_prefix and t.startswith(UA_PREFIX)) else t
        if tid not in beads or el[i] == "H":
            continue
        n_h, mass, _ = beads[tid]
        hs = [j for j in mol.neighbors(i) if el[j] == "H"]
        if len(hs) < n_h:
            plan.problems.append(
                f"atom {i + 1} ({el[i]}) has {len(hs)} hydrogen(s) but type "
                f"{tid} is a CH{n_h if n_h != 1 else ''} bead that absorbs {n_h}")
            continue
        plan.remove.extend(sorted(hs)[:n_h])
        plan.beads[i] = (tid, n_h, mass)
    return plan


def absorb_hydrogens(mol: Molecule, plan: AbsorptionPlan) -> Tuple[Molecule, Dict[int, int]]:
    """A copy of ``mol`` without the planned hydrogens; returns ``(mol, old->new)``."""
    drop = set(plan.remove)
    keep = [a for a in mol.atoms if a.index not in drop]
    remap = {a.index: k for k, a in enumerate(keep)}
    new = Molecule(name=mol.name, source_path=mol.source_path)
    for k, a in enumerate(keep):
        na = Atom(index=k, element=a.element, xyz=np.array(a.xyz, dtype=float),
                  name=a.name or f"{a.element}{k + 1}", charge=a.charge,
                  ff_type=a.ff_type)
        new.atoms.append(na)
    new.bonds = [(remap[i], remap[j], o) for i, j, o in mol.bonds
                 if i in remap and j in remap]
    # unique names after renumbering
    seen = set()
    for a in new.atoms:
        if a.name in seen or not a.name:
            a.name = f"{a.element}{a.index + 1}"
        seen.add(a.name)
    return new, remap


# ------------------------------------------------------------ bridge library
def _pair_substyle(lt_path: Path) -> str:
    """The pair sub-style name a library's pair_coeff lines use (e.g. lj/charmm/coul/long)."""
    text = Path(lt_path).read_text(errors="replace")
    m = re.search(r"^\s*pair_coeff\s+@atom:\S+\s+@atom:\S+\s+([a-z][\w/]*)\s", text, re.M)
    if m:
        return m.group(1)
    # Plain (non-hybrid) pair_style: coefficient lines carry no style token.
    return ""


def _equivalence_suffix(lt_path: Path, type_id: str) -> str:
    """``_bCT_aCT_dCT_iCT`` from ``replace{ @atom:136 @atom:136_bCT_aCT_dCT_iCT }``."""
    text = Path(lt_path).read_text(errors="replace")
    m = re.search(rf"replace\{{\s*@atom:{re.escape(type_id)}\s+@atom:{re.escape(type_id)}(\S+)\s*\}}", text)
    return m.group(1) if m else ""


def _lj_params(lt_path: Path, type_id: str) -> Optional[Tuple[float, float]]:
    text = Path(lt_path).read_text(errors="replace")
    m = re.search(rf"^\s*pair_coeff\s+@atom:{re.escape(type_id)}\s+@atom:{re.escape(type_id)}"
                  rf"\s+(?:[a-z][\w/]*\s+)?([-\d.eE+]+)\s+([-\d.eE+]+)", text, re.M)
    return (float(m.group(1)), float(m.group(2))) if m else None


def bridge_type_name(tid: str) -> str:
    return "UA_" + re.sub(r"[^A-Za-z0-9]", "_", tid)


def write_bridge_lt(out_dir: Path, primary_ff, ua_ff, beads_used: Dict[str, Tuple[int, float, str]],
                    sp3_type: str) -> Path:
    """Generate ``paaf_ua_bridge.lt`` declaring the UA beads inside the AA force field.

    ``sp3_type`` is the primary library's all-atom sp3 carbon type (OPLS
    ``136``) whose bonded equivalence classes the beads borrow.
    """
    ua_lt = ua_ff.bundled_path()
    aa_lt = primary_ff.bundled_path()
    sub = _pair_substyle(aa_lt)
    suffix = _equivalence_suffix(aa_lt, sp3_type)
    if not suffix:
        raise RuntimeError(
            f"{primary_ff.display_name} has no bonded-equivalence table "
            f"(replace{{}} lines) for type {sp3_type}, so united-atom beads "
            f"cannot borrow its bonded parameters. Use OPLS-AA 2024 as the "
            f"all-atom force field for UA/AA mixing, or OPLS-UA 2024 as the "
            f"united-atom one (its beads are already part of OPLS-AA 2024).")
    lines = [
        f"# Generated by PAAF: {ua_ff.display_name} beads inside {primary_ff.display_name}.",
        "# Non-bonded (LJ, mass) from the united-atom library, charge 0;",
        f"# bonded terms borrowed from the all-atom sp3 carbon class of type {sp3_type}",
        "# so bonds/angles/dihedrals across the UA/AA boundary resolve.",
        f'import "{primary_ff.lt_include}"',
        "",
        # Re-open the all-atom force field's own namespace: the library's
        # "Bonds/Angles/Dihedrals By Type" wildcards only match types that
        # live inside it, so the beads must be declared there, not in a
        # child object.
        f"{primary_ff.inherit} {{",
        '  write_once("Data Masses") {',
    ]
    for tid, (n_h, mass, desc) in sorted(beads_used.items()):
        lines.append(f"    @atom:{bridge_type_name(tid)} {mass:.4f}   # {ua_ff.key} {tid}: {desc}")
    lines += ["  }", '  write_once("In Charges") {']
    for tid in sorted(beads_used):
        lines.append(f"    set type @atom:{bridge_type_name(tid)} charge 0.0")
    lines += ["  }", '  write_once("In Settings") {']
    for tid in sorted(beads_used):
        lj = _lj_params(ua_lt, tid)
        if lj is None:
            raise RuntimeError(f"no pair_coeff for {ua_ff.key} type {tid} in {ua_lt.name}")
        eps, sig = lj
        name = bridge_type_name(tid)
        lines.append(f"    pair_coeff @atom:{name} @atom:{name} {sub} {eps:.6f} {sig:.4f}".rstrip())
    lines += ["  }"]
    for tid in sorted(beads_used):
        name = bridge_type_name(tid)
        lines.append(f"  replace{{ @atom:{name} @atom:{name}{suffix} }}")
    lines += [f"}}  # {primary_ff.inherit} (+ PAAF united-atom beads)", ""]
    p = Path(out_dir) / BRIDGE_FILE
    p.write_text("\n".join(lines))
    log.info("Wrote UA/AA bridge library %s (%d bead types)", p, len(beads_used))
    return p


# ------------------------------------------------------------ orchestration
@dataclass
class HybridResult:
    chain: Molecule
    inherit: Optional[str] = None       # object the chain should inherit from
    lt_include: Optional[str] = None    # file the chain .lt should import
    bond_type: Optional[str] = None     # explicit bond type (TraPPE has no Bonds By Type)
    bead_masses: Dict[str, float] = field(default_factory=dict)   # final type id -> mass
    removed_h: int = 0
    n_beads: int = 0
    remap: Dict[int, int] = field(default_factory=dict)


def apply_hybrid(chain: Molecule, primary_ff, ua_ff, out_dir: Path,
                 say=None) -> HybridResult:
    """Absorb hydrogens for every UA-typed atom and prepare the export.

    ``ua_ff`` may be the primary force field itself (UA-only run) or a
    secondary united-atom library (hybrid). Atom types on ``chain`` are
    already assigned; UA types from a *secondary* library carry ``UA:``.
    """
    say = say or (lambda s: None)
    from .ff_registry import get_ff
    from .typers.generic import FF_MAPS
    ua_only = ua_ff.key == primary_ff.key
    beads = bead_table(ua_ff.bundled_path())
    types = {a.index: a.ff_type for a in chain.atoms if a.ff_type}
    same_family = (ua_ff.key == "oplsua_2024" and primary_ff.key in _OPLS_AA_KEYS)
    if ua_only:
        plan = plan_absorption(chain, types, beads, strip_prefix=False)
    elif same_family:
        # The UA block (66-134) is part of OPLS-AA itself, so a bead may have
        # been picked from the all-atom table without the UA: tag. Either
        # spelling is the same united-atom type and absorbs its hydrogens.
        plan = plan_absorption(chain, {i: t for i, t in types.items()
                                       if t.startswith(UA_PREFIX)
                                       or t in beads}, beads)
    else:
        plan = plan_absorption(chain, {i: t for i, t in types.items()
                                       if t.startswith(UA_PREFIX)}, beads)
    if plan.problems:
        raise RuntimeError("United-atom typing problems:\n  " + "\n  ".join(plan.problems))
    if not plan.beads:
        say("No united-atom bead types assigned — nothing to absorb.")
        return HybridResult(chain=chain)

    new, remap = absorb_hydrogens(chain, plan)
    # H left over on a bead (e.g. CH bead on a CH2 carbon) must be AA-typed
    # by the primary library; hydrogens elsewhere already are.
    res = HybridResult(chain=new, removed_h=len(plan.remove), n_beads=len(plan.beads),
                       remap=remap)

    if ua_only:
        # Types are native to the chosen library.
        for a in new.atoms:
            if a.ff_type and a.ff_type in beads:
                res.bead_masses[a.ff_type] = beads[a.ff_type][1]
        if ua_ff.key == "trappe_ua":
            res.bond_type = "saturated"        # trappe1998.lt has no Bonds By Type
        untyped_h = [a for a in new.atoms if a.element == "H"]
        if untyped_h:
            raise RuntimeError(
                f"{ua_ff.display_name} models no explicit hydrogens, but "
                f"{len(untyped_h)} hydrogen(s) remain (e.g. atom "
                f"{untyped_h[0].index + 1}): their carbon was not given a bead "
                f"type. Type every carbon as a CHn bead, or mix with an "
                f"all-atom force field (Advanced typing).")
    elif same_family:
        for a in new.atoms:
            if a.ff_type and a.ff_type.startswith(UA_PREFIX):
                a.ff_type = a.ff_type[len(UA_PREFIX):]
            if a.ff_type in beads and a.element == "C":
                res.bead_masses[a.ff_type] = beads[a.ff_type][1]
        say(f"OPLS-UA beads used inside OPLS-AA 2024 (same library; mixed "
            f"bonded terms such as CT-C2 are Jorgensen's own).")
    else:
        used = {}
        for a in new.atoms:
            if a.ff_type and a.ff_type.startswith(UA_PREFIX):
                tid = a.ff_type[len(UA_PREFIX):]
                used[tid] = beads[tid]
                a.ff_type = bridge_type_name(tid)
                res.bead_masses[a.ff_type] = beads[tid][1]
        sp3 = FF_MAPS.get(primary_ff.key, ({}, {}))[0].get("alkane_CH2", "")
        write_bridge_lt(Path(out_dir), primary_ff, ua_ff, used, sp3)
        res.inherit = primary_ff.inherit
        res.lt_include = BRIDGE_FILE
        say(f"Bridge library {BRIDGE_FILE}: {len(used)} {ua_ff.display_name} bead "
            f"type(s) declared inside {primary_ff.display_name}; bonded terms "
            f"borrowed from all-atom type {sp3}.")
    say(f"United-atom absorption: {res.n_beads} bead(s), {res.removed_h} "
        f"hydrogen(s) removed; chain now {len(new.atoms)} atoms.")
    return res


def implicit_ua_library(primary_ff, chain: Molecule) -> Optional[str]:
    """The UA library whose beads are already on ``chain`` without a UA: tag.

    OPLS-AA 2024 contains the OPLS-UA block, so a user can pick type 71
    (CH2 bead) from the all-atom table. That is a united-atom choice all the
    same: return ``"oplsua_2024"`` so the hydrogens get absorbed.
    """
    if primary_ff.key not in _OPLS_AA_KEYS:
        return None
    from .ff_registry import get_ff
    try:
        beads = bead_table(get_ff("oplsua_2024").bundled_path())
    except Exception:
        return None
    for a in chain.atoms:
        t = a.ff_type or ""
        if a.element == "C" and (t in beads or
                                 (t.startswith(UA_PREFIX) and t[len(UA_PREFIX):] in beads)):
            return "oplsua_2024"
    return None


def patch_data_masses(data_file: Path, bead_masses: Dict[str, float]) -> int:
    """Set bead masses in a moltemplate ``system.data`` (Masses lines carry
    ``# <type>_<equivalence>`` comments, which is how types are recognised).
    Returns the number of lines changed."""
    if not bead_masses:
        return 0
    p = Path(data_file)
    lines = p.read_text(errors="replace").splitlines()
    changed = 0
    in_masses = False
    for k, line in enumerate(lines):
        s = line.strip()
        if s == "Masses" or s.startswith("Masses "):
            in_masses = True; continue
        if in_masses and s and s.split()[0] in ("Atoms", "Bonds", "Pair", "Bond", "Angle",
                                                  "Dihedral", "Improper", "Velocities"):
            break
        if not in_masses or "#" not in line:
            continue
        comment = line.split("#", 1)[1].strip()
        tname = re.split(r"[_~\s]", comment, 1)[0]
        if tname in bead_masses:
            parts = line.split("#", 1)[0].split()
            if len(parts) >= 2:
                lines[k] = f"{parts[0]} {bead_masses[tname]:.4f}  # {comment}"
                changed += 1
    if changed:
        p.write_text("\n".join(lines) + "\n")
        log.info("Patched %d bead mass line(s) in %s", changed, p)
    return changed
