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
2. **Hybrid** (any all-atom OPLS library + OPLS-UA or TraPPE-UA beads,
   whether the bead was picked from the UA list or, for the OPLS-UA block
   that lives inside OPLS-AA 2024, straight from the all-atom table): a
   small bridge library ``paaf_ua_bridge.lt`` is generated. It re-opens the
   all-atom force field's namespace and declares each bead as a new atom
   type with the UA library's own Lennard-Jones parameters, charge and
   mass, and maps its bonded behaviour onto the all-atom carbon class of
   the same hybridisation (sp3 -> CT, sp2 -> CM, aromatic -> CA) through
   ``replace{}`` equivalence. Every bond/angle/dihedral across the UA/AA
   boundary — to O, N, S, Si, halogens, anything the all-atom library
   knows — therefore resolves. Cross non-bonded terms follow the all-atom
   force field's mixing rule, exactly as DL_FIELD does between force
   fields. (Jorgensen's native UA bonded classes are not used: they lack
   terms for many neighbours, e.g. Si–C2, and the harmonic parameters are
   the same as the CT ones to within a few percent anyway.)

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
    if element and element != "C":
        return None
    # 1. OPLS bonded-class keys name the bead outright (C3 = CH3, C2 = CH2,
    #    CH, C4 = CH4, C9 = =CH2, C8 = =CH-, CD = aromatic CH, C7 = =C<).
    cls = (type_key or "").strip()
    n = _OPLS_UA_CLASS_H.get(cls)
    if n is not None:
        return n
    # 2. Name / description starting with CHn (TraPPE "CH2", "CH3 IN METHANOL").
    for text in (description or "", type_key or ""):
        text = text.strip()
        if not text:
            continue
        m = re.match(r"^(?:@atom:)?CH(\d)?(?![A-Za-z0-9])", text)
        if m:
            return int(m.group(1)) if m.group(1) else 1
    # 3. CHn as a word anywhere ("ETHER CH3 (-O)", "CH2 Methylenechloride").
    m = re.search(r"(?<![A-Za-z0-9])CH(\d)?(?![A-Za-z0-9])", description or "")
    if m:
        return int(m.group(1)) if m.group(1) else 1
    # 4. bare "C (SP3) NEOPENTANE", "C IN CH3CN" — a carbon bead with no H.
    for text in (description or "", type_key or ""):
        if re.match(r"^(?:@atom:)?C(?![A-Za-z0-9])", text.strip()):
            return 0
    return None


# OPLS united-atom bonded classes -> hydrogens the bead stands for.
_OPLS_UA_CLASS_H = {"C3": 3, "C2": 2, "CH": 1, "C4": 4, "C9": 2, "C8": 1,
                    "CD": 1, "C7": 0}


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
        # Names are regenerated from the new index: moltemplate merges atoms
        # that share a $atom name, and chains assembled from copies of one
        # monomer can carry duplicate names.
        na = Atom(index=k, element=a.element, xyz=np.array(a.xyz, dtype=float),
                  name=f"{a.element}{k + 1}", charge=a.charge, ff_type=a.ff_type)
        new.atoms.append(na)
    new.bonds = [(remap[i], remap[j], o) for i, j, o in mol.bonds
                 if i in remap and j in remap]
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
    t = re.escape(type_id)
    m = re.search(rf"^\s*pair_coeff\s+@atom:{t}(?:_\S*)?\s+@atom:{t}(?:_\S*)?"
                  rf"\s+(?:[a-z][\w/]*\s+)?([-\d.eE+]+)\s+([-\d.eE+]+)", text, re.M)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _charge(lt_path: Path, type_id: str) -> float:
    text = Path(lt_path).read_text(errors="replace")
    m = re.search(rf"^\s*set\s+type\s+@atom:{re.escape(type_id)}\s+charge\s+([-\d.eE+]+)", text, re.M)
    return float(m.group(1)) if m else 0.0


def _ua_class(lt_path: Path, type_id: str) -> str:
    """OPLS bonded class of a UA type from its charge-line comment (C3, C2, CH, C9 …)."""
    text = Path(lt_path).read_text(errors="replace")
    m = re.search(rf"^\s*set\s+type\s+@atom:{re.escape(type_id)}\s+charge\s+\S+\s*#\s*\S+\s*-\s*(\S+)", text, re.M)
    return m.group(1) if m else ""


# All-atom type whose bonded classes a bead borrows, by the bead's own
# hybridisation: sp3 -> CT (136), sp2 -> CM (141), aromatic -> CA (145).
_CLASS_TO_AA_REP = {"C3": "136", "C2": "136", "CH": "136", "C4": "136", "CT": "136",
                    "C9": "141", "C8": "141", "C7": "141", "CD": "145"}


def bridge_type_name(tid: str) -> str:
    return "UA_" + re.sub(r"[^A-Za-z0-9]", "_", tid)


def write_bridge_lt(out_dir: Path, primary_ff, ua_ff, beads_used: Dict[str, Tuple[int, float, str]],
                    sp3_type: str, extra_sources=()) -> Path:
    """Generate ``paaf_ua_bridge.lt`` declaring the UA beads inside the AA force field.

    ``sp3_type`` is the primary library's all-atom sp3 carbon type (OPLS
    ``136``). Each bead borrows the bonded classes of the all-atom carbon
    with its own hybridisation (sp3 -> CT, sp2 -> CM, aromatic -> CA), so
    every bond/angle/dihedral/improper it takes part in — to any neighbour
    the all-atom library knows, Si and S included — resolves.
    """
    aa_lt = primary_ff.bundled_path()
    sub = _pair_substyle(aa_lt)
    suffix = _equivalence_suffix(aa_lt, sp3_type)
    # (ua_ff, ua_lt, beads) per united-atom source; beads from the all-atom
    # library's own UA block come as a second source.
    sources = [(ua_ff, ua_ff.bundled_path(), dict(beads_used))]
    for ff2, beads2 in extra_sources:
        if beads2:
            sources.append((ff2, ff2.bundled_path(), dict(beads2)))
    if not suffix:
        raise RuntimeError(
            f"{primary_ff.display_name} has no bonded-equivalence table "
            f"(replace{{}} lines) for type {sp3_type}, so united-atom beads "
            f"cannot borrow its bonded parameters. Use OPLS-AA 2024 as the "
            f"all-atom force field for UA/AA mixing, or OPLS-UA 2024 as the "
            f"united-atom one (its beads are already part of OPLS-AA 2024).")
    lines = [
        f"# Generated by PAAF: united-atom beads inside {primary_ff.display_name}.",
        "# Non-bonded (LJ, charge, mass) from the united-atom library;",
        "# bonded terms borrowed from the all-atom carbon class of matching",
        f"# hybridisation (sp3: type {sp3_type} / CT, sp2: CM, aromatic: CA)",
        "# so bonds/angles/dihedrals across the UA/AA boundary resolve.",
        f'import "{primary_ff.lt_include}"',
        "",
        # Re-open the all-atom force field's own namespace: the library's
        # "Bonds/Angles/Dihedrals By Type" wildcards only match types that
        # live inside it, so the beads must be declared there, not in a
        # child object.
        f"{primary_ff.inherit} {{",
    ]
    lines.append('  write_once("Data Masses") {')
    for ff_i, lt_i, beads_i in sources:
        for tid, (n_h, mass, desc) in sorted(beads_i.items()):
            # No ':' in the comment: moltemplate copies it into the data
            # file's Masses line and cleanup_moltemplate.sh / ltemplify.py
            # read that comment back as a type name ("at most one ':'").
            clean = re.sub(r"[:\s]+", " ", f"{ff_i.key} {tid} {desc}").strip()
            lines.append(f"    @atom:{bridge_type_name(tid)} {mass:.4f}   # {bridge_type_name(tid)} {clean}")
    lines += ["  }", '  write_once("In Charges") {']
    for ff_i, lt_i, beads_i in sources:
        for tid in sorted(beads_i):
            lines.append(f"    set type @atom:{bridge_type_name(tid)} charge {_charge(lt_i, tid):.4f}")
    lines += ["  }", '  write_once("In Settings") {']
    for ff_i, lt_i, beads_i in sources:
        for tid in sorted(beads_i):
            lj = _lj_params(lt_i, tid)
            if lj is None:
                raise RuntimeError(f"no pair_coeff for {ff_i.key} type {tid} in {lt_i.name}")
            eps, sig = lj
            name = bridge_type_name(tid)
            lines.append(f"    pair_coeff @atom:{name} @atom:{name} {sub} {eps:.6f} {sig:.4f}".rstrip())
    lines += ["  }"]
    for ff_i, lt_i, beads_i in sources:
        for tid in sorted(beads_i):
            name = bridge_type_name(tid)
            rep_id = _CLASS_TO_AA_REP.get(_ua_class(lt_i, tid), sp3_type)
            sfx = _equivalence_suffix(aa_lt, rep_id) or suffix
            lines.append(f"  replace{{ @atom:{name} @atom:{name}{sfx} }}")
    lines += [f"}}  # {primary_ff.inherit} (+ PAAF united-atom beads)", ""]
    p = Path(out_dir) / BRIDGE_FILE
    p.write_text("\n".join(lines))
    log.info("Wrote UA/AA bridge library %s (%d bead types)", p,
             sum(len(b) for _, _, b in sources))
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
    # Beads the all-atom library carries itself (the OPLS-UA block inside
    # OPLS-AA). A type picked from the all-atom table without the UA: tag is
    # the same united-atom choice and absorbs its hydrogens all the same —
    # whichever secondary library (if any) was chosen.
    own_beads = primary_own_beads(primary_ff) if not ua_only else {}
    if ua_only:
        plan = plan_absorption(chain, types, beads, strip_prefix=False)
    else:
        tagged = {i: t for i, t in types.items() if t.startswith(UA_PREFIX)}
        plain = {i: t for i, t in types.items()
                 if not t.startswith(UA_PREFIX) and t in own_beads}
        plan = plan_absorption(chain, tagged, beads)
        plan_own = plan_absorption(chain, plain, own_beads, strip_prefix=False)
        plan.remove.extend(plan_own.remove)
        plan.beads.update(plan_own.beads)
        plan.problems.extend(plan_own.problems)
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
    else:
        # Every bead — from the secondary library (UA:) or the all-atom
        # library's own UA block — is declared through the bridge: its own
        # LJ/charge/mass, bonded classes of the matching all-atom carbon.
        # Using the UA block's native bonded classes instead fails the
        # moment a bead sits next to Si, S or another atom Jorgensen never
        # paired with a united-atom carbon; the all-atom classes cover all.
        used_sec: Dict[str, Tuple[int, float, str]] = {}
        used_own: Dict[str, Tuple[int, float, str]] = {}
        for a in new.atoms:
            if a.ff_type and a.ff_type.startswith(UA_PREFIX):
                tid = a.ff_type[len(UA_PREFIX):]
                used_sec[tid] = beads[tid]
                a.ff_type = bridge_type_name(tid)
                res.bead_masses[a.ff_type] = beads[tid][1]
            elif a.ff_type in own_beads and a.element == "C":
                tid = a.ff_type
                used_own[tid] = own_beads[tid]
                a.ff_type = bridge_type_name(tid)
                res.bead_masses[a.ff_type] = own_beads[tid][1]
        sp3 = FF_MAPS.get(primary_ff.key, ({}, {}))[0].get("alkane_CH2", "")
        own_ff = get_ff("oplsua_2024")
        if ua_ff.key == own_ff.key:
            write_bridge_lt(Path(out_dir), primary_ff, ua_ff, {**used_sec, **used_own}, sp3)
        else:
            write_bridge_lt(Path(out_dir), primary_ff, ua_ff, used_sec, sp3,
                            extra_sources=[(own_ff, used_own)])
        res.inherit = primary_ff.inherit
        res.lt_include = BRIDGE_FILE
        say(f"Bridge library {BRIDGE_FILE}: {len(used_sec) + len(used_own)} united-atom "
            f"bead type(s) declared inside {primary_ff.display_name} (LJ/charge/mass "
            f"from the UA library, bonded terms from the all-atom CT/CM/CA classes).")
    say(f"United-atom absorption: {res.n_beads} bead(s), {res.removed_h} "
        f"hydrogen(s) removed; chain now {len(new.atoms)} atoms.")
    return res


def primary_own_beads(primary_ff) -> Dict[str, Tuple[int, float, str]]:
    """United-atom bead types the all-atom library itself contains.

    OPLS-AA 2024 ships the OPLS-UA block (types 66-134); nothing else does.
    """
    if primary_ff.key not in _OPLS_AA_KEYS:
        return {}
    from .ff_registry import get_ff
    try:
        return bead_table(get_ff("oplsua_2024").bundled_path())
    except Exception:
        return {}


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
