"""Type an amorphous cell with DL_FIELD one distinct chain at a time.

Why not type the whole cell in one DL_FIELD run
------------------------------------------------
DL_FIELD has fixed array sizes compiled in (``dlf_notation.h``,
``dl_field.h``): ``MAX_FN 12250`` functional groups of one kind,
``MAX_ATOM 50000`` atoms per molecular group, and caps on bonds, angles and
dihedrals. A 200-chain PE cell has 40,000 aliphatic carbons, and DL_FIELD
stops with "Error: too many C identified." — no .data, nothing typed. It
also perceives bonds from distance, so an inter-chain contact left by
construction can be read as a bond and break the typing of a cell that is
otherwise fine.

What this does instead
----------------------
1. Group the cell's chains by their exact bonded graph (element sequence plus
   bond list). Chains in a group are the same molecule with the same atom
   order, so one typing serves them all.
2. Type one representative per group. Representatives are unwrapped, spread
   out on a grid far enough apart that DL_FIELD cannot see a bond between
   them, and sent to DL_FIELD in batches kept well under its limits.
3. Check the typing: DL_FIELD's bond list for each representative must be
   exactly the graph's bond list, and the atom counts must line up.
4. Copy types, charges and bonded terms onto every chain, with the cell's own
   coordinates, and write one LAMMPS data/input pair and/or one GROMACS
   .gro/.top/.itp set.

Cross-pair coefficients come from DL_FIELD itself: all representatives in a
batch are typed together, so DL_FIELD writes every pair between their types
with the force field's own mixing rule. When several batches are needed, a
pair that never met in one batch is refused by name rather than guessed.
"""
from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..logging_utils import get_logger

log = get_logger(__name__)

#: Atoms per DL_FIELD run. Far below MAX_ATOM (50,000 per molecular group)
#: and low enough that no single functional-group count (MAX_FN 12,250,
#: MAX_CGI 9,999) can be reached by one batch.
BATCH_ATOMS = 8000
#: Empty space between representatives in a batch, in Å.
GAP = 15.0
#: DL_FIELD refuses a cell whose half-box is below the pair cutoff.
MIN_BOX = 40.0

__all__ = ["ChainTypingError", "ChainTyping", "type_cell_by_chain",
           "group_chains"]


class ChainTypingError(RuntimeError):
    """The cell could not be typed chain by chain; the message says why."""


@dataclass
class _Template:
    index: int
    first_chain: int
    start: int               # first atom of the representative chain
    n_atoms: int
    elements: Tuple[str, ...]
    bonds: Tuple[Tuple[int, int], ...]      # local indices, i < j
    chains: List[int] = field(default_factory=list)
    name: str = ""
    # filled from DL_FIELD
    out_local: List[int] = field(default_factory=list)   # output atom -> local input atom


@dataclass
class ChainTyping:
    """Files written by :func:`type_cell_by_chain`."""
    data_file: Optional[Path] = None
    input_file: Optional[Path] = None
    gro_file: Optional[Path] = None
    top_file: Optional[Path] = None
    itp_files: List[Path] = field(default_factory=list)
    n_templates: int = 0
    n_batches: int = 0
    messages: List[str] = field(default_factory=list)


# ====================================================================== graph
def group_chains(molecule, atoms_per_chain: Sequence[int],
                 chain_names: Optional[Sequence[str]] = None
                 ) -> Tuple[List[_Template], List[int]]:
    """Distinct chain graphs, and the template index of every chain."""
    n_total = len(molecule.atoms)
    if sum(atoms_per_chain) != n_total:
        raise ChainTypingError(
            f"The cell has {n_total} atoms but its chains add up to "
            f"{sum(atoms_per_chain)}; cannot tell which atom is in which "
            f"chain.")
    owner = np.empty(n_total, dtype=np.int64)
    starts = []
    s = 0
    for c, n in enumerate(atoms_per_chain):
        owner[s:s + n] = c
        starts.append(s)
        s += n
    local_bonds: List[List[Tuple[int, int]]] = [[] for _ in atoms_per_chain]
    for b in molecule.bonds:
        i, j = int(b[0]), int(b[1])
        ci, cj = owner[i], owner[j]
        if ci != cj:
            raise ChainTypingError(
                f"Atoms {i + 1} and {j + 1} are bonded but belong to "
                f"different chains; chain-by-chain typing needs separate "
                f"molecules.")
        a, z = i - starts[ci], j - starts[ci]
        local_bonds[ci].append((min(a, z), max(a, z)))

    by_key: "OrderedDict[tuple, _Template]" = OrderedDict()
    chain_template: List[int] = []
    for c, n in enumerate(atoms_per_chain):
        st = starts[c]
        elements = tuple(a.element for a in molecule.atoms[st:st + n])
        bonds = tuple(sorted(set(local_bonds[c])))
        key = (elements, bonds)
        t = by_key.get(key)
        if t is None:
            t = _Template(index=len(by_key), first_chain=c, start=st,
                          n_atoms=n, elements=elements, bonds=bonds)
            by_key[key] = t
        t.chains.append(c)
        chain_template.append(t.index)

    templates = list(by_key.values())
    # Molecule names: the species name, numbered when a species needs more
    # than one template (random copolymer sequences, differing DP).
    bases = []
    for t in templates:
        base = "POLY"
        if chain_names and t.first_chain < len(chain_names):
            base = re.sub(r"[^A-Za-z0-9_]", "_",
                          str(chain_names[t.first_chain])) or "POLY"
        bases.append(base)
    seen: Dict[str, int] = {}
    for t, base in zip(templates, bases):
        if bases.count(base) == 1:
            t.name = base
        else:
            seen[base] = seen.get(base, 0) + 1
            t.name = f"{base}_{seen[base]}"
    return templates, chain_template


def _unwrap(coords: np.ndarray, bonds: Sequence[Tuple[int, int]],
            dims: Optional[np.ndarray]) -> np.ndarray:
    """Make a chain continuous: follow bonds with the minimum image."""
    xyz = coords.copy()
    if dims is None or not np.all(np.asarray(dims) > 0):
        return xyz
    n = len(xyz)
    adj: List[List[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    seen = np.zeros(n, dtype=bool)
    for root in range(n):
        if seen[root]:
            continue
        seen[root] = True
        stack = [root]
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if seen[j]:
                    continue
                d = xyz[j] - xyz[i]
                xyz[j] = xyz[i] + d - np.round(d / dims) * dims
                seen[j] = True
                stack.append(j)
    return xyz


# ==================================================================== DL_FIELD
def _batches(templates: List[_Template]) -> List[List[_Template]]:
    out: List[List[_Template]] = []
    cur: List[_Template] = []
    size = 0
    for t in templates:
        if cur and size + t.n_atoms > BATCH_ATOMS:
            out.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += t.n_atoms
    if cur:
        out.append(cur)
    return out


def _write_batch_xyz(batch: List[_Template], molecule, dims, path: Path) -> float:
    """Spread the representatives on a grid; return the cubic box edge."""
    coords = molecule.coords()
    placed = []
    span = 0.0
    for t in batch:
        xyz = _unwrap(coords[t.start:t.start + t.n_atoms], t.bonds, dims)
        xyz = xyz - xyz.mean(axis=0)
        span = max(span, float((xyz.max(axis=0) - xyz.min(axis=0)).max()))
        placed.append(xyz)
    step = span + GAP
    k = max(1, math.ceil(len(batch) ** (1.0 / 3.0) - 1e-9))
    box = max(MIN_BOX, k * step)
    lines = [str(sum(t.n_atoms for t in batch)), "paaf chain templates"]
    for m, (t, xyz) in enumerate(zip(batch, placed)):
        ix, iy, iz = m % k, (m // k) % k, m // (k * k)
        centre = (np.array([ix, iy, iz], dtype=float) + 0.5) * step
        for el, p in zip(t.elements, xyz + centre):
            lines.append(f"{el} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return float(box)


def _dlfield_error(work: Path) -> str:
    log_file = work / "dl_field.log"
    try:
        text = log_file.read_text(errors="replace")
    except OSError:
        return ""
    errs = [l.strip() for l in text.splitlines()
            if l.strip().lower().startswith(("error", "unknown", "please"))]
    return " ".join(errs[:3])


def _map_output_atoms(batch: List[_Template], n_out: int, where: str) -> None:
    """Which input atom each DL_FIELD output atom is, per template."""
    n_in = sum(t.n_atoms for t in batch)
    heavy = sum(sum(1 for e in t.elements if e != "H") for t in batch)
    for t in batch:
        if n_out == n_in:
            t.out_local = list(range(t.n_atoms))
        elif n_out == heavy:
            # United-atom schemes fold hydrogens into their carbons and keep
            # the heavy atoms in input order.
            t.out_local = [i for i, e in enumerate(t.elements) if e != "H"]
        else:
            raise ChainTypingError(
                f"DL_FIELD wrote {n_out} atoms for {where}, but the chains "
                f"sent have {n_in} atoms ({heavy} without hydrogens). The "
                f"typed atoms cannot be matched back to the cell.")


# ================================================================ LAMMPS parse
_TOPO = ("Bonds", "Angles", "Dihedrals", "Impropers")
_TOPO_ATOMS = {"Bonds": 2, "Angles": 3, "Dihedrals": 4, "Impropers": 4}
_KIND = {"Bonds": "bond", "Angles": "angle", "Dihedrals": "dihedral",
         "Impropers": "improper"}
#: Coefficient sections indexed by each kind of type (class2 adds cross terms).
_COEFF_GROUPS = {
    "atom": ("Pair Coeffs",),
    "bond": ("Bond Coeffs",),
    "angle": ("Angle Coeffs", "BondBond Coeffs", "BondAngle Coeffs"),
    "dihedral": ("Dihedral Coeffs", "MiddleBondTorsion Coeffs",
                 "EndBondTorsion Coeffs", "AngleTorsion Coeffs",
                 "AngleAngleTorsion Coeffs", "BondBond13 Coeffs"),
    "improper": ("Improper Coeffs", "AngleAngle Coeffs"),
}
_SECTIONS = {"Masses", "Atoms", "Velocities", "PairIJ Coeffs", *_TOPO,
             *(s for g in _COEFF_GROUPS.values() for s in g)}


def _parse_data(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    current = None
    for raw in Path(path).read_text(errors="replace").splitlines():
        head = raw.split("#")[0].strip()
        if head in _SECTIONS:
            current = head
            out[current] = []
            continue
        if current is None or not raw.strip():
            continue
        out[current].append(raw.strip())
    return out


def _body(line: str) -> List[str]:
    return line.split("#")[0].split()


def _comment(line: str) -> str:
    return line.split("#", 1)[1].strip() if "#" in line else ""


@dataclass
class _LmpBatch:
    labels: Dict[int, str]            # local atom type -> label
    masses: Dict[int, str]            # local atom type -> mass text
    coeffs: Dict[str, Dict[int, str]]   # section -> local type -> params
    atoms: List[Tuple[int, float]]    # per output atom: (local type, charge)
    topo: Dict[str, List[Tuple[int, List[int]]]]  # section -> (type, atoms 0-based)
    pair: Dict[Tuple[int, int], str]  # (i<=j local types) -> params after i j
    header_in: List[str]              # lammps.in without read_data / pair_coeff / group


def _parse_lammps_batch(data: Path, inp: Optional[Path]) -> _LmpBatch:
    sec = _parse_data(data)
    labels, masses = {}, {}
    for ln in sec.get("Masses", []):
        p = _body(ln)
        labels[int(p[0])] = _comment(ln) or f"type{p[0]}"
        masses[int(p[0])] = p[1]
    coeffs: Dict[str, Dict[int, str]] = {}
    for group in _COEFF_GROUPS.values():
        for s in group:
            if s in sec:
                coeffs[s] = {int(_body(l)[0]): " ".join(_body(l)[1:])
                             for l in sec[s]}
    rows = []
    for ln in sec.get("Atoms", []):
        p = _body(ln)
        rows.append((int(p[0]), int(p[2]), float(p[3])))
    rows.sort()
    atoms = [(t, q) for _, t, q in rows]
    topo = {}
    for s in _TOPO:
        items = []
        for ln in sec.get(s, []):
            p = _body(ln)
            k = _TOPO_ATOMS[s]
            items.append((int(p[1]), [int(x) - 1 for x in p[2:2 + k]]))
        topo[s] = items
    pair: Dict[Tuple[int, int], str] = {}
    header: List[str] = []
    if inp is not None and Path(inp).is_file():
        for raw in Path(inp).read_text(errors="replace").splitlines():
            p = _body(raw)
            if p and p[0] == "pair_coeff":
                try:
                    i, j = int(p[1]), int(p[2])
                except ValueError:
                    raise ChainTypingError(
                        f"{inp.name}: cannot place '{raw.strip()}' — a "
                        f"wildcard pair_coeff cannot be renumbered.")
                pair[(min(i, j), max(i, j))] = " ".join(p[3:])
                continue
            if p and p[0] in ("read_data", "group"):
                continue
            if raw.lstrip().startswith("# Input structure"):
                continue
            header.append(raw.rstrip())
    return _LmpBatch(labels, masses, coeffs, atoms, topo, pair, header)


# ============================================================== LAMMPS write
def _assemble_lammps(templates: List[_Template], chain_template: List[int],
                     parsed: List[Tuple[List[_Template], _LmpBatch]],
                     molecule, dims: np.ndarray, out_dir: Path
                     ) -> Tuple[Path, Path]:
    # ---- global atom types, keyed by DL_FIELD label ---------------------
    atom_types: "OrderedDict[str, int]" = OrderedDict()
    atom_mass: Dict[str, str] = {}
    atom_pair_sec: Dict[str, str] = {}
    kinds = ("bond", "angle", "dihedral", "improper")
    type_ids: Dict[str, "OrderedDict[tuple, int]"] = {k: OrderedDict() for k in kinds}
    sections_present = set()
    per_template = {}          # template index -> dict of mapped data
    pair: Dict[Tuple[int, int], str] = {}
    pair_seen: Dict[Tuple[int, int], str] = {}
    header_in: Optional[List[str]] = None

    for batch, lb in parsed:
        if header_in is None:
            header_in = lb.header_in
        elif _style_lines(header_in) != _style_lines(lb.header_in):
            raise ChainTypingError(
                "DL_FIELD wrote different LAMMPS styles for different "
                "batches of chains; they cannot be combined into one input.")
        gmap: Dict[int, int] = {}
        for lt, label in lb.labels.items():
            if label not in atom_types:
                atom_types[label] = len(atom_types) + 1
                atom_mass[label] = lb.masses[lt]
                if "Pair Coeffs" in lb.coeffs:
                    atom_pair_sec[label] = lb.coeffs["Pair Coeffs"].get(lt, "")
            elif abs(float(atom_mass[label]) - float(lb.masses[lt])) > 1e-4:
                raise ChainTypingError(
                    f"Atom type {label} has mass {atom_mass[label]} in one "
                    f"batch and {lb.masses[lt]} in another.")
            gmap[lt] = atom_types[label]
        for (i, j), params in lb.pair.items():
            gi, gj = sorted((gmap[i], gmap[j]))
            if (gi, gj) in pair and pair[(gi, gj)] != params:
                raise ChainTypingError(
                    f"DL_FIELD gave pair {lb.labels[i]}-{lb.labels[j]} two "
                    f"different coefficient sets: '{pair[(gi, gj)]}' and "
                    f"'{params}'.")
            pair[(gi, gj)] = params
            pair_seen[(gi, gj)] = f"{lb.labels[i]} {lb.labels[j]}"

        # bonded types: identical parameters in every indexed section = same type
        tmap: Dict[str, Dict[int, int]] = {k: {} for k in kinds}
        for kind in kinds:
            secs = [s for s in _COEFF_GROUPS[kind] if s in lb.coeffs]
            sections_present.update(secs)
            local_types = set()
            for s in secs:
                local_types.update(lb.coeffs[s])
            for lt in sorted(local_types):
                key = tuple((s, lb.coeffs[s].get(lt, "")) for s in secs)
                ids = type_ids[kind]
                if key not in ids:
                    ids[key] = len(ids) + 1
                tmap[kind][lt] = ids[key]

        # split the batch's atoms and topology per template
        offset = 0
        for t in batch:
            n_out = len(t.out_local)
            rng = range(offset, offset + n_out)
            atoms = [(gmap[lb.atoms[k][0]], lb.atoms[k][1]) for k in rng]
            topo = {}
            for s in _TOPO:
                items = []
                for typ, idx in lb.topo.get(s, []):
                    if idx and offset <= idx[0] < offset + n_out:
                        if not all(offset <= x < offset + n_out for x in idx):
                            raise ChainTypingError(
                                f"DL_FIELD made a {s[:-1].lower()} between "
                                f"two different chains in the typing batch; "
                                f"the chains were too close.")
                        items.append((tmap[_KIND[s]].get(typ, typ),
                                      [x - offset for x in idx]))
                topo[s] = items
            # DL_FIELD's bonds must be the graph's bonds, or the typing is
            # of a different molecule than the one in the cell.
            got = sorted(tuple(sorted((t.out_local[a], t.out_local[b])))
                         for _, (a, b) in topo["Bonds"])
            if n_out == t.n_atoms and got != sorted(t.bonds):
                raise ChainTypingError(
                    f"DL_FIELD perceived {len(got)} bonds in chain "
                    f"{t.first_chain + 1} ({t.name}) where the structure has "
                    f"{len(t.bonds)}. Its geometry is too distorted to type "
                    f"reliably; rebuild the cell (lower 'Build at').")
            per_template[t.index] = (atoms, topo)
            offset += n_out

    n_types = len(atom_types)
    missing = [(i, j) for i in range(1, n_types + 1)
               for j in range(i, n_types + 1) if (i, j) not in pair]
    if pair and missing:
        names = {v: k for k, v in atom_types.items()}
        shown = ", ".join(f"{names[i]}-{names[j]}" for i, j in missing[:6])
        raise ChainTypingError(
            f"{len(missing)} pair coefficient(s) were never produced "
            f"together by DL_FIELD ({shown}{' …' if len(missing) > 6 else ''}). "
            f"They would have to be guessed from a mixing rule, so the cell "
            f"is not typed.")

    # ---- write cell.data ----------------------------------------------------
    coords = molecule.coords()
    L = np.asarray(dims, dtype=float)
    label_of = {gid: label for label, gid in atom_types.items()}
    atom_lines, topo_lines = [], {s: [] for s in _TOPO}
    atom_id = 0
    counts = {s: 0 for s in _TOPO}
    starts = np.cumsum([0] + [templates[ti].n_atoms for ti in chain_template])
    for c, ti in enumerate(chain_template):
        t = templates[ti]
        atoms, topo = per_template[ti]
        base_in = int(starts[c])
        first_id = atom_id + 1
        for k, (typ, q) in enumerate(atoms):
            atom_id += 1
            x, y, z = coords[base_in + t.out_local[k]]
            label = label_of[typ]
            atom_lines.append(f"{atom_id} {c + 1} {typ} {q:.6f} "
                              f"{x:.6f} {y:.6f} {z:.6f} # {label}")
        for s in _TOPO:
            for typ, idx in topo[s]:
                counts[s] += 1
                ids = " ".join(str(first_id + x) for x in idx)
                topo_lines[s].append(f"{counts[s]} {typ} {ids}")

    out_dir.mkdir(parents=True, exist_ok=True)
    data = out_dir / "cell.data"
    L_lo = np.zeros(3)
    lines = ["LAMMPS data file: amorphous cell typed chain by chain by PAAF "
             "(DL_FIELD)", "",
             f"{atom_id} atoms"]
    for s in _TOPO:
        lines.append(f"{counts[s]} {s.lower()}")
    lines.append(f"{n_types} atom types")
    for kind in kinds:
        if type_ids[kind]:
            lines.append(f"{len(type_ids[kind])} {kind} types")
    lines += ["",
              f"{L_lo[0]:.6f} {L[0]:.6f} xlo xhi",
              f"{L_lo[1]:.6f} {L[1]:.6f} ylo yhi",
              f"{L_lo[2]:.6f} {L[2]:.6f} zlo zhi", "", "Masses", ""]
    for label, gid in atom_types.items():
        lines.append(f"{gid} {atom_mass[label]} # {label}")
    if atom_pair_sec:
        lines += ["", "Pair Coeffs", ""]
        for label, gid in atom_types.items():
            lines.append(f"{gid} {atom_pair_sec.get(label, '')} # {label}")
    for kind in kinds:
        ids = type_ids[kind]
        if not ids:
            continue
        for s in _COEFF_GROUPS[kind]:
            if s not in sections_present:
                continue
            lines += ["", s, ""]
            for key, gid in ids.items():
                params = dict(key).get(s, "")
                lines.append(f"{gid} {params}")
    lines += ["", "Atoms  # full", ""] + atom_lines
    for s in _TOPO:
        if topo_lines[s]:
            lines += ["", s, ""] + topo_lines[s]
    data.write_text("\n".join(lines) + "\n")

    # ---- write cell.in -------------------------------------------------------
    inp = out_dir / "cell.in"
    body: List[str] = []
    wrote_data = False
    for raw in header_in or []:
        body.append(raw)
        if (not wrote_data and _body(raw) and _body(raw)[0] == "special_bonds"):
            body += ["", "read_data cell.data", ""]
            body += _pair_lines(pair, atom_types)
            wrote_data = True
    if not wrote_data:
        body += ["", "read_data cell.data", ""] + _pair_lines(pair, atom_types)
    inp.write_text("# LAMMPS input for the PAAF amorphous cell; styles from "
                   "DL_FIELD, types merged chain by chain\n"
                   + "\n".join(body) + "\n")
    return data, inp


def _pair_lines(pair: Dict[Tuple[int, int], str],
                atom_types: "OrderedDict[str, int]") -> List[str]:
    names = {v: k for k, v in atom_types.items()}
    return [f"pair_coeff {i} {j} {params} # {names[i]} {names[j]}"
            for (i, j), params in sorted(pair.items())]


def _style_lines(header: List[str]) -> List[str]:
    return [" ".join(_body(l)) for l in header
            if _body(l) and _body(l)[0].endswith("_style")]


# ============================================================== GROMACS parse
_ITP_ATOMS = {"bonds": 2, "pairs": 2, "angles": 3, "dihedrals": 4,
              "constraints": 2, "settles": 1, "position_restraints": 1,
              "cmap": 5}


def _parse_itp_blocks(text: str) -> List[Tuple[str, List[str]]]:
    blocks: List[Tuple[str, List[str]]] = []
    for raw in text.splitlines():
        line = raw.split(";")[0].strip()
        m = re.match(r"\[\s*(\S+)\s*\]", line)
        if m:
            blocks.append((m.group(1), []))
            continue
        if line and blocks:
            blocks[-1][1].append(line)
    return blocks


@dataclass
class _GmxBatch:
    defaults: List[str]
    atomtypes: List[str]
    nrexcl: str
    atoms: List[List[str]]                         # tokens per output atom
    topo: List[Tuple[str, List[Tuple[List[int], List[str]]]]]  # (section, [(atoms0, rest)])


def _parse_gromacs_batch(top: Path) -> _GmxBatch:
    from ..gromacs_blend import parse_top
    sections = parse_top(top)
    itp_text = ""
    for inc in sections.get("#include", []):
        p = top.parent / inc
        if p.is_file():
            itp_text += p.read_text(errors="replace") + "\n"
    blocks = _parse_itp_blocks(itp_text) if itp_text else []
    if not blocks:
        blocks = _parse_itp_blocks(top.read_text(errors="replace"))
    nrexcl = "3"
    atoms: List[List[str]] = []
    topo: List[Tuple[str, List[Tuple[List[int], List[str]]]]] = []
    n_moltypes = sum(1 for name, _ in blocks if name == "moleculetype")
    if n_moltypes != 1:
        raise ChainTypingError(
            f"{top.name}: expected one [moleculetype] from DL_FIELD, found "
            f"{n_moltypes}.")
    for name, lines in blocks:
        if name == "moleculetype":
            if lines:
                nrexcl = lines[0].split()[1] if len(lines[0].split()) > 1 else "3"
        elif name == "atoms":
            atoms = [l.split() for l in lines]
        elif name in _ITP_ATOMS:
            k = _ITP_ATOMS[name]
            items = []
            for l in lines:
                p = l.split()
                items.append(([int(x) - 1 for x in p[:k]], p[k:]))
            topo.append((name, items))
        elif name == "exclusions":
            topo.append((name, [([int(x) - 1 for x in l.split()], [])
                                for l in lines]))
        elif name in ("system", "molecules", "defaults", "atomtypes"):
            continue
        else:
            raise ChainTypingError(
                f"{top.name}: section [{name}] is not one PAAF can split "
                f"per chain.")
    atoms.sort(key=lambda p: int(p[0]))
    return _GmxBatch(sections.get("defaults", []),
                     sections.get("atomtypes", []), nrexcl, atoms, topo)


def _assemble_gromacs(templates: List[_Template], chain_template: List[int],
                      parsed: List[Tuple[List[_Template], _GmxBatch]],
                      molecule, dims: np.ndarray, out_dir: Path
                      ) -> Tuple[Path, Path, Path]:
    defaults: Optional[List[str]] = None
    atomtypes: "OrderedDict[str, str]" = OrderedDict()
    moltypes: Dict[int, str] = {}
    atom_names: Dict[int, List[str]] = {}

    for batch, gb in parsed:
        if defaults is None:
            defaults = gb.defaults
        elif [l.split() for l in gb.defaults] != [l.split() for l in defaults]:
            raise ChainTypingError(
                "DL_FIELD wrote different GROMACS [defaults] for different "
                "batches of chains.")
        for line in gb.atomtypes:
            name = line.split()[0]
            if name in atomtypes and atomtypes[name].split() != line.split():
                raise ChainTypingError(
                    f"GROMACS atom type {name} has different parameters in "
                    f"two typing batches:\n  {atomtypes[name]}\n  {line}")
            atomtypes.setdefault(name, line)

        offset = 0
        for t in batch:
            n_out = len(t.out_local)
            lo, hi = offset, offset + n_out
            rows = gb.atoms[lo:hi]
            if len(rows) != n_out:
                raise ChainTypingError(
                    f"DL_FIELD's [atoms] list is shorter than the chains "
                    f"sent for typing ({len(gb.atoms)} atoms).")
            cg0 = int(rows[0][5]) if rows and len(rows[0]) > 5 else 1
            out = [f"[ moleculetype ]", "; name  nrexcl",
                   f"{t.name}    {gb.nrexcl}", "", "[ atoms ]",
                   "; nr type resnr residue atom cgnr charge mass"]
            names = []
            for k, p in enumerate(rows):
                p = list(p)
                p[0] = str(k + 1)
                if len(p) > 5:
                    p[5] = str(int(p[5]) - cg0 + 1)
                names.append(p[4] if len(p) > 4 else "X")
                out.append("  ".join(p))
            atom_names[t.index] = names
            bonds_seen = []
            for section, items in gb.topo:
                mine = []
                for idx, rest in items:
                    if not idx or not (lo <= idx[0] < hi):
                        continue
                    if not all(lo <= x < hi for x in idx):
                        raise ChainTypingError(
                            f"DL_FIELD made a [{section}] entry between two "
                            f"different chains in the typing batch.")
                    local = [x - lo for x in idx]
                    if section == "bonds":
                        bonds_seen.append(tuple(sorted(
                            (t.out_local[local[0]], t.out_local[local[1]]))))
                    mine.append("  ".join([str(x + 1) for x in local] + rest))
                out += ["", f"[ {section} ]"] + mine
            if n_out == t.n_atoms and sorted(bonds_seen) != sorted(t.bonds):
                raise ChainTypingError(
                    f"DL_FIELD perceived {len(bonds_seen)} bonds in chain "
                    f"{t.first_chain + 1} ({t.name}) where the structure has "
                    f"{len(t.bonds)}; not typed. Rebuild the cell (lower "
                    f"'Build at').")
            moltypes[t.index] = "\n".join(out) + "\n"
            offset += n_out

    out_dir.mkdir(parents=True, exist_ok=True)
    itp = out_dir / "cell.itp"
    itp.write_text("; molecule types of the PAAF amorphous cell, typed by "
                   "DL_FIELD one distinct chain at a time\n\n"
                   + "\n".join(moltypes[t.index] for t in templates))

    # [molecules]: consecutive chains of one template collapse into one line
    runs: List[List] = []
    for ti in chain_template:
        if runs and runs[-1][0] == ti:
            runs[-1][1] += 1
        else:
            runs.append([ti, 1])
    top = out_dir / "cell.top"
    lines = ["; GROMACS topology of the PAAF amorphous cell (DL_FIELD types)",
             "", "[ defaults ]", "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ"]
    lines += defaults or []
    lines += ["", "[ atomtypes ]"] + list(atomtypes.values())
    lines += ["", f'#include "{itp.name}"', "", "[ system ]",
              "PAAF amorphous cell", "", "[ molecules ]"]
    lines += [f"{templates[ti].name}    {n}" for ti, n in runs]
    top.write_text("\n".join(lines) + "\n")

    coords = molecule.coords()
    starts = np.cumsum([0] + [templates[ti].n_atoms for ti in chain_template])
    gro_lines = ["PAAF amorphous cell (typed chain by chain)", ""]
    n = 0
    for c, ti in enumerate(chain_template):
        t = templates[ti]
        resname = t.name[:5]
        for k, local in enumerate(t.out_local):
            n += 1
            x, y, z = coords[int(starts[c]) + local] / 10.0
            gro_lines.append(
                f"{(c + 1) % 100000:5d}{resname:<5s}"
                f"{atom_names[ti][k][:5]:>5s}{n % 100000:5d}"
                f"{x:8.3f}{y:8.3f}{z:8.3f}")
    gro_lines[1] = str(n)
    L = np.asarray(dims, dtype=float) / 10.0
    gro_lines.append(f"{L[0]:10.5f}{L[1]:10.5f}{L[2]:10.5f}")
    gro = out_dir / "cell.gro"
    gro.write_text("\n".join(gro_lines) + "\n")
    return gro, top, itp


# ==================================================================== driver
def type_cell_by_chain(molecule, atoms_per_chain: Sequence[int],
                       dims: Sequence[float], ff_key: str,
                       dl_field_dir: Optional[Path], work_dir: Path,
                       out_dir: Path, run_dlfield: Callable,
                       engines: Sequence[str] = ("lammps",),
                       chain_names: Optional[Sequence[str]] = None,
                       emit: Optional[Callable[[str], None]] = None,
                       cancel=None) -> ChainTyping:
    """Type ``molecule`` (a wrapped cell) with DL_FIELD chain by chain.

    ``run_dlfield`` is :func:`paaf.cell.cell_export._run_dlfield`. Raises
    :class:`ChainTypingError` with a user-facing reason when the cell cannot
    be typed; a cancel propagates as the caller's cancellation exception.
    """
    emit = emit or (lambda _m: None)
    dims = np.asarray(dims, dtype=float)
    templates, chain_template = group_chains(molecule, atoms_per_chain,
                                             chain_names)
    batches = _batches(templates)
    res = ChainTyping(n_templates=len(templates), n_batches=len(batches))
    emit(f"  typing {len(atoms_per_chain)} chain(s) through "
         f"{len(templates)} distinct chain structure(s) in "
         f"{len(batches)} DL_FIELD run(s) per format …")

    for engine in engines:
        parsed = []
        for b, batch in enumerate(batches):
            work = Path(work_dir) / f"{engine}_batch{b + 1}"
            xyz = work.parent / f"templates_{b + 1}.xyz"
            box = _write_batch_xyz(batch, molecule, dims, xyz)
            out = run_dlfield(xyz, work, ff_key, dl_field_dir, emit,
                              output_engine=engine, box_ang=(box, box, box),
                              cancel=cancel)
            if out is None:
                why = _dlfield_error(work)
                raise ChainTypingError(
                    f"DL_FIELD could not type chain structure(s) "
                    f"{', '.join(t.name for t in batch)} for {engine.upper()}"
                    + (f": {why}" if why else "")
                    + f". See {work / 'dl_field.log'}.")
            out = Path(out)
            if engine == "lammps":
                inp = None
                for cand in sorted(out.parent.glob("*.in")):
                    if "pair_style" in cand.read_text(errors="ignore"):
                        inp = cand
                        break
                lb = _parse_lammps_batch(out, inp)
                _map_output_atoms(batch, len(lb.atoms), f"batch {b + 1}")
                parsed.append((batch, lb))
            else:
                tops = sorted(out.parent.glob("*.top"))
                if not tops:
                    raise ChainTypingError(
                        f"DL_FIELD wrote {out.name} but no .top for batch "
                        f"{b + 1}; GROMACS output cannot be assembled.")
                gb = _parse_gromacs_batch(tops[0])
                _map_output_atoms(batch, len(gb.atoms), f"batch {b + 1}")
                parsed.append((batch, gb))

        if engine == "lammps":
            res.data_file, res.input_file = _assemble_lammps(
                templates, chain_template, parsed, molecule, dims, out_dir)
            emit(f"  wrote {res.data_file.name} + {res.input_file.name} "
                 f"({sum(len(t.out_local) * len(t.chains) for t in templates):,} atoms)")
        else:
            res.gro_file, res.top_file, itp = _assemble_gromacs(
                templates, chain_template, parsed, molecule, dims, out_dir)
            res.itp_files = [itp]
            emit(f"  wrote {res.gro_file.name} + {res.top_file.name} + "
                 f"{itp.name}")
    return res
