"""Type a repeat unit in the environment it will actually have.

The problem
-----------
A monomer on its own is not the thing that ends up in the chain. Isoprene's
polymerisation SMILES caps both link carbons with a hydrogen, so the isolated
molecule is ``CC=C(C)C`` — and those two carbons are genuinely CH3 there. In
the chain the cap is deleted and replaced by a bond to the next unit, making
them CH2.

OPLS-AA distinguishes the two: 135 is a CH3 carbon, 136 a CH2. So typing the
isolated monomer gives 135 for atoms that need 136, and the automatic typer is
*correct* about the molecule it was shown — it was shown the wrong molecule.

There is a second, quieter consequence. The chain has the cap hydrogens
removed, so monomer atom *k* is not chain atom *k*. Any per-atom typing keyed
by monomer index drifts out of alignment the moment the first cap is dropped.

The fix
-------
Build a trimer, A–A–A, and type its **middle** unit. Every atom there sits in
its true in-chain environment: the link carbons have two heavy neighbours, so
an automatic typer says CH2 without being told, and there are no cap hydrogens
to account for. The two outer units are not padding — they are the real chain
ends, and a chain end genuinely does need different types from a chain middle.

This module builds that trimer and records where every atom came from, so a
type assigned in the view can be mapped back to the monomer and forward to
every unit of the chain.

Copolymers
----------
For a copolymer the middle environment is not unique: isoprene beside an
epoxide is not isoprene beside isoprene. :func:`build_trimer` takes the
neighbours explicitly so A–B–A can be built as easily as A–A–A, and
:attr:`TrimerContext.junction_note` records which case was shown, rather than
letting a homopolymer trimer quietly stand in for a copolymer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["TrimerContext", "build_trimer", "ROLES", "ROLE_BASE", "CAP_UNIT",
           "role_key", "split_role_key", "split_by_role", "merge_roles"]

#: The kinds of unit a chain has, in the order they appear, plus the end cap.
#:
#: ``cap`` is not a unit. It is the handful of atoms the builder adds AFTER
#: assembly — for a polyester, turning the terminal -C(=O)H into -C(=O)OH.
#: They belong to no monomer, so they need a role of their own or they cannot
#: be shown, typed, or carried to the chain.
ROLES = ("head", "middle", "tail", "cap")

#: A type map is keyed by monomer atom index. Once the chain ends get their own
#: types there are three such maps, and they have to travel together through a
#: config file, a text box and a pipeline that all expect ONE ``{int: str}``.
#:
#: Rather than change every one of those, the role is folded into the key. The
#: repeat unit keeps its plain index, so everything written before this existed
#: keeps working untouched and unaware; the two end units are offset far beyond
#: any real atom index. A monomer would need a million atoms to collide.
ROLE_BASE = {"middle": 0, "head": 1_000_000, "tail": 2_000_000,
             "cap": 3_000_000}

#: Unit number used in provenance for atoms that belong to no repeat unit —
#: the terminal group the builder adds once the chain is finished, e.g. the
#: -OH that turns a polyester's terminal -C(=O)H into -C(=O)OH.
CAP_UNIT = -1


def role_key(role: str, monomer_index: int) -> int:
    """The single-integer key for ``monomer_index`` in a ``role`` unit."""
    if role not in ROLE_BASE:
        raise ValueError(f"unknown unit role {role!r}; expected one of {ROLES}")
    return ROLE_BASE[role] + int(monomer_index)


def split_role_key(key: int) -> Tuple[str, int]:
    """Inverse of :func:`role_key` — ``(role, monomer index)``."""
    key = int(key)
    for role in ("cap", "tail", "head"):
        if key >= ROLE_BASE[role]:
            return role, key - ROLE_BASE[role]
    return "middle", key


def split_by_role(types: Dict[int, str]) -> Dict[str, Dict[int, str]]:
    """Unfold a role-encoded map into ``{role: {monomer atom: type}}``.

    Maps written before end typing existed contain only plain indices, so they
    come back as pure ``middle`` — which is exactly how they used to behave.
    """
    out: Dict[str, Dict[int, str]] = {r: {} for r in ROLES}
    for key, value in (types or {}).items():
        role, index = split_role_key(key)
        out[role][index] = value
    return out


def merge_roles(by_role: Dict[str, Dict[int, str]]) -> Dict[int, str]:
    """Fold ``{role: {monomer atom: type}}`` back into one keyed map."""
    out: Dict[int, str] = {}
    for role in ROLES:
        for index, value in (by_role.get(role) or {}).items():
            out[role_key(role, index)] = value
    return out


@dataclass
class TrimerContext:
    """A three-unit oligomer and the provenance of every atom in it."""
    molecule: object
    #: ``trimer index -> (unit number 0/1/2, atom index in that monomer)``
    provenance: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    #: Atoms of the middle unit — the ones the user should be typing.
    middle: List[int] = field(default_factory=list)
    #: Atoms of the two outer units, which are the real chain ends.
    ends: List[int] = field(default_factory=list)
    #: The two bonds joining the units, as ``(i, j)`` trimer indices.
    link_bonds: List[Tuple[int, int]] = field(default_factory=list)
    #: Cap hydrogens still present at the far ends of the trimer.
    caps: List[int] = field(default_factory=list)
    #: Atoms of the terminal group the builder adds after assembly — for a
    #: polyester, the -OH that makes the chain end a carboxylic acid. They
    #: belong to no monomer, and until they existed here the view showed an
    #: aldehyde where the real chain has an acid.
    terminal_cap: List[int] = field(default_factory=list)
    junction_note: str = ""

    # ------------------------------------------------------------------
    def monomer_index(self, trimer_index: int) -> Optional[int]:
        """Which monomer atom a trimer atom came from."""
        entry = self.provenance.get(int(trimer_index))
        return None if entry is None else entry[1]

    def middle_to_monomer(self) -> Dict[int, int]:
        """``{trimer index: monomer index}`` for the middle unit only."""
        return {i: self.provenance[i][1] for i in self.middle}

    def types_by_monomer_index(self, types_by_trimer: Dict[int, str]
                               ) -> Dict[int, str]:
        """Fold types assigned on the trimer back onto monomer indices.

        Only the middle unit is used. An end unit's types describe the chain
        *ends*, which are a different thing and must not be written back as if
        they were the repeat unit.
        """
        out: Dict[int, str] = {}
        for trimer_index in self.middle:
            value = types_by_trimer.get(trimer_index)
            if value:
                out[self.provenance[trimer_index][1]] = value
        return out

    def types_by_role(self, types_by_trimer: Dict[int, str]
                      ) -> Dict[str, Dict[int, str]]:
        """Split trimer types into ``head`` / ``middle`` / ``tail`` sets.

        A chain has three kinds of unit and they genuinely need different
        types: the first keeps its head cap, the last keeps its tail cap, and
        everything between is the repeat unit. For isoprene that is 135 at the
        two ends where the middle wants 136 — but the rule is not about
        isoprene, it is about the cap, so this works for any polymer without
        knowing anything about its chemistry.

        Each set is keyed by **monomer** atom index, so a caller can apply the
        right one to each unit of a chain of any length.
        """
        out: Dict[str, Dict[int, str]] = {r: {} for r in ROLES}
        role_of = {0: "head", 1: "middle", 2: "tail", CAP_UNIT: "cap"}
        for trimer_index, (unit, monomer_index) in self.provenance.items():
            value = types_by_trimer.get(trimer_index)
            if value:
                out[role_of[unit]][monomer_index] = value
        return out

    def summary(self) -> str:
        return (f"trimer: {len(self.molecule.atoms)} atoms; "
                f"middle unit {len(self.middle)} atoms, "
                f"ends {len(self.ends)}, "
                f"{len(self.link_bonds)} junctions, "
                f"{len(self.caps)} terminal caps"
                + (f"; {self.junction_note}" if self.junction_note else ""))


# =====================================================================
#: The trimer is laid out along +x. Any fixed axis would do; what matters is
#: that every unit is aligned to the SAME one.
_AXIS = np.array([1.0, 0.0, 0.0])


def _rotation_aligning(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """A rotation matrix taking unit vector ``source`` onto ``target``."""
    v = np.cross(source, target)
    c = float(np.dot(source, target))
    s = float(np.linalg.norm(v))
    if s < 1e-12:                     # already parallel, or exactly opposed
        if c > 0:
            return np.eye(3)
        # 180 degrees: rotate about any axis perpendicular to source.
        perp = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, perp)
        axis = axis / np.linalg.norm(axis)
        return _roll_about(axis, np.pi)
    kmat = np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1.0 - c) / (s * s))


def _roll_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation of ``angle`` radians about ``axis`` (Rodrigues)."""
    axis = axis / np.linalg.norm(axis)
    kmat = np.array([[0.0, -axis[2], axis[1]],
                     [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])
    return (np.eye(3) + np.sin(angle) * kmat
            + (1.0 - np.cos(angle)) * (kmat @ kmat))


#: Roll angles tried when seating each unit. 24 is a 15-degree step: fine
#: enough to find a clear orientation, coarse enough to stay instant.
_ROLL_STEPS = 24

#: Clearances a seated unit must reach, in angstrom. Two heavy atoms closer
#: than this are drawn inside one another; hydrogens are allowed nearer
#: because they are small and drawn small.
_MIN_HEAVY_GAP = 2.2
_MIN_ANY_GAP = 1.5

#: How far a unit may be nudged along the axis to find room, and in what
#: steps. The junction bond stretches by this much, so the cap is deliberately
#: tight: this is a trade between a slightly long bond and atoms drawn inside
#: one another, and it was chosen by measuring both across nine polymers.
#: Allowing 3 A of nudge cleared everything but stretched junctions to 3.0 A,
#: which reads as a gap rather than a bond. At 0.75 A the worst heavy-atom
#: contact is 2.2 A and the longest junction bond 2.3 A -- both readable.
_MAX_NUDGE = 0.75
_NUDGE_STEP = 0.25


#: Tilting each unit about its head atom was tried here, on the reasoning that
#: it would move a unit clear without stretching the junction bond at all. It
#: did not: the contacts that matter are AT the junction, right at the pivot,
#: where a rotation about the head barely moves anything. Measured across nine
#: polymers it was neutral at best and made PMMA and the epoxide worse, so it
#: was removed rather than kept as plausible-looking machinery.


def _seat_unit(surviving, elements, head, align, placement,
               placed: List[np.ndarray], placed_heavy: List[bool],
               skip_local: Optional[set] = None,
               skip_placed: Optional[set] = None):
    """Choose a roll and an axial nudge that keep this unit clear.

    Returns ``(rotation_matrix, nudge_angstrom)``.

    Aligning each unit to the chain axis fixes its direction but not its spin,
    and spin alone is not always enough: a monomer embedded in a coiled
    conformation extends past its own head along the axis, so it reaches into
    its neighbour no matter how it is turned. Polystyrene and PBS both did.

    So the roll is searched first, and only if the best roll still leaves
    atoms inside one another is the unit slid along the axis — by the smallest
    amount that clears it. Nothing is moved that does not need to be.
    """
    if not placed:
        return align, 0.0      # first unit: nothing to avoid

    existing = np.asarray(placed)
    existing_heavy = np.asarray(placed_heavy)
    # Raw offsets from the head atom. The alignment is part of the candidate
    # rotation built below, so applying it here too would align twice.
    local = np.asarray([a.xyz - head for a in surviving])
    is_heavy = np.asarray([e != "H" for e in elements])
    heavy_pairs = is_heavy[:, None] & existing_heavy[None, :]

    # The two atoms about to be BONDED across this junction are not clashing —
    # they are doing their job. Counting that pair as a clash is what made
    # every junction bond come out at 2.79 A instead of 1.54: the seater kept
    # sliding the unit away to "fix" a contact it was supposed to create.
    #
    # ONLY that one pair is exempt. Exempting their neighbours too was tried
    # and was far too generous: it let a polystyrene ring carbon sit 1.66 A
    # from the previous unit unnoticed. Genuine 1-3 contacts across a junction
    # are ~2.5 A, which clears the threshold below on its own merits.
    considered = np.ones((len(surviving), len(placed)), dtype=bool)
    for i in (skip_local or set()):
        for j in (skip_placed or set()):
            considered[i, j] = False
    heavy_pairs = heavy_pairs & considered
    if not considered.any():
        return align, 0.0

    def orientation(roll: float) -> np.ndarray:
        return _roll_about(_AXIS, roll) @ align

    def clearances(rotation: np.ndarray, nudge: float):
        moved = local @ rotation.T + placement + _AXIS * nudge
        gaps = np.linalg.norm(moved[:, None, :] - existing[None, :, :], axis=2)
        heavy = float(gaps[heavy_pairs].min()) if heavy_pairs.any() else 1e9
        return float(gaps[considered].min()), heavy

    rolls = [2.0 * np.pi * s / _ROLL_STEPS for s in range(_ROLL_STEPS)]

    # Tried in order of increasing distortion: every roll at no nudge first,
    # then progressively larger nudges. The first arrangement that clears
    # wins, so an easy monomer keeps its ideal bond length and only an awkward
    # one pays for the room it needs.
    best_fallback = (np.eye(3), 0.0, -1e9)
    nudge = 0.0
    while nudge <= _MAX_NUDGE + 1e-9:
        for roll in rolls:
            rotation = orientation(roll)
            any_gap, heavy_gap = clearances(rotation, nudge)
            if heavy_gap >= _MIN_HEAVY_GAP and any_gap >= _MIN_ANY_GAP:
                return rotation, nudge
            score = min(heavy_gap - _MIN_HEAVY_GAP, any_gap - _MIN_ANY_GAP)
            if score > best_fallback[2]:
                best_fallback = (rotation, nudge, score)
        nudge += _NUDGE_STEP

    # Nothing fully clears — take the roomiest found rather than the first
    # tried, and say so, because a cramped picture is worth knowing about.
    rotation, nudge, _score = best_fallback
    log.info("Trimer unit could not be seated with full clearance; using the "
             "roomiest orientation found (nudged %.2f A).", nudge)
    return rotation, nudge


def _is_carbonyl_carbon(mol, index: Optional[int]) -> bool:
    """True if ``index`` is a carbon carrying a double-bonded oxygen."""
    if index is None or not (0 <= index < len(mol.atoms)):
        return False
    if mol.atoms[index].element != "C":
        return False
    for i, j, order in mol.bonds:
        if i == index and mol.atoms[j].element == "O" and float(order) == 2.0:
            return True
        if j == index and mol.atoms[i].element == "O" and float(order) == 2.0:
            return True
    return False


def _apply_carboxyl_cap(mol, tail_atom: Optional[int],
                        cap_h: Optional[int]) -> List[int]:
    """Turn a terminal ``-C(=O)H`` into ``-C(=O)OH``. Returns the cap atoms.

    This mirrors what the chain builder does once a polyester chain is
    finished. Doing it here too is the difference between typing the molecule
    you will get and typing an aldehyde that never exists: PAAF has to present
    the acid end as an aldehyde so there is a hydrogen to remove when linking,
    but that is a device for building, not the chain's chemistry.

    Returns ``[]`` for anything that is not a polyester end, so polyethylene
    and the rest are untouched.
    """
    from .structure import Atom

    if cap_h is None or not _is_carbonyl_carbon(mol, tail_atom):
        return []
    if mol.atoms[cap_h].element != "H":
        return []

    carbon = mol.atoms[tail_atom].xyz
    direction = mol.atoms[cap_h].xyz - carbon
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return []
    direction = direction / norm

    # The aldehyde hydrogen BECOMES the hydroxyl oxygen, in place, so every
    # other index is untouched; the acid's hydrogen is then appended.
    mol.atoms[cap_h].element = "O"
    mol.atoms[cap_h].xyz = carbon + direction * 1.34
    mol.atoms[cap_h].name = ""
    mol.atoms[cap_h].ff_type = None

    h_index = len(mol.atoms)
    mol.atoms.append(Atom(index=h_index, element="H",
                          xyz=mol.atoms[cap_h].xyz + direction * 0.97))
    mol.bonds.append((cap_h, h_index, 1.0))
    return [cap_h, h_index]


def build_trimer(monomers: Sequence, bond_length: float = 1.54,
                 cap_carboxyl_end: bool = True) -> TrimerContext:
    """Join three monomers head-to-tail and record where each atom came from.

    ``monomers`` is one :class:`~paaf.monomer.Monomer` (repeated three times)
    or exactly three, which is how an A–B–A copolymer junction is built.

    The construction mirrors the chain builder: cap atoms are removed at each
    junction and kept at the two far ends, so the middle unit has precisely
    the composition it will have in the chain.
    """
    from .structure import Atom, Molecule

    seq = list(monomers)
    if len(seq) == 1:
        seq = seq * 3
    if len(seq) != 3:
        raise ValueError(f"build_trimer needs 1 or 3 monomers, got {len(seq)}")

    atoms: List = []
    bonds: List[Tuple[int, int, float]] = []
    provenance: Dict[int, Tuple[int, int]] = {}
    removed: set = set()
    link_pairs: List[Tuple[int, int]] = []

    offset = 0
    prev_tail: Optional[int] = None
    placement = np.zeros(3)           # where the next unit's head atom goes
    placed: List[np.ndarray] = []     # coordinates already committed
    placed_heavy: List[bool] = []     # is each committed atom a heavy atom?
    prev_junction: List[int] = []     # placed-list positions of the last tail

    for unit, m in enumerate(seq):
        mol = m.molecule
        head = mol.atoms[m.head_index].xyz
        tail = mol.atoms[m.tail_index].xyz
        direction = tail - head
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            raise ValueError(f"Monomer {m.name}: head and tail coincide")

        # Each unit is ROTATED so its head->tail vector lies along the chain
        # axis, then placed after the previous one.
        #
        # This used to be a translation only, which left every copy in the
        # conformation RDKit happened to embed and simply moved it along the
        # previous unit's head->tail direction. For a small monomer that looks
        # fine; for a folded one it is badly wrong. Measured on PBS the units
        # came within 1.75 A of each other — atoms sitting inside atoms — which
        # is the tangled, dangling picture the 3D view was showing. Aligning
        # each unit to a common axis makes the trimer a chain rather than three
        # molecules dropped near one another.
        align = _rotation_aligning(direction / norm, _AXIS)

        # Which of this unit's atoms will still be there once the junction
        # caps are consumed — those are the ones that must not collide.
        doomed = set()
        if unit > 0:
            doomed.update(int(h) for h in (m.head_removes or []))
        if unit < 2:
            doomed.update(int(t) for t in (m.tail_removes or []))
        surviving = [a for a in mol.atoms if a.index not in doomed]

        # Spin the unit about the chain axis to the angle that keeps it
        # furthest from what is already placed.
        #
        # Aligning alone is not enough: it fixes the direction of the unit but
        # not its roll, and an arbitrary roll drives side groups and junction
        # hydrogens straight into the previous unit. A fixed quarter-turn per
        # unit measured WORSE than no rotation at all (0.55 A contacts on
        # polystyrene), which is what a guess deserves. This picks the roll on
        # the only criterion that matters.
        # The head of this unit bonds to the tail of the previous one, so
        # neither they nor their immediate neighbours count as clashes.
        local_pos = {a.index: k for k, a in enumerate(surviving)}
        skip_local = ({local_pos[m.head_index]}
                      if m.head_index in local_pos else set())
        skip_placed = set(prev_junction) if unit > 0 else set()

        rotation, nudge = _seat_unit([a for a in surviving],
                                     [a.element for a in surviving],
                                     head, align, placement,
                                     placed, placed_heavy,
                                     skip_local=skip_local,
                                     skip_placed=skip_placed)
        placement = placement + _AXIS * nudge

        for a in mol.atoms:
            xyz = rotation @ (a.xyz - head) + placement
            atoms.append(Atom(index=a.index + offset, element=a.element,
                              xyz=xyz, name=a.name,
                              charge=a.charge, ff_type=a.ff_type))
            provenance[a.index + offset] = (unit, a.index)
        # Remember where this unit's tail (and its neighbours) landed in the
        # placed list, so the NEXT unit knows which contacts are its junction.
        tail_group = {m.tail_index}
        prev_junction = []
        for a in surviving:
            if a.index in tail_group:
                prev_junction.append(len(placed))
            placed.append(rotation @ (a.xyz - head) + placement)
            placed_heavy.append(a.element != "H")

        # The next unit's head goes one bond length on from where THIS unit's
        # tail actually ended up. Measuring from the real tail position rather
        # than from an arithmetic cursor is what keeps the junction bond at
        # exactly ``bond_length`` however far the unit was tilted.
        tail_pos = rotation @ (tail - head) + placement
        placement = tail_pos + _AXIS * bond_length
        for i, j, order in mol.bonds:
            bonds.append((i + offset, j + offset, order))

        if prev_tail is not None:
            link_pairs.append((prev_tail, m.head_index + offset))
            bonds.append((prev_tail, m.head_index + offset, 1.0))

        # Caps vanish at a junction and survive at the two far ends — which is
        # exactly what makes the middle unit's link carbons CH2 and the outer
        # units' CH3.
        if unit > 0:
            removed.update(h + offset for h in m.head_removes)
        if unit < 2:
            removed.update(t + offset for t in m.tail_removes)

        prev_tail = m.tail_index + offset
        offset += len(mol.atoms)

    # ---- drop the consumed caps and renumber ---------------------------
    keep = [a for a in atoms if a.index not in removed]
    remap = {a.index: k for k, a in enumerate(keep)}
    new_prov = {remap[a.index]: provenance[a.index] for a in keep}
    for k, a in enumerate(keep):
        a.index = k
    new_bonds = [(remap[i], remap[j], o) for (i, j, o) in bonds
                 if i in remap and j in remap]

    trimer = Molecule(atoms=keep, bonds=new_bonds, name="typing-trimer")

    # ---- give the chain its real terminal group -------------------------
    terminal_cap: List[int] = []
    if cap_carboxyl_end:
        tail_atom = remap.get(seq[2].tail_index + offset - len(seq[2].molecule.atoms))
        cap_h = None
        for c in (seq[2].tail_removes or []):
            original = c + offset - len(seq[2].molecule.atoms)
            if original in remap:
                cap_h = remap[original]
                break
        terminal_cap = _apply_carboxyl_cap(trimer, tail_atom, cap_h)
        for k, index in enumerate(terminal_cap):
            new_prov[index] = (CAP_UNIT, k)

    ctx = TrimerContext(molecule=trimer, provenance=new_prov)
    ctx.terminal_cap = terminal_cap
    ctx.middle = sorted(i for i, (u, _k) in new_prov.items() if u == 1)
    ctx.ends = sorted(i for i, (u, _k) in new_prov.items() if u != 1)
    ctx.link_bonds = [(remap[i], remap[j]) for i, j in link_pairs
                      if i in remap and j in remap]

    # Cap hydrogens still present: those the outer units kept.
    caps: List[int] = []
    for k, m in ((0, seq[0]), (2, seq[2])):
        for c in (m.head_removes if k == 0 else m.tail_removes):
            base = 0 if k == 0 else len(seq[0].molecule.atoms) + len(seq[1].molecule.atoms)
            original = c + base
            if original in remap:
                caps.append(remap[original])
    ctx.caps = sorted(set(caps))

    names = [getattr(m, "name", "?") for m in seq]
    if len(set(names)) == 1:
        ctx.junction_note = f"homopolymer junction {names[0]}-{names[0]}"
    else:
        ctx.junction_note = (f"copolymer junction {names[0]}-{names[1]}-"
                             f"{names[2]}; other neighbour pairings will "
                             f"differ")

    log.info("Typing context: %s", ctx.summary())
    return ctx
