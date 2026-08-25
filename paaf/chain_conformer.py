"""Give a built chain a realistic shape before it is minimised.

Why minimisation is not enough
------------------------------
:func:`paaf.chain_builder.simple_backend` places each repeat unit by rigid
translation along one fixed direction. The result is a perfectly collinear
rod: end-to-end distance equal to contour length, every backbone torsion
identical. mBuild's port-based assembly is better but still close to
all-trans.

Handing that to MMFF94 or UFF does not help, and this is the part that looks
like a bug and is not. Conjugate-gradient minimisation follows the energy
gradient downhill. Going from *trans* to *gauche* means climbing a barrier of
roughly 3 kcal/mol — uphill — so the minimiser cannot do it. An extended chain
is a genuine local minimum. The optimiser runs, reports a real energy drop
from fixing bond lengths and angles, and leaves the chain straight.

So the conformation has to be generated, not minimised into existence.

What this does
--------------
Walks the backbone and assigns each rotatable bond a torsion drawn from the
rotational-isomeric-state populations of the chain — *trans* near 180° and
two *gauche* states near ±60°, with Boltzmann weights at the build
temperature. Each rotation is applied to everything downstream of the bond,
and rejected if it drives two non-bonded atoms closer than a cutoff, which is
what keeps the chain from passing through itself.

This is the same physics the amorphous-cell builder already uses; the
difference is that the cell builder grows a chain bond-by-bond with these
statistics, while here the chain already exists and only its torsions are
re-drawn.

After this, minimisation does what it is good at: cleaning up bond lengths,
angles and any residual close contacts, without being asked to solve a
sampling problem it cannot solve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["ConformerSettings", "ConformerResult", "randomise_backbone",
           "rebuild_backbone", "backbone_path", "chain_dimensions",
           "backbone_angles"]

_R_KCAL = 1.98720425e-3          # kcal / (mol K)

#: Torsional states of an sp3–sp3 backbone bond, with the energy of each
#: relative to *trans*. The gauche penalty is the standard n-butane value;
#: it is what makes trans about twice as likely as either gauche at 300 K,
#: and hence what sets the chain's stiffness.
_STATES: Tuple[Tuple[float, float], ...] = (
    (180.0, 0.00),      # trans
    (60.0, 0.55),       # gauche+
    (-60.0, 0.55),      # gauche-
)


@dataclass
class ConformerSettings:
    """How the torsions are drawn and how hard self-overlap is enforced."""
    enabled: bool = True
    temperature_k: float = 300.0
    #: Minimum separation allowed between atoms more than three bonds apart.
    min_contact_a: float = 2.2
    #: Attempts per bond before accepting the least-bad state and moving on.
    tries_per_bond: int = 12
    seed: Optional[int] = None
    #: Leave this many bonds at each chain end alone; end groups are short and
    #: rotating them buys nothing but overlap failures.
    skip_ends: int = 1
    #: Backbone valence angle used when the built chain's own angle is
    #: unphysical. ``simple_backend`` places each unit by translating along the
    #: monomer's head-to-tail vector, which makes every backbone angle exactly
    #: 180° — a straight line no torsion can bend, because the atoms lie on
    #: the rotation axis. 112° is the sp3 C–C–C value.
    backbone_angle_deg: float = 112.0
    #: Draw torsions from the RIS populations. Turning this off keeps the
    #: chain all-trans (extended) while STILL repairing unphysical valence
    #: angles — a 180° backbone is not a molecule, whatever shape is wanted.
    sample_torsions: bool = True
    #: Angles above this are treated as the degenerate case above and replaced.
    #: Below it the built geometry is kept, so a good mBuild chain is not
    #: overwritten with a generic one.
    straight_angle_threshold_deg: float = 150.0


@dataclass
class ConformerResult:
    ran: bool = False
    n_rotatable: int = 0
    n_rotated: int = 0
    n_rejected: int = 0
    r_end_to_end_before: float = 0.0
    r_end_to_end_after: float = 0.0
    contour_length: float = 0.0
    n_angles_fixed: int = 0
    mean_angle_before: float = 0.0
    mean_angle_after: float = 0.0
    message: str = ""

    def summary(self) -> str:
        if not self.ran:
            return f"conformer sampling not run — {self.message}"
        before = (self.r_end_to_end_before / self.contour_length
                  if self.contour_length else 0.0)
        after = (self.r_end_to_end_after / self.contour_length
                 if self.contour_length else 0.0)
        parts = [f"sampled {self.n_rotated}/{self.n_rotatable} backbone "
                 f"torsions ({self.n_rejected} rejected for overlap)"]
        if self.n_angles_fixed:
            parts.append(f"replaced {self.n_angles_fixed} straight backbone "
                         f"angles ({self.mean_angle_before:.0f}° -> "
                         f"{self.mean_angle_after:.0f}°)")
        parts.append(f"R/L {before:.3f} -> {after:.3f}")
        return "; ".join(parts)


# =====================================================================
def _adjacency(n_atoms: int, bonds: Sequence[Tuple[int, int, float]]
               ) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n_atoms)]
    for i, j, _o in bonds:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def backbone_path(molecule) -> List[int]:
    """The longest chain of heavy atoms through the molecule.

    Found as the graph diameter over heavy atoms only: two breadth-first
    sweeps, the first to find an extremal atom and the second to find the
    atom furthest from it. Hydrogens and side groups are excluded, so what
    comes back is the backbone rather than the longest path through a
    substituent.
    """
    atoms = molecule.atoms
    heavy = [k for k, a in enumerate(atoms) if a.element != "H"]
    if len(heavy) < 4:
        return []
    heavy_set = set(heavy)
    adj = _adjacency(len(atoms), molecule.bonds)

    def _bfs(start: int) -> Tuple[int, Dict[int, int]]:
        seen = {start: -1}
        queue = [start]
        last = start
        while queue:
            node = queue.pop(0)
            last = node
            for nb in adj[node]:
                if nb in heavy_set and nb not in seen:
                    seen[nb] = node
                    queue.append(nb)
        return last, seen

    far_a, _ = _bfs(heavy[0])
    far_b, parents = _bfs(far_a)

    path = [far_b]
    while parents[path[-1]] != -1:
        path.append(parents[path[-1]])
    path.reverse()
    return path


def chain_dimensions(molecule, path: Optional[Sequence[int]] = None
                     ) -> Tuple[float, float]:
    """``(end-to-end distance, contour length)`` along the backbone, in Å."""
    path = list(path) if path is not None else backbone_path(molecule)
    if len(path) < 2:
        return 0.0, 0.0
    xyz = np.array([molecule.atoms[k].xyz for k in path], dtype=float)
    r = float(np.linalg.norm(xyz[-1] - xyz[0]))
    contour = float(sum(np.linalg.norm(xyz[i + 1] - xyz[i])
                        for i in range(len(xyz) - 1)))
    return r, contour


def _side_of_bond(n_atoms: int, adj: Sequence[Sequence[int]],
                  anchor: int, moving: int) -> Set[int]:
    """Every atom reachable from ``moving`` without going back through ``anchor``.

    That is the half of the molecule a rotation about ``anchor–moving`` should
    carry. If the search comes back to ``anchor``, the bond is inside a ring
    and cannot be rotated independently — the caller is told by returning an
    empty set.
    """
    seen = {moving}
    stack = [moving]
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if nb == anchor:
                if node != moving:
                    return set()          # ring: rotation is not free
                continue
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen


def _dihedral(p0, p1, p2, p3) -> float:
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    norm = np.linalg.norm(b1)
    if norm < 1e-9:
        return 0.0
    b1 = b1 / norm
    v = b0 - b0.dot(b1) * b1
    w = b2 - b2.dot(b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v).dot(w), v.dot(w))))


def _rotate(xyz: np.ndarray, indices: Sequence[int], origin: np.ndarray,
            axis: np.ndarray, degrees: float) -> None:
    """Rotate ``indices`` about ``axis`` through ``origin``, in place."""
    norm = np.linalg.norm(axis)
    if norm < 1e-9 or abs(degrees) < 1e-9:
        return
    k = axis / norm
    theta = np.radians(degrees)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    idx = np.fromiter(indices, dtype=int)
    v = xyz[idx] - origin
    # Rodrigues' rotation formula.
    xyz[idx] = (v * cos_t
                + np.cross(k, v) * sin_t
                + k * np.outer(v.dot(k), np.ones(3)) * (1.0 - cos_t)
                + origin)


def _worst_contact(xyz: np.ndarray, moved: Sequence[int],
                   fixed: Sequence[int], exclude: Set[Tuple[int, int]]
                   ) -> float:
    """Closest approach between a moved atom and a fixed one, in Å."""
    if not len(moved) or not len(fixed):
        return np.inf
    a = xyz[np.fromiter(moved, dtype=int)]
    b = xyz[np.fromiter(fixed, dtype=int)]
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    if exclude:
        mv = list(moved)
        fx = list(fixed)
        pos_m = {v: i for i, v in enumerate(mv)}
        pos_f = {v: i for i, v in enumerate(fx)}
        for i, j in exclude:
            if i in pos_m and j in pos_f:
                d[pos_m[i], pos_f[j]] = np.inf
            if j in pos_m and i in pos_f:
                d[pos_m[j], pos_f[i]] = np.inf
    return float(d.min())


def _neighbours_within(adj: Sequence[Sequence[int]], depth: int
                       ) -> Set[Tuple[int, int]]:
    """Atom pairs separated by ``depth`` bonds or fewer.

    Their distances are set by the bonded terms, not by packing, so they must
    not be judged against a non-bonded contact cutoff.
    """
    pairs: Set[Tuple[int, int]] = set()
    for start in range(len(adj)):
        frontier = {start}
        seen = {start}
        for _ in range(depth):
            nxt: Set[int] = set()
            for node in frontier:
                for nb in adj[node]:
                    if nb not in seen:
                        seen.add(nb)
                        nxt.add(nb)
            frontier = nxt
        for other in seen:
            if other != start:
                pairs.add((start, other))
    return pairs


# =====================================================================
def backbone_angles(molecule, path: Optional[Sequence[int]] = None
                    ) -> np.ndarray:
    """Valence angles along the backbone, in degrees."""
    path = list(path) if path is not None else backbone_path(molecule)
    if len(path) < 3:
        return np.zeros(0)
    xyz = np.array([molecule.atoms[k].xyz for k in path], dtype=float)
    out = []
    for i in range(1, len(xyz) - 1):
        u = xyz[i - 1] - xyz[i]
        v = xyz[i + 1] - xyz[i]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-9 or nv < 1e-9:
            continue
        out.append(np.degrees(np.arccos(np.clip(u.dot(v) / (nu * nv), -1, 1))))
    return np.array(out)


def _own_side_groups(n_atoms: int, adj: Sequence[Sequence[int]],
                     path: Sequence[int]) -> Dict[int, List[int]]:
    """Assign every non-backbone atom to the backbone atom it hangs off.

    A multi-source breadth-first sweep outwards from the backbone, so each
    substituent goes to its nearest backbone attachment point and the whole
    group travels together when that atom moves.
    """
    on_path = {k: i for i, k in enumerate(path)}
    owner: Dict[int, int] = {}
    queue: List[Tuple[int, int]] = [(k, i) for k, i in on_path.items()]
    seen = set(on_path)
    while queue:
        node, home = queue.pop(0)
        for nb in adj[node]:
            if nb in seen:
                continue
            seen.add(nb)
            owner[nb] = home
            queue.append((nb, home))
    groups: Dict[int, List[int]] = {i: [] for i in range(len(path))}
    for atom, home in owner.items():
        groups[home].append(atom)
    return groups


def _local_frame(xyz: np.ndarray, prev_i: Optional[int], this_i: int,
                 next_i: Optional[int]) -> np.ndarray:
    """An orthonormal 3x3 basis attached to one backbone atom.

    Built from the bonds to its backbone neighbours, so it rotates with the
    chain. Side-group positions are stored in this basis and rewritten from
    the rebuilt one, which moves each group rigidly — bond lengths and angles
    inside it are preserved exactly.
    """
    origin = xyz[this_i]
    if next_i is not None:
        e1 = xyz[next_i] - origin
    else:
        e1 = origin - xyz[prev_i]                     # type: ignore[index]
    n1 = np.linalg.norm(e1)
    e1 = e1 / n1 if n1 > 1e-9 else np.array([1.0, 0.0, 0.0])

    ref = (xyz[prev_i] - origin) if prev_i is not None else np.array([0., 0., 1.])
    e2 = ref - ref.dot(e1) * e1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-9:                    # collinear: any perpendicular will do
        trial = np.array([0.0, 0.0, 1.0])
        if abs(trial.dot(e1)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        e2 = trial - trial.dot(e1) * e1
        n2 = np.linalg.norm(e2)
    e2 = e2 / n2
    return np.array([e1, e2, np.cross(e1, e2)])


def _place_nerf(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                bond: float, angle_deg: float, torsion_deg: float) -> np.ndarray:
    """Natural Extension Reference Frame: the next atom from internal coords."""
    theta = np.radians(angle_deg)
    phi = np.radians(torsion_deg)
    d2 = np.array([-bond * np.cos(theta),
                   bond * np.sin(theta) * np.cos(phi),
                   bond * np.sin(theta) * np.sin(phi)])
    bc = c - b
    n_bc = np.linalg.norm(bc)
    bc = bc / n_bc if n_bc > 1e-9 else np.array([1.0, 0.0, 0.0])
    n = np.cross(b - a, bc)
    n_n = np.linalg.norm(n)
    n = n / n_n if n_n > 1e-9 else np.array([0.0, 0.0, 1.0])
    m = np.array([bc, np.cross(n, bc), n]).T
    return c + m.dot(d2)


def rebuild_backbone(molecule, settings: Optional[ConformerSettings] = None
                     ) -> ConformerResult:
    """Rebuild the backbone with real valence angles and sampled torsions.

    This is the one that actually works on a chain from
    :func:`~paaf.chain_builder.simple_backend`. That backend translates each
    repeat unit along the previous unit's head-to-tail vector, so consecutive
    backbone bonds are parallel and every valence angle is exactly 180°. The
    chain is a straight rod, and no amount of torsional rotation will bend it:
    the atoms lie *on* the rotation axis, so they do not move.

    Bond lengths are taken from the built chain and preserved. Angles are
    replaced only where they are unphysically straight, so a well-built
    mBuild chain keeps its own geometry. Torsions are drawn from the
    rotational-isomeric-state populations, with a self-avoidance check that
    rejects a state driving the new atom into one already placed.

    Side groups are carried rigidly in a frame attached to their backbone
    atom, so their internal geometry is untouched.
    """
    settings = settings or ConformerSettings()
    res = ConformerResult()
    if not settings.enabled:
        res.message = "switched off"
        return res

    path = backbone_path(molecule)
    if len(path) < 5:
        res.message = f"backbone is only {len(path)} atoms long"
        return res

    xyz = np.array([a.xyz for a in molecule.atoms], dtype=float)
    adj = _adjacency(len(molecule.atoms), molecule.bonds)
    res.r_end_to_end_before, res.contour_length = chain_dimensions(molecule, path)
    before = backbone_angles(molecule, path)
    res.mean_angle_before = float(before.mean()) if len(before) else 0.0

    # --- remember every side group in its own backbone atom's frame ------
    groups = _own_side_groups(len(molecule.atoms), adj, path)
    local: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for i, k in enumerate(path):
        frame = _local_frame(xyz, path[i - 1] if i > 0 else None, k,
                             path[i + 1] if i + 1 < len(path) else None)
        local[i] = [(atom, frame.dot(xyz[atom] - xyz[k]))
                    for atom in groups.get(i, [])]

    # --- bond lengths and angles from the built chain --------------------
    bonds = [float(np.linalg.norm(xyz[path[i + 1]] - xyz[path[i]]))
             for i in range(len(path) - 1)]
    angles = []
    for i, value in enumerate(before):
        if value >= settings.straight_angle_threshold_deg:
            angles.append(settings.backbone_angle_deg)
            res.n_angles_fixed += 1
        else:
            angles.append(float(value))

    # --- regrow the backbone ---------------------------------------------
    rng = np.random.default_rng(settings.seed)
    weights = np.array([np.exp(-e / (_R_KCAL * max(settings.temperature_k, 1.0)))
                        for _a, e in _STATES], dtype=float)
    weights /= weights.sum()
    state_angles = np.array([a for a, _e in _STATES], dtype=float)

    new = np.zeros((len(path), 3), dtype=float)
    new[0] = np.zeros(3)
    new[1] = new[0] + np.array([bonds[0], 0.0, 0.0])
    theta0 = np.radians(angles[0] if angles else settings.backbone_angle_deg)
    new[2] = new[1] + bonds[1] * np.array([-np.cos(theta0), np.sin(theta0), 0.0])

    for i in range(3, len(path)):
        res.n_rotatable += 1
        best, best_gap = None, -np.inf
        placed = new[:i]
        tries = max(1, settings.tries_per_bond) if settings.sample_torsions else 1
        for _ in range(tries):
            phi = (float(rng.choice(state_angles, p=weights))
                   if settings.sample_torsions else 180.0)
            cand = _place_nerf(new[i - 3], new[i - 2], new[i - 1],
                               bonds[i - 1], angles[i - 2], phi)
            # Only atoms more than three bonds back can clash meaningfully.
            gap = (np.linalg.norm(placed[:-3] - cand, axis=1).min()
                   if i > 3 else np.inf)
            if gap >= settings.min_contact_a:
                best, best_gap = cand, gap
                break
            res.n_rejected += 1
            if gap > best_gap:
                best, best_gap = cand, gap
        new[i] = best if best is not None else _place_nerf(
            new[i - 3], new[i - 2], new[i - 1], bonds[i - 1],
            angles[i - 2], 180.0)
        res.n_rotated += 1

    # --- write the backbone back, then carry the side groups -------------
    for i, k in enumerate(path):
        xyz[k] = new[i]
    for i, k in enumerate(path):
        frame = _local_frame(xyz, path[i - 1] if i > 0 else None, k,
                             path[i + 1] if i + 1 < len(path) else None)
        for atom, offset in local[i]:
            xyz[atom] = xyz[k] + frame.T.dot(offset)

    for k, atom in enumerate(molecule.atoms):
        atom.xyz[:] = xyz[k]

    res.r_end_to_end_after, _ = chain_dimensions(molecule, path)
    after = backbone_angles(molecule, path)
    res.mean_angle_after = float(after.mean()) if len(after) else 0.0
    res.ran = True
    log.info("Conformer: %s", res.summary())
    return res


def randomise_backbone(molecule, settings: Optional[ConformerSettings] = None
                       ) -> ConformerResult:
    """Draw new backbone torsions for ``molecule``, in place.

    Returns a :class:`ConformerResult`. The molecule is left untouched — and
    the result says so — when there is no backbone to speak of, so a small
    monomer passed here by mistake is not silently mangled.
    """
    settings = settings or ConformerSettings()
    res = ConformerResult()

    if not settings.enabled:
        res.message = "switched off"
        return res

    path = backbone_path(molecule)
    if len(path) < 5:
        res.message = (f"backbone is only {len(path)} atoms long; nothing "
                       f"worth sampling")
        return res

    xyz = np.array([a.xyz for a in molecule.atoms], dtype=float)
    adj = _adjacency(len(molecule.atoms), molecule.bonds)
    res.r_end_to_end_before, res.contour_length = chain_dimensions(molecule, path)

    rng = np.random.default_rng(settings.seed)
    weights = np.array([np.exp(-e / (_R_KCAL * max(settings.temperature_k, 1.0)))
                        for _angle, e in _STATES], dtype=float)
    weights /= weights.sum()
    angles = np.array([a for a, _e in _STATES], dtype=float)

    close_pairs = _neighbours_within(adj, 3)

    start = 1 + max(0, settings.skip_ends)
    stop = len(path) - 2 - max(0, settings.skip_ends)
    for pos in range(start, stop):
        a, b, c, d = path[pos - 1], path[pos], path[pos + 1], path[pos + 2]
        moving = _side_of_bond(len(molecule.atoms), adj, b, c)
        if not moving:
            continue                        # ring bond: not freely rotatable
        moving.discard(c)
        moving = {m for m in moving if m != b}
        if not moving:
            continue
        res.n_rotatable += 1

        fixed = [k for k in range(len(molecule.atoms)) if k not in moving]
        current = _dihedral(xyz[a], xyz[b], xyz[c], xyz[d])
        moved_list = sorted(moving)

        best_delta, best_contact = None, -np.inf
        for _ in range(max(1, settings.tries_per_bond)):
            target = float(rng.choice(angles, p=weights))
            delta = target - current
            _rotate(xyz, moved_list, xyz[c], xyz[c] - xyz[b], delta)
            contact = _worst_contact(xyz, moved_list, fixed, close_pairs)
            if contact >= settings.min_contact_a:
                best_delta, best_contact = delta, contact
                break
            # Undo and try another state.
            _rotate(xyz, moved_list, xyz[c], xyz[c] - xyz[b], -delta)
            res.n_rejected += 1
            if contact > best_contact:
                best_delta, best_contact = delta, contact

        if best_delta is None:
            continue
        if best_contact < settings.min_contact_a:
            # Every state overlapped. Take the least-bad one rather than
            # leaving this bond all-trans, and let minimisation clean it up.
            _rotate(xyz, moved_list, xyz[c], xyz[c] - xyz[b], best_delta)
        res.n_rotated += 1

    for k, atom in enumerate(molecule.atoms):
        atom.xyz[:] = xyz[k]

    res.r_end_to_end_after, _ = chain_dimensions(molecule, path)
    res.ran = True
    log.info("Conformer: %s", res.summary())
    return res
