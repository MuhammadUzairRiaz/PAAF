"""Remove residual close contacts from a grown bead cell.

Why this exists
---------------
Chain growth (:mod:`paaf.cell.grow`) is a sequential random walk, and a walk
into a crowded box can reach a step where every rotational isomeric state is
blocked. Backtracking resolves nearly all of those. The rare ones it cannot —
a chain tip caged by its neighbours, or a density the user asked for that is
simply very high — used to abort the whole build. The grower now places the
least-crowded state instead and records it; this module then pushes those
contacts apart, so a build always finishes and always says how clean it is.

What it does
------------
FIRE minimisation of a purely repulsive, bounded penalty

.. math::  U = \\tfrac12 k_{nb} \\sum_{r_{ij} < d_{ij}} (d_{ij} - r_{ij})^2
          + \\tfrac12 k_b \\sum (r - l)^2 + \\tfrac12 k_\\theta \\sum (r_{13} - d_{13})^2

where :math:`d_{ij}` is exactly the overlap criterion growth uses
(:class:`paaf.cell.packing.NeighbourGrid`), bonds are held at the RIS bond
length and angles through their 1-3 distance. Torsions are free: that is the
degree of freedom a real chain uses to get out of the way.

The penalty is harmonic, never singular, so a deep overlap cannot launch a
bead across the box; per-step displacement is capped as well. When the
contacts are gone, bond lengths and angles are projected back onto their exact
values (SHAKE), so the geometry the back-mapper relies on is preserved.

It is a construction aid, not a force field. Energies here mean nothing
physically; the real relaxation happens later, under the chosen force field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .packing import CancelToken, check_cancel

__all__ = ["ContactReport", "ChainTopology", "measure_contacts",
           "relieve_contacts"]


@dataclass
class ChainTopology:
    """Bonded structure of a bead cell, as flat arrays.

    ``chain`` and ``index`` give each bead's chain and its position along it;
    ``bond_length`` and ``d13`` are per bead, taken from the chain's RIS model
    (every bead in one chain shares them).
    """
    chain: np.ndarray            # (N,) int
    index: np.ndarray            # (N,) int, position along the chain
    radius: np.ndarray           # (N,) effective bead radius, Å
    bond_length: np.ndarray      # (N,) Å
    d13: np.ndarray              # (N,) 1-3 distance for the RIS bond angle, Å

    @classmethod
    def from_chains(cls, lengths: Sequence[int], radii: Sequence[np.ndarray],
                    bond_lengths: Sequence[float],
                    bond_angles_deg: Sequence[float]) -> "ChainTopology":
        chain, index, rad, bl, d13 = [], [], [], [], []
        for ci, (n, r, l, th) in enumerate(zip(lengths, radii, bond_lengths,
                                               bond_angles_deg)):
            chain.append(np.full(n, ci, dtype=np.int64))
            index.append(np.arange(n, dtype=np.int64))
            rad.append(np.asarray(r, dtype=float)[:n])
            bl.append(np.full(n, float(l)))
            d13.append(np.full(n, float(
                l * np.sqrt(2.0 * (1.0 - np.cos(np.radians(th)))))))
        return cls(np.concatenate(chain), np.concatenate(index),
                   np.concatenate(rad), np.concatenate(bl),
                   np.concatenate(d13))

    # ------------------------------------------------------------ bonded
    def pairs_at_separation(self, sep: int) -> np.ndarray:
        """Index pairs ``(i, i + sep)`` that lie in the same chain."""
        n = len(self.chain)
        if n <= sep:
            return np.zeros((0, 2), dtype=np.int64)
        i = np.arange(n - sep)
        ok = self.chain[i] == self.chain[i + sep]
        i = i[ok]
        return np.stack([i, i + sep], axis=1)


@dataclass
class ContactReport:
    """How clean a bead cell is, measured under minimum image."""
    n_contacts: int = 0              # non-bonded pairs inside the criterion
    min_distance: float = float("inf")
    worst_ratio: float = 1.0         # min r / criterion; >= 1 means clean
    overlap: float = 0.0             # summed (criterion - r) over contacts, Å
    iterations: int = 0
    max_bond_error: float = 0.0
    max_d13_error: float = 0.0


_SLACK = 1e-3


def _wrap(pos: np.ndarray, dims: np.ndarray) -> np.ndarray:
    out = pos - dims * np.floor(pos / dims)
    # floor can round a coordinate onto the upper face; cKDTree rejects that.
    return np.where(out >= dims, out - dims, out)


def _min_image(d: np.ndarray, dims: np.ndarray) -> np.ndarray:
    return d - dims * np.round(d / dims)


def _nonbonded_pairs(pos: np.ndarray, dims: np.ndarray, topo: ChainTopology,
                     rmax: float) -> np.ndarray:
    """All non-bonded pairs within ``rmax``; same-chain 1-2/1-3/1-4 dropped.

    The 1-4 pair is excluded for the same reason growth excludes it: its
    distance is set by the torsion, and gauche legitimately brings it close.
    """
    from scipy.spatial import cKDTree
    # A periodic KD-tree needs the query radius to be under half the box.
    r = float(min(rmax, 0.5 * float(dims.min()) - 1e-6))
    tree = cKDTree(_wrap(pos, dims), boxsize=dims)
    pairs = tree.query_pairs(r, output_type="ndarray")
    if pairs.size == 0:
        return pairs.reshape(0, 2)
    i, j = pairs[:, 0], pairs[:, 1]
    bonded = ((topo.chain[i] == topo.chain[j])
              & (np.abs(topo.index[i] - topo.index[j]) <= 3))
    return pairs[~bonded]


def _criterion(pairs: np.ndarray, topo: ChainTopology, tolerance: float,
               scale: float) -> np.ndarray:
    """The overlap distance growth enforces, per pair.

    Different chains: ``max(tolerance, ½(r_i + r_j)·scale)``. Same chain:
    ``tolerance`` alone — exactly the self-avoidance test in growth.
    """
    i, j = pairs[:, 0], pairs[:, 1]
    other = np.maximum(tolerance,
                       0.5 * (topo.radius[i] + topo.radius[j]) * scale)
    return np.where(topo.chain[i] == topo.chain[j], tolerance, other)


def measure_contacts(pos: np.ndarray, dims: np.ndarray, topo: ChainTopology,
                     tolerance: float, scale: float) -> ContactReport:
    dims = np.asarray(dims, dtype=float)
    rmax = float(max(tolerance, topo.radius.max() * max(scale, 0.0)))
    # Search past the criterion so the closest pair is reported even in a
    # clean cell, not just counted when it violates.
    pairs = _nonbonded_pairs(pos, dims, topo, rmax + 2.0)
    rep = ContactReport()
    if len(pairs):
        d = _min_image(pos[pairs[:, 1]] - pos[pairs[:, 0]], dims)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        cut = _criterion(pairs, topo, tolerance, scale)
        ratio = r / cut
        # 0.1% slack: SHAKE's last correction is ~1e-6 Å, and a pair at
        # 1.6995 Å against a 1.70 Å guard is not an overlap in any sense
        # that matters downstream.
        bad = ratio < 1.0 - _SLACK
        rep.n_contacts = int(np.count_nonzero(bad))
        rep.overlap = float((cut[bad] - r[bad]).sum())
        rep.min_distance = float(r.min())
        rep.worst_ratio = float(min(1.0, ratio.min()))
    rep.max_bond_error, rep.max_d13_error = _bonded_errors(pos, dims, topo)
    return rep


def _bonded_errors(pos, dims, topo):
    errs = []
    for sep, target in ((1, topo.bond_length), (2, topo.d13)):
        p = topo.pairs_at_separation(sep)
        if len(p) == 0:
            errs.append(0.0)
            continue
        d = _min_image(pos[p[:, 1]] - pos[p[:, 0]], dims)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        errs.append(float(np.abs(r - target[p[:, 0]]).max()))
    return errs[0], errs[1]


def _accumulate(force: np.ndarray, idx: np.ndarray, vec: np.ndarray) -> None:
    for ax in range(3):
        force[:, ax] += np.bincount(idx, weights=vec[:, ax],
                                    minlength=len(force))


def _shake(pos: np.ndarray, dims: np.ndarray, topo: ChainTopology,
           tol: float = 1e-6, max_sweeps: int = 500) -> None:
    """Project bond lengths and 1-3 distances back to their exact values.

    Constraints are split into classes that share no bead — bonds by the
    parity of their first index, 1-3 pairs by that index mod 4 — so each class
    is corrected exactly and at once, Gauss–Seidel between classes.
    """
    classes = []
    for sep, target, period in ((1, topo.bond_length, 2), (2, topo.d13, 4)):
        p = topo.pairs_at_separation(sep)
        if len(p) == 0:
            continue
        for c in range(period):
            sel = p[topo.index[p[:, 0]] % period == c]
            if len(sel):
                classes.append((sel, target[sel[:, 0]]))
    for _ in range(max_sweeps):
        worst = 0.0
        for sel, target in classes:
            a, b = sel[:, 0], sel[:, 1]
            d = _min_image(pos[b] - pos[a], dims)
            r = np.sqrt(np.einsum("ij,ij->i", d, d))
            err = r - target
            worst = max(worst, float(np.abs(err).max()))
            corr = (0.5 * err / np.maximum(r, 1e-9))[:, None] * d
            pos[a] += corr
            pos[b] -= corr
        if worst < tol:
            return


def relieve_contacts(pos: np.ndarray, dims, topo: ChainTopology, *,
                     tolerance: float, scale: float, margin: float = 1.03,
                     max_iter: int = 20000, patience: int = 1500,
                     cancel: Optional[CancelToken] = None
                     ) -> tuple:
    """Push non-bonded contacts apart while holding bonds and angles.

    Returns ``(positions, report)``. ``positions`` are wrapped into the box.
    The input array is not modified, and a cell with no contacts is returned
    untouched — so a clean construction keeps its exact RIS torsions.

    Minimisation is FIRE (Bitzek et al., *Phys. Rev. Lett.* **97** (2006)
    170201), which on this kind of penalty converges an order of magnitude
    faster than steepest descent. It stops when the cell is clean, or when
    the contact count has not improved for ``patience`` iterations — the sign
    that the request cannot be met at this density and tolerance.

    ``margin`` aims a little past the criterion so the SHAKE projection at
    the end, which nudges beads by a few thousandths of an Ångström, cannot
    tip a pair back under it.
    """
    dims = np.asarray(dims, dtype=float)
    pos = _wrap(np.array(pos, dtype=float), dims)
    rep = measure_contacts(pos, dims, topo, tolerance, scale)
    if rep.n_contacts == 0 or len(pos) < 2:
        return pos, rep

    bonded = []
    for sep, target, k in ((1, topo.bond_length, 4.0), (2, topo.d13, 2.0)):
        p = topo.pairs_at_separation(sep)
        if len(p):
            bonded.append((p, target[p[:, 0]], k))

    rmax = float(max(tolerance, topo.radius.max() * max(scale, 0.0))) * margin
    skin = 1.0
    max_move = 0.1                   # Å per iteration, per bead

    def neighbours(x):
        prs = _nonbonded_pairs(x, dims, topo, rmax + skin)
        return prs, (_criterion(prs, topo, tolerance, scale)
                     if len(prs) else np.zeros(0))

    def forces(x, prs, cut, stiff):
        f_tot = np.zeros_like(x)
        n_viol = 0
        if len(prs):
            i, j = prs[:, 0], prs[:, 1]
            d = _min_image(x[j] - x[i], dims)
            r = np.sqrt(np.einsum("ij,ij->i", d, d))
            target = cut * margin
            push = r < target
            # Converge on the full criterion; the report allows _SLACK, so
            # the final SHAKE nudge has room.
            n_viol = int(np.count_nonzero(r < cut))
            if np.any(push):
                rv = np.maximum(r[push], 1e-3)
                f = ((target[push] - rv) / rv)[:, None] * d[push]
                _accumulate(f_tot, i[push], -f)
                _accumulate(f_tot, j[push], f)
        for sel, tgt, k in bonded:
            a, b = sel[:, 0], sel[:, 1]
            d = _min_image(x[b] - x[a], dims)
            r = np.maximum(np.sqrt(np.einsum("ij,ij->i", d, d)), 1e-6)
            f = (stiff * k * (r - tgt) / r)[:, None] * d
            _accumulate(f_tot, a, f)
            _accumulate(f_tot, b, -f)
        return f_tot, n_viol

    it = 0
    # "Best" is least total overlap depth, not fewest contacts: for a request
    # that cannot be met, many shallow contacts beat a few deep ones.
    best = (rep.overlap, pos.copy())
    n_cycles = 10
    for cycle in range(n_cycles):
        # Each pass that ends with SHAKE undoing the fix means the contacts
        # were cleared by STRETCHING bonds. Stiffen the bonds so the next pass
        # has to clear them the physical way — by turning torsions and moving
        # whole segments. The FIRE step shrinks with the stiffness to stay
        # stable.
        stiff = 4.0 ** cycle
        pairs, cut = neighbours(pos)
        ref = pos.copy()
        vel = np.zeros_like(pos)
        dt_max = min(0.25, 0.5 / np.sqrt(8.0 * stiff))
        dt, alpha, n_pos = 0.2 * dt_max, 0.1, 0
        best_viol, since_best = np.inf, 0
        for _ in range(max_iter):
            it += 1
            if it % 250 == 0:
                check_cancel(cancel)
            force, n_viol = forces(pos, pairs, cut, stiff)
            if n_viol == 0:
                break
            if n_viol < best_viol:
                best_viol, since_best = n_viol, 0
            else:
                since_best += 1
                if since_best > patience:
                    break
            # FIRE: steer the velocity toward the force while going downhill,
            # stop dead and shrink the step the moment it goes uphill.
            power = float(np.vdot(force, vel))
            if power > 0.0:
                fn = np.sqrt(float(np.vdot(force, force)))
                vn = np.sqrt(float(np.vdot(vel, vel)))
                vel = (1.0 - alpha) * vel + alpha * vn * force / max(fn, 1e-12)
                n_pos += 1
                if n_pos > 5:
                    dt = min(dt * 1.1, dt_max)
                    alpha *= 0.99
            else:
                vel[:] = 0.0
                dt *= 0.5
                alpha, n_pos = 0.1, 0
            vel += dt * force
            move = dt * vel
            norm = np.sqrt(np.einsum("ij,ij->i", move, move))
            too_far = norm > max_move
            if np.any(too_far):
                move[too_far] *= (max_move / norm[too_far])[:, None]
            pos += move

            drift = _min_image(pos - ref, dims)
            if float(np.einsum("ij,ij->i", drift, drift).max()) > (0.5 * skin) ** 2:
                pos = _wrap(pos, dims)
                ref = pos.copy()
                pairs, cut = neighbours(pos)

        _shake(pos, dims, topo)
        pos = _wrap(pos, dims)
        rep = measure_contacts(pos, dims, topo, tolerance, scale)
        if rep.overlap < best[0]:
            best = (rep.overlap, pos.copy())
        # Clean, or stalled: another cycle from the same place would repeat
        # the same stall.
        if rep.n_contacts == 0 or since_best > patience:
            break

    if rep.overlap > best[0]:
        pos = best[1]
        rep = measure_contacts(pos, dims, topo, tolerance, scale)
    rep.iterations = it
    return pos, rep
