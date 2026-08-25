"""Shared primitives for cell packing: neighbour search, progress, cancellation.

Ported from the reference amorphous-cell implementation so the growth builder
(:mod:`paaf.cell.grow`) and any future packer share one overlap criterion and
one neighbour search rather than each carrying its own subtly different copy.

The overlap criterion
---------------------
Two atoms clash when

.. math::  r_{ij} < \\max(\\text{tolerance},\\; 0.5\\,(v_i + v_j)\\,\\text{scale})

``tolerance`` is an absolute floor in Å; the second term lets van der Waals
spheres interpenetrate by a factor ``scale``. Real melts are *dense* — atoms
sit well inside each other's nominal vdW radii — so ``scale`` around 0.55 is
what makes melt densities reachable at all. Demanding full vdW separation
would leave a structure at roughly half the density of the real material.

Minimum image
-------------
Distances are evaluated under the minimum-image convention: for each pair we
take the nearest periodic replica. Exact for orthorhombic cells. For triclinic
cells this is applied in the axis-aligned bounding box, which is an
approximation — flagged here rather than hidden, because a strongly skewed
cell will under-count contacts across the skewed faces.
"""
from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PackProgress", "CancelToken", "PackError", "PackFailed", "PackCancelled",
    "ProgressFn", "NeighbourGrid", "vdw_radius", "atomic_mass", "wrap",
]


# ============================================================ atom data
#: Van der Waals radii in Å (Bondi-style values, as used by the reference tree).
VDW_RADII: Dict[str, float] = {
    "H": 1.20, "He": 1.40, "Li": 1.82, "Be": 1.53, "B": 1.92, "C": 1.70,
    "N": 1.55, "O": 1.52, "F": 1.47, "Ne": 1.54, "Na": 2.27, "Mg": 1.73,
    "Al": 1.84, "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75, "Ar": 1.88,
    "K": 2.75, "Ca": 2.31, "Fe": 2.04, "Ni": 1.97, "Cu": 1.96, "Zn": 2.01,
    "Ga": 1.87, "Ge": 2.11, "As": 1.85, "Se": 1.90, "Br": 1.85, "Kr": 2.02,
    "Ag": 2.11, "Cd": 2.18, "Sn": 2.17, "Sb": 2.06, "Te": 2.06, "I": 1.98,
    "Xe": 2.16, "Pt": 2.13, "Au": 2.14, "Hg": 2.23, "Pb": 2.02, "U": 1.86,
}
_DEFAULT_VDW = 2.00

ATOMIC_MASS: Dict[str, float] = {
    "H": 1.008, "He": 4.003, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990,
    "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06,
    "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078, "Fe": 55.845,
    "Cu": 63.546, "Zn": 65.38, "Br": 79.904, "I": 126.904,
}


def vdw_radius(symbol: str) -> float:
    return VDW_RADII.get(symbol, _DEFAULT_VDW)


def atomic_mass(symbol: str) -> float:
    return ATOMIC_MASS.get(symbol, 12.011)


# ======================================================= progress / cancel
@dataclass
class PackProgress:
    """Snapshot handed to a ``progress`` callback.

    Same shape as the reference packer's, so a GUI worker can drive either
    without special-casing.
    """
    placed: int = 0
    total: int = 0
    attempts: int = 0
    current_species: str = ""
    message: str = ""
    fraction: float = 0.0


class CancelToken:
    """Thread-safe cancellation flag.

    A plain :class:`threading.Event` wrapper rather than a Qt object, so the
    builder runs head-less in tests and notebooks while a GUI worker thread
    can hold the very same token.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()


class PackError(RuntimeError):
    """Base class for packing failures."""


class PackFailed(PackError):
    """The cell could not be filled (density too high / box too small)."""


class PackCancelled(PackError):
    """The caller cancelled through a :class:`CancelToken`."""


ProgressFn = Callable[[PackProgress], None]


def emit(progress: Optional[ProgressFn], **kw) -> None:
    if progress is None:
        return
    try:
        progress(PackProgress(**kw))
    except Exception:
        # A misbehaving callback must never take the builder down.
        pass


def check_cancel(cancel: Optional[CancelToken]) -> None:
    if cancel is not None and cancel.is_cancelled():
        raise PackCancelled("cancelled by caller")


def wrap(points: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Fold Cartesian points into the primary cell ``[0, L)`` per axis."""
    return points - dims * np.floor(points / dims)


# ============================================================ neighbour grid
class NeighbourGrid:
    """Uniform cell list over a periodic box.

    Cells are at least ``cutoff`` wide, so every atom within ``cutoff`` of a
    query point lies in the query cell or one of its 26 neighbours — hence the
    fixed 3×3×3 stencil. Cell indices are taken modulo the cell count, so the
    stencil wraps across the periodic boundary for free.

    Growth needs one capability a rigid-body packer does not: the ability to
    ignore *specific atoms* rather than a whole molecule, because the last few
    atoms of the chain being grown are its own 1-2 / 1-3 neighbours and must
    not count as clashes. ``exclude_slots`` provides that.
    """

    _STENCIL = list(itertools.product((-1, 0, 1), repeat=3))

    def __init__(self, dims: np.ndarray, cutoff: float,
                 capacity: int = 1024) -> None:
        self.dims = np.asarray(dims, dtype=float)
        self.cutoff = float(cutoff)
        # At least one cell per axis; a box thinner than the cutoff degenerates
        # to "scan everything", which is slow but still correct.
        self.ncell = np.maximum(
            1, np.floor(self.dims / self.cutoff).astype(np.int64))
        self.cell = self.dims / self.ncell
        self._cells: Dict[Tuple[int, int, int], List[int]] = {}
        cap = max(capacity, 8)
        self._pos = np.zeros((cap, 3), dtype=float)
        self._rad = np.zeros(cap, dtype=float)
        self._mol = np.full(cap, -1, dtype=np.int64)
        self._n = 0

    # ------------------------------------------------------------ internals
    def _grow(self, extra: int) -> None:
        need = self._n + extra
        if need <= self._pos.shape[0]:
            return
        cap = max(need, self._pos.shape[0] * 2)
        pos = np.zeros((cap, 3), dtype=float)
        rad = np.zeros(cap, dtype=float)
        mol = np.full(cap, -1, dtype=np.int64)
        pos[: self._n] = self._pos[: self._n]
        rad[: self._n] = self._rad[: self._n]
        mol[: self._n] = self._mol[: self._n]
        self._pos, self._rad, self._mol = pos, rad, mol

    def _key(self, point: np.ndarray) -> Tuple[int, int, int]:
        idx = np.floor(point / self.cell).astype(np.int64) % self.ncell
        return (int(idx[0]), int(idx[1]), int(idx[2]))

    def _candidates(self, point: np.ndarray) -> Optional[np.ndarray]:
        base = np.floor(point / self.cell).astype(np.int64)
        seen: set = set()
        out: List[int] = []
        nx, ny, nz = (int(v) for v in self.ncell)
        for dx, dy, dz in self._STENCIL:
            key = ((base[0] + dx) % nx, (base[1] + dy) % ny, (base[2] + dz) % nz)
            # With fewer than 3 cells on an axis the stencil revisits cells;
            # dedupe so no pair is tested twice.
            if key in seen:
                continue
            seen.add(key)
            bucket = self._cells.get(key)
            if bucket:
                out.extend(bucket)
        if not out:
            return None
        return np.asarray(out, dtype=np.int64)

    # --------------------------------------------------------------- public
    @property
    def n_atoms(self) -> int:
        return self._n

    def add(self, positions: np.ndarray, radii: np.ndarray,
            mol_id: int = -1) -> List[int]:
        """Insert already-wrapped positions; returns their slot indices."""
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        radii = np.atleast_1d(np.asarray(radii, dtype=float))
        self._grow(len(positions))
        slots: List[int] = []
        for p, r in zip(positions, radii):
            i = self._n
            self._pos[i] = p
            self._rad[i] = r
            self._mol[i] = mol_id
            self._cells.setdefault(self._key(p), []).append(i)
            self._n = i + 1
            slots.append(i)
        return slots

    def positions(self, slots: Sequence[int]) -> np.ndarray:
        return self._pos[np.asarray(slots, dtype=np.int64)]

    def min_distance(self, point: np.ndarray,
                     exclude_slots: Optional[set] = None) -> float:
        """Closest stored atom to ``point`` under minimum image (Å)."""
        if self._n == 0:
            return float("inf")
        cand = self._candidates(point)
        if cand is None:
            return float("inf")
        if exclude_slots:
            cand = np.asarray([c for c in cand if int(c) not in exclude_slots],
                              dtype=np.int64)
            if cand.size == 0:
                return float("inf")
        d = self._pos[cand] - point
        d -= self.dims * np.round(d / self.dims)
        return float(np.sqrt(np.einsum("ij,ij->i", d, d).min()))

    def overlaps(self, positions: np.ndarray, radii: np.ndarray,
                 tolerance: float, scale: float,
                 exclude_mol: Optional[int] = None,
                 exclude_slots: Optional[set] = None) -> bool:
        """True if any of ``positions`` clashes with a stored atom.

        ``exclude_mol`` skips a whole molecule (rigid trial moves);
        ``exclude_slots`` skips individual atoms, which is what chain growth
        needs so a new bead does not "clash" with the neighbours it is bonded
        to.
        """
        if self._n == 0:
            return False
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        radii = np.atleast_1d(np.asarray(radii, dtype=float))
        dims = self.dims
        for k in range(len(positions)):
            p = positions[k]
            cand = self._candidates(p)
            if cand is None:
                continue
            if exclude_mol is not None:
                cand = cand[self._mol[cand] != exclude_mol]
            if exclude_slots:
                cand = np.asarray(
                    [c for c in cand if int(c) not in exclude_slots],
                    dtype=np.int64)
            if cand.size == 0:
                continue
            d = self._pos[cand] - p
            d -= dims * np.round(d / dims)
            d2 = np.einsum("ij,ij->i", d, d)
            cut = np.maximum(tolerance,
                             0.5 * (self._rad[cand] + radii[k]) * scale)
            if np.any(d2 < cut * cut):
                return True
        return False

    def soft_energy(self, point: np.ndarray, radius: float,
                    exclude_slots: Optional[set] = None,
                    epsilon: float = 0.2) -> float:
        """A cheap repulsive non-bonded energy at ``point`` (kcal/mol).

        Theodorou–Suter weight each candidate torsion by
        ``exp(-E_nb / kT)``. A full Lennard-Jones sum would be the faithful
        choice, but it needs per-atom LJ parameters that are not available
        until the force field is assigned — which happens *after* the cell is
        built. So this uses a purely repulsive soft-core term keyed on vdW
        contact distance:

            E = epsilon * sum_j [ (sigma_ij / r_ij)^12 ]  for r_ij < sigma_ij

        DEVIATION FROM THE PAPER, stated plainly: this reproduces the
        *qualitative* effect that matters for construction — candidates that
        bury the new bead in existing material are strongly disfavoured —
        without pretending to a quantitative energy. Absolute values from this
        function are not physical energies and should not be reported as such.
        """
        if self._n == 0:
            return 0.0
        cand = self._candidates(point)
        if cand is None:
            return 0.0
        if exclude_slots:
            cand = np.asarray([c for c in cand if int(c) not in exclude_slots],
                              dtype=np.int64)
            if cand.size == 0:
                return 0.0
        d = self._pos[cand] - point
        d -= self.dims * np.round(d / self.dims)
        r2 = np.einsum("ij,ij->i", d, d)
        sigma = 0.5 * (self._rad[cand] + radius)
        s2 = sigma * sigma
        close = r2 < s2
        if not np.any(close):
            return 0.0
        ratio = s2[close] / np.maximum(r2[close], 1e-6)
        # Cap the exponent: a near-zero separation would otherwise overflow.
        return float(epsilon * np.sum(np.minimum(ratio ** 6, 1e6)))
