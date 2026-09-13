"""Detect and repair clashed chain geometries before force-field typing.

Why this exists
---------------
mBuild joins monomers on a straight line. For polymers with bulky side
groups (poly(1-heptene), polystyrene, PMMA …) the side chains of neighbouring
units land on top of each other. A local minimiser started from that
structure stays entangled: atoms remain 0.3–1.0 Å apart and some bonds get
stretched past 2 Å.

DL_FIELD is given an xyz file and infers bonds *from distances*. Fed a
clashed geometry it "sees" the wrong molecule, and fails with

    Fail to decide type of unsaturated C atom.

which points at the force field when the real problem is the geometry. The
topology PAAF holds (``Molecule.bonds``) is correct; only the coordinates
are wrong. So the fix is to relax coordinates *against that topology*: a
soft push-off (harmonic bonds/angles + a finite repulsion ramped up in
stages, like LAMMPS ``pair_style soft``) lets overlapping atoms pass through
each other and settle apart in seconds for thousands of atoms. If that ever
leaves residue on a small chain, RDKit's ETKDG embedding rebuilds it from
the bond graph alone.

Everything is topology-driven and element-agnostic, so it works for any
polymer PAAF can build.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .logging_utils import get_logger
from .structure import Molecule

log = get_logger(__name__)

# Covalent radii (Å), Cordero et al. 2008 — enough for organic/silicone.
_RCOV = {"H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
         "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
         "Na": 1.66, "K": 2.03, "Li": 1.28}


def _rcov(el: str) -> float:
    return _RCOV.get(el, 0.9)


def find_clashes(mol: Molecule, nonbonded_factor: float = 1.15,
                 bond_stretch: float = 1.15) -> Tuple[List[Tuple[int, int, float]],
                                                       List[Tuple[int, int, float]]]:
    """Return ``(clashes, bad_bonds)``.

    A *clash* is a non-bonded pair closer than ``nonbonded_factor`` × the sum
    of covalent radii — i.e. close enough that a distance-based bond
    perception (DL_FIELD's) would call it a bond. A *bad bond* is a real bond
    longer than ``bond_stretch`` × that sum or shorter than 0.6 × it — long
    enough that the same perception would drop it. 1.15 reproduces what
    dl_field 4.13 did on real failures: a 1.86 Å C–C bond was not seen
    (1.15 × 1.52 = 1.75), 1.68 Å was.
    """
    xyz = mol.coords()
    n = len(xyz)
    if n < 2:
        return [], []
    el = mol.elements()
    r = np.array([_rcov(e) for e in el])
    xyz = np.asarray(xyz, dtype=float)
    # Neighbour search instead of dense N×N arrays: memory stays linear in N.
    bonded = set()
    bad_bonds: List[Tuple[int, int, float]] = []
    for i, j, _ in mol.bonds:
        bonded.add((min(i, j), max(i, j)))
        dij = float(np.linalg.norm(xyz[i] - xyz[j]))
        rs = r[i] + r[j]
        if dij > bond_stretch * rs or dij < 0.6 * rs:
            bad_bonds.append((i, j, dij))
    from scipy.spatial import cKDTree
    pairs = cKDTree(xyz).query_pairs(nonbonded_factor * 2.0 * float(r.max()),
                                     output_type="ndarray")
    clashes: List[Tuple[int, int, float]] = []
    if len(pairs):
        ii, jj = pairs[:, 0], pairs[:, 1]
        swap = ii > jj
        ii, jj = np.where(swap, jj, ii), np.where(swap, ii, jj)
        d = np.linalg.norm(xyz[ii] - xyz[jj], axis=1)
        keep = d < nonbonded_factor * (r[ii] + r[jj])
        for i, j, dij in zip(ii[keep], jj[keep], d[keep]):
            if (int(i), int(j)) not in bonded:
                clashes.append((int(i), int(j), float(dij)))
        clashes.sort()
    return clashes, bad_bonds


def is_clean(mol: Molecule) -> bool:
    c, b = find_clashes(mol)
    return not c and not b


def _to_rdkit(mol: Molecule):
    from rdkit import Chem
    rw = Chem.RWMol()
    for a in mol.atoms:
        at = Chem.Atom(a.element)
        at.SetNoImplicit(True)          # every H is explicit in a PAAF chain
        rw.AddAtom(at)
    order_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                 3: Chem.BondType.TRIPLE}
    for i, j, o in mol.bonds:
        bt = order_map.get(int(round(o)), Chem.BondType.SINGLE)
        if abs(o - 1.5) < 1e-6:
            bt = Chem.BondType.AROMATIC
        rw.AddBond(int(i), int(j), bt)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(m)
    except Exception as exc:
        log.warning("geometry_repair: partial sanitisation only (%s)", exc)
        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                         | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                         | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION,
                         catchErrors=True)
    return m



# --------------------------------------------------------------- push-off
def _angle_triplets(mol: Molecule):
    nbrs = [[] for _ in mol.atoms]
    for i, j, _ in mol.bonds:
        nbrs[i].append(j); nbrs[j].append(i)
    el = mol.elements()
    trip = []      # (i, center, k, cos(theta0))
    for c, ns in enumerate(nbrs):
        if len(ns) < 2:
            continue
        if len(ns) >= 4:
            th = 109.47
        elif len(ns) == 3:
            th = 120.0 if el[c] in ("C", "N", "B") else 107.0
        else:
            th = 180.0 if el[c] == "C" and any(o >= 2.5 for i, j, o in mol.bonds
                                               if c in (i, j)) else 109.47
        c0 = float(np.cos(np.radians(th)))
        for a in range(len(ns)):
            for b in range(a + 1, len(ns)):
                trip.append((ns[a], c, ns[b], c0))
    return nbrs, trip


def relax_topology(mol: Molecule, seed: int = 7, stages=(0.5, 2.0, 8.0, 30.0),
                   kb: float = 200.0, ka: float = 40.0, rc_pad: float = 1.2,
                   iters: int = 400, cancel=None) -> bool:
    """Fast, size-independent clash removal driven by the bond graph.

    Minimises harmonic bonds (to covalent-radius lengths), harmonic angle
    cosines (tetrahedral / trigonal / linear by neighbour count) and a
    *soft* repulsion ``A(1+cos(pi r/rc))`` between non-1-2/1-3 pairs, with
    ``A`` ramped over ``stages`` so overlapping atoms can pass through each
    other before the wall hardens — the same idea as LAMMPS ``pair_style
    soft`` push-off used by the amorphous-cell builder. Starts from the
    current coordinates, so the overall chain shape is kept. Returns True
    when :func:`find_clashes` is clean afterwards. Coincident atoms are
    jittered first (a zero distance has no gradient direction).
    """
    try:
        from scipy.optimize import minimize
        from scipy.spatial import cKDTree
    except Exception:
        log.warning("geometry_repair: scipy not available — no push-off")
        return False
    n = len(mol.atoms)
    if n < 3:
        return True
    rng = np.random.default_rng(seed)
    x = mol.coords().copy()
    el = mol.elements()
    r = np.array([_rcov(e) for e in el])
    # jitter exact duplicates
    tree = cKDTree(x)
    for i, j in tree.query_pairs(0.05):
        x[j] += rng.normal(0, 0.3, 3)
    bi = np.array([[i, j] for i, j, _ in mol.bonds], dtype=int)
    r0 = r[bi[:, 0]] + r[bi[:, 1]]
    nbrs, trip = _angle_triplets(mol)
    ti = np.array([[a, c, b] for a, c, b, _ in trip], dtype=int).reshape(-1, 3)
    tc0 = np.array([c0 for *_, c0 in trip])
    excl = set()
    for i, j in bi:
        excl.add((min(i, j), max(i, j)))
    for a, c, b, _ in trip:
        excl.add((min(a, b), max(a, b)))
    rc_max = float(2 * r.max() + rc_pad)

    def energy(flat, A, pairs):
        X = flat.reshape(-1, 3)
        g = np.zeros_like(X)
        E = 0.0
        # bonds
        v = X[bi[:, 1]] - X[bi[:, 0]]
        d = np.linalg.norm(v, axis=1) + 1e-12
        diff = d - r0
        E += kb * np.sum(diff ** 2)
        f = (2 * kb * diff / d)[:, None] * v
        np.add.at(g, bi[:, 1], f); np.add.at(g, bi[:, 0], -f)
        # angles (cos form)
        if len(ti):
            u = X[ti[:, 0]] - X[ti[:, 1]]; w = X[ti[:, 2]] - X[ti[:, 1]]
            lu = np.linalg.norm(u, axis=1) + 1e-12; lw = np.linalg.norm(w, axis=1) + 1e-12
            cos = np.sum(u * w, axis=1) / (lu * lw)
            dc = cos - tc0
            E += ka * np.sum(dc ** 2)
            pref = (2 * ka * dc)[:, None]
            dcos_du = (w / (lu * lw)[:, None]) - (cos / lu ** 2)[:, None] * u
            dcos_dw = (u / (lu * lw)[:, None]) - (cos / lw ** 2)[:, None] * w
            np.add.at(g, ti[:, 0], pref * dcos_du)
            np.add.at(g, ti[:, 2], pref * dcos_dw)
            np.add.at(g, ti[:, 1], -pref * (dcos_du + dcos_dw))
        # soft repulsion
        if len(pairs):
            pi_, pj = pairs[:, 0], pairs[:, 1]
            v = X[pj] - X[pi_]
            d = np.linalg.norm(v, axis=1) + 1e-12
            rc = r[pi_] + r[pj] + rc_pad
            m = d < rc
            if m.any():
                dd, rr, vv = d[m], rc[m], v[m]
                E += A * np.sum(1 + np.cos(np.pi * dd / rr))
                f = (A * np.pi / rr * np.sin(np.pi * dd / rr) / dd)[:, None] * vv
                np.add.at(g, pj[m], -f); np.add.at(g, pi_[m], f)
        return E, g.ravel()

    for A in stages:
        for _ in range(3):
            if cancel is not None and cancel.is_cancelled():
                from .cell.packing import PackCancelled
                raise PackCancelled("cancelled during geometry repair")
            tree = cKDTree(x)
            pr = np.array([p for p in tree.query_pairs(rc_max) if p not in excl],
                          dtype=int).reshape(-1, 2)
            res = minimize(energy, x.ravel(), args=(A, pr), jac=True,
                           method="L-BFGS-B", options={"maxiter": iters})
            x = res.x.reshape(-1, 3)
    for k, a in enumerate(mol.atoms):
        a.xyz = np.array(x[k], dtype=float)
    c, b = find_clashes(mol)
    log.info("geometry_repair: push-off on %d atoms of %s -> %d clashes, %d bad bonds",
             n, mol.name, len(c), len(b))
    return not c and not b


def rebuild_coordinates(mol: Molecule, seed: int = 7,
                        max_attempts: int = 3) -> bool:
    """Regenerate all coordinates for ``mol``'s topology in place.

    Uses RDKit ETKDG distance geometry followed by MMFF (UFF fallback)
    minimisation. Returns True when the result passes :func:`find_clashes`.
    Coordinates are left untouched if RDKit cannot embed the molecule.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception:
        log.warning("geometry_repair: RDKit not available — cannot rebuild")
        return False

    m = _to_rdkit(mol)
    best_xyz: Optional[np.ndarray] = None
    best_score = None
    for attempt in range(max_attempts):
        params = AllChem.ETKDGv3()
        params.randomSeed = seed + attempt
        params.useRandomCoords = True
        params.maxIterations = 2000
        cid = AllChem.EmbedMolecule(m, params)
        if cid < 0:
            log.warning("geometry_repair: embedding attempt %d failed", attempt + 1)
            continue
        try:
            if AllChem.MMFFHasAllMoleculeParams(m):
                AllChem.MMFFOptimizeMolecule(m, maxIters=5000)
            else:
                AllChem.UFFOptimizeMolecule(m, maxIters=5000)
        except Exception as exc:
            log.warning("geometry_repair: minimisation error (%s); using raw embed", exc)
        xyz = m.GetConformer().GetPositions()
        trial = Molecule(atoms=[type(a)(index=a.index, element=a.element,
                                        xyz=np.array(xyz[k]), name=a.name)
                                for k, a in enumerate(mol.atoms)],
                         bonds=list(mol.bonds), name=mol.name)
        c, b = find_clashes(trial)
        score = len(c) + len(b)
        if best_score is None or score < best_score:
            best_score, best_xyz = score, np.array(xyz)
        if score == 0:
            break
    if best_xyz is None:
        return False
    for k, a in enumerate(mol.atoms):
        a.xyz = np.array(best_xyz[k], dtype=float)
    log.info("geometry_repair: rebuilt %d atoms of %s (residual clashes+bad bonds: %d)",
             len(mol.atoms), mol.name, best_score)
    return best_score == 0


def ensure_clean_geometry(mol: Molecule, log_fn=None, seed: int = 7,
                          cancel=None) -> bool:
    """Check ``mol``; rebuild its coordinates if clashed. Returns cleanliness.

    ``log_fn`` receives human-readable progress lines (the pipeline's ``_p``).
    """
    say = log_fn or (lambda s: None)
    clashes, bad = find_clashes(mol)
    if not clashes and not bad:
        return True
    el = mol.elements()
    ex = ", ".join(f"{el[i]}{i+1}–{el[j]}{j+1} {d:.2f} Å" for i, j, d in clashes[:4])
    say(f"Geometry check: {len(clashes)} overlapping non-bonded pair(s) and "
        f"{len(bad)} distorted bond(s) in {mol.name} (e.g. {ex or 'bond only'}). "
        f"Relaxing against the bond topology (soft push-off) — "
        f"a distance-based typer such as dl_field would misread this structure.")
    ok = relax_topology(mol, seed=seed, cancel=cancel)
    rnd = 0
    while not ok and rnd < 3:
        # Residual overlaps: nudge the offending atoms apart and harden the
        # wall further. Cheap, and resolves the last few pairs.
        rnd += 1
        rng = np.random.default_rng(seed + rnd)
        for i, j, _ in find_clashes(mol)[0]:
            mol.atoms[i].xyz = mol.atoms[i].xyz + rng.normal(0, 0.4, 3)
        ok = relax_topology(mol, seed=seed + rnd, stages=(8.0, 30.0, 100.0),
                            cancel=cancel)
    if not ok and len(mol.atoms) <= 700:
        say("Geometry check: push-off left residual overlaps — re-embedding "
            "from scratch (RDKit ETKDG).")
        ok = rebuild_coordinates(mol, seed=seed)
    c2, b2 = find_clashes(mol)
    if ok:
        say("Geometry check: clean after rebuild.")
    else:
        say(f"Geometry check: still {len(c2)} clash(es) / {len(b2)} bad bond(s) "
            f"after rebuild — dl_field typing may fail.")
    return ok
