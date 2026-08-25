"""Rotational Isomeric State (RIS) model for polymer backbone torsions.

References
----------
Flory, P. J., *Statistical Mechanics of Chain Molecules*, Interscience, 1969.
Mattice, W. L. & Suter, U. W., *Conformational Theory of Large Molecules:
The Rotational Isomeric State Model*, Wiley, 1994.

Why this module exists
----------------------
A backbone torsion is not free. Rotation about a C–C bond has three minima —
*trans* (~180°) and *gauche*\\ ± (~±60°) — separated by barriers of a few
kcal/mol. Two facts follow, and both matter for chain dimensions:

1. **The minima are not equally weighted.** Gauche costs an extra energy
   :math:`E_\\sigma`, so :math:`\\sigma = \\exp(-E_\\sigma / RT) < 1`.
2. **Neighbouring torsions are correlated.** A g\\ :sup:`+`\\ g\\ :sup:`-` pair
   drives the two chain segments either side into each other — the *pentane
   effect*. It carries a large extra penalty :math:`\\omega`.

Encode both in a 3×3 statistical weight matrix ``U``, indexed
``U[previous_state, current_state]``.

The observable that tells you whether the model is real
-------------------------------------------------------
The characteristic ratio

.. math::  C_n = \\frac{\\langle R^2 \\rangle}{n\\,l^2}

measures how expanded the coil is relative to a random walk of the same bond
count. Its limiting value :math:`C_\\infty` is tabulated for real polymers.
Three regimes, and the gaps between them are the whole point:

=========================  ==========  ================================
model                       C for PE    what it captures
=========================  ==========  ================================
freely jointed              1.0         nothing
freely rotating (θ fixed)   ~2.2        bond angle only
RIS (θ fixed, weighted φ)   ~6.7        + torsional preference
=========================  ==========  ================================

So if an implementation returns ≈2 it has a bond angle but no working RIS; if
it returns ≈1 it has neither. Reproducing ≈6.7 is the test that the torsion
statistics are actually being applied.

Sampling
--------
Torsions are sampled with their **exact** finite-chain conditional
probabilities, not with the naive :math:`P(k) \\propto U[j,k]`. The difference
matters: the naive form ignores how many conformations a choice leaves
available downstream, and biases the ends of short chains. The partition
function is accumulated backwards,

.. math::  Z^{(i)} = U\\,Z^{(i+1)},\\qquad Z^{(n)} = \\mathbf{1}

after which the forward conditional is exact:

.. math::  P(k \\mid j) = \\frac{U[j,k]\\,Z^{(i+1)}_k}{\\sum_{k'} U[j,k']\\,Z^{(i+1)}_{k'}}

This is Flory's matrix method used as a sampler rather than only as a way to
evaluate ``Z``.

Honesty about parameters
------------------------
Parameters below are the conventional literature values for each polymer, with
the energies they came from recorded on the model. Where no parameters are
available :func:`generic_ris` returns a clearly-labelled generic model — it is
*not* silently passed off as parameterised. Check ``model.is_generic`` and
``model.provenance`` before trusting a number that came out of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "RISModel",
    "RIS_LIBRARY",
    "get_ris",
    "generic_ris",
    "sample_torsions",
    "build_backbone",
    "characteristic_ratio",
    "mean_square_end_to_end",
    "GAS_CONSTANT_KCAL",
]

# Gas constant in kcal/(mol·K) — energies throughout are kcal/mol.
GAS_CONSTANT_KCAL = 1.987204259e-3


@dataclass
class RISModel:
    """Discrete torsion states and their statistical weights.

    Attributes
    ----------
    name :
        Polymer this parameterisation is for.
    state_labels, state_angles_deg :
        The rotational isomeric states. Angles follow the IUPAC convention,
        trans = 180°.
    e_sigma_kcal :
        Energy of a gauche state relative to trans.
    e_omega_kcal :
        Extra energy of a g\\ :sup:`+`\\ g\\ :sup:`-` (pentane) pair.
    bond_length_a, bond_angle_deg :
        Backbone geometry. ``bond_angle_deg`` is the *bond angle* at the
        skeletal atom (e.g. 112° for C–C–C), not the supplement.
    is_generic :
        True when this is the fallback rather than a real parameterisation.
    provenance :
        Where the numbers came from. Printed in results so a reader can tell
        a literature value from a guess.
    """

    name: str
    state_labels: Tuple[str, ...] = ("t", "g+", "g-")
    state_angles_deg: Tuple[float, ...] = (180.0, 60.0, -60.0)
    e_sigma_kcal: float = 0.5
    e_omega_kcal: float = 2.0
    bond_length_a: float = 1.53
    bond_angle_deg: float = 112.0
    is_generic: bool = False
    provenance: str = ""
    reference_c_inf: Optional[float] = None
    reference_c_inf_temp_k: Optional[float] = None

    # ------------------------------------------------------------------
    def weights(self, temperature: float) -> Tuple[float, float]:
        """``(sigma, omega)`` Boltzmann factors at ``temperature`` (K)."""
        rt = GAS_CONSTANT_KCAL * temperature
        return (float(np.exp(-self.e_sigma_kcal / rt)),
                float(np.exp(-self.e_omega_kcal / rt)))

    def u_matrix(self, temperature: float) -> np.ndarray:
        """Statistical weight matrix ``U[prev, cur]`` at ``temperature``.

        The standard three-state form (Flory 1969, ch. V). Rows are the state
        of bond *i-1*, columns the state of bond *i*::

                      t      g+       g-
            t   [     1      σ        σ    ]
            g+  [     1      σ        σω   ]
            g-  [     1      σω       σ    ]

        Every row starts at 1 because trans is the reference state. ``ω``
        appears only on g\\ :sup:`+`\\ g\\ :sup:`-` and g\\ :sup:`-`\\ g\\ :sup:`+`
        — those are the pairs whose substituents collide.
        """
        sigma, omega = self.weights(temperature)
        return np.array([
            [1.0, sigma, sigma],
            [1.0, sigma, sigma * omega],
            [1.0, sigma * omega, sigma],
        ], dtype=float)

    def state_angles_rad(self) -> np.ndarray:
        return np.radians(np.asarray(self.state_angles_deg, dtype=float))

    @property
    def n_states(self) -> int:
        return len(self.state_labels)

    def describe(self, temperature: float = 300.0) -> str:
        sigma, omega = self.weights(temperature)
        tag = "  [GENERIC — not a real parameterisation]" if self.is_generic else ""
        return (f"{self.name}{tag}\n"
                f"  states      : {', '.join(self.state_labels)} at "
                f"{', '.join(f'{a:.0f}°' for a in self.state_angles_deg)}\n"
                f"  E_sigma     : {self.e_sigma_kcal:.3f} kcal/mol  "
                f"-> sigma = {sigma:.4f} at {temperature:.0f} K\n"
                f"  E_omega     : {self.e_omega_kcal:.3f} kcal/mol  "
                f"-> omega = {omega:.4f}\n"
                f"  geometry    : l = {self.bond_length_a:.3f} Å, "
                f"theta = {self.bond_angle_deg:.1f}°\n"
                f"  provenance  : {self.provenance or 'unspecified'}")


# =====================================================================
# Parameter library
# =====================================================================
# The energies below are the conventional values used to reproduce measured
# characteristic ratios. Two honest caveats:
#
#   * Published RIS parameter sets differ between authors, and several are
#     quoted over a range (E_sigma for PE appears as 0.4-0.6 kcal/mol in
#     different treatments). The values here sit inside those ranges and were
#     chosen to reproduce the reference C_inf; they are not the unique answer.
#   * C_inf is temperature dependent. The reference temperature is recorded
#     alongside each value, because comparing a 300 K calculation against a
#     413 K measurement is a real (and easily missed) error.

RIS_LIBRARY: Dict[str, RISModel] = {
    "PE": RISModel(
        name="polyethylene",
        e_sigma_kcal=0.50,
        e_omega_kcal=2.10,
        bond_length_a=1.53,
        bond_angle_deg=112.0,
        provenance=("Flory, Statistical Mechanics of Chain Molecules (1969), "
                    "three-state PE model; E_sigma ~0.4-0.6, E_omega ~2.0-2.2 "
                    "kcal/mol depending on the treatment."),
        reference_c_inf=6.7,
        reference_c_inf_temp_k=413.0,
    ),
    "PP": RISModel(
        name="polypropylene (atactic)",
        e_sigma_kcal=0.20,
        e_omega_kcal=1.65,
        bond_length_a=1.53,
        bond_angle_deg=114.0,
        provenance=("Suter & Flory, Macromolecules 8 (1975) 765, atactic PP. "
                    "Stereochemistry is NOT modelled here — a single averaged "
                    "state set stands in for the meso/racemo distinction, so "
                    "this is an approximation to the published treatment."),
        reference_c_inf=5.5,
        reference_c_inf_temp_k=413.0,
    ),
    "PS": RISModel(
        name="polystyrene (atactic)",
        e_sigma_kcal=0.85,
        e_omega_kcal=2.40,
        bond_length_a=1.53,
        bond_angle_deg=114.0,
        provenance=("Yoon, Sundararajan & Flory, Macromolecules 8 (1975) 776. "
                    "As for PP, tacticity is averaged rather than modelled."),
        reference_c_inf=10.0,
        reference_c_inf_temp_k=413.0,
    ),
    "PMMA": RISModel(
        name="poly(methyl methacrylate) (atactic)",
        e_sigma_kcal=0.70,
        e_omega_kcal=2.20,
        bond_length_a=1.53,
        bond_angle_deg=116.0,
        provenance=("Sundararajan & Flory, JACS 96 (1974) 5025. The wide "
                    "skeletal angle is the quaternary-carbon effect."),
        reference_c_inf=8.2,
        reference_c_inf_temp_k=413.0,
    ),
}


def generic_ris(bond_length_a: float = 1.53,
                bond_angle_deg: float = 112.0) -> RISModel:
    """Fallback for polymers with no parameterisation.

    Deliberately labelled ``is_generic``. It gives a physically reasonable
    trans-rich chain, but the resulting dimensions carry no claim of
    quantitative accuracy for any specific polymer — callers should surface
    that to the user rather than quoting C_inf as if it were measured.
    """
    return RISModel(
        name="generic 3-state",
        e_sigma_kcal=0.50,
        e_omega_kcal=2.00,
        bond_length_a=bond_length_a,
        bond_angle_deg=bond_angle_deg,
        is_generic=True,
        provenance=("GENERIC fallback — no polymer-specific parameters. "
                    "Values are typical of an sp3 carbon backbone; treat "
                    "resulting chain dimensions as qualitative."),
    )


def get_ris(key: str) -> RISModel:
    """Look a model up by key (case-insensitive), else the generic fallback."""
    if not key:
        return generic_ris()
    return RIS_LIBRARY.get(key.upper().strip(), generic_ris())


# =====================================================================
# Sampling
# =====================================================================
def _backward_partition(u: np.ndarray, n_torsions: int) -> List[np.ndarray]:
    """``Z[i]`` = partition function over torsions ``i..n-1`` given state ``i``.

    Recursion ``Z[i] = U @ Z[i+1]`` with ``Z[n-1] = 1``. Normalised at each
    step because the product underflows for long chains — only ratios are ever
    used, so the scaling is free.
    """
    z: List[np.ndarray] = [np.ones(u.shape[0], dtype=float)]
    for _ in range(n_torsions - 1):
        nxt = u @ z[-1]
        nxt = nxt / nxt.max()          # guard against underflow
        z.append(nxt)
    z.reverse()
    return z


def sample_torsions(model: RISModel, n_torsions: int, temperature: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Sample a correlated torsion-state sequence. Returns state indices.

    Uses exact finite-chain conditionals (see the module docstring), so the
    realised state populations reproduce the RIS weights rather than only
    approximating them for long chains.
    """
    if n_torsions <= 0:
        return np.zeros(0, dtype=int)

    u = model.u_matrix(temperature)
    z = _backward_partition(u, n_torsions)
    states = np.empty(n_torsions, dtype=int)

    # First torsion: no predecessor, so weight by the a-priori single-bond
    # weight (row of U from a trans reference) times its downstream partition
    # function.
    first = u[0] * z[0]
    states[0] = rng.choice(model.n_states, p=first / first.sum())

    for i in range(1, n_torsions):
        w = u[states[i - 1]] * z[i]
        states[i] = rng.choice(model.n_states, p=w / w.sum())
    return states


def state_populations(states: np.ndarray, n_states: int = 3) -> np.ndarray:
    """Realised fraction of each torsion state."""
    if states.size == 0:
        return np.zeros(n_states)
    counts = np.bincount(states, minlength=n_states).astype(float)
    return counts / counts.sum()


def expected_populations(model: RISModel, temperature: float) -> np.ndarray:
    """Equilibrium state populations from the dominant eigenvector of ``U``.

    For a long chain the state distribution converges to the normalised
    element-wise product of the left and right principal eigenvectors of
    ``U`` — the standard transfer-matrix result. This is what a correct
    sampler must reproduce, and is used as the reference in the tests.
    """
    u = model.u_matrix(temperature)
    evals, evecs = np.linalg.eig(u)
    k = int(np.argmax(evals.real))
    right = np.abs(evecs[:, k].real)
    evals_l, evecs_l = np.linalg.eig(u.T)
    kl = int(np.argmax(evals_l.real))
    left = np.abs(evecs_l[:, kl].real)
    p = left * right
    return p / p.sum()


# =====================================================================
# Geometry
# =====================================================================
def place_atom(a: np.ndarray, b: np.ndarray, c: np.ndarray,
               bond_length: float, bond_angle_deg: float,
               torsion_deg: float) -> np.ndarray:
    """Place atom D given A-B-C and internal coordinates (NeRF).

    Natural Extension Reference Frame: build an orthonormal frame on the
    B→C bond and place D by spherical coordinates within it. Exact by
    construction, so bond lengths and angles cannot drift — which is what
    makes fixed-geometry growth possible.
    """
    theta = np.radians(bond_angle_deg)
    phi = np.radians(torsion_deg)

    bc = c - b
    bc /= np.linalg.norm(bc)
    ab = b - a
    n = np.cross(ab, bc)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        # A, B, C collinear: any perpendicular will do for the frame.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(bc @ helper) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        n = np.cross(bc, helper)
        n_norm = np.linalg.norm(n)
    n /= n_norm
    m = np.cross(n, bc)

    # Local coordinates: -x along the bond, then rotate by the torsion.
    d2 = np.array([
        -bond_length * np.cos(theta),
        bond_length * np.sin(theta) * np.cos(phi),
        bond_length * np.sin(theta) * np.sin(phi),
    ])
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def build_backbone(model: RISModel, states: np.ndarray) -> np.ndarray:
    """Cartesian backbone from a torsion-state sequence.

    ``len(states)`` torsions give ``len(states) + 3`` skeletal atoms: the
    first three fix the initial frame, then each torsion adds one atom.
    """
    l = model.bond_length_a
    theta = model.bond_angle_deg
    angles = np.asarray(model.state_angles_deg, dtype=float)

    n_atoms = len(states) + 3
    pos = np.zeros((n_atoms, 3), dtype=float)
    pos[0] = (0.0, 0.0, 0.0)
    pos[1] = (l, 0.0, 0.0)
    # Third atom closes the bond angle in the xy-plane.
    ang = np.radians(180.0 - theta)
    pos[2] = pos[1] + l * np.array([np.cos(ang), np.sin(ang), 0.0])

    for i, s in enumerate(states):
        pos[i + 3] = place_atom(pos[i], pos[i + 1], pos[i + 2],
                                l, theta, angles[s])
    return pos


# =====================================================================
# Observables
# =====================================================================
def mean_square_end_to_end(positions: np.ndarray) -> float:
    """⟨R²⟩ for one chain, in Å²."""
    r = positions[-1] - positions[0]
    return float(r @ r)


def characteristic_ratio(positions: np.ndarray, bond_length: float) -> float:
    """C_n = ⟨R²⟩ / (n l²) for a single backbone.

    ``n`` is the number of *bonds*, i.e. ``len(positions) - 1``. Averaging
    over many chains is the caller's job — one chain is far too noisy.
    """
    n_bonds = len(positions) - 1
    if n_bonds <= 0:
        return 0.0
    return mean_square_end_to_end(positions) / (n_bonds * bond_length ** 2)


def measure_c_infinity(model: RISModel, n_bonds: int, temperature: float,
                       n_chains: int = 200, seed: int = 12345,
                       random_torsions: bool = False) -> Tuple[float, float]:
    """Average C_n over ``n_chains`` independently grown chains.

    Returns ``(mean, standard_error)``.

    ``random_torsions=True`` replaces RIS sampling with uniform selection
    among the same three states — the control that isolates what the
    statistical weights contribute. It should land near the freely-rotating
    value (~2.2 for a 112° backbone), not near C_inf.
    """
    rng = np.random.default_rng(seed)
    n_torsions = max(n_bonds - 2, 1)
    vals = np.empty(n_chains, dtype=float)
    for k in range(n_chains):
        if random_torsions:
            states = rng.integers(0, model.n_states, size=n_torsions)
        else:
            states = sample_torsions(model, n_torsions, temperature, rng)
        pos = build_backbone(model, states)
        vals[k] = characteristic_ratio(pos, model.bond_length_a)
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(n_chains))


def freely_rotating_c_infinity(bond_angle_deg: float) -> float:
    """Analytic C_inf for a freely rotating chain — the no-RIS baseline.

    With ``alpha`` the angle between consecutive bond *vectors*
    (``180° - bond_angle``), ``C = (1 + cos alpha) / (1 - cos alpha)``.
    For a 112° backbone this is ≈2.2, which is why a chain built with
    unweighted torsions lands there.
    """
    alpha = np.radians(180.0 - bond_angle_deg)
    ca = np.cos(alpha)
    return float((1.0 + ca) / (1.0 - ca))
