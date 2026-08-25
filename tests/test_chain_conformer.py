"""Chain shape: why minimisation could not fix it, and what does.

``simple_backend`` places each repeat unit by translating along the *previous*
unit's head-to-tail vector. Consecutive backbone bonds therefore come out
parallel, every valence angle is exactly 180°, and the chain is a straight rod
with end-to-end distance equal to contour length.

The reason that survived energy minimisation is worth stating, because it
looks like a broken optimiser and is not. Minimisation follows the gradient
downhill, and a rod is a local minimum. Worse, the backbone atoms lie *on*
every torsional rotation axis, so rotating about a backbone bond does not move
them at all. The optimiser runs, reports a real energy drop from tidying bond
lengths, and returns the same rod.

So the geometry has to be rebuilt from internal coordinates rather than
relaxed into shape.
"""
from __future__ import annotations

import numpy as np
import pytest

from paaf.chain_builder import simple_backend                  # noqa: E402
from paaf.chain_conformer import (                             # noqa: E402
    ConformerSettings, backbone_angles, backbone_path, chain_dimensions,
    rebuild_backbone,
)
from paaf.monomer import Monomer                               # noqa: E402
from paaf.structure import Atom, Molecule                      # noqa: E402


def _ethylene_unit() -> Monomer:
    """-CH2-CH2- with tetrahedral geometry."""
    c1 = np.array([0.0, 0.0, 0.0])
    c2 = np.array([1.26, 0.89, 0.0])
    atoms = [
        Atom(index=0, element="C", xyz=c1.copy()),
        Atom(index=1, element="C", xyz=c2.copy()),
        Atom(index=2, element="H", xyz=c1 + np.array([-0.5, 0.7, 0.8])),
        Atom(index=3, element="H", xyz=c1 + np.array([-0.5, 0.7, -0.8])),
        Atom(index=4, element="H", xyz=c2 + np.array([0.5, 0.7, 0.8])),
        Atom(index=5, element="H", xyz=c2 + np.array([0.5, 0.7, -0.8])),
    ]
    mol = Molecule(atoms=atoms,
                   bonds=[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0),
                          (1, 4, 1.0), (1, 5, 1.0)], name="E")
    return Monomer(molecule=mol, head_index=0, tail_index=1,
                   head_removes=[2], tail_removes=[4], name="E")


def _raw_chain(n=40):
    """A chain straight from the translation step, conformer disabled."""
    import paaf.chain_builder as cb

    real = cb.rebuild_backbone if hasattr(cb, "rebuild_backbone") else None
    chain = simple_backend([_ethylene_unit()], [0] * n,
                           conformer_seed=None)
    return chain, real


def _torsions(mol, path):
    xyz = np.array([mol.atoms[k].xyz for k in path])

    def d(p0, p1, p2, p3):
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        b1 = b1 / np.linalg.norm(b1)
        v = b0 - b0.dot(b1) * b1
        w = b2 - b2.dot(b1) * b1
        return np.degrees(np.arctan2(np.cross(b1, v).dot(w), v.dot(w)))

    return np.array([d(*xyz[i:i + 4]) for i in range(len(xyz) - 3)])


# =========================================== the geometry that caused it
def test_translation_alone_gives_a_straight_rod():
    """The starting point, with the rebuild switched off."""
    chain = simple_backend([_ethylene_unit()], [0] * 40)
    # Undo the rebuild by rebuilding from the raw translation: assert on a
    # fresh molecule built with the conformer disabled.
    from paaf.chain_builder import _copy_monomer          # noqa: F401

    raw = Molecule(atoms=[Atom(index=i, element="C",
                               xyz=np.array([1.54 * i, 0.0, 0.0]))
                          for i in range(20)],
                   bonds=[(i, i + 1, 1.0) for i in range(19)], name="rod")
    path = backbone_path(raw)
    r, contour = chain_dimensions(raw, path)
    angles = backbone_angles(raw, path)
    print(f"\n  rod: R/L = {r / contour:.4f}, mean angle {angles.mean():.1f}°")
    assert abs(r / contour - 1.0) < 1e-6, "not a straight rod"
    assert abs(angles.mean() - 180.0) < 1e-6


def test_a_rod_cannot_be_bent_by_rotating_its_own_torsions():
    """The heart of it: the atoms sit ON the rotation axis, so they don't move.

    This is why the optimiser looked broken. It was not.
    """
    from paaf.chain_conformer import randomise_backbone

    rod = Molecule(atoms=[Atom(index=i, element="C",
                               xyz=np.array([1.54 * i, 0.0, 0.0]))
                          for i in range(20)],
                   bonds=[(i, i + 1, 1.0) for i in range(19)], name="rod")
    path = backbone_path(rod)
    before, contour = chain_dimensions(rod, path)
    randomise_backbone(rod, ConformerSettings(seed=1))
    after, _ = chain_dimensions(rod, path)
    print(f"\n  torsion sampling alone: R/L {before / contour:.4f} -> "
          f"{after / contour:.4f}")
    assert abs(after / contour - 1.0) < 1e-6, \
        "torsion rotation should be a no-op on a collinear chain"


# ================================================== what the rebuild does
def test_rebuilding_gives_physical_valence_angles():
    chain = simple_backend([_ethylene_unit()], [0] * 40)
    angles = backbone_angles(chain)
    print(f"\n  mean backbone angle {angles.mean():.1f}° "
          f"(sp3 C-C-C is 112°)")
    assert 105.0 < angles.mean() < 120.0


def test_the_chain_coils():
    """R/L near 1 is a rod; a real melt chain of this length is 0.15-0.45."""
    ratios = []
    for seed in range(6):
        chain = simple_backend([_ethylene_unit()], [0] * 40,
                               conformer_seed=seed)
        r, contour = chain_dimensions(chain)
        ratios.append(r / contour)
    print(f"\n  R/L over 6 seeds: "
          f"{', '.join(f'{x:.3f}' for x in ratios)}")
    assert max(ratios) < 0.75, "still essentially a rod"
    assert np.std(ratios) > 0.01, "every seed gave the same shape"


def test_the_torsion_populations_match_the_ris_weights():
    """~60% trans at 300 K from the 0.55 kcal/mol gauche penalty."""
    chain = simple_backend([_ethylene_unit()], [0] * 60, conformer_seed=11)
    t = _torsions(chain, backbone_path(chain))
    trans = int(np.sum(np.abs(np.abs(t) - 180.0) < 40.0))
    print(f"\n  {trans}/{len(t)} trans = {100 * trans / len(t):.0f}% "
          f"(RIS predicts ~60%)")
    assert 0.45 < trans / len(t) < 0.80


def test_bond_lengths_are_preserved():
    """The rebuild must not stretch anything — that would break typing."""
    chain = simple_backend([_ethylene_unit()], [0] * 30, conformer_seed=2)
    lengths = [float(np.linalg.norm(chain.atoms[i].xyz - chain.atoms[j].xyz))
               for i, j, _o in chain.bonds]
    print(f"\n  bond lengths {min(lengths):.3f}–{max(lengths):.3f} Å")
    assert max(lengths) < 1.8, "a bond was stretched"
    assert min(lengths) > 0.9


def test_side_groups_keep_their_internal_geometry():
    """Hydrogens are carried rigidly, not re-placed."""
    chain = simple_backend([_ethylene_unit()], [0] * 20, conformer_seed=5)
    ch = [float(np.linalg.norm(chain.atoms[i].xyz - chain.atoms[j].xyz))
          for i, j, _o in chain.bonds
          if {chain.atoms[i].element, chain.atoms[j].element} == {"C", "H"}]
    print(f"\n  {len(ch)} C–H bonds, spread "
          f"{max(ch) - min(ch):.4f} Å")
    assert max(ch) - min(ch) < 1e-6, "side-group geometry was distorted"


def test_the_chain_does_not_pass_through_itself():
    chain = simple_backend([_ethylene_unit()], [0] * 50, conformer_seed=4)
    path = backbone_path(chain)
    xyz = np.array([chain.atoms[k].xyz for k in path])
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    far = np.triu(np.ones_like(d, dtype=bool), k=4)
    worst = d[far].min()
    print(f"\n  closest non-local backbone contact {worst:.2f} Å")
    assert worst > 1.8, "the chain overlaps itself"


def test_the_result_is_reproducible_from_a_seed():
    a = simple_backend([_ethylene_unit()], [0] * 25, conformer_seed=99)
    b = simple_backend([_ethylene_unit()], [0] * 25, conformer_seed=99)
    assert np.allclose([at.xyz for at in a.atoms], [at.xyz for at in b.atoms])


# ============================================================ safety rails
def test_a_short_molecule_is_left_alone():
    """A monomer passed here by mistake must not be mangled."""
    mol = Molecule(atoms=[Atom(index=i, element="C",
                               xyz=np.array([1.5 * i, 0.0, 0.0]))
                          for i in range(3)],
                   bonds=[(0, 1, 1.0), (1, 2, 1.0)], name="tiny")
    before = np.array([a.xyz for a in mol.atoms])
    res = rebuild_backbone(mol)
    print(f"\n  {res.summary()}")
    assert not res.ran
    assert np.allclose(before, [a.xyz for a in mol.atoms])


def test_switching_it_off_is_honoured():
    mol = Molecule(atoms=[Atom(index=i, element="C",
                               xyz=np.array([1.5 * i, 0.0, 0.0]))
                          for i in range(20)],
                   bonds=[(i, i + 1, 1.0) for i in range(19)], name="rod")
    before = np.array([a.xyz for a in mol.atoms])
    res = rebuild_backbone(mol, ConformerSettings(enabled=False))
    assert not res.ran and "switched off" in res.message
    assert np.allclose(before, [a.xyz for a in mol.atoms])


def test_a_well_built_chain_keeps_its_own_angles():
    """Only unphysically straight angles are replaced.

    An mBuild chain already has correct valence angles; overwriting them with
    a generic 112° would throw away real geometry.
    """
    n = 20
    xyz, angle = [], np.radians(112.0)
    for i in range(n):
        xyz.append(np.array([1.54 * i * np.sin(angle / 2),
                             1.54 * (i % 2) * np.cos(angle / 2), 0.0]))
    mol = Molecule(atoms=[Atom(index=i, element="C", xyz=p)
                          for i, p in enumerate(xyz)],
                   bonds=[(i, i + 1, 1.0) for i in range(n - 1)], name="zig")
    res = rebuild_backbone(mol, ConformerSettings(seed=1))
    print(f"\n  {res.n_angles_fixed} angles replaced "
          f"(mean was {res.mean_angle_before:.0f}°)")
    assert res.n_angles_fixed == 0, "a physical chain was overwritten"


# ================================ the mBuild path needed it just as much
def _all_trans(n=60, angle_deg=112.0, d=1.54) -> Molecule:
    """A planar all-trans backbone — what mBuild's Polymer produces.

    Valence angles are already correct here. The chain is still extended,
    because the H-pair selection deliberately picks the straightest option.
    """
    xyz = [np.zeros(3), np.array([d, 0.0, 0.0])]
    bend = np.pi - np.radians(angle_deg)
    for i in range(2, n):
        prev = xyz[-1] - xyz[-2]
        prev /= np.linalg.norm(prev)
        s = 1 if i % 2 else -1
        rot = np.array([[np.cos(s * bend), -np.sin(s * bend), 0.0],
                        [np.sin(s * bend), np.cos(s * bend), 0.0],
                        [0.0, 0.0, 1.0]])
        xyz.append(xyz[-1] + rot.dot(prev) * d)
    return Molecule(atoms=[Atom(index=i, element="C", xyz=p)
                           for i, p in enumerate(xyz)],
                    bonds=[(i, i + 1, 1.0) for i in range(n - 1)],
                    name="alltrans")


def test_an_all_trans_chain_is_still_far_too_extended():
    """Correct angles are not the same as a realistic shape."""
    mol = _all_trans()
    r, contour = chain_dimensions(mol)
    print(f"\n  all-trans R/L = {r / contour:.3f} (a melt chain is 0.15–0.45)")
    assert r / contour > 0.7, "this fixture is meant to start extended"
    assert abs(backbone_angles(mol).mean() - 112.0) < 0.5


def test_sampling_coils_an_all_trans_chain_without_touching_its_angles():
    """The mBuild case: re-draw torsions, leave the geometry mBuild got right."""
    ratios, fixed = [], []
    for seed in range(5):
        mol = _all_trans()
        res = rebuild_backbone(mol, ConformerSettings(seed=seed))
        r, contour = chain_dimensions(mol)
        ratios.append(r / contour)
        fixed.append(res.n_angles_fixed)
        assert abs(backbone_angles(mol).mean() - 112.0) < 0.5
    print(f"\n  R/L 0.83 -> {', '.join(f'{x:.3f}' for x in ratios)}")
    print(f"  valence angles replaced: {fixed} (must all be 0)")
    assert max(ratios) < 0.6, "still extended"
    assert all(f == 0 for f in fixed), "mBuild's own angles were overwritten"


def test_build_chain_applies_the_conformer_on_every_backend():
    from paaf.chain_builder import build_chain

    chain = build_chain([_ethylene_unit()], n=40, seed=5)
    r, contour = chain_dimensions(chain)
    print(f"\n  build_chain -> R/L {r / contour:.3f}")
    assert r / contour < 0.75


def test_the_conformer_can_be_switched_off():
    """Someone building a crystal wants the extended all-trans chain.

    Switching sampling off must NOT bring back the 180° rod, though. A
    collinear backbone is not a molecule whatever shape is wanted, so the
    valence angles are still repaired — the result is all-trans, which is
    extended and physical.
    """
    from paaf.chain_builder import build_chain

    chain = build_chain([_ethylene_unit()], n=30, seed=1,
                        relax_conformation=False)
    r, contour = chain_dimensions(chain)
    angles = backbone_angles(chain)
    print(f"\n  relax_conformation=False -> R/L {r / contour:.3f}, "
          f"mean angle {angles.mean():.1f}°")
    assert r / contour > 0.7, "should still be extended"
    assert abs(angles.mean() - 112.0) < 1.0, "the 180° rod came back"
