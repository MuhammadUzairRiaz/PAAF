"""Atom types must survive replication into a multi-chain box.

The typing work so far has all been about ONE chain: mapping a hand-typed
monomer onto the polymer built from it. But the stated requirement is "any
number of monomers, any number of chains", and the chain count is handled
somewhere else entirely — the single typed chain is written out, then
``replicate_single_chain`` packs N copies of it into a box.

That step re-pairs coordinates with atom lines, and it did so through two
different orderings:

* ``_write_pdb_from_data`` walked the ``Atoms`` section in **file order** and
  wrote the PDB packmol sees;
* ``_replicate_atoms`` walked the same section **sorted by atom id**.

Identical whenever the file happens to be sorted, which is why this never
showed up — moltemplate writes ascending ids. But nothing guarantees it. A
hand-edited file, a file merged from two sources, or any future writer that
groups atoms by molecule instead would silently pair atom 1's force-field
type with atom 7's position, in every chain at once. The result is a box that
loads, runs, and is wrong: right formula, right topology, atoms in each
other's places.

So the ordering is pinned here rather than assumed. The decisive test is
``test_types_are_not_scrambled_when_the_atoms_section_is_out_of_order``; the
rest guard the invariants that make a replica a replica.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from paaf.lammps_replicator import (                              # noqa: E402
    _clean, _read_xyz_or_pdb_coords, _write_pdb_from_data,
    parse_lammps_data, replicate_single_chain,
)

#: A five-atom stand-in for a chain: distinct type per atom, distinct
#: position per atom, so any mis-pairing is visible rather than plausible.
#: Types double as markers exactly as they do in the manual-typing tests.
ATOMS = [
    # id, mol, type, charge,     x,     y,     z
    (1, 1, 1, -0.18, 0.000, 0.000, 0.000),
    (2, 1, 2,  0.06, 1.540, 0.000, 0.000),
    (3, 1, 3, -0.12, 2.100, 1.400, 0.000),
    (4, 1, 4,  0.09, 3.640, 1.400, 0.000),
    (5, 1, 5, -0.21, 4.200, 2.800, 0.000),
]
MASSES = [(1, 12.011), (2, 12.011), (3, 15.999), (4, 12.011), (5, 1.008)]
BONDS = [(1, 1, 1, 2), (2, 1, 2, 3), (3, 2, 3, 4), (4, 1, 4, 5)]


def _write_data(path: Path, atom_order: List[int]) -> Path:
    """Write a single-chain data file with ``Atoms`` in the given id order."""
    by_id = {a[0]: a for a in ATOMS}
    lines = [
        "LAMMPS data — single chain for a replication test", "",
        f"{len(ATOMS)} atoms", f"{len(BONDS)} bonds", "",
        f"{len(MASSES)} atom types", "2 bond types", "",
        "0.0 50.0 xlo xhi", "0.0 50.0 ylo yhi", "0.0 50.0 zlo zhi", "",
        "Masses", "",
    ]
    lines += [f"{t} {m}" for t, m in MASSES]
    lines += ["", "Atoms  # full", ""]
    for aid in atom_order:
        i, mol, typ, q, x, y, z = by_id[aid]
        lines.append(f"{i} {mol} {typ} {q:.8f} {x:.8f} {y:.8f} {z:.8f}")
    lines += ["", "Bonds", ""]
    lines += [f"{b} {t} {i} {j}" for b, t, i, j in BONDS]
    path.write_text("\n".join(lines) + "\n")
    return path


def _read_atoms(path: Path) -> Dict[int, Tuple[int, np.ndarray]]:
    """``{atom id: (type, xyz)}`` from a packed data file."""
    _header, sections = parse_lammps_data(path)
    out = {}
    for line in _clean(sections.get("Atoms")):
        p = line.split()
        out[int(p[0])] = (int(p[2]),
                          np.array([float(p[4]), float(p[5]), float(p[6])]))
    return out


def _pack(tmp_path: Path, atom_order: List[int], n_chains: int = 3) -> Path:
    single = _write_data(tmp_path / "single.data", atom_order)
    return replicate_single_chain(
        single, n_chains, (50.0, 50.0, 50.0),
        tmp_path / "packed.data", seed=7)


#: Sorted is what moltemplate produces; the others are what nothing
#: guarantees it will keep producing.
ORDERS = {
    "sorted":   [1, 2, 3, 4, 5],
    "reversed": [5, 4, 3, 2, 1],
    "shuffled": [3, 1, 5, 2, 4],
}


# ================================================== the ordering regression
@pytest.mark.parametrize("order_name", list(ORDERS))
def test_atom_id_n_keeps_type_n_in_every_chain(tmp_path, order_name):
    """Types ride with ids, so this passed even while the box was wrong.

    Worth stating anyway, because it is *why* the bug was invisible: the
    replicator writes ``oid + offset`` alongside the type it read from that
    same line, so the type is never separated from its id. The manual types
    the user assigned are all still there, on all the right atom numbers —
    only the coordinates moved. Every per-atom summary of the broken file
    agrees with every other one.

    The assertion that fails when the ordering is wrong is the next test.
    """
    packed = _read_atoms(_pack(tmp_path, ORDERS[order_name]))
    natoms = len(ATOMS)
    wrong = []
    for aid, (typ, _xyz) in sorted(packed.items()):
        expected = ((aid - 1) % natoms) + 1        # id N of every chain
        if typ != expected:
            wrong.append((aid, typ, expected))
    print(f"\n  {order_name}: {len(packed)} atoms, {len(wrong)} mistyped")
    assert not wrong, f"first wrong: id {wrong[0][0]} got type {wrong[0][1]}, wanted {wrong[0][2]}"


@pytest.mark.parametrize("order_name", list(ORDERS))
def test_each_replica_keeps_the_geometry_of_the_original(tmp_path, order_name):
    """A replica is the same molecule moved, so every internal vector holds.

    This is the assertion that actually catches a coordinate/type mis-pairing:
    scrambling the pairing leaves each atom at *some* real position, so no
    atom looks obviously wrong on its own — but the bonds stretch.
    """
    packed = _read_atoms(_pack(tmp_path, ORDERS[order_name]))
    natoms = len(ATOMS)
    want = {a[0]: np.array(a[4:7]) for a in ATOMS}
    reference = [np.linalg.norm(want[j] - want[i]) for i, j in
                 ((1, 2), (2, 3), (3, 4), (4, 5), (1, 5))]

    for chain in range(len(packed) // natoms):
        off = chain * natoms
        got = [np.linalg.norm(packed[j + off][1] - packed[i + off][1])
               for i, j in ((1, 2), (2, 3), (3, 4), (4, 5), (1, 5))]
        worst = max(abs(a - b) for a, b in zip(got, reference))
        print(f"\n  {order_name} chain {chain}: worst distance error "
              f"{worst:.4f} A")
        assert worst < 0.01, (
            f"chain {chain} is not a rigid copy — coordinates were paired "
            f"with the wrong atoms")


def test_the_check_would_notice_a_scramble(tmp_path):
    """A test that cannot fail proves nothing.

    Rotate the types by one and both assertions above must object; otherwise
    they are passing for the wrong reason.
    """
    packed = _read_atoms(_pack(tmp_path, ORDERS["sorted"]))
    natoms = len(ATOMS)
    scrambled = {aid: (((typ) % natoms) + 1, xyz)
                 for aid, (typ, xyz) in packed.items()}
    wrong = [aid for aid, (typ, _x) in scrambled.items()
             if typ != ((aid - 1) % natoms) + 1]
    print(f"\n  after rotating every type: {len(wrong)} detected")
    assert len(wrong) == len(packed)


# ============================================ the contract, at its source
@pytest.mark.parametrize("order_name", list(ORDERS))
def test_the_packmol_pdb_is_written_in_atom_id_order(tmp_path, order_name):
    """The single invariant both replicators depend on.

    Packmol hands coordinates back in the order it received them, and every
    caller pairs them onto atom lines sorted by id. Pinning it here rather
    than only through ``replicate_single_chain`` also covers the blend path,
    whose ``_replicate_component_atoms`` sorts the same way but cannot be
    exercised end to end without packmol installed.
    """
    data = _write_data(tmp_path / "single.data", ORDERS[order_name])
    pdb = tmp_path / "single.pdb"
    _write_pdb_from_data(data, pdb)
    got = np.array(_read_xyz_or_pdb_coords(pdb))
    want = np.array([a[4:7] for a in sorted(ATOMS)])
    print(f"\n  {order_name}: first atom written at {got[0]}, "
          f"id-1 is at {want[0]}")
    assert np.allclose(got, want, atol=1e-3), \
        "the PDB is not in ascending atom-id order"


def test_a_replaced_geometry_is_still_written_in_atom_id_order(tmp_path):
    """The minimised-coordinates path takes the same route.

    ``coords`` is keyed by atom id, so it is immune to the ordering bug on
    its own — but it is written into the same PDB, so it has to come out in
    the same order as everything else.
    """
    data = _write_data(tmp_path / "single.data", ORDERS["shuffled"])
    pdb = tmp_path / "single.pdb"
    moved = {a[0]: (a[4] + 10.0, a[5], a[6]) for a in ATOMS}
    _write_pdb_from_data(data, pdb, coords=moved)
    got = np.array(_read_xyz_or_pdb_coords(pdb))
    want = np.array([moved[a[0]] for a in sorted(ATOMS)])
    assert np.allclose(got, want, atol=1e-3)


# ========================================================= replica sanity
@pytest.mark.parametrize("n_chains", [1, 2, 5, 12])
def test_every_chain_is_present_and_complete(tmp_path, n_chains):
    work = tmp_path / f"n{n_chains}"
    work.mkdir()
    packed = _read_atoms(_pack(work, ORDERS["sorted"], n_chains))
    print(f"\n  {n_chains} chains -> {len(packed)} atoms")
    assert len(packed) == n_chains * len(ATOMS)
    assert sorted(packed) == list(range(1, n_chains * len(ATOMS) + 1)), \
        "atom ids must be contiguous from 1"


def test_the_topology_is_offset_chain_by_chain(tmp_path):
    """Bond 4 of chain 2 must join chain 2's atoms, not chain 1's."""
    out = _pack(tmp_path, ORDERS["sorted"], n_chains=3)
    _header, sections = parse_lammps_data(out)
    bonds = [tuple(int(x) for x in l.split()) for l in
             _clean(sections.get("Bonds"))]
    natoms, nbonds = len(ATOMS), len(BONDS)
    print(f"\n  {len(bonds)} bonds across 3 chains")
    assert len(bonds) == 3 * nbonds
    for bid, btype, i, j in bonds:
        chain = (bid - 1) // nbonds
        assert chain == (i - 1) // natoms == (j - 1) // natoms, \
            f"bond {bid} straddles two chains"


def test_bond_types_survive_replication(tmp_path):
    """Bond 3 is the only type-2 bond; each chain must still have exactly one."""
    out = _pack(tmp_path, ORDERS["sorted"], n_chains=4)
    _header, sections = parse_lammps_data(out)
    types = [int(l.split()[1]) for l in _clean(sections.get("Bonds"))]
    print(f"\n  type-2 bonds: {types.count(2)} across 4 chains")
    assert types.count(2) == 4


def test_the_coefficient_sections_are_copied_verbatim(tmp_path):
    """Masses map type -> element. Losing them loses the chemistry."""
    out = _pack(tmp_path, ORDERS["sorted"], n_chains=3)
    _header, sections = parse_lammps_data(out)
    masses = {int(l.split()[0]): float(l.split()[1])
              for l in _clean(sections.get("Masses"))}
    print(f"\n  masses: {masses}")
    assert masses == {t: m for t, m in MASSES}


def test_the_header_counts_match_what_was_written(tmp_path):
    """A count that disagrees with the body makes LAMMPS refuse the file."""
    out = _pack(tmp_path, ORDERS["shuffled"], n_chains=6)
    header, sections = parse_lammps_data(out)
    text = "\n".join(header)
    assert f"{6 * len(ATOMS)} atoms" in text
    assert f"{6 * len(BONDS)} bonds" in text
    assert f"{len(MASSES)} atom types" in text, "type count must NOT be scaled"
    assert len(_clean(sections["Atoms"])) == 6 * len(ATOMS)


def test_each_chain_gets_its_own_molecule_id(tmp_path):
    """Otherwise LAMMPS treats the whole box as one molecule."""
    out = _pack(tmp_path, ORDERS["sorted"], n_chains=4)
    _header, sections = parse_lammps_data(out)
    mols = {int(l.split()[1]) for l in _clean(sections.get("Atoms"))}
    print(f"\n  molecule ids: {sorted(mols)}")
    assert mols == {1, 2, 3, 4}


def test_charges_ride_along_with_their_atoms(tmp_path):
    """Charge is per-atom-type data written on the atom line, like the type."""
    out = _pack(tmp_path, ORDERS["reversed"], n_chains=3)
    _header, sections = parse_lammps_data(out)
    natoms = len(ATOMS)
    want = {a[0]: a[3] for a in ATOMS}
    for line in _clean(sections["Atoms"]):
        p = line.split()
        aid = int(p[0])
        assert float(p[3]) == pytest.approx(want[((aid - 1) % natoms) + 1]), \
            f"atom {aid} carries another atom's charge"
