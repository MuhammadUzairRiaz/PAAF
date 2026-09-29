"""Packing the box with LAMMPS instead of packmol, on several cores.

packmol has no periodic boundaries and optimises every copy at once, so long
polymer chains at melt density can keep it busy for a very long time. The
LAMMPS packer places the chains, pushes them apart as rigid bodies, then
finishes with a flexible push-off whose bonds, angles and torsions are held at
their built values — periodic, parallel, and with a fixed step budget.

Measured while developing it (100 PIB chains, 24,200 atoms, 0.9 g/cm3, 4
OpenMP threads): 11 s, no two chains closer than 2.0 Å, chain RMSD 1.6 Å from
the builder's conformation. packmol on the same box ran for many minutes.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from paaf.config import BoxCfg, Config
from paaf.lammps_packer import (LammpsBuild, _read_chain, launch_command,
                                place_chains)


# ------------------------------------------------------------ a test chain
def _zigzag_chain_data(path: Path, n: int = 40) -> Path:
    """A united-atom zigzag chain: n beads, bonds, angles, dihedrals."""
    xyz = []
    for i in range(n):
        xyz.append((1.27 * i, 0.44 * (i % 2), 0.0))
    bonds = [(i, i + 1) for i in range(n - 1)]
    angles = [(i, i + 1, i + 2) for i in range(n - 2)]
    dihs = [(i, i + 1, i + 2, i + 3) for i in range(n - 3)]
    lines = ["single chain", "", f"{n} atoms", f"{len(bonds)} bonds",
             f"{len(angles)} angles", f"{len(dihs)} dihedrals", "",
             "1 atom types", "1 bond types", "1 angle types",
             "1 dihedral types", "",
             "-60 60 xlo xhi", "-60 60 ylo yhi", "-60 60 zlo zhi", "",
             "Masses", "", "1 14.027", "", "Atoms  # full", ""]
    lines += [f"{i + 1} 1 1 0.0 {x:.4f} {y:.4f} {z:.4f}"
              for i, (x, y, z) in enumerate(xyz)]
    for title, terms in (("Bonds", bonds), ("Angles", angles),
                         ("Dihedrals", dihs)):
        lines += ["", title, ""]
        lines += [f"{k + 1} 1 " + " ".join(str(a + 1) for a in t)
                  for k, t in enumerate(terms)]
    path.write_text("\n".join(lines) + "\n")
    return path


def _min_interchain_distance(xyz: np.ndarray, per: int, L: float) -> float:
    from scipy.spatial import cKDTree
    wrapped = xyz % L
    mol = np.repeat(np.arange(len(xyz) // per), per)
    pairs = cKDTree(wrapped, boxsize=L).query_pairs(3.0, output_type="ndarray")
    pairs = pairs[mol[pairs[:, 0]] != mol[pairs[:, 1]]]
    if not len(pairs):
        return math.inf
    d = wrapped[pairs[:, 0]] - wrapped[pairs[:, 1]]
    d -= L * np.round(d / L)
    return float(np.linalg.norm(d, axis=1).min())


# ------------------------------------------------------------ config
def test_old_gromacs_target_atoms_applies_to_every_packer():
    box = BoxCfg(gmx_target_atoms=50000)
    assert box.target_atoms == 50000
    assert box.chain_count(242) == 207          # nearest, not floor (206)


def test_chain_count_is_n_chains_without_a_target():
    assert BoxCfg(n_chains=12).chain_count(242) == 12


def test_packmol_off_in_an_old_config_still_means_the_grid():
    assert BoxCfg(packmol=False).packer == "grid"
    assert BoxCfg().packer == "packmol"


def test_new_fields_round_trip_through_a_saved_config(tmp_path):
    cfg = Config(box=BoxCfg(packer="lammps", nprocs=6, target_atoms=30000))
    back = Config.load(cfg.save(tmp_path / "c.json"))
    assert (back.box.packer, back.box.nprocs, back.box.target_atoms) \
        == ("lammps", 6, 30000)


# ------------------------------------------------------------ launching
def test_a_serial_build_is_never_started_under_mpirun():
    """mpirun -np 4 on an MPI-STUBS build runs four copies of the same job."""
    serial = LammpsBuild("lmp", has_mpi=False, has_openmp=False)
    cmd, how = launch_command(serial, 4)
    assert cmd == ["lmp"] and "1 core" in how


def test_openmp_build_uses_threads(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    omp = LammpsBuild("lmp", has_mpi=False, has_openmp=True)
    cmd, how = launch_command(omp, 4)
    assert cmd == ["lmp", "-sf", "omp", "-pk", "omp", "4"]
    assert how == "4 OpenMP threads"


def test_mpi_build_uses_mpirun(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    mpi = LammpsBuild("lmp_mpi", has_mpi=True, has_openmp=False)
    cmd, how = launch_command(mpi, 4)
    assert cmd == ["/usr/bin/mpirun", "-np", "4", "lmp_mpi"]


def test_gromacs_minimisation_uses_the_cores(monkeypatch):
    from paaf.gromacs_packer import mdrun_command
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    assert mdrun_command("/opt/bin/gmx", 4) == ["/opt/bin/gmx", "mdrun",
                                                "-nt", "4"]
    assert mdrun_command("/opt/bin/gmx", 1) == ["/opt/bin/gmx", "mdrun"]


# ------------------------------------------------------------ placement
def test_placement_keeps_every_copy_rigid_and_spreads_them(tmp_path):
    chain = _read_chain(_zigzag_chain_data(tmp_path / "c.data"))
    placed = place_chains(chain.xyz, 27, (60.0, 60.0, 60.0), seed=3,
                          placement="lattice")
    assert placed.shape == (27, 40, 3)
    ref = np.linalg.norm(chain.xyz[1:] - chain.xyz[:-1], axis=1)
    for copy in placed:
        got = np.linalg.norm(copy[1:] - copy[:-1], axis=1)
        assert np.allclose(got, ref, atol=1e-9)
    centres = placed.mean(axis=1)
    assert centres.min() >= 0.0 and centres.max() <= 60.0
    # A lattice spreads them: no two centres on top of each other.
    d = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    assert d[np.triu_indices(27, 1)].min() > 5.0


def test_default_placement_is_random_anywhere_like_packmol(tmp_path):
    chain = _read_chain(_zigzag_chain_data(tmp_path / "c.data"))
    a = place_chains(chain.xyz, 200, (60.0, 60.0, 60.0), seed=5)
    b = place_chains(chain.xyz, 200, (60.0, 60.0, 60.0), seed=5)
    assert np.array_equal(a, b)                  # same seed, same start
    centres = a.mean(axis=1)
    assert centres.min() >= 0.0 and centres.max() <= 60.0
    # Uniform over the cell: every octant gets chains, none gets most.
    octant = (centres >= 30.0) @ np.array([1, 2, 4])
    counts = np.bincount(octant, minlength=8)
    assert counts.min() > 5 and counts.max() < 50
    with pytest.raises(ValueError):
        place_chains(chain.xyz, 3, (60.0, 60.0, 60.0), seed=1,
                     placement="spiral")


# ------------------------------------------------------------ real LAMMPS
def _lammps_or_skip():
    from paaf.cell.relax import find_lammps
    from paaf.lammps_packer import probe_lammps
    exe = find_lammps("")
    if exe is None:
        pytest.skip("no LAMMPS executable")
    b = probe_lammps(str(exe))
    if b.packages and not {"MOLECULE", "RIGID"} <= set(b.packages):
        pytest.skip("LAMMPS lacks MOLECULE/RIGID")
    return exe


def test_lammps_packer_writes_a_packed_box_with_no_close_contacts(tmp_path):
    _lammps_or_skip()
    from paaf.lammps_replicator import (_clean, parse_lammps_data,
                                        replicate_single_chain)
    single = _zigzag_chain_data(tmp_path / "single.data")
    L = 40.0                     # 60 × 40 beads of CH2 ≈ 0.6 g/cm3
    out = replicate_single_chain(single, 60, (L, L, L),
                                 tmp_path / "packed_box.data",
                                 packer="lammps", nprocs=2, seed=11,
                                 tolerance=2.0)
    header, sec = parse_lammps_data(out)
    atoms = sorted((l.split() for l in _clean(sec["Atoms"])),
                   key=lambda p: int(p[0]))
    assert len(atoms) == 60 * 40
    assert len(_clean(sec["Bonds"])) == 60 * 39
    assert len(_clean(sec["Dihedrals"])) == 60 * 37
    xyz = np.array([[float(v) for v in p[4:7]] for p in atoms])
    # Bonds intact...
    bonds = np.array([[int(v) for v in l.split()[2:4]]
                      for l in _clean(sec["Bonds"])]) - 1
    lengths = np.linalg.norm(xyz[bonds[:, 0]] - xyz[bonds[:, 1]], axis=1)
    assert np.abs(lengths - lengths[0]).max() < 0.2
    # ...and no two chains closer than the tolerance, checked independently
    # of LAMMPS, through the periodic boundaries.
    assert _min_interchain_distance(xyz, 40, L) >= 1.99
    assert (tmp_path / "lammps_pack" / "pack.log").exists()


def test_a_missing_lammps_says_what_to_do(tmp_path, monkeypatch):
    from paaf import lammps_packer
    monkeypatch.setattr("paaf.cell.relax.find_lammps", lambda *_a, **_k: None)
    single = _zigzag_chain_data(tmp_path / "single.data")
    with pytest.raises(RuntimeError, match="choose Packmol"):
        lammps_packer.lammps_pack(single, 4, (40.0, 40.0, 40.0))
