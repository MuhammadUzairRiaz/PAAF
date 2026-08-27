"""The Kremer-Grest CG builder: physics pinned numerically.

sigma from the melt condition, canonical FENE constants, the Langevin
inversion for angle stiffness, both mappings, both unit systems — each
asserted with numbers, because a CG model that is silently mis-scaled
runs fine and answers wrong.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from paaf.cell.cg_model import (CGSettings, build_cg_cell, fit_angle_k,
                                langevin)
from paaf.cell.composition import Component, from_chain_counts
from paaf.cell.grow import grow_amorphous_cell


def _cell(smiles="[*]CC[*]", dp=10, chains=2, density=0.85, name="PE"):
    comp = from_chain_counts(
        [Component(name=name, repeat_unit=smiles,
                   degree_of_polymerisation=dp, n_chains=chains,
                   ris_key="PE")], density)
    res = grow_amorphous_cell(comp.grow_specs(), comp.box(),
                              temperature=413.0, seed=7)
    return res, comp.grow_specs()


# ------------------------------------------------------------ angle fit
def test_langevin_inversion_roundtrips():
    for cn in (2.0, 5.0, 8.0):
        k = fit_angle_k(cn)
        l = langevin(k)
        got = (1 + l) / (1 - l)
        print(f"  C_inf {cn} -> K/eps {k:.3f} -> C_inf {got:.3f}")
        assert got == pytest.approx(cn, rel=1e-3)


def test_floppy_targets_need_no_angle_term():
    assert fit_angle_k(1.0) == 0.0
    assert fit_angle_k(0.5) == 0.0


# ------------------------------------------------------------ backbone map
def test_backbone_mapping_lj_units(tmp_path):
    res, specs = _cell()
    data, inp = build_cg_cell(res, specs, tmp_path, settings=CGSettings(
        units="lj", mapping="backbone", angle_mode="off"))
    text = data.read_text()
    # 2 chains x DP 10 x 2 skeletal atoms
    assert "40 atoms" in text
    assert "38 bonds" in text          # 19 per chain
    assert "angles" not in text
    lines = inp.read_text()
    assert "units           lj" in lines
    # canonical FENE: k = 30 eps/sigma^2 with sigma = eps = 1 -> "30.0000 1.5"
    assert "bond_coeff      1 30.0000 1.5000 1.0000 1.0000" in lines
    # KG melt condition: number density 0.85 sigma^-3 exactly, by design
    import re
    m = re.search(r"0.0 ([\d.]+) xlo", text)
    L = float(m.group(1))
    assert 40 / L ** 3 == pytest.approx(0.85, rel=1e-6)


def test_monomer_mapping_halves_the_beads(tmp_path):
    res, specs = _cell()
    data, _ = build_cg_cell(res, specs, tmp_path, settings=CGSettings(
        units="lj", mapping="monomer", angle_mode="off"))
    text = data.read_text()
    assert "20 atoms" in text           # one bead per repeat unit
    assert "18 bonds" in text
    # bead mass in lj units is 1 (reference bead)
    assert "1 1.0000  # PE" in text


def test_real_units_carry_the_temperature(tmp_path):
    res, specs = _cell()
    st = CGSettings(units="real", mapping="backbone", angle_mode="off",
                    temperature_k=413.0)
    data, inp = build_cg_cell(res, specs, tmp_path, settings=st)
    text = data.read_text()
    assert "1 14.0270  # PE" in text    # CH2 bead mass in amu
    lines = inp.read_text()
    assert "units           real" in lines
    eps = 0.0019872041 * 413.0
    assert f"{eps:.4f}" in lines        # eps = kT in kcal/mol


def test_auto_angles_fit_the_measured_cn(tmp_path):
    res, specs = _cell()
    said = []
    _data, inp = build_cg_cell(res, specs, tmp_path, settings=CGSettings(
        units="lj", angle_mode="auto"), measured_cn=6.9,
        progress=said.append)
    assert any("fitted" in m for m in said)
    lines = inp.read_text()
    assert "angle_style     cosine" in lines
    k = fit_angle_k(6.9)
    assert f"angle_coeff     1 {k:.4f}" in lines


def test_two_species_get_two_bead_types(tmp_path):
    comp = from_chain_counts(
        [Component(name="PE", repeat_unit="[*]CC[*]",
                   degree_of_polymerisation=8, n_chains=2, ris_key="PE"),
         Component(name="PS", repeat_unit="[*]CC([*])c1ccccc1",
                   degree_of_polymerisation=8, n_chains=1, ris_key="PS")],
        0.9)
    res = grow_amorphous_cell(comp.grow_specs(), comp.box(),
                              temperature=413.0, seed=3)
    data, inp = build_cg_cell(res, comp.grow_specs(), tmp_path,
                              settings=CGSettings(units="real",
                                                  angle_mode="off"))
    text = data.read_text()
    assert "2 atom types" in text
    assert "# PE" in text and "# PS" in text
    # PS bead (52 amu) must be FATTER than PE bead (14 amu)
    import re
    sigmas = re.search(r"sigma/species: PE=([\d.]+), PS=([\d.]+)",
                       inp.read_text())
    pe, ps = float(sigmas.group(1)), float(sigmas.group(2))
    print(f"  sigma PE {pe:.3f} A, PS {ps:.3f} A")
    assert ps > pe
    # mixed pair coeff (1-2) present
    assert "pair_coeff      1 2" in inp.read_text()


def test_beads_are_inside_the_box(tmp_path):
    res, specs = _cell()
    data, _ = build_cg_cell(res, specs, tmp_path, settings=CGSettings(
        units="real", angle_mode="off"))
    import numpy as np
    rows = []
    in_atoms = False
    for line in data.read_text().splitlines():
        if line.startswith("Atoms"):
            in_atoms = True; continue
        if in_atoms:
            p = line.split()
            if not p:
                continue
            if not p[0].isdigit():
                break
            rows.append([float(x) for x in p[3:6]])
    xyz = np.array(rows)
    assert (xyz >= 0).all()
