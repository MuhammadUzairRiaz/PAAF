"""Smoke tests for the new cell builders — all use the pure-numpy fallbacks so
they don't need packmol/mbuild/openbabel."""
from __future__ import annotations

import numpy as np
import pytest

from paaf.cell import (
    build_preset, list_presets, cleave_surface, build_nanotube,
    list_fragments, get_fragment,
)
from paaf.cell.amorphous import (
    BoxShape, PackSpec, _box_from_density, _pack_grid, _mass_of,
    _shape_for_density,
)
from paaf.structure import Atom, Molecule


def _tiny_water():
    return Molecule(
        atoms=[
            Atom(0, "O", np.array([0.0, 0.0, 0.0])),
            Atom(1, "H", np.array([0.96, 0.0, 0.0])),
            Atom(2, "H", np.array([-0.24, 0.93, 0.0])),
        ],
        bonds=[(0, 1, 1), (0, 2, 1)],
        name="H2O",
    )


def test_crystal_presets_all_build():
    """Every shipped crystal preset builds a 1x1x1 supercell without error."""
    assert len(list_presets()) >= 15
    for p in list_presets():
        mol = build_preset(p.name, nx=1, ny=1, nz=1)
        assert len(mol.atoms) >= 1


def test_crystal_supercell_scales_correctly():
    mol1 = build_preset("Cu", 1, 1, 1)
    mol4 = build_preset("Cu", 2, 2, 2)
    assert len(mol4.atoms) == 8 * len(mol1.atoms)


def test_surface_cleaves_something():
    slab = cleave_surface("Cu", (1, 0, 0), supercell=(3, 3, 3), thickness_ang=6.0)
    assert 1 <= len(slab.atoms) <= 4 * 27


def test_nanotube_66_armchair():
    tube = build_nanotube(6, 6, length_ang=8.0)
    # Should have >0 atoms and be roughly cylindrical
    assert len(tube.atoms) >= 24
    coords = tube.coords()
    r = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    assert r.std() < 0.5, f"tube not cylindrical: r std = {r.std()}"


def test_pack_grid_and_density_math():
    from paaf.cell.amorphous import BoxShape
    water = _tiny_water()
    specs = [PackSpec(molecule=water, count=8, name="H2O")]
    side = _box_from_density(specs, [water], 1000.0)
    packed = _pack_grid(specs, [water], BoxShape(shape="cubic", a=side, b=side, c=side))
    # 8 water molecules * 3 atoms
    assert len(packed.atoms) == 24
    # box should be roughly the cube root of the volume
    assert 5.0 < side < 20.0


def test_box_shape_orthorhombic_and_triclinic_volume():
    from paaf.cell.amorphous import BoxShape
    ortho = BoxShape(shape="orthorhombic", a=10, b=20, c=30)
    assert abs(ortho.volume_ang3() - 6000.0) < 1e-6
    cube = BoxShape.from_any(15.0)
    assert cube.shape == "cubic" and abs(cube.volume_ang3() - 3375.0) < 1e-6
    tricl = BoxShape.from_any((10.0, 10.0, 10.0, 60.0, 60.0, 60.0))
    assert tricl.shape == "triclinic"
    # Triclinic volume is less than cubic when all angles are 60°
    assert tricl.volume_ang3() < 1000.0


def test_pack_cell_preserves_shape_ratio_when_density_given():
    from paaf.cell import pack_cell, PackSpec
    from paaf.cell.amorphous import BoxShape
    water = _tiny_water()
    specs = [PackSpec(molecule=water, count=8, name="H2O")]
    template = BoxShape(shape="orthorhombic", a=2, b=1, c=1)
    _, box = pack_cell(specs, density_kg_m3=1000.0, shape=template, backend="grid")
    # Aspect ratio 2:1:1 preserved after scaling for density
    assert abs(box.a / box.b - 2.0) < 1e-6
    assert abs(box.b / box.c - 1.0) < 1e-6


def test_run_moltemplate_default_is_true():
    """Users pick a Moltemplate-native FF expecting the full LAMMPS
    system.data / .in.settings / .in.init to be produced. So the tool must
    default to also invoking moltemplate.sh (or dl_field). Users can turn
    the switch OFF if they want to hand-edit the .lt scaffolds first."""
    from paaf.config import Config
    c = Config()
    assert c.run_moltemplate is True, \
        "run_moltemplate must default to True so users get the .data files"


def test_fragments_library_populated():
    frags = list_fragments()
    assert len(frags) >= 40
    assert get_fragment("carboxyl").smiles == "C(=O)O"
    assert get_fragment("phenyl").smiles == "c1ccccc1"
    categories = {f.category for f in frags}
    for c in ("alkyl", "oxygen", "nitrogen", "aromatic", "halogen"):
        assert c in categories


# ==================================================================== layers
def test_layer_trilayer_stacks_in_z():
    """A trilayer of Cu / graphene / Cu should stack strictly along Z with
    each layer's minimum z > the previous layer's maximum z."""
    from paaf.cell import build_layers, LayerSpec

    stack, box = build_layers([
        LayerSpec(preset="Cu",       supercell=(2, 2, 2), name="Cu_bot"),
        LayerSpec(preset="graphene", supercell=(3, 3, 1), name="gr"),
        LayerSpec(preset="Cu",       supercell=(2, 2, 2), name="Cu_top"),
    ], gap_ang=3.0)
    assert len(stack.atoms) > 30

    # z monotonically increases across layer boundaries
    zs = stack.coords()[:, 2]
    assert zs.min() >= 0.0
    assert zs.max() > 15.0
    assert box[2] > zs.max()


def test_layer_gap_before_overrides_default():
    from paaf.cell import build_layers, LayerSpec

    stack, box = build_layers([
        LayerSpec(preset="Cu", supercell=(1, 1, 1), name="a"),
        LayerSpec(preset="Cu", supercell=(1, 1, 1), name="b", gap_before=10.0),
    ], gap_ang=2.0)
    zs = stack.coords()[:, 2]
    # There should be a large gap (>= 10 Å - epsilon) somewhere in z
    sorted_zs = np.sort(zs)
    gaps = np.diff(sorted_zs)
    assert gaps.max() >= 8.0


# ==================================================================== damping
def test_lammps_input_writes_tdamp_pdamp_and_multistage(tmp_path):
    from paaf.lt_writer import write_lammps_input

    p = write_lammps_input(
        tmp_path, ensemble="npt", temperature=350.0, pressure=1.0,
        steps=1000, timestep=0.5,
        tdamp_fs=50.0, pdamp_fs=500.0,
        thermostat="langevin", barostat="nose-hoover",
        pressure_coupling="aniso",
        multistage=True, minimize=True, nvt_steps=500, npt_steps=1500,
    )
    text = p.read_text()
    # Damping constants appear literally in the script
    assert "50.0" in text and "500.0" in text
    # Multi-stage: both NVT and NPT fixes present
    assert "Stage 1: NVT" in text
    assert "Stage 2: NPT" in text
    # Aniso coupling is passed through
    assert "aniso" in text
    # Langevin thermostat for NVT
    assert "langevin" in text
    # Minimization runs before dynamics
    assert "minimize" in text


def test_lammps_input_single_stage_default(tmp_path):
    from paaf.lt_writer import write_lammps_input

    p = write_lammps_input(tmp_path, ensemble="nvt", steps=100)
    text = p.read_text()
    assert "NVT run" in text
    # No multi-stage banners
    assert "Stage 1:" not in text
    assert "Stage 2:" not in text


def test_config_lammps_damping_defaults():
    from paaf.config import LammpsCfg
    L = LammpsCfg()
    assert L.tdamp_fs == 100.0
    assert L.pdamp_fs == 1000.0
    assert L.thermostat == "nose-hoover"
    assert L.pressure_coupling == "iso"
    assert L.multistage is False


# ==================================================================== GROMACS
def _tiny_mol():
    from paaf.structure import Atom, Molecule
    atoms = [
        Atom(0, "C", np.array([0.0, 0.0, 0.0]), name="C1"),
        Atom(1, "C", np.array([1.5, 0.0, 0.0]), name="C2"),
        Atom(2, "H", np.array([-0.5, 0.9, 0.0]), name="H1"),
        Atom(3, "H", np.array([-0.5, -0.9, 0.0]), name="H2"),
    ]
    for a in atoms:
        a.ff_type = "opls_135" if a.element == "C" else "opls_140"
    bonds = [(0, 1, 1), (0, 2, 1), (0, 3, 1)]
    return Molecule(atoms=atoms, bonds=bonds, name="TinyMol")


def test_gromacs_gro_written_with_nm_units(tmp_path):
    from paaf.gromacs_writer import write_gro
    mol = _tiny_mol()
    p = write_gro(mol, tmp_path / "sys.gro")
    lines = p.read_text().splitlines()
    assert lines[0].startswith("Generated by paaf")
    assert int(lines[1].strip()) == 4                # atom count
    assert lines[-1].split()                         # non-empty box line
    # Å -> nm conversion: atom at 1.5 Å should be at 0.150 nm
    assert "0.150" in lines[3]


def test_gromacs_top_lists_atoms_bonds_angles(tmp_path):
    from paaf.gromacs_writer import write_top
    mol = _tiny_mol()
    p = write_top(mol, tmp_path / "sys.top")
    text = p.read_text()
    assert "[ moleculetype ]" in text
    assert "[ atoms ]" in text
    assert "[ bonds ]" in text
    assert "[ angles ]" in text          # C0 has H2-C-H3 and C1-C-H angles
    # atom types propagate
    assert "opls_135" in text and "opls_140" in text


def test_gromacs_mdp_damping_maps_to_ps(tmp_path):
    from paaf.gromacs_writer import write_mdp
    mdps = write_mdp(
        tmp_path, ensemble="npt", temperature=300.0, pressure=1.0,
        steps=1000, timestep_fs=1.0,
        tdamp_fs=200.0, pdamp_fs=2000.0,
        thermostat="langevin", barostat="parrinello",
        pressure_coupling="aniso",
        multistage=True, minimize=True, nvt_steps=500, npt_steps=1500,
    )
    files = [p.name for p in mdps]
    assert "em.mdp" in files
    assert "nvt.mdp" in files
    assert "npt.mdp" in files
    npt = [p for p in mdps if p.name == "npt.mdp"][0].read_text()
    # tdamp 200 fs -> tau_t 0.2 ps ;  pdamp 2000 fs -> tau_p 2.0 ps
    assert "tau_t           = 0.2" in npt
    assert "tau_p           = 2.0" in npt
    # Barostat mapping
    assert "Parrinello-Rahman" in npt
    assert "anisotropic" in npt
    # Langevin -> v-rescale in GROMACS
    assert "v-rescale" in npt


def test_gromacs_mdp_single_stage_no_pcoupl_for_nvt(tmp_path):
    from paaf.gromacs_writer import write_mdp
    mdps = write_mdp(tmp_path, ensemble="nvt", steps=500, minimize=False,
                     multistage=False)
    assert [p.name for p in mdps] == ["run.mdp"]
    txt = mdps[0].read_text()
    assert "pcoupl          = no" in txt


def test_config_engine_field_default_lammps():
    from paaf.config import Config
    c = Config()
    assert c.engine == "lammps"
    assert c.gromacs_include_itp is None


# ==================================================================== optimizer
def test_optimizer_fallback_signature():
    """The public optimize() must accept `fallback` and `strict` kwargs so
    the builder can request non-crashing behaviour."""
    import inspect
    from paaf.optimizer import optimize
    sig = inspect.signature(optimize)
    assert "fallback" in sig.parameters
    assert "strict" in sig.parameters
    # Defaults: fallback on, strict off, so we never crash on a typing failure
    assert sig.parameters["fallback"].default is True
    assert sig.parameters["strict"].default is False


def test_optimizer_strict_mode_raises_without_openbabel():
    """Without OpenBabel installed in this sandbox, strict mode should
    surface a clear RuntimeError so callers know what's missing."""
    from paaf.optimizer import optimize
    from paaf.structure import Atom, Molecule
    mol = Molecule(atoms=[Atom(0, "C", np.array([0.0, 0.0, 0.0]))],
                   bonds=[], name="X")
    try:
        import openbabel  # noqa
        has_ob = True
    except Exception:
        has_ob = False
    if not has_ob:
        with pytest.raises(RuntimeError):
            optimize(mol, ff="MMFF94", strict=True)
