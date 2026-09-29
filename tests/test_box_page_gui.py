"""Box page: choose packmol, LAMMPS or the grid; chains or a target atom count.

The GROMACS group only shows when GROMACS files are written, and the LAMMPS
packer choice only when LAMMPS files are, since that is the box it packs.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 not installed")

from PyQt5.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


def test_box_page_offers_the_packers_and_carries_them_into_the_config():
    from paaf.gui.main_window import MainWindow
    w = MainWindow()
    assert [w.box_packer.itemData(i) for i in range(w.box_packer.count())] \
        == ["packmol", "lammps", "grid"]
    assert w.box_packer.currentData() == "packmol"      # packmol stays default
    assert w.box_nprocs.value() == 4

    w.box_packer.setCurrentIndex(w.box_packer.findData("lammps"))
    w.chains_mode.setCurrentIndex(w.chains_mode.findData("atoms"))
    w.box_target_atoms.setValue(40000)
    cfg = w._build_config()
    assert cfg.box.packer == "lammps" and cfg.box.nprocs == 4
    assert cfg.box.lammps_placement == "random"          # packmol-like start
    assert cfg.box.target_atoms == 40000
    assert not w.n_chains.isEnabled()

    w2 = MainWindow()
    w2._apply_config(cfg)
    assert w2.box_packer.currentData() == "lammps"
    assert w2.chains_mode.currentData() == "atoms"
    assert w2.box_target_atoms.value() == 40000


def test_gromacs_group_shows_only_for_gromacs_output():
    from paaf.gui.main_window import MainWindow
    w = MainWindow()
    w.engine_combo.setCurrentText("lammps")
    w._sync_gmx_visibility()
    assert w.gb_gmx.isHidden() and not w.box_packer.isHidden()
    w.engine_combo.setCurrentText("gromacs")
    w._sync_gmx_visibility()
    assert not w.gb_gmx.isHidden()
    assert w.box_packer.isHidden()       # packmol/LAMMPS: LAMMPS box only
    assert not w.box_nprocs.isHidden()           # cores still drive mdrun
