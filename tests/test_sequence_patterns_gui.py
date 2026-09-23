"""The copolymer GUIs expose gradient, multiblock, 3+ monomers and a seed."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 not installed")
pytest.importorskip("rdkit", reason="copolymer dialog needs RDKit for masses")

from PyQt5.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

ISOPRENE = "[*]C/C=C(C)\\C[*]"
EPOXIDE = "[*]CC1(C)OC1C[*]"


def test_dialog_multiblock_pattern_sets_composition_and_preview():
    from paaf.gui.copolymer_dialog import CopolymerDialog
    dlg = CopolymerDialog(dp=15)
    dlg.arrangement_box.setCurrentIndex(dlg.arrangement_box.findData("multiblock"))
    dlg.pattern_edit.setText("AAAAA-BBBB-BBB-AAA")
    print(f"\n  preview: {dlg.preview.text()!r}")
    assert "A5-B7-A3" in dlg.preview.text()
    mons = dlg.monomers()
    assert abs(mons[0].fraction - 8 / 15) < 1e-9
    assert dlg.pattern() == "AAAAA-BBBB-BBB-AAA" and dlg.fill() == "repeat"


def test_dialog_refuses_a_bad_pattern():
    from paaf.gui.copolymer_dialog import CopolymerDialog
    dlg = CopolymerDialog()
    dlg.arrangement_box.setCurrentIndex(dlg.arrangement_box.findData("multiblock"))
    dlg.pattern_edit.setText("A5-D5")
    dlg.accept()
    assert dlg.result() != dlg.Accepted
    assert "monomer D" in dlg.summary.text()


def test_dialog_seed_reaches_the_grow_spec():
    from paaf.gui.amorphous_tab import ComponentRow
    from paaf.cell.sequence import Monomer
    row = ComponentRow(1)
    row.set_copolymer([Monomer(ISOPRENE, 0.5, "iso"),
                       Monomer(EPOXIDE, 0.5, "ep")], "gradient", seed=42)
    comp = row.to_component(weight_mode=True)
    assert comp.arrangement == "gradient" and comp.sequence_seed == 42


def test_builder_adds_a_third_monomer_and_emits_its_fractions():
    from paaf.gui.builder_tab import BuilderTab
    tab = BuilderTab()
    tab.rb_cop.setChecked(True)
    got = []
    tab.copolymer_settings_changed.connect(got.append)
    tab._add_extra_panel()
    assert tab.n_copolymer_monomers() == 3
    tab.co_fractions_multi.setText("2,1,1")
    tab.co_mode.setCurrentText("multiblock")
    tab.co_pattern.setText("A4-B3-C3")
    tab.co_seed.setText("17")
    s = got[-1]
    print(f"\n  settings: {s}\n  preview: {tab.co_preview.text()!r}")
    assert s["n_monomers"] == 3
    assert [round(f, 3) for f in s["fractions"]] == [0.5, 0.25, 0.25]
    assert s["mode"] == "multiblock" and s["pattern"] == "A4-B3-C3"
    assert s["seed"] == 17
    assert "A4-B3-C3" in tab.co_preview.text()
    tab._remove_extra_panel()
    assert tab.n_copolymer_monomers() == 2


def test_builder_settings_reach_the_chain_page_and_config():
    from paaf.gui.main_window import MainWindow
    w = MainWindow()
    b = w.builder_tab
    b.rb_cop.setChecked(True)
    b._add_extra_panel()
    b.co_fractions_multi.setText("0.5,0.3,0.2")
    b.co_mode.setCurrentText("gradient")
    b.co_seed.setText("0")
    assert w.chain_mode.currentText() == "gradient"
    assert w.chain_fractions.text() == "0.5,0.3,0.2"
    assert w.chain_seed.text() == "0"
    b.co_mode.setCurrentText("multiblock")
    b.co_pattern.setText("AAAAA-BBBB-BBB-AAA")
    cfg = w._build_config()
    assert cfg.chain.mode == "multiblock"
    assert cfg.chain.block_pattern == "AAAAA-BBBB-BBB-AAA"
    assert cfg.chain.seed == 0
    w.chain_pattern.clear()
    w._apply_config(cfg)
    assert w.chain_pattern.text() == "AAAAA-BBBB-BBB-AAA"
