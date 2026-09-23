"""Gradient and multiblock copolymers, multi-monomer chains, and seeds.

Both sequence engines — the single-chain Builder (chain_builder) and the
amorphous cell (cell.sequence) — must agree on every arrangement.
"""
import pytest

from paaf.chain_builder import _make_sequence
from paaf.sequence_patterns import (
    gradient_order, multiblock_order, parse_block_pattern, run_length,
)

ISOPRENE = "[*]C/C=C(C)\\C[*]"
EPOXIDE = "[*]CC1(C)OC1C[*]"
PE = "[*]CC[*]"


def _spell(order):
    return "".join(chr(ord("A") + i) for i in order)


# ------------------------------------------------------------- multiblock
def test_multiblock_accepts_spelled_out_sections():
    sections = parse_block_pattern("AAAAA-BBBB-BBB-AAA", 2)
    assert sections == [(0, 5), (1, 4), (1, 3), (0, 3)]
    assert _spell(multiblock_order(sections, 15)) == "AAAAABBBBBBBAAA"


def test_multiblock_letter_length_and_groups():
    assert parse_block_pattern("A5-B4-B3-A3") == [(0, 5), (1, 4), (1, 3), (0, 3)]
    assert parse_block_pattern("(A2-B1)x2 C3", 3) == [
        (0, 2), (1, 1), (0, 2), (1, 1), (2, 3)]
    assert parse_block_pattern("a:2, b:3") == [(0, 2), (1, 3)]


def test_multiblock_repeats_or_stretches_to_the_chain():
    s = parse_block_pattern("A5-B5")
    assert run_length(multiblock_order(s, 30, "repeat")) == "A5-B5-A5-B5-A5-B5"
    assert run_length(multiblock_order(s, 30, "stretch")) == "A15-B15"
    # longer than the chain: cut, never overrun
    assert len(multiblock_order(s, 7)) == 7


@pytest.mark.parametrize("bad", ["", "A5-X", "AB3", "A0", "(A5-B5", "C4"])
def test_multiblock_rejects_bad_patterns(bad):
    with pytest.raises(ValueError):
        parse_block_pattern(bad, 2)


# --------------------------------------------------------------- gradient
def test_gradient_keeps_exact_counts_and_drifts_from_a_to_b():
    order = gradient_order([0.5, 0.5], 100, seed=4)
    assert order.count(0) == 50 and order.count(1) == 50
    first, last = order[:25], order[-25:]
    assert first.count(0) > 20 and last.count(1) > 20
    assert run_length(order).count("-") > 1, "gradient came out as a sharp block"


def test_gradient_is_reproducible_by_seed():
    assert gradient_order([0.3, 0.7], 60, seed=11) == \
        gradient_order([0.3, 0.7], 60, seed=11)
    assert gradient_order([0.3, 0.7], 60, seed=11) != \
        gradient_order([0.3, 0.7], 60, seed=12)


def test_gradient_works_for_three_monomers():
    order = gradient_order([1, 1, 1], 60, seed=0)
    assert [order.count(i) for i in range(3)] == [20, 20, 20]
    assert order[:10].count(0) > order[-10:].count(0)
    assert order[-10:].count(2) > order[:10].count(2)


# ------------------------------------------------------ Builder engine
def test_builder_engine_supports_new_modes_and_three_monomers():
    assert _spell(_make_sequence(2, 15, "multiblock", None, None, None,
                                 block_pattern="AAAAA-BBBB-BBB-AAA")) \
        == "AAAAABBBBBBBAAA"
    assert _make_sequence(3, 6, "alternating", None, None, None) == [0, 1, 2] * 2
    g = _make_sequence(3, 30, "gradient", [0.5, 0.3, 0.2], None, 5)
    assert [g.count(i) for i in range(3)] == [15, 9, 6]
    r1 = _make_sequence(3, 40, "random", [0.5, 0.3, 0.2], None, 9)
    r2 = _make_sequence(3, 40, "random", [0.5, 0.3, 0.2], None, 9)
    assert r1 == r2 and set(r1) == {0, 1, 2}
    with pytest.raises(ValueError, match="monomer C"):
        _make_sequence(2, 10, "multiblock", None, None, None,
                       block_pattern="A5-C5")


def test_chain_config_round_trips_the_pattern(tmp_path):
    from paaf.config import ChainCfg, Config
    cfg = Config(chain=ChainCfg(n_monomers=15, mode="multiblock",
                                block_pattern="A5-B4-B3-A3",
                                block_fill="stretch", seed=0))
    path = cfg.save(tmp_path / "c.yaml")
    back = Config.load(path)
    assert back.chain.block_pattern == "A5-B4-B3-A3"
    assert back.chain.block_fill == "stretch"
    assert back.chain.seed == 0


# --------------------------------------------------------- cell engine
def test_cell_engine_multiblock_and_gradient():
    pytest.importorskip("rdkit")
    from paaf.cell.sequence import Monomer, build_sequence
    mons = [Monomer(ISOPRENE, 0.5, "iso"), Monomer(EPOXIDE, 0.5, "ep"),
            Monomer(PE, 0.0, "pe")]
    # multiblock keeps a zero-fraction monomer: the pattern names it
    seq = build_sequence(mons, 12, "multiblock", pattern="A4-C4-B4")
    assert _spell(seq.order) == "AAAACCCCBBBB"
    g1 = build_sequence(mons[:2], 40, "gradient", seed=3)
    g2 = build_sequence(mons[:2], 40, "gradient", seed=3)
    assert g1.order == g2.order and g1.order.count(0) == 20


def test_grow_spec_carries_seed_and_pattern_per_chain():
    pytest.importorskip("rdkit")
    from paaf.cell.composition import Component
    from paaf.cell.sequence import Monomer
    comp = Component(name="ABA", repeat_unit=ISOPRENE,
                     degree_of_polymerisation=15,
                     monomers=[Monomer(ISOPRENE, 0.5, "iso"),
                               Monomer(EPOXIDE, 0.5, "ep")],
                     arrangement="multiblock",
                     block_pattern="AAAAA-BBBB-BBB-AAA", sequence_seed=7)
    comp.resolve_chemistry()
    from paaf.cell.composition import CellComposition
    spec = CellComposition(components=[comp], chain_counts=[2]).grow_specs()[0]
    assert spec.block_pattern == "AAAAA-BBBB-BBB-AAA"
    assert spec.sequence_seed == 7
    assert _spell(spec.build_sequence(0).order) == "AAAAABBBBBBBAAA"
    # composition-weighted mass follows the pattern (8 A : 7 B), not 50/50
    iso, ep = comp.monomers
    expect = (8 * iso.mass_amu + 7 * ep.mass_amu) / 15
    assert abs(comp.unit_mass - expect) < 1e-6
