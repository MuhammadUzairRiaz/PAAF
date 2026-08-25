"""SMILES-based polymer chain builder — a faithful port of the user's
``create_chain.py`` / ``create_chain_pbs_cooh_mbuild_fixed.py``.

Why this exists
---------------
PAAF's original chain builder went SMILES → OpenBabel Gen3D → PDB →
internal ``Molecule`` → mBuild ``Compound`` → ``Polymer``. That round-trip
corrupted geometry (backbone C-C bonds stretched to ~2 Å), which then made
DL_FIELD's distance-based bond perception fail to connect the backbone and
abort atom typing.

The robust approach (this module) builds the mBuild compound **directly
from SMILES** with ``mb.load(smiles=True)`` — a clean, properly bonded 3D
structure — then uses RDKit + NetworkX graph isomorphism to locate the two
``[*]`` connection atoms, picks the straightest head/tail hydrogen pair,
and assembles the chain with ``mbuild.lib.recipes.polymer.Polymer``.

For polyester ends (right ``[*]`` on a carbonyl carbon, e.g. PBS) the final
chain's terminal ``-C(=O)H`` is capped to ``-C(=O)OH``.

Requires: rdkit, mbuild, networkx, openbabel. If any is missing, the caller
should fall back to the legacy builder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .logging_utils import get_logger
from .structure import Atom, Molecule

log = get_logger(__name__)


def available() -> bool:
    """True if all deps for the SMILES-based builder are importable."""
    try:
        import rdkit          # noqa: F401
        import mbuild         # noqa: F401
        import networkx       # noqa: F401
        from openbabel import openbabel  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- SMILES prep
def _rdkit_and_mbuild_smiles(poly_smiles: str) -> Tuple[str, str]:
    """Derive the RDKit (with ``*``) and mBuild (stripped) SMILES from a
    polymerization SMILES, exactly like ``create_chain.py``:

        rdkit  = smiles.replace('[*]','*').replace('H2','').replace('H3','').replace('H','')
        mbuild = smiles.replace('[*]','').replace('*','').replace('()','')
    """
    rdkit_smiles = (poly_smiles.replace("[*]", "*")
                    .replace("H2", "").replace("H3", "").replace("H", ""))
    mbuild_smiles = (poly_smiles.replace("[*]", "")
                     .replace("*", "").replace("()", ""))
    return rdkit_smiles, mbuild_smiles


# ---------------------------------------------------------------- graph match
def _rdkit_to_graph(rdkit_smiles: str):
    from rdkit import Chem
    import networkx as nx
    mol = Chem.MolFromSmiles(rdkit_smiles)
    if mol is None:
        raise ValueError(f"Invalid RDKit SMILES: {rdkit_smiles!r}")
    mol = Chem.AddHs(mol)
    G = nx.Graph()
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == "*":
            sym = "H"     # treat wildcard as hydrogen for matching
        G.add_node(atom.GetIdx(), element=sym)
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G, mol


def _mbuild_to_graph(comp):
    import networkx as nx
    G = nx.Graph()
    particles = list(comp.particles())
    for i, p in enumerate(particles):
        G.add_node(i, element=p.name)
    for a, b in comp.bonds():
        G.add_edge(particles.index(a), particles.index(b))
    return G, particles


def _match(rdkit_smiles: str, comp):
    from networkx.algorithms import isomorphism as iso
    G_rdkit, mol = _rdkit_to_graph(rdkit_smiles)
    G_mb, particles = _mbuild_to_graph(comp)
    nm = iso.categorical_node_match("element", None)
    matcher = iso.GraphMatcher(G_rdkit, G_mb, node_match=nm)
    if not matcher.is_isomorphic():
        raise ValueError(
            "RDKit ↔ mBuild graph mismatch — SMILES parsed inconsistently.")
    return matcher.mapping, mol, particles


# ---------------------------------------------------------------- PBS cap
def _right_dummy_is_carbonyl(poly_smiles: str) -> bool:
    """True if the RIGHT ``[*]`` is bonded to a carbonyl carbon (C=O)."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(poly_smiles)
    if mol is None:
        return False
    dummies = sorted(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
    if len(dummies) != 2:
        return False
    right = mol.GetAtomWithIdx(dummies[-1])
    if right.GetDegree() != 1:
        return False
    nbr = right.GetNeighbors()[0]
    if nbr.GetSymbol() != "C":
        return False
    return any(b.GetBondType() == Chem.BondType.DOUBLE
               and b.GetOtherAtom(nbr).GetSymbol() == "O"
               for b in nbr.GetBonds())


def _cap_terminal_carboxylic_acid(chain) -> None:
    """Convert the terminal PBS ``-C(=O)H`` end to ``-C(=O)OH`` in mBuild.

    Finds the unique C with exactly (1 O, 1 C, 1 H) heavy/H neighbours —
    the aldehyde end left after the right ``[*]`` H is removed. mBuild
    coords are in nm: C-O = 0.134 nm, O-H = 0.097 nm.
    """
    import mbuild as mb
    particles = list(chain.particles())
    bonds = list(chain.bonds())

    def neighbours(atom):
        out = []
        for a, b in bonds:
            if a is atom: out.append(b)
            elif b is atom: out.append(a)
        return out

    candidates = []
    for atom in particles:
        if atom.name != "C":
            continue
        nbs = neighbours(atom)
        o = [n for n in nbs if n.name == "O"]
        c = [n for n in nbs if n.name == "C"]
        h = [n for n in nbs if n.name == "H"]
        heavy = [n for n in nbs if n.name != "H"]
        if len(o) == 1 and len(c) == 1 and len(h) == 1 and len(heavy) == 2:
            candidates.append((atom, h[0], c[0]))

    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one terminal -C(=O)H site, found {len(candidates)}")

    carbon, old_h, chain_nb = candidates[0]
    vec = np.asarray(old_h.pos) - np.asarray(carbon.pos)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        vec = np.asarray(carbon.pos) - np.asarray(chain_nb.pos)
        norm = float(np.linalg.norm(vec)) or 1.0
    unit = vec / norm
    o_pos = np.asarray(carbon.pos) + 0.134 * unit
    h_pos = o_pos + 0.097 * unit
    chain.remove(old_h)
    new_o = mb.Particle(name="O", pos=o_pos)
    new_h = mb.Particle(name="H", pos=h_pos)
    chain.add(new_o); chain.add(new_h)
    chain.add_bond((carbon, new_o))
    chain.add_bond((new_o, new_h))
    log.info("Capped terminal -C(=O)H → -C(=O)OH (polyester end).")


# ------------------------------------------------------------ monomer prep
def _prepare_monomer(poly_smiles: str, separation: float):
    """Load one ``[*]…[*]`` monomer as a clean mBuild compound and return
    ``(compound, (h1, h2))`` — the compound plus the indices of the head/tail
    hydrogens to replace with inter-monomer ports.

    Same logic as ``create_chain.py``: mb.load(smiles) → RDKit/mBuild graph
    match → locate the two ``*`` connection atoms → pick the straightest
    (most antiparallel) removable-H pair.
    """
    import mbuild as mb

    rdkit_smiles, mbuild_smiles = _rdkit_and_mbuild_smiles(poly_smiles)
    log.info("SMILES monomer: rdkit=%r  mbuild=%r", rdkit_smiles, mbuild_smiles)

    comp = mb.load(mbuild_smiles, smiles=True)
    mapping, mol, _ = _match(rdkit_smiles, comp)

    star_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == "*"]
    connected = [a.GetNeighbors()[0] for a in star_atoms]
    rdkit_heavy = [a.GetIdx() for a in connected]
    mbuild_heavy = [mapping[i] for i in rdkit_heavy]

    particles = list(comp.particles())
    bonds = list(comp.bonds())

    def bonded(atom):
        out = []
        for a, b in bonds:
            if a is atom: out.append(b)
            elif b is atom: out.append(a)
        return out

    h_per_heavy = {}
    for idx in mbuild_heavy:
        heavy = particles[idx]
        h_per_heavy[idx] = [particles.index(p) for p in bonded(heavy) if p.name == "H"]

    c1, c2 = mbuild_heavy
    if not h_per_heavy[c1] or not h_per_heavy[c2]:
        raise ValueError(
            "A connection atom has no removable hydrogen — cannot polymerise. "
            "Check the SMILES has an implicit-H site at each [*].")

    def angle(v1, v2):
        v1 = v1 / np.linalg.norm(v1); v2 = v2 / np.linalg.norm(v2)
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

    best_pair, best_ang = None, 181.0
    for h1 in h_per_heavy[c1]:
        for h2 in h_per_heavy[c2]:
            v1 = particles[h1].pos - particles[c1].pos
            v2 = particles[h2].pos - particles[c2].pos
            ang = angle(v1, -v2)   # want v1 and -v2 parallel → 0°
            if ang < best_ang:
                best_ang, best_pair = ang, (h1, h2)
    return comp, best_pair


# ---------------------------------------------------------------- main entry
def build_chain_from_smiles(
    poly_smiles: str,
    n: int,
    separation: float = 0.15,
    optimize: bool = True,
    opt_ff: str = "UFF",
    opt_steps: int = 10000,
    opt_tol: float = 1.0e-6,
    cap_carboxyl_end: bool = True,
    name: str = "chain",
) -> Molecule:
    """Build an n-unit homopolymer chain from a ``[*]…[*]`` SMILES.

    Returns an internal :class:`Molecule` with correct connectivity and
    geometry. Mirrors ``create_chain.py``: mb.load(smiles) → graph match →
    straightest H pair → Polymer.build → (PBS cap) → UFF optimise.
    """
    from mbuild.lib.recipes.polymer import Polymer

    comp, best_pair = _prepare_monomer(poly_smiles, separation)

    # assemble the homopolymer
    chain = Polymer()
    chain.add_monomer(compound=comp, indices=list(best_pair),
                      separation=separation, replace=True)
    chain.build(n=n, sequence="A")

    # PBS-style terminal carboxyl cap
    if cap_carboxyl_end and _right_dummy_is_carbonyl(poly_smiles):
        try:
            _cap_terminal_carboxylic_acid(chain)
        except RuntimeError as e:
            log.warning("Carboxyl capping skipped: %s", e)

    # convert to internal Molecule (nm → Å) and UFF-optimise
    out = _mbuild_to_internal(chain, name=name)
    if optimize:
        from . import optimizer
        optimizer.optimize(out, ff=opt_ff, steps=opt_steps, tol=opt_tol)
    return out


# ------------------------------------------------------------ copolymer entry
def _seq_letters(sequence):
    """Map an explicit monomer-index sequence to mBuild's letter alphabet.

    Returns ``(unique_indices, idx_to_letter, seq_str)``. Only the monomer
    indices that actually appear get a letter, so the number of unique
    sequence letters matches the number of monomers added to the Polymer
    (mBuild raises otherwise). Deterministic: letters follow sorted index order.

    >>> _seq_letters([0, 1, 0, 1])
    ([0, 1], {0: 'A', 1: 'B'}, 'ABAB')
    >>> _seq_letters([1, 1, 0])
    ([0, 1], {0: 'A', 1: 'B'}, 'BBA')
    """
    seq = [int(i) for i in sequence]
    if not seq:
        raise ValueError("Empty copolymer sequence.")
    unique = sorted(set(seq))
    idx_to_letter = {mi: chr(ord("A") + k) for k, mi in enumerate(unique)}
    seq_str = "".join(idx_to_letter[i] for i in seq)
    return unique, idx_to_letter, seq_str


def build_copolymer_from_smiles(
    poly_smiles_list,
    sequence,
    separation: float = 0.15,
    optimize: bool = True,
    opt_ff: str = "UFF",
    opt_steps: int = 10000,
    opt_tol: float = 1.0e-6,
    cap_carboxyl_end: bool = True,
    name: str = "copolymer",
) -> Molecule:
    """Build a multi-monomer copolymer chain from per-monomer ``[*]…[*]`` SMILES.

    Parameters
    ----------
    poly_smiles_list : list[str]
        Polymerization SMILES for each distinct monomer (index 0, 1, ...).
    sequence : list[int]
        The explicit monomer order along the chain, e.g. ``[0,1,0,1,...]`` for
        alternating, or a block/random pattern. Its length is the total number
        of monomer units.

    Uses the same clean ``mb.load(smiles)`` path as the homopolymer builder but
    assembles with mBuild's multi-monomer ``Polymer`` recipe so the units are
    properly bonded head-to-tail — no PDB round-trip, no clumped/dangling
    fragments.
    """
    from mbuild.lib.recipes.polymer import Polymer

    seq = [int(i) for i in sequence]
    # Only add the monomers that actually appear, so the number of unique
    # sequence letters matches the number of monomers added (mBuild requires it).
    unique, _idx_to_letter, seq_str = _seq_letters(seq)

    chain = Polymer()
    for mi in unique:
        comp, pair = _prepare_monomer(poly_smiles_list[mi], separation)
        chain.add_monomer(compound=comp, indices=list(pair),
                          separation=separation, replace=True)

    log.info("SMILES copolymer: %d monomer types, %d units, sequence=%s%s",
             len(unique), len(seq), seq_str[:48],
             "..." if len(seq_str) > 48 else "")

    # n=1 with the full explicit sequence string builds exactly this order
    # (mBuild builds n * len(sequence) units, cycling the sequence).
    chain.build(n=1, sequence=seq_str)

    # Cap only if the LAST unit along the chain is a polyester right-end.
    if cap_carboxyl_end and _right_dummy_is_carbonyl(poly_smiles_list[seq[-1]]):
        try:
            _cap_terminal_carboxylic_acid(chain)
        except RuntimeError as e:
            log.warning("Carboxyl capping skipped: %s", e)

    out = _mbuild_to_internal(chain, name=name)
    if optimize:
        from . import optimizer
        optimizer.optimize(out, ff=opt_ff, steps=opt_steps, tol=opt_tol)
    return out


def _mbuild_to_internal(comp, name: str = "chain") -> Molecule:
    parts = list(comp.particles())
    atoms = [Atom(index=i, element=p.name,
                  xyz=np.asarray(p.pos) * 10.0, name=f"{p.name}{i + 1}")
             for i, p in enumerate(parts)]
    bonds = []
    for a, b in comp.bonds():
        # mBuild carries no bond order, so everything arrives single and the
        # monomer's C=C is lost. Recovered below — without it the SMARTS typer
        # cannot see an alkene and types every carbon as saturated.
        bonds.append((parts.index(a), parts.index(b), 1.0))
    mol = Molecule(atoms=atoms, bonds=bonds, name=name)
    try:
        from .bond_orders import order_summary, perceive_bond_orders
        raised = perceive_bond_orders(mol)
        if raised:
            log.info("%s: recovered %d multiple bonds (%s)",
                     name, raised, order_summary(mol))
    except Exception as exc:                            # pragma: no cover
        log.warning("%s: bond orders could not be recovered (%s); alkene and "
                    "carbonyl atoms will be typed as saturated.", name, exc)
    return mol
