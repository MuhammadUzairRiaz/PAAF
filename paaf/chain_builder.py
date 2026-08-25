"""Polymer chain builder.

Two backends are provided:

- ``mbuild_backend``: uses mbuild.lib.recipes.polymer.Polymer for chain assembly
  (recommended, matches the user's existing scripts).
- ``simple_backend``: a self-contained numpy-only backend that translates
  monomers along the head->tail vector, removes cap atoms, and adds bonds.
  Used as a fallback if mbuild is not installed and for unit tests.

Both produce a :class:`Molecule` representing the full chain.
"""
from __future__ import annotations

import random
from typing import List, Literal, Optional, Sequence

import numpy as np

from .logging_utils import get_logger
from .monomer import Monomer
from .structure import Atom, Molecule

log = get_logger(__name__)

SequenceMode = Literal["homopolymer", "block", "alternating", "random"]


# ================================================================ SIMPLE BACKEND
def _copy_monomer(m: Monomer, offset: int) -> tuple[list[Atom], list[tuple[int, int, float]]]:
    atoms = []
    for a in m.molecule.atoms:
        atoms.append(
            Atom(
                index=a.index + offset,
                element=a.element,
                xyz=a.xyz.copy(),
                name=f"{a.name}",
                charge=a.charge,
                ff_type=a.ff_type,
            )
        )
    bonds = [(i + offset, j + offset, o) for (i, j, o) in m.molecule.bonds]
    return atoms, bonds


def simple_backend(
    monomers: Sequence[Monomer],
    sequence: Sequence[int],
    bond_length: float = 1.54,
    conformer_seed: Optional[int] = None,
    sample_torsions: bool = True,
) -> Molecule:
    """Assemble a chain by rigid translation, then give it a real shape.

    ``sequence`` is a list of monomer indices (into ``monomers``) giving the
    order of monomers in the chain.

    This is the no-dependency fallback used when mBuild is not installed. The
    translation step alone produces a chain that is chemically impossible —
    see the note at the end of the function — so the backbone geometry is
    rebuilt before the chain is returned.
    """
    atoms: List[Atom] = []
    bonds: List[tuple[int, int, float]] = []
    prev_tail_idx: Optional[int] = None
    prev_tail_pos: Optional[np.ndarray] = None
    prev_tail_dir: Optional[np.ndarray] = None
    removed_global: set[int] = set()
    offset = 0

    for seq_pos, mono_idx in enumerate(sequence):
        m = monomers[mono_idx]
        n = m.n_atoms

        head_pos = m.molecule.atoms[m.head_index].xyz
        tail_pos = m.molecule.atoms[m.tail_index].xyz
        head_to_tail = tail_pos - head_pos
        htlen = np.linalg.norm(head_to_tail)
        if htlen < 1e-9:
            raise ValueError(f"Monomer {m.name}: head==tail?")
        head_to_tail_dir = head_to_tail / htlen

        # Where to place this monomer's head
        if seq_pos == 0:
            translate = -head_pos  # origin
        else:
            target = prev_tail_pos + prev_tail_dir * bond_length  # type: ignore[operator]
            translate = target - head_pos

        atoms_new, bonds_new = _copy_monomer(m, offset)
        for a in atoms_new:
            a.xyz += translate
        atoms.extend(atoms_new)
        bonds.extend(bonds_new)

        # Bond previous tail -> current head
        if prev_tail_idx is not None:
            bonds.append((prev_tail_idx, m.head_index + offset, 1.0))

        # Mark cap atoms for removal.  Head caps are only removed when this is
        # not the first monomer; tail caps only when not the last.
        if seq_pos > 0:
            for hc in m.head_removes:
                removed_global.add(hc + offset)
        if seq_pos < len(sequence) - 1:
            for tc in m.tail_removes:
                removed_global.add(tc + offset)

        prev_tail_idx = m.tail_index + offset
        prev_tail_pos = atoms_new[m.tail_index].xyz + 0.0  # already translated
        prev_tail_dir = head_to_tail_dir
        offset += n

    # Compact indices after removing capped atoms
    keep = [a for a in atoms if a.index not in removed_global]
    remap = {a.index: i for i, a in enumerate(keep)}
    for i, a in enumerate(keep):
        a.index = i
    kept_bonds = [
        (remap[i], remap[j], o)
        for (i, j, o) in bonds
        if i in remap and j in remap
    ]
    chain = Molecule(atoms=keep, bonds=kept_bonds, name="chain")

    # Every unit above was placed by translating along the PREVIOUS unit's
    # head-to-tail direction. That makes consecutive backbone bonds parallel,
    # so every backbone valence angle comes out at exactly 180° and the chain
    # is a straight rod with end-to-end distance equal to contour length.
    #
    # No amount of energy minimisation repairs this. Minimisation follows the
    # gradient downhill, and a rod is a local minimum: the backbone atoms sit
    # *on* every torsional rotation axis, so rotating about a backbone bond
    # does not move them at all. The optimiser runs, reports a genuine energy
    # drop from tidying bond lengths, and hands back the same rod.
    #
    # So the geometry is rebuilt here instead: real valence angles, and
    # torsions drawn from rotational-isomeric-state populations. Minimisation
    # afterwards then does what it is actually good at.
    try:
        from .chain_conformer import ConformerSettings, rebuild_backbone
        result = rebuild_backbone(chain, ConformerSettings(
            seed=conformer_seed, sample_torsions=sample_torsions))
        if result.ran:
            log.info("simple_backend: %s", result.summary())
    except Exception as exc:                            # pragma: no cover
        log.warning("Could not rebuild backbone geometry (%s); the chain is "
                    "a straight rod with 180° backbone angles and energy "
                    "minimisation will NOT fix that.", exc)
    return chain


# ================================================================ MBUILD BACKEND
def mbuild_backend(
    monomers: Sequence[Monomer],
    sequence: Sequence[int],
    cap_carboxyl_end: bool = True,
) -> Molecule:
    """Use mbuild.recipes.Polymer for chain assembly.

    Two quality improvements over the naïve version, ported from the user's
    ``create_chain_pbs_cooh_mbuild_fixed.py``:

    1. **Straight-chain H-pair selection** — when the monomer has multiple
       hydrogens on its head or tail heavy atoms, we try every (head_h,
       tail_h) pair and pick the one whose H-vectors are closest to
       antiparallel. That produces a linear chain instead of a bent one.

    2. **Terminal carboxyl capping** — if the monomer's tail heavy atom is
       a carbonyl carbon (as in PBS: ``[*]OCCCCOC(=O)CCC(=O)[*]``), the
       polymer's right end would otherwise finish as -C(=O)H (aldehyde).
       We detect this and cap it as -C(=O)OH (carboxylic acid) directly
       in the mBuild compound before serialising.
    """
    try:
        import mbuild as mb  # noqa: F401
        from mbuild.lib.recipes.polymer import Polymer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "mbuild is required for the mbuild backend. "
            "Install with `conda install -c conda-forge mbuild`."
        ) from exc

    if len(set(sequence)) != 1:
        log.warning(
            "mbuild backend currently supports only single-monomer chains; "
            "falling back to simple backend for copolymers."
        )
        return simple_backend(monomers, sequence)

    m = monomers[sequence[0]]
    comp = _molecule_to_mbuild(m.molecule)
    n = len(sequence)

    # ---------- pick the straightest head/tail H pair ---------------------
    indices = _pick_straight_h_pair(m, comp)
    if indices is None:
        # Fall back to the first available H on each end.
        indices = []
        if m.head_removes:
            indices.append(m.head_removes[0])
        if m.tail_removes:
            indices.append(m.tail_removes[0])
    if len(indices) != 2:
        raise ValueError("mbuild backend requires exactly one head_h and one tail_h index")

    chain = Polymer()
    chain.add_monomer(compound=comp, indices=indices, separation=0.15, replace=True)
    chain.build(n=n, sequence="A")

    # ---------- carboxyl-end capping if the tail is a carbonyl -----------
    # Only fires when (a) the flag is on AND (b) the tail heavy atom is
    # structurally a C=O carbon. So it stays a no-op for polyethylene,
    # polypropylene, polystyrene, PVC, PMMA, polyisoprene, etc.
    if cap_carboxyl_end and _tail_is_carbonyl(m):
        try:
            _cap_terminal_carboxylic_acid_mbuild(chain)
            log.info("Capped terminal carbonyl end as -C(=O)OH (polyester end).")
        except RuntimeError as e:
            log.warning("Terminal -COOH capping failed: %s (chain kept as -CHO).", e)
    elif not cap_carboxyl_end and _tail_is_carbonyl(m):
        log.info("Tail is C=O but cap_carboxyl_end=False → leaving chain as -C(=O)H.")

    return _mbuild_to_molecule(chain, name="chain")


def _pick_straight_h_pair(monomer: Monomer, comp) -> Optional[list]:
    """Choose the (head_h, tail_h) pair whose H-vectors from their heavy
    atoms are closest to 180°. Only useful when a monomer has multiple
    candidate Hs on either end; otherwise returns None so the caller can
    fall back."""
    if not monomer.head_removes or not monomer.tail_removes:
        return None
    if len(monomer.head_removes) == 1 and len(monomer.tail_removes) == 1:
        return [monomer.head_removes[0], monomer.tail_removes[0]]
    try:
        particles = list(comp.particles())
        head_heavy = particles[monomer.head_index].pos
        tail_heavy = particles[monomer.tail_index].pos
    except Exception:
        return None
    best_pair, best_angle = None, 181.0
    for h1 in monomer.head_removes:
        for h2 in monomer.tail_removes:
            v1 = np.asarray(particles[h1].pos) - np.asarray(head_heavy)
            v2 = np.asarray(particles[h2].pos) - np.asarray(tail_heavy)
            n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            # We want v1 and -v2 to be parallel → v1·(-v2)/(|v1||v2|) close to 1.
            cosang = float(np.clip(np.dot(v1, -v2) / (n1 * n2), -1.0, 1.0))
            ang = float(np.degrees(np.arccos(cosang)))
            if ang < best_angle:
                best_angle, best_pair = ang, [h1, h2]
    return best_pair


# ================================================================ carboxyl cap
def _tail_is_carbonyl(monomer: Monomer) -> bool:
    """Return True if the monomer's tail heavy atom is a carbonyl carbon
    (has at least one C=O bond) — matches the user's ``right_dummy_is_carbonyl``.
    """
    mol = monomer.molecule
    tail = monomer.tail_index
    if not (0 <= tail < len(mol.atoms)):
        return False
    if mol.atoms[tail].element != "C":
        return False
    for (i, j, order) in mol.bonds:
        if i == tail and mol.atoms[j].element == "O" and order == 2:
            return True
        if j == tail and mol.atoms[i].element == "O" and order == 2:
            return True
    return False


def _cap_terminal_carboxylic_acid_mbuild(chain) -> None:
    """Locate the terminal PBS-style -C(=O)H site in the mBuild chain and
    replace the aldehydic H with -O-H (carboxylic acid).

    Adapted from the user's ``cap_terminal_carboxylic_acid_mbuild``: pick
    the C atom with exactly 1 oxygen + 1 carbon + 1 hydrogen neighbour
    (that's the aldehyde end after mBuild wraps up the polymer).
    """
    import mbuild as mb
    particles = list(chain.particles())
    bond_list = list(chain.bonds())

    def _neighbours(atom):
        out = []
        for a, b in bond_list:
            if a is atom:   out.append(b)
            elif b is atom: out.append(a)
        return out

    candidates = []
    for atom in particles:
        if atom.name != "C":
            continue
        nbs = _neighbours(atom)
        oxygens   = [n for n in nbs if n.name == "O"]
        carbons   = [n for n in nbs if n.name == "C"]
        hydrogens = [n for n in nbs if n.name == "H"]
        heavy     = [n for n in nbs if n.name != "H"]
        if (len(oxygens) == 1 and len(carbons) == 1
                and len(hydrogens) == 1 and len(heavy) == 2):
            candidates.append((atom, hydrogens[0], carbons[0]))

    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one -C(=O)H terminal site, found {len(candidates)}")

    carbon, old_h, chain_neighbour = candidates[0]

    # mBuild positions are in nm.
    vec = np.asarray(old_h.pos) - np.asarray(carbon.pos)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        vec = np.asarray(carbon.pos) - np.asarray(chain_neighbour.pos)
        norm = float(np.linalg.norm(vec)) or 1.0
    unit = vec / norm

    o_pos = np.asarray(carbon.pos) + 0.134 * unit     # ≈ 1.34 Å
    h_pos = o_pos + 0.097 * unit                       # ≈ 0.97 Å

    chain.remove(old_h)
    new_o = mb.Particle(name="O", pos=o_pos)
    new_h = mb.Particle(name="H", pos=h_pos)
    chain.add(new_o); chain.add(new_h)
    chain.add_bond((carbon, new_o))
    chain.add_bond((new_o, new_h))


def _molecule_to_mbuild(mol: Molecule):
    import mbuild as mb
    comp = mb.Compound(name=mol.name)
    parts = []
    for a in mol.atoms:
        p = mb.Particle(name=a.element, pos=a.xyz * 0.1)  # A -> nm
        comp.add(p)
        parts.append(p)
    for i, j, _ in mol.bonds:
        comp.add_bond((parts[i], parts[j]))
    return comp


def _mbuild_to_molecule(comp, name: str = "chain") -> Molecule:
    parts = list(comp.particles())
    atoms = [
        Atom(index=i, element=p.name, xyz=np.array(p.pos) * 10.0, name=f"{p.name}{i + 1}")
        for i, p in enumerate(parts)
    ]
    bonds = []
    for a, b in comp.bonds():
        i = parts.index(a)
        j = parts.index(b)
        # mBuild's Compound records connectivity but NOT bond order, so every
        # bond arrives here as single and the monomer's C=C is destroyed.
        # Orders are recovered below from valence and geometry; without that
        # the SMARTS typer's alkene rules can never match and every carbon
        # falls through to the alkane types.
        bonds.append((i, j, 1.0))
    mol = Molecule(atoms=atoms, bonds=bonds, name=name)
    _recover_bond_orders(mol, name)
    return mol


def _recover_bond_orders(mol, name: str = "chain") -> None:
    """Put back the bond orders the assembly step could not carry."""
    try:
        from .bond_orders import order_summary, perceive_bond_orders
        raised = perceive_bond_orders(mol)
        if raised:
            log.info("%s: recovered %d multiple bonds (%s)",
                     name, raised, order_summary(mol))
    except Exception as exc:                            # pragma: no cover
        log.warning(
            "%s: bond orders could not be recovered (%s). Every bond is "
            "single, so alkene/carbonyl atoms will be typed as saturated.",
            name, exc)


# ================================================================ TOP-LEVEL
def build_chain(
    monomers: Sequence[Monomer],
    n: int,
    mode: SequenceMode = "homopolymer",
    fractions: Optional[Sequence[float]] = None,
    block_sizes: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
    backend: Literal["mbuild", "simple", "auto"] = "auto",
    cap_carboxyl_end: bool = True,
    relax_conformation: bool = True,
    conformer_seed: Optional[int] = None,
) -> Molecule:
    """High-level chain builder.

    Parameters
    ----------
    monomers : list of Monomer
        One entry for homopolymer, N entries for copolymers.
    n : int
        Total number of monomer units in the chain.
    mode :
        - homopolymer: always monomer[0]
        - block: A_{k}B_{k}... using block_sizes (default n/len(monomers) each)
        - alternating: ABAB...
        - random: sampled with given `fractions`
    fractions : list of float, sums to 1
        Only used for mode="random". Defaults to equal fractions.
    block_sizes : list of int
        Only used for mode="block".
    seed : int, optional
        Seed for random sequence generation.
    """
    seq = _make_sequence(len(monomers), n, mode, fractions, block_sizes, seed)
    log.info("Chain sequence: %s", "".join(chr(ord("A") + i) for i in seq[:32]) + ("..." if n > 32 else ""))

    def _finish(chain: Molecule) -> Molecule:
        """Give the assembled chain a melt-like conformation.

        Every backend produces an *extended* chain. mBuild's port-based
        assembly gives correct 112° valence angles but leaves the chain
        essentially all-trans — and the H-pair selection deliberately picks
        the straightest option, so it is extended by design. R/L comes out
        around 0.8 where a real melt chain of the same length is 0.15–0.45.

        Energy minimisation afterwards does not fix that. Going trans →
        gauche means climbing a ~3 kcal/mol torsional barrier, and a
        minimiser only goes downhill, so an all-trans chain is a local
        minimum it cannot leave. The conformation has to be *sampled*.

        Valence angles are preserved wherever they are already physical, so
        this re-draws torsions on an mBuild chain and does not touch the
        geometry mBuild got right.
        """
        try:
            from .chain_conformer import ConformerSettings, rebuild_backbone
            result = rebuild_backbone(chain, ConformerSettings(
                seed=conformer_seed if conformer_seed is not None else seed,
                sample_torsions=relax_conformation))
            if result.ran:
                log.info("Chain conformation: %s", result.summary())
        except Exception as exc:                        # pragma: no cover
            log.warning("Conformer sampling failed (%s); the chain is left "
                        "extended. Energy minimisation will NOT coil it.", exc)
        return chain

    # -- Preferred path: SMILES-based mBuild builder (ported from the user's
    #    create_chain.py). Works for both homopolymers and copolymers as long
    #    as every monomer carries a polymerization SMILES (from the .polysmi
    #    sidecar). This avoids the PDB round-trip that stretched backbone bonds
    #    (broke DL_FIELD typing) and, for copolymers, the geometry-translation
    #    fallback that produced clumped/dangling fragments.
    if len(set(seq)) == 1:
        m0 = monomers[seq[0]]
        poly_smiles = getattr(m0, "poly_smiles", None)
        if poly_smiles and "[*]" in poly_smiles:
            try:
                from . import chain_smiles
                if chain_smiles.available():
                    log.info("Using SMILES-based chain builder for %s "
                             "(poly=%s)", m0.name, poly_smiles)
                    return _finish(chain_smiles.build_chain_from_smiles(
                        poly_smiles, n=n,
                        cap_carboxyl_end=cap_carboxyl_end,
                        name=getattr(m0, "name", "chain"),
                    ))
                log.info("SMILES-builder deps missing; using legacy mBuild path.")
            except Exception as e:
                log.warning("SMILES-based builder failed (%s); "
                            "falling back to legacy mBuild path.", e)
    else:
        # Copolymer: use the SMILES multi-monomer builder when every monomer
        # has a polymerization SMILES. Otherwise fall through to the legacy
        # backends below.
        poly_list = [getattr(m, "poly_smiles", None) for m in monomers]
        if all(ps and "[*]" in ps for ps in poly_list):
            try:
                from . import chain_smiles
                if chain_smiles.available():
                    log.info("Using SMILES-based copolymer builder "
                             "(%d monomer types, %d units)",
                             len(set(seq)), len(seq))
                    cname = "-".join(getattr(m, "name", "M") for m in monomers)
                    return _finish(chain_smiles.build_copolymer_from_smiles(
                        poly_list, seq,
                        cap_carboxyl_end=cap_carboxyl_end,
                        name=cname or "copolymer",
                    ))
                log.info("SMILES-builder deps missing; using legacy copolymer path.")
            except Exception as e:
                log.warning("SMILES-based copolymer builder failed (%s); "
                            "falling back to legacy backend.", e)

    if backend == "auto":
        try:
            import mbuild  # noqa: F401
            backend = "mbuild"
        except Exception:
            backend = "simple"
    if backend == "mbuild" and len(set(seq)) == 1:
        return _finish(mbuild_backend(monomers, seq,
                                      cap_carboxyl_end=cap_carboxyl_end))

    # Reaching here means both preferred builders were unavailable. That used
    # to be recorded at info level only, so a chain built by the fallback was
    # indistinguishable from an mBuild one — which is how straight chains
    # reached the output without anyone knowing which code produced them.
    log.warning(
        "Chain built with the no-dependency fallback backend, not mBuild. "
        "Install mbuild (and openbabel, networkx, rdkit) for the "
        "port-based builder: conda install -c conda-forge mbuild. "
        "Backbone geometry has been rebuilt from internal coordinates, but "
        "the mBuild path is the one that matches your reference script.")
    return simple_backend(monomers, seq, conformer_seed=seed,
                          sample_torsions=relax_conformation)


def _make_sequence(
    n_monomers: int,
    n: int,
    mode: SequenceMode,
    fractions: Optional[Sequence[float]],
    block_sizes: Optional[Sequence[int]],
    seed: Optional[int],
) -> List[int]:
    if mode == "homopolymer" or n_monomers == 1:
        return [0] * n
    if mode == "alternating":
        return [i % n_monomers for i in range(n)]
    if mode == "block":
        block_sizes = list(block_sizes or [max(1, n // n_monomers)] * n_monomers)
        seq: List[int] = []
        i = 0
        while len(seq) < n:
            for m, k in enumerate(block_sizes):
                for _ in range(k):
                    if len(seq) >= n:
                        break
                    seq.append(m)
                    i += 1
                if len(seq) >= n:
                    break
        return seq[:n]
    if mode == "random":
        fractions = list(fractions or [1.0 / n_monomers] * n_monomers)
        rng = random.Random(seed)
        return rng.choices(range(n_monomers), weights=fractions, k=n)
    raise ValueError(f"Unknown sequence mode {mode!r}")
