"""Monomer definition + head/tail atom detection.

A monomer needs two designated atoms that will form the polymer bond:
- head_index  -> atom on the incoming end (bonded to previous monomer's tail)
- tail_index  -> atom on the outgoing end (bonded to next monomer's head)

The user can specify these manually (from Avogadro: hover over an atom to see
its 1-based index) or let the tool guess. Guessing modes:

- terminal_h:   pick the two hydrogens whose C-H vectors are closest to 180°
                (works for vinyl/CH2=CH- style monomers).
- carboxyl_hydroxyl: for ester monomers, remove -COOH's -OH and the alcohol -H
                (used by the user's PBS script).
- explicit smarts: pass a SMARTS query per side.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np

from .logging_utils import get_logger
from .structure import Molecule, load_structure

log = get_logger(__name__)

CapMode = Literal["terminal_h", "carboxyl_hydroxyl", "explicit", "user"]


@dataclass
class Monomer:
    molecule: Molecule
    head_index: int
    tail_index: int
    # Atoms that must be *removed* when this monomer is joined at its head
    # (e.g. the H on the head carbon). Zero-indexed.
    head_removes: List[int] = field(default_factory=list)
    tail_removes: List[int] = field(default_factory=list)
    name: str = "MON"
    # Polymerization SMILES with [*] connection points, e.g. PBS's
    # "[*]OCCCCOC(=O)CCC(=O)[*]". When present, the chain builder uses the
    # robust SMILES-based mBuild path (graph isomorphism → best H pair →
    # Polymer) instead of the PDB round-trip. Filled from the .polysmi
    # sidecar when a library monomer was picked.
    poly_smiles: Optional[str] = None

    @property
    def n_atoms(self) -> int:
        return len(self.molecule.atoms)


def _wildcard_neighbour_symbols(poly_smiles: str) -> Optional[Tuple[str, str]]:
    """Return (head_neighbour_element, tail_neighbour_element) for the two
    heavy atoms bonded to the two ``[*]`` markers of a polymerization
    SMILES. Prefers RDKit; falls back to a small hand-written parser so
    the function works even when RDKit isn't installed. Returns None if
    the SMILES doesn't have exactly two wildcards.
    """
    # -- RDKit path -----------------------------------------------------
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(poly_smiles)
        if m is not None:
            stars = [a for a in m.GetAtoms() if a.GetAtomicNum() == 0]
            if len(stars) == 2:
                def _nb(a):
                    nbs = list(a.GetNeighbors())
                    return nbs[0].GetSymbol() if len(nbs) == 1 else None
                hs = _nb(stars[0]); ts = _nb(stars[1])
                if hs is not None and ts is not None:
                    return (hs, ts)
    except Exception:
        pass

    # -- RDKit-free fallback -------------------------------------------
    # Handle the common shape `[*]<HEAD>...<TAIL>[*]`. For each `[*]`
    # occurrence we look at the character(s) immediately following the
    # closing `]` (head case) or immediately preceding the opening `[`
    # (tail case) to find the heavy-atom symbol.
    if poly_smiles.count("[*]") != 2:
        return None
    _AROMATIC = {"c", "n", "o", "s", "p"}
    def _read_symbol_forward(s: str, i: int) -> Optional[str]:
        """Read an atom symbol starting at s[i]. Handles two-letter symbols
        like Cl, Br, and bracket atoms like [NH+]."""
        if i >= len(s):
            return None
        ch = s[i]
        if ch == "[":
            # bracket atom, read until ']'
            j = s.find("]", i)
            if j < 0:
                return None
            inside = s[i + 1:j]
            # Element symbol is the leading letters (skip isotope digits).
            k = 0
            while k < len(inside) and inside[k].isdigit(): k += 1
            elem = ""
            while k < len(inside) and inside[k].isalpha(): elem += inside[k]; k += 1
            if not elem:
                return None
            if elem[0].islower() and elem in _AROMATIC:
                return elem[0].upper()
            return elem[:1].upper() + elem[1:2].lower()
        if ch.isalpha():
            if ch in _AROMATIC:
                return ch.upper()
            if ch.isupper() and i + 1 < len(s) and s[i + 1].islower():
                # Cl, Br, Si, etc.
                two = ch + s[i + 1]
                if two in ("Cl", "Br", "Si", "Se", "As", "Li", "Na", "Mg",
                            "Al", "Ca", "Fe", "Zn"):
                    return two
            return ch.upper()
        return None

    def _read_symbol_backward(s: str, i: int) -> Optional[str]:
        """Read the symbol whose last character is s[i]. Walks left over
        ring-bond digits, bond markers, and balanced ``(...)`` branches.

        Example: for ``[*]OCCCCOC(=O)CCC(=O)[*]`` and i pointing at the
        character just before the tail ``[*]`` (which is ``)``), we want
        to return ``"C"`` — the carbon that owns the ``(=O)`` branch."""
        while i >= 0:
            ch = s[i]
            if ch.isdigit() or ch in "=#-\\/@:%":
                i -= 1; continue
            if ch == ")":
                # Skip the whole balanced group to its matching '('.
                depth = 1; i -= 1
                while i >= 0 and depth > 0:
                    if s[i] == ")": depth += 1
                    elif s[i] == "(": depth -= 1
                    i -= 1
                # Now i is at the char BEFORE the '(' — usually the owner atom.
                continue
            break
        if i < 0:
            return None
        if s[i] == "]":
            j = s.rfind("[", 0, i)
            if j < 0:
                return None
            return _read_symbol_forward(s, j)
        # single character; check for a leading uppercase (Cl, Br…)
        if i > 0 and s[i - 1].isupper() and s[i].islower():
            return _read_symbol_forward(s, i - 1)
        return _read_symbol_forward(s, i)

    def _neighbour_symbol(pos: int) -> Optional[str]:
        """Return the element symbol of the atom bonded to the ``[*]`` at
        SMILES index ``pos``. Handles three cases:

        * ``[*]X...``      (leading)   → the atom right after
        * ``...Y[*]``      (trailing)  → the atom right before
        * ``...C([*])...`` (in-branch) → the atom that OWNS the branch
        """
        before_ch = poly_smiles[pos - 1] if pos > 0 else ""
        after_ch  = poly_smiles[pos + 3] if pos + 3 < len(poly_smiles) else ""
        # In-branch: (...[*]...)  — the atom is the one right before "("
        if before_ch == "(" and after_ch == ")":
            return _read_symbol_backward(poly_smiles, pos - 2)
        # Leading (or after a branch open, i.e. wildcard is first in branch)
        if pos == 0 or before_ch == "(":
            return _read_symbol_forward(poly_smiles, pos + 3)
        # Trailing (default)
        return _read_symbol_backward(poly_smiles, pos - 1)

    first_pos = poly_smiles.find("[*]")
    tail_pos  = poly_smiles.rfind("[*]")
    head_sym = _neighbour_symbol(first_pos)
    tail_sym = _neighbour_symbol(tail_pos)
    if head_sym and tail_sym:
        return (head_sym, tail_sym)
    return None


def _capped_from_wildcards(poly_smiles: str):
    """The repeat unit with each ``[*]`` turned into its cap hydrogen.

    Returns ``(rdkit mol with explicit H, [cap H indices], [link indices])``,
    or ``None``.

    A polymerisation SMILES already states exactly where the chain joins.
    Turning the wildcard atom itself into a hydrogen — rather than deleting it
    and letting an implicit H appear somewhere — keeps that statement: the
    wildcard IS the hydrogen the junction consumes, at an index we know.
    """
    try:
        from rdkit import Chem
    except Exception:                             # pragma: no cover
        return None

    mol = Chem.MolFromSmiles(poly_smiles)
    if mol is None:
        return None
    editable = Chem.RWMol(mol)
    dummies = [a.GetIdx() for a in editable.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        return None

    links: List[int] = []
    for d in dummies:
        neighbours = [n.GetIdx() for n in editable.GetAtomWithIdx(d).GetNeighbors()]
        if len(neighbours) != 1:
            return None
        links.append(neighbours[0])
        atom = editable.GetAtomWithIdx(d)
        atom.SetAtomicNum(1)
        atom.SetNoImplicit(False)
        atom.SetIsotope(0)
        atom.SetFormalCharge(0)
    try:
        out = editable.GetMol()
        Chem.SanitizeMol(out)
        out = Chem.AddHs(out)
    except Exception:
        return None
    return out, dummies, links


def _links_from_wildcards(mol: Molecule, poly_smiles: str
                          ) -> Optional[Tuple[int, int, List[int], List[int]]]:
    """Head/tail read from the ``[*]`` positions, not inferred from shape.

    The failure this replaces
    -------------------------
    ``_pick_head_tail_by_element`` knows only the ELEMENT beside each wildcard
    and then looks for TERMINAL atoms of that element. For a polyester that is
    enough — the links really are at the ends. For a vinyl polymer it is not.
    Poly(1-butene) is ``[*]CC(CC)[*]``: capped, that is n-butane, and its two
    link carbons are the two INTERIOR ones. Terminal carbons of the right
    element exist — the two methyls — so a plausible wrong answer was always
    available, and the result was a chain joined end to end through the
    methyls. That is polyethylene with a four-carbon repeat, not poly(1-butene)
    — no branch, no methine, and every atom type assigned against a backbone
    that does not exist. The same applies to polypropylene, polystyrene, PVC,
    PMMA and every other vinyl entry in the library.

    The wildcards were never ambiguous. What was missing was a way to carry
    their position across from the SMILES numbering into the 3D file's, which
    is a graph-matching problem and is now solved as one.

    Returns ``None`` — leaving the caller to fall back — whenever the 3D file
    is not the molecule the SMILES describes. The curated PBS entry is one
    such case on purpose: its stored monomer is the acid, while capping the
    wildcards gives the aldehyde PAAF links through.
    """
    from .graph_match import match_graphs

    built = _capped_from_wildcards(poly_smiles)
    if built is None:
        return None
    reference, caps, links = built

    ref_elements = {a.GetIdx(): a.GetSymbol() for a in reference.GetAtoms()}
    ref_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                 for b in reference.GetBonds()]
    loaded_elements = {a.index: a.element for a in mol.atoms}
    loaded_bonds = [(int(b[0]), int(b[1])) for b in mol.bonds]

    pairing = match_graphs(ref_elements, ref_bonds,
                           loaded_elements, loaded_bonds,
                           label=f"wildcard links for {mol.name}")
    if pairing is None:
        return None

    try:
        head, tail = pairing[links[0]], pairing[links[1]]
    except KeyError:
        return None
    if head == tail:
        return None

    # The cap hydrogens are taken from the LINK ATOM'S OWN NEIGHBOURS, not
    # from the pairing.
    #
    # Atoms that share an environment are paired in sorted order, which is
    # fine for anything that only cares what an atom is, and wrong for
    # anything that cares where it is. Ethane is the smallest example: all six
    # hydrogens are one class, so the wildcard hydrogen of one carbon was
    # quite happily paired with a hydrogen of the other, and the check that
    # the cap is bonded to its link atom then rejected the whole molecule.
    # Eight library polymers fell out that way, polyethylene among them.
    #
    # Any hydrogen on the link atom will do, precisely because they are
    # indistinguishable — which is what made the pairing ambiguous to begin
    # with.
    def _cap_for(link: int, taken: Optional[int]) -> Optional[int]:
        for j in mol.neighbors(link):
            if mol.atoms[j].element == "H" and j != taken:
                return j
        return None

    head_h = _cap_for(head, None)
    tail_h = _cap_for(tail, head_h)
    if head_h is None or tail_h is None:
        return None

    # Colour refinement is not a proof of isomorphism, so the link atoms are
    # checked against what the SMILES actually said they were.
    for link, source in ((head, links[0]), (tail, links[1])):
        if mol.atoms[link].element != ref_elements[source]:
            return None

    log.info("Head/tail read from the polymerisation SMILES wildcards: "
             "%s%d and %s%d (caps %d, %d)",
             mol.atoms[head].element, head, mol.atoms[tail].element, tail,
             head_h, tail_h)
    _warn_if_link_is_implausible(mol, head, tail)
    return head, tail, [head_h], [tail_h]


def _pick_head_tail_by_element(mol: Molecule,
                              head_elem: str, tail_elem: str
                             ) -> Optional[Tuple[int, int, List[int], List[int]]]:
    """From the closed-shell 3D molecule pick two TERMINAL heavy atoms
    matching (head_elem, tail_elem), each carrying at least one H.
    Handles three shapes of polymer end:

    * simple terminal atom (-OH, -NH2, -SH): heavy-degree 1
    * carbonyl-style end (-C(=O)H after wildcard cap): heavy-degree 2 but
      one of the two heavy neighbours is a double-bonded O — this is the
      polyester tail case (PBS, PET, PLA…)
    * symmetric case (HO-…-OH): fall back to the antiparallel-vector rule

    Returns (head_heavy, tail_heavy, [head_h], [tail_h]) or None.
    """
    def _heavy_deg(i: int) -> int:
        return sum(1 for j in mol.neighbors(i) if mol.atoms[j].element != "H")
    def _H_neighbour(i: int) -> Optional[int]:
        for j in mol.neighbors(i):
            if mol.atoms[j].element == "H":
                return j
        return None
    def _has_carbonyl_O(i: int) -> bool:
        """True if atom i has a double-bonded O neighbour (=O)."""
        for (a, b, order) in mol.bonds:
            if a == i and mol.atoms[b].element == "O" and order == 2:
                return True
            if b == i and mol.atoms[a].element == "O" and order == 2:
                return True
        return False
    def _is_terminal(i: int, elem: str) -> bool:
        if _H_neighbour(i) is None:
            return False
        hd = _heavy_deg(i)
        if hd == 1:
            return True
        # Carbonyl-C polyester tail: heavy_deg=2 with one double-bond O.
        if elem == "C" and hd == 2 and _has_carbonyl_O(i):
            return True
        return False
    head_c = [a.index for a in mol.atoms
              if a.element == head_elem and _is_terminal(a.index, head_elem)]
    tail_c = [a.index for a in mol.atoms
              if a.element == tail_elem and _is_terminal(a.index, tail_elem)]
    if head_elem == tail_elem:
        cand = sorted(set(head_c) | set(tail_c))
        if len(cand) < 2:
            return None
        coords = mol.coords()
        best = None; best_score = -2.0
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                a, b = cand[i], cand[j]
                ha, hb = _H_neighbour(a), _H_neighbour(b)
                if ha is None or hb is None: continue
                va = coords[ha] - coords[a]; vb = coords[hb] - coords[b]
                na = np.linalg.norm(va); nb = np.linalg.norm(vb)
                if na < 1e-9 or nb < 1e-9: continue
                score = -float(np.dot(va, vb) / (na * nb))
                if score > best_score:
                    best_score, best = score, (a, b, ha, hb)
        return None if best is None else (best[0], best[1], [best[2]], [best[3]])
    if len(head_c) != 1 or len(tail_c) != 1:
        return None
    h = head_c[0]; t = tail_c[0]
    return (h, t, [_H_neighbour(h)], [_H_neighbour(t)])


# ---------------------------------------------------------------- factories
def monomer_from_file(
    path: str | Path,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    head_h: Optional[int] = None,
    tail_h: Optional[int] = None,
    cap_mode: CapMode = "terminal_h",
    name: Optional[str] = None,
    poly_smiles: Optional[str] = None,
) -> Monomer:
    """Create a Monomer, converting 1-based Avogadro indices to 0-based.

    ``poly_smiles`` (optional) is the polymerization SMILES with ``[*]``
    wildcards, e.g. PBS's ``[*]OCCCCOC(=O)CCC(=O)[*]``. When supplied,
    PAAF uses the wildcards to identify head/tail atoms by ELEMENT match
    (bypasses the fragile geometry-based ``_guess_terminal_h``). If we
    can auto-load a sidecar ``<path>.polysmi`` file, we do that too.
    """
    mol = load_structure(path, name=name)
    # Auto-load sidecar polymerization SMILES if not explicitly given.
    if poly_smiles is None:
        try:
            side = Path(str(path) + ".polysmi")
            if side.exists():
                poly_smiles = side.read_text().strip() or None
        except Exception:
            poly_smiles = None

    if head is not None and tail is not None:
        h_idx = head - 1
        t_idx = tail - 1
        head_removes = [head_h - 1] if head_h else []
        tail_removes = [tail_h - 1] if tail_h else []
        cap_mode = "user"
    elif poly_smiles:
        # Position first, element second. The wildcards say WHERE the chain
        # joins; matching on the element beside them only says what it is
        # made of, and for any vinyl polymer several atoms answer to that.
        picked = _links_from_wildcards(mol, poly_smiles)
        if picked is None:
            symbols = _wildcard_neighbour_symbols(poly_smiles)
            if symbols is not None:
                picked = _pick_head_tail_by_element(mol, symbols[0], symbols[1])
        if picked is None:
            log.warning("Wildcard-based head/tail failed for %s; "
                        "falling back to geometry guess.", mol.name)
            h_idx, t_idx, head_removes, tail_removes = _guess_terminal_h(mol)
            cap_mode = "terminal_h"
        else:
            h_idx, t_idx, head_removes, tail_removes = picked
            cap_mode = "wildcard"
    elif cap_mode == "terminal_h":
        h_idx, t_idx, head_removes, tail_removes = _guess_terminal_h(mol)
    else:
        raise ValueError(f"Unsupported cap_mode {cap_mode!r}; please pass head/tail")
    log.info(
        "Monomer %s: head=%d tail=%d, remove head:%s tail:%s (%s)",
        mol.name, h_idx, t_idx, head_removes, tail_removes, cap_mode,
    )
    return Monomer(mol, h_idx, t_idx, head_removes, tail_removes,
                   name=name or mol.name, poly_smiles=poly_smiles)


# ----------------------------------------------------------- head/tail guess
def _bond_distances(mol: Molecule, source: int) -> Dict[int, int]:
    """Shortest path length in BONDS from ``source`` to every other atom."""
    from collections import deque

    seen = {int(source): 0}
    queue = deque([int(source)])
    while queue:
        current = queue.popleft()
        for neighbour in mol.neighbors(current):
            if neighbour not in seen:
                seen[neighbour] = seen[current] + 1
                queue.append(neighbour)
    return seen


def _guess_terminal_h(mol: Molecule) -> Tuple[int, int, List[int], List[int]]:
    """Pick the two ends of the molecule, as head and tail link atoms.

    Returns (head_heavy, tail_heavy, [head_h], [tail_h]).

    Why this is topological and not geometric
    -----------------------------------------
    This used to choose the pair of heavy atoms whose C-H vectors pointed most
    nearly opposite one another. That is a statement about one conformer, not
    about the molecule, and it fails badly on anything folded: for PBS entered
    as ``OCCCCOC(=O)CCC(=O)O`` it selected atoms 8 and 9 — two ADJACENT CH2
    carbons in the middle of the succinate — because their hydrogens happened
    to point apart. Linking there polymerises through the middle of the
    monomer, which is what produced the tangled, dangling structure in the 3D
    view. Nothing complained, because two carbons with hydrogens is a perfectly
    legal answer to the question that was being asked.

    The question worth asking is which atoms are at opposite ENDS, and that is
    a property of the bond graph, not of the coordinates. So the pair is chosen
    to maximise the number of bonds between them, with the old anti-parallel
    score kept only to break ties. A folded conformer no longer matters.
    """
    coords = mol.coords()
    candidates = []          # (heavy_idx, h_idx, unit_vec)
    for a in mol.atoms:
        if a.element == "H":
            continue
        for j in mol.neighbors(a.index):
            if mol.atoms[j].element == "H":
                v = coords[j] - coords[a.index]
                n = float(np.linalg.norm(v))
                if n < 1e-9:
                    continue
                candidates.append((a.index, j, v / n))
                break            # one removable H per heavy atom is enough

    if len(candidates) < 2:
        raise ValueError(
            "Could not auto-detect terminal hydrogens for monomer. "
            "Please pass head/tail atom indices explicitly."
        )

    distances = {c[0]: _bond_distances(mol, c[0]) for c in candidates}

    best = None
    best_score = (-1, -2.0)
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a_heavy, _a_h, a_vec = candidates[i]
            b_heavy, _b_h, b_vec = candidates[j]
            if a_heavy == b_heavy:
                continue
            separation = distances[a_heavy].get(b_heavy)
            if separation is None:
                continue          # different fragments; not a chain
            score = (separation, -float(np.dot(a_vec, b_vec)))
            if score > best_score:
                best_score = score
                best = (candidates[i], candidates[j])

    if best is None:
        raise ValueError("Could not find a suitable head/tail hydrogen pair")
    (h_heavy, h_h, _), (t_heavy, t_h, _) = best
    log.info("Head/tail guessed from the bond graph: %s%d and %s%d, "
             "%d bonds apart",
             mol.atoms[h_heavy].element, h_heavy,
             mol.atoms[t_heavy].element, t_heavy, best_score[0])
    _warn_if_link_is_implausible(mol, h_heavy, t_heavy)
    return h_heavy, t_heavy, [h_h], [t_h]


#: Bonds that linking would create which are almost never what was meant.
#: Two hydroxyls joined head-to-tail give a peroxide, not an ester.
_IMPLAUSIBLE_LINKS = {
    ("O", "O"): "a peroxide (O-O), not an ester",
    ("N", "N"): "a hydrazine (N-N), not an amide",
    ("O", "N"): "a hydroxylamine (N-O) link",
}


def _warn_if_link_is_implausible(mol: Molecule, head: int, tail: int) -> None:
    """Say so when joining the chosen ends would give absurd chemistry.

    A condensation monomer written in its acid form — PBS as
    ``OCCCCOC(=O)CCC(=O)O`` — has an -OH at both ends, so the only atoms with
    a removable hydrogen at the two extremes are both oxygens. PAAF links by
    deleting one H from each end and bonding what remains, which turns that
    into an O-O peroxide backbone. The geometry is valid, the file is well
    formed, and the polymer is nonsense.
    """
    pair = tuple(sorted((mol.atoms[head].element, mol.atoms[tail].element)))
    problem = _IMPLAUSIBLE_LINKS.get(pair)
    if not problem:
        return
    log.warning(
        "Linking %s%d to %s%d would create %s. This usually means the monomer "
        "was written in its condensation form (e.g. PBS as "
        "'OCCCCOC(=O)CCC(=O)O', with an -OH at both ends). PAAF joins monomers "
        "by removing ONE hydrogen from each end, which cannot perform a "
        "condensation. Use the polymerisation form with [*] markers "
        "(e.g. '[*]OCCCCOC(=O)CCC(=O)[*]'), or set the head/tail atoms "
        "explicitly.",
        mol.atoms[head].element, head, mol.atoms[tail].element, tail, problem)


# ----------------------------------------------------------- multi-monomer
def load_monomers(specs: Sequence[dict]) -> List[Monomer]:
    """Load a list of monomer specs (e.g. from YAML)."""
    out: List[Monomer] = []
    for s in specs:
        out.append(
            monomer_from_file(
                path=s["file"],
                head=s.get("head"),
                tail=s.get("tail"),
                head_h=s.get("head_h"),
                tail_h=s.get("tail_h"),
                cap_mode=s.get("cap_mode", "terminal_h"),
                name=s.get("name"),
            )
        )
    return out
