"""Force-field-independent chemical-environment classification.

Given only PAAF's internal :class:`~paaf.structure.Molecule` (element symbols,
the bond graph and bond orders) this module works out a human-readable
*chemistry* label for every atom — the real chemical environment (epoxide CH,
aromatic C, methyl CH3, ester / amide / carboxyl carbonyl C, vinyl =CH-, ...) —
**independent of any force field**.

Why this lives in its own module (and not only inside the OPLS-AA typer):

* The manual atom-typing dialog can now show the SAME authentic chemistry
  mapping for *every* Moltemplate-native force field (OPLS-AA 2024/2008,
  L-OPLS-AA, OPLS-UA, GAFF/GAFF2, DREIDING, COMPASS, TraPPE, SDK, MARTINI …),
  not just OPLS-AA 2024.

* For **hydrogens** the label reflects the *parent heavy atom's* environment,
  because force fields assign a DIFFERENT hydrogen type depending on what the
  H is attached to.  In OPLS-AA an epoxide C-H, an aromatic C-H and a methyl
  C-H are three different atom types even though all three are "just" a C-H.
  So a hydrogen here is reported as e.g. ``"H on epoxide CH"`` /
  ``"H on aromatic CH"`` / ``"H on CH3 (methyl)"``.

The module also offers :func:`suggest_ff_type`, a generic keyword matcher that
ranks the atom-type library of ANY force field against an atom's chemical
environment, giving the user a short-list to pick from when no exact SMARTS
typer exists for that particular force field.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

__all__ = [
    "detect_context",
    "group_label",
    "atom_group",
    "all_groups",
    "all_groups_with_links",
    "suggest_ff_type",
]


# =====================================================================
# 1. Geometry / bond-graph context detection
# =====================================================================
def detect_context(mol, atom_index: int) -> dict:
    """Return the extra bond/ring info needed to classify ``atom_index``.

    Uses only the internal Molecule representation (no OpenBabel/RDKit).

    Keys returned
    -------------
    has_double_bond : bool
        Any incident bond of order 2.
    has_triple_bond : bool
        Any incident bond of order 3.
    double_bond_partners : list[str]
        Element symbols of atoms double-bonded to this atom (lets us tell a
        carbonyl ``C=O`` apart from an alkene ``C=C``).
    in_small_ring : int
        Size of the smallest ring containing this atom (0 = acyclic).
    ring_neighbors : list[str]
        Element symbols of the atoms forming that smallest ring.
    aromatic : bool
        True when the atom sits in a 5- or 6-membered ring in which
        essentially every member is sp2 (has double-bond character).
    """
    has_2 = False
    has_3 = False
    dbl_partners: List[str] = []
    for i, j, order in mol.bonds:
        if i == atom_index or j == atom_index:
            o = int(round(order))
            if o == 2:
                has_2 = True
                other = j if i == atom_index else i
                dbl_partners.append(mol.atoms[other].element)
            if o == 3:
                has_3 = True

    # Smallest ring via BFS on the bond graph, back to atom_index.  Limited to
    # size 6 to stay cheap; epoxides (3) and aromatics (5/6) are what matter.
    min_ring = 0
    best_path: List[int] = []
    for start_nbr in mol.neighbors(atom_index):
        queue = deque([(start_nbr, [start_nbr])])
        seen = {start_nbr}
        while queue:
            cur, path = queue.popleft()
            if len(path) > 6:
                continue
            for nxt in mol.neighbors(cur):
                if nxt == atom_index and cur != start_nbr:
                    cycle_len = len(path) + 1
                    if min_ring == 0 or cycle_len < min_ring:
                        min_ring = cycle_len
                        best_path = path
                    continue
                if nxt in seen or nxt == atom_index:
                    continue
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))

    ring_atoms = [atom_index] + best_path
    ring_neigh_elems = [mol.atoms[i].element for i in best_path]

    # Aromatic heuristic: a 5- or 6-membered ring in which (almost) every atom
    # bears double-bond character.  Benzene (Kekulé single/double or aromatic
    # order-1.5 bonds) -> every ring atom has an incident double bond -> aromatic.
    # Cyclohexane (no double bonds) or cyclohexene (2 sp2 atoms) -> not aromatic.
    aromatic = False
    if min_ring in (5, 6):
        sp2_count = 0
        for ai in ring_atoms:
            for i, j, order in mol.bonds:
                if (i == ai or j == ai) and int(round(order)) == 2:
                    sp2_count += 1
                    break
        aromatic = sp2_count >= (len(ring_atoms) - 1)

    return {
        "has_double_bond": has_2,
        "has_triple_bond": has_3,
        "double_bond_partners": dbl_partners,
        "in_small_ring": min_ring,
        "ring_neighbors": ring_neigh_elems,
        "aromatic": aromatic,
    }


# =====================================================================
# 2. Human-readable chemistry label
# =====================================================================
def group_label(element: str, neighbor_elements: list, *,
                has_double_bond: bool = False,
                has_triple_bond: bool = False,
                double_bond_partners: Optional[list] = None,
                in_small_ring: int = 0,
                ring_neighbors: Optional[list] = None,
                aromatic: bool = False,
                parent_group: Optional[str] = None) -> str:
    """Return an authentic chemistry group label for one heavy atom or hydrogen.

    ``parent_group`` is used only for hydrogens: when supplied (the label of
    the heavy atom the H is bonded to) the hydrogen is reported as
    ``"H on <parent_group>"`` so its force-field environment is explicit.
    """
    e = element
    ne = list(neighbor_elements)
    n_h = sum(1 for x in ne if x == "H")
    n_heavy = sum(1 for x in ne if x != "H")
    n_o = sum(1 for x in ne if x == "O")
    n_n = sum(1 for x in ne if x == "N")
    rn = ring_neighbors or []
    dbl = double_bond_partners or ([] if not has_double_bond else ["?"])

    # ---- Hydrogen: describe by its parent heavy atom's environment ----
    if e == "H":
        heavy = next((x for x in ne if x != "H"), "")
        if heavy == "C":
            if parent_group:
                return f"H on {parent_group}"
            return "H (aliphatic C-H)"
        if heavy == "O":
            return "H (hydroxyl O-H)"
        if heavy == "N":
            return "H (amine/amide N-H)"
        if heavy == "S":
            return "H (thiol S-H)"
        return "H"

    # ---- Epoxide (3-membered ring containing an O) ----
    # For the ring oxygen itself the ring-neighbour list holds the two carbons,
    # so also treat a 3-ring oxygen as the epoxide O.
    if in_small_ring == 3 and (e == "O" or any(x == "O" for x in rn)):
        if e == "O":
            return "epoxide -O- (oxirane)"
        if e == "C":
            if n_h == 2:
                return "epoxide CH2"
            if n_h == 1:
                return "epoxide CH"
            return "epoxide C (quaternary)"

    # ---- Aromatic ring carbon ----
    if e == "C" and aromatic:
        if n_h == 1:
            return "aromatic CH"
        return "aromatic C"

    # ---- Carbon with a double bond: distinguish carbonyl (C=O) from alkene ----
    if e == "C" and has_double_bond and "O" in dbl:
        if n_n >= 1:
            return "amide carbonyl C (N-C=O)"
        if n_o >= 2:
            return "ester/acid carbonyl C (O-C=O)"
        return "carbonyl C (C=O)"

    if e == "C" and has_double_bond and ("C" in dbl or "?" in dbl):
        if n_h == 2:
            return "=CH2 (terminal alkene)"
        if n_h == 1:
            return "=CH- (vinyl)"
        return ">C= (internal alkene)"

    if e == "C" and has_double_bond and "N" in dbl:
        return "imine/enamine C (C=N)"

    # ---- Aliphatic / sp carbon ----
    if e == "C":
        if has_triple_bond:
            return "-C# (alkyne / nitrile C)"
        if n_h == 4:
            return "CH4 (methane)"
        if n_h == 3:
            return "CH3 (methyl)"
        if n_h == 2:
            return "CH2 (methylene)"
        if n_h == 1:
            return "CH (methine)"
        return "C (quaternary)"

    # ---- Oxygen ----
    if e == "O":
        if has_double_bond:
            if n_n >= 1:
                return "=O (amide carbonyl)"
            return "=O (carbonyl)"
        if n_h == 1:
            # OH on a carbonyl carbon is a carboxylic-acid OH
            return "OH (alcohol / acid)"
        if n_heavy == 2:
            return "O- (ether / ester -O-)"
        return "O"

    # ---- Nitrogen ----
    if e == "N":
        if has_triple_bond:
            return "N# (nitrile N)"
        # amide N: bonded to a carbonyl carbon
        if any(x == "C" for x in ne) and n_n == 0:
            if n_h == 2:
                return "NH2 (amide/amine)"
            if n_h == 1:
                return "NH (amide / 2° amine)"
        if n_h == 3:
            return "NH3 (ammonium?)"
        if n_h == 2:
            return "NH2 (primary amine)"
        if n_h == 1:
            return "NH (secondary amine)"
        return "N (tertiary amine)"

    # ---- Sulfur / phosphorus / halogens ----
    if e == "S":
        if any(x == "O" for x in ne) and ne.count("O") >= 2:
            return "S (sulfone / sulfonate)"
        if n_h == 1:
            return "SH (thiol)"
        return "S (sulfide / thioether)"
    if e == "P":
        return "P"
    if e in {"F", "Cl", "Br", "I"}:
        host = "aromatic" if aromatic else "aliphatic"
        return f"{e} (halide on {host} C)"
    return e


# =====================================================================
# 3. High-level per-atom classification (handles H -> parent environment)
# =====================================================================
def atom_group(mol, index: int, _depth: int = 0) -> str:
    """Chemistry group label for atom ``index`` in ``mol``.

    For hydrogens the parent heavy atom's own group is computed first and the
    hydrogen is reported as ``"H on <parent_group>"``.
    """
    a = mol.atoms[index]
    nbr_elems = [mol.atoms[j].element for j in mol.neighbors(index)]
    ctx = detect_context(mol, index)

    parent_group = None
    if a.element == "H" and _depth == 0:
        heavy = next((j for j in mol.neighbors(index)
                      if mol.atoms[j].element != "H"), None)
        if heavy is not None and mol.atoms[heavy].element == "C":
            parent_group = atom_group(mol, heavy, _depth + 1)

    return group_label(a.element, nbr_elems, parent_group=parent_group, **ctx)


def all_groups(mol) -> Dict[int, str]:
    """Return ``{atom_index: chemistry_group_label}`` for every atom."""
    return {a.index: atom_group(mol, a.index) for a in mol.atoms}


def _link_carbon_group(mol, ci: int, removed_h: Optional[int]) -> str:
    """Chemistry label for a polymer *connection* carbon as it exists IN the
    chain: one capping hydrogen is replaced by a bond to the neighbouring
    monomer.  So an isolated CH3-looking connection carbon is really a
    backbone ``-CH2-`` once polymerised.
    """
    nbr_indices = list(mol.neighbors(ci))
    if removed_h is not None and removed_h in nbr_indices:
        nbr_indices = [j for j in nbr_indices if j != removed_h]
        nbrs = [mol.atoms[j].element for j in nbr_indices]
    else:
        nbrs = [mol.atoms[j].element for j in nbr_indices]
        if "H" in nbrs:          # drop one generic H for the inter-monomer bond
            nbrs.remove("H")
    nbrs.append("C")             # the new bond to the adjacent monomer
    ctx = detect_context(mol, ci)
    base = group_label(mol.atoms[ci].element, nbrs, **ctx)
    return f"{base} [backbone chain-link]"


def all_groups_with_links(mol, link_carbons: Optional[Dict[int, Optional[int]]] = None
                          ) -> Dict[int, str]:
    """Like :func:`all_groups`, but aware of polymer head/tail connection atoms.

    ``link_carbons`` maps ``{connection_carbon_index: removed_cap_H_index}``
    (the H may be ``None`` if unknown).  Connection carbons are relabelled to
    their *in-chain* environment (e.g. the isoprene backbone carbons become
    ``"CH2 (methylene) [backbone chain-link]"`` instead of ``"CH3 (methyl)"``),
    the cap hydrogens are flagged as removed on polymerisation, and hydrogens
    still attached to a connection carbon report that carbon's in-chain group.
    """
    link_carbons = dict(link_carbons or {})
    removed = {h for h in link_carbons.values() if h is not None}

    # Cache each connection carbon's in-chain label.
    link_group: Dict[int, str] = {
        ci: _link_carbon_group(mol, ci, h) for ci, h in link_carbons.items()
    }

    out: Dict[int, str] = {}
    for a in mol.atoms:
        i = a.index
        if a.element == "H":
            if i in removed:
                out[i] = "H (cap — removed when chain-linked)"
                continue
            heavy = next((j for j in mol.neighbors(i)
                          if mol.atoms[j].element != "H"), None)
            if heavy is not None and heavy in link_group:
                out[i] = f"H on {link_group[heavy]}"
                continue
            out[i] = atom_group(mol, i)
        elif i in link_group:
            out[i] = link_group[i]
        else:
            out[i] = atom_group(mol, i)
    return out


# =====================================================================
# 4. Generic force-field-type suggester (works for ANY .lt library)
# =====================================================================
# Map fragments of a chemistry group label -> keywords likely to appear in a
# force field's own atom-type key/description.  Used to rank candidate types
# from any Moltemplate .lt when no dedicated SMARTS typer exists for that FF.
def _keywords_for(group: str, element: str) -> set:
    g = group.lower()
    kw: set = set()

    def add(*words):
        kw.update(words)

    if "epoxide" in g or "oxiran" in g:
        add("epoxide", "oxiran", "ether", "ring")
    if "aromatic" in g:
        add("aromatic", "benzene", "aryl", "ar", "phenyl", "ca")
    if "carbonyl" in g or "c=o" in g:
        add("carbonyl", "c=o", "keto")
    if "amide" in g:
        add("amide", "peptide", "carbonyl")
    if "ester" in g or "acid" in g or "carboxyl" in g or "cooh" in g:
        add("ester", "acid", "carboxyl", "carbonyl")
    if "alkene" in g or "vinyl" in g or "=ch" in g or ">c=" in g:
        add("alkene", "vinyl", "sp2", "olefin", "=c", "cm")
    if "alkyne" in g or "-c#" in g:
        add("alkyne", "sp", "cz")
    if "nitrile" in g or "n#" in g:
        add("nitrile", "cyano", "cz", "nz")
    if "methyl" in g or "ch3" in g:
        add("methyl", "ch3", "alkane", "sp3", "ct")
    if "methylene" in g or "ch2" in g:
        add("methylene", "ch2", "alkane", "sp3", "ct")
    if "methine" in g or (g.startswith("ch ") or g == "ch"):
        add("methine", "ch", "alkane", "sp3", "ct")
    if "quaternary" in g:
        add("quaternary", "alkane", "sp3", "ct")
    if "alcohol" in g or "hydroxyl" in g or "oh (" in g or " o-h" in g:
        add("alcohol", "hydroxyl", "oh", "ho")
    if "ether" in g:
        add("ether", "os", "oxygen")
    if "amine" in g:
        add("amine", "amino", "nitrogen", "nh", "n3")
    if "thiol" in g:
        add("thiol", "sh", "mercapto", "sulfur")
    if "sulfone" in g or "sulfonate" in g:
        add("sulfone", "sulfonyl", "sulfur")
    if "sulfide" in g or "thioether" in g:
        add("sulfide", "thioether", "sulfur")
    if "halide" in g:
        add("halide", "halogen")
    if element in {"F", "Cl", "Br", "I"}:
        add(element.lower(), "halide", "halogen")
    return kw


def suggest_ff_type(group: str, element: str, ff_types, top_n: int = 1):
    """Rank a force field's own atom types against a chemistry group label.

    ``ff_types`` is a list of objects with ``.ff_id``, ``.element``, ``.key``
    and ``.description`` (i.e. :class:`paaf.lt_parser.AtomTypeInfo`).  Returns a
    list of ``(ff_id, score)`` best matches (highest score first), constrained
    to types whose element matches ``element``.  Empty when nothing scores.

    This is a *suggestion* aid for force fields without a dedicated SMARTS
    typer — the user still confirms the choice in the dialog.
    """
    kw = _keywords_for(group, element)
    if not kw:
        return []
    el = (element or "").lower()
    scored = []
    for t in ff_types:
        t_el = (t.element or "").lower()
        # Element gate: the FF type's element token should start with our symbol
        # (OPLS keys are bare element symbols; GAFF/DREIDING keys like "c3",
        # "ca" also start with the element letter).
        if el and not (t_el == el or t_el.startswith(el) or
                       (t.key or "").lower().startswith(el)):
            continue
        hay = f"{(t.key or '')} {(t.description or '')}".lower()
        score = sum(1 for w in kw if w and w in hay)
        if score > 0:
            scored.append((t.ff_id, score))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:top_n]
