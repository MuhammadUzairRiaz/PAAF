"""Assign force-field atom types to a Molecule.

Three strategies are supported:

- ``openbabel``:   OBForceField typing for OPLS/GAFF/MMFF (whatever OB knows).
- ``dlfield_sf``:  parse a DL_FIELD .sf file and use the SMARTS-like patterns
                   inside to type each atom.  Since DL_FIELD .sf uses a
                   custom mini-language, we implement a *substring* matcher
                   that covers the very common `element -connectivity- symbol`
                   patterns and falls back to element mapping for anything
                   ambiguous.  A more complete matcher can be plugged in.
- ``antechamber``: shells out to antechamber (part of AmberTools) for GAFF.
- ``manual``:      trust ``Atom.ff_type`` if it is already set (or a mapping
                   dict passed by the user).

For any pattern the matcher can't resolve, the tool falls back to the
element symbol and logs a warning listing the unresolved atoms so the user
can add explicit rules to their config.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .ff_registry import ForceField
from .logging_utils import get_logger
from .structure import Molecule

log = get_logger(__name__)


# ==================================================================== main
def assign(
    mol: Molecule,
    ff: ForceField,
    dl_lib_dir: Optional[Path] = None,
    manual_types: Optional[Dict[int, str]] = None,
    monomer: Optional[object] = None,
    n_units: int = 0,
) -> Molecule:
    """Assign ff.ff_type to every atom in `mol` in place.

    ``manual_types`` is keyed by **monomer** atom index — that is what the
    typing dialog shows and what the user assigns against. ``mol`` here is the
    whole chain. Writing one straight onto the other typed only the first
    repeat unit and left the other nineteen to the automatic typer, which is
    how a 20-unit polyisoprene came out with a single alkene carbon in it.
    So the map is expanded across every unit first.
    """
    strategy = ff.atom_typer
    if manual_types:
        expanded = expand_manual_types(
            mol, manual_types, monomer=monomer, n_units=n_units)
        # Nothing placed? Try matching on chemical environment before giving
        # up. The dialog seeds every atom from the automatic typer and the
        # user overrides only what it got wrong, so the map is COMPLETE --
        # discarding it discards the automatic types too and the chain gets
        # retyped from scratch by the very typer being corrected.
        if not expanded and monomer is not None:
            from .chain_provenance import types_by_environment
            from .typing_context import split_by_role

            expanded = types_by_environment(
                mol, monomer, split_by_role(manual_types))
        if not expanded:
            raise RuntimeError(
                f"You assigned {len(manual_types)} atom types, but none of "
                f"them could be placed on this chain -- not by mapping it to "
                f"the monomer, and not by matching chemical environments. "
                f"The chain would have been typed automatically and your "
                f"assignments silently ignored, so nothing was written.\n\n"
                f"The log line beginning 'Chain has N atoms' gives the atom "
                f"count the chain has versus what {n_units} units predict.")

        # Write them, and hand the CHAIN-keyed map on so the automatic typer
        # below fills only what is left rather than overwriting these.
        for index, value in expanded.items():
            if 0 <= index < len(mol.atoms):
                mol.atoms[index].ff_type = value
        manual_types = expanded
        log.info("Manual atom types applied to %d of %d chain atoms; the rest "
                 "are left to the automatic typer.",
                 len(expanded), len(mol.atoms))

    if strategy == "openbabel":
        # For OPLS-AA / L-OPLS-AA use the dedicated SMARTS-based OPLS typer
        # that produces valid numeric OPLS-AA 2024 type IDs (@atom:135 etc.)
        # that moltemplate.sh actually recognizes when importing oplsaa2024.lt.
        if ff.key in ("oplsaa", "oplsaa2008", "loplsaa", "loplsaa2008"):
            _assign_opls_smarts(mol, ff, manual_types)
        else:
            _assign_openbabel(mol, ff)
    elif strategy == "dlfield_sf":
        if dl_lib_dir is None:
            raise ValueError("dlfield_sf typer requires dl_lib_dir")
        _assign_from_dlfield(mol, ff, dl_lib_dir)
    elif strategy == "antechamber":
        _assign_antechamber(mol, ff)
    elif strategy == "manual":
        if not all(a.ff_type for a in mol.atoms):
            missing = [a.index for a in mol.atoms if not a.ff_type]
            raise ValueError(
                f"manual atom typing requested but atoms {missing} have no type set"
            )
    else:
        raise ValueError(f"Unknown atom typer {strategy!r}")

    _warn_unresolved(mol)
    return mol


# ==================================================================== helpers
def expand_manual_types(mol: Molecule, manual_types: Dict[int, str],
                        *, monomer: Optional[object] = None,
                        n_units: int = 0) -> Dict[int, str]:
    """Place a monomer's manual types onto every unit of the chain.

    ``manual_types`` is keyed by **monomer** atom index — that is what the
    typing dialog shows. ``mol`` is the whole chain, whose numbering differs
    because linking deletes a cap hydrogen at every junction.

    The two are reconciled through an exact, element-verified mapping
    (:func:`paaf.chain_provenance.unit_provenance`). When that mapping cannot
    be verified, **nothing is applied** and the automatic typer handles the
    chain. That is deliberate: an earlier version guessed, by spreading types
    across chemical equivalence classes, and put a hydrogen type onto two
    backbone carbons — which were then written with mass 1.008. A chain typed
    from a mapping that is off by one looks plausible and is wrong, which is
    worse than one that was never hand-typed at all.

    A map already covering the whole chain is returned unchanged, so this is
    safe to call twice.
    """
    if not manual_types:
        return {}

    # A chain has three kinds of unit and they need different types: the first
    # and last keep a cap hydrogen, so their link atoms are a terminal CH3
    # where the interior is a backbone CH2. The dialog collects all three sets
    # and folds the unit's role into the key, so they arrive here as one map.
    from .typing_context import ROLE_BASE, split_by_role

    by_role = split_by_role(manual_types)
    repeat_types = by_role["middle"]
    head_types = by_role["head"] or None
    tail_types = by_role["tail"] or None
    cap_types = by_role["cap"] or None

    # A map already covering the whole chain is passed through untouched. Only
    # plain keys can be chain indices — a role-encoded key is a monomer index
    # wearing an offset, and counting it here would let a three-unit chain look
    # "already complete" and skip the mapping entirely.
    if (not head_types and not tail_types and not cap_types
            and len(repeat_types) >= len(mol.atoms)):
        return dict(repeat_types)

    if monomer is None or n_units < 1:
        log.warning(
            "Manual atom types were supplied but the monomer they refer to is "
            "not known here, so they cannot be placed on the chain reliably "
            "and have been IGNORED. The automatic typer will type the chain.")
        return {}

    from .chain_provenance import expand_by_provenance, unit_provenance

    provenance = unit_provenance(mol, monomer, n_units)
    if provenance is None:
        log.warning(
            "Manual atom types could not be matched to this chain (it is not "
            "a clean repetition of the monomer), so they have been IGNORED "
            "rather than applied to the wrong atoms.")
        return {}

    return expand_by_provenance(mol, repeat_types, provenance,
                                head_types=head_types, tail_types=tail_types,
                                cap_types=cap_types, n_units=n_units)


def _warn_unresolved(mol: Molecule) -> None:
    unresolved = [a for a in mol.atoms if not a.ff_type or a.ff_type == a.element]
    if unresolved:
        log.warning(
            "%d atoms have no specific FF type; fell back to element symbol. "
            "Add manual_types to your config to override.",
            len(unresolved),
        )


# ------------------------------------------------------------ OPLS SMARTS typer
def _assign_opls_smarts(mol: Molecule, ff: ForceField,
                        manual_types: Optional[Dict[int, str]]) -> None:
    """Assign OPLS-AA 2024 numeric type IDs via SMARTS rules.

    Preserves any atoms already typed by ``manual_types``; overrides
    everything else via :func:`paaf.typers.oplsaa.type_oplsaa`.
    """
    from .typers.oplsaa import type_oplsaa
    types = type_oplsaa(mol)
    for a in mol.atoms:
        if manual_types and a.index in manual_types:
            a.ff_type = manual_types[a.index]
        else:
            a.ff_type = types.get(a.index, a.element)


# ------------------------------------------------------------ OpenBabel typer
def _assign_openbabel(mol: Molecule, ff: ForceField) -> None:
    try:
        from openbabel import openbabel, pybel
    except Exception as exc:
        raise RuntimeError("OpenBabel required for openbabel atom typer") from exc

    # Emit temp mol2 -> re-read -> use OBAtomTyper via ObForceField.GetAtomType
    from .structure import write
    tmp = Path(f"/tmp/_mta_type_{id(mol)}.mol2")
    write(mol, tmp)
    obmol = next(pybel.readfile("mol2", str(tmp)))

    ob_ff_name = "GAFF" if ff.key.startswith("gaff") else "MMFF94"
    if ff.key.startswith("opls"):
        # OB's OPLS typing is limited but sufficient for the moltemplate lookup
        ob_ff_name = "Ghemical"
    ob_ff = openbabel.OBForceField.FindForceField(ob_ff_name)
    if ob_ff is None:
        log.warning("OpenBabel FF %s not available; falling back to element", ob_ff_name)
        for a in mol.atoms:
            a.ff_type = a.element
        return
    ob_ff.Setup(obmol.OBMol)
    for i, a in enumerate(mol.atoms):
        ob_atom = obmol.OBMol.GetAtom(i + 1)
        try:
            t = ob_atom.GetType()
        except Exception:
            t = a.element
        a.ff_type = t
    try:
        tmp.unlink()
    except OSError:
        pass


# ------------------------------------------------------------ DL_FIELD typer
_SF_RULE_RE = re.compile(
    r"^\s*(?P<key>\S+)\s+(?P<elem>[A-Za-z]{1,2})\s+"
    r"(?P<nbonds>\d+)\s+(?P<neigh>\S+)"
)


def _parse_dlfield_sf(sf_path: Path) -> List[dict]:
    """Return a list of typing rules extracted from a DL_FIELD .sf file.

    The full .sf syntax is not trivial; this parser handles the most common
    entries of the form:

        <atom_key> <element> <n_bonds> <neighbor_elements> <comment>

    where neighbor_elements is a hyphen-separated list e.g. "C-C-H-H".
    Rules that don't match this shape are skipped (they can still be added
    to a ``manual_types`` map).
    """
    rules: List[dict] = []
    with sf_path.open() as f:
        for raw in f:
            line = raw.split("!", 1)[0].split("#", 1)[0].strip()
            if not line:
                continue
            m = _SF_RULE_RE.match(line)
            if not m:
                continue
            neigh = m.group("neigh")
            rules.append(
                {
                    "key": m.group("key"),
                    "element": m.group("elem"),
                    "nbonds": int(m.group("nbonds")),
                    "neighbors": tuple(sorted(neigh.upper().split("-"))),
                }
            )
    return rules


def _assign_from_dlfield(mol: Molecule, ff: ForceField, dl_lib_dir: Path) -> None:
    sf = Path(dl_lib_dir) / (ff.dlfield_sf or "")
    if not sf.exists():
        log.warning("DL_FIELD .sf not found at %s; using element fallback", sf)
        for a in mol.atoms:
            a.ff_type = a.element
        return
    rules = _parse_dlfield_sf(sf)
    for a in mol.atoms:
        nbrs = tuple(sorted(mol.atoms[j].element.upper() for j in mol.neighbors(a.index)))
        best = None
        for r in rules:
            if r["element"].upper() != a.element.upper():
                continue
            if r["nbonds"] != len(nbrs):
                continue
            if r["neighbors"] == nbrs:
                best = r
                break
            # weaker match: same multiset ignoring aromatic/valence detail
            if set(r["neighbors"]) == set(nbrs):
                best = r if best is None else best
        a.ff_type = best["key"] if best else a.element


# ------------------------------------------------------------ antechamber
def _assign_antechamber(mol: Molecule, ff: ForceField) -> None:
    """Type via antechamber; requires AmberTools on PATH."""
    from .structure import write

    if not _which("antechamber"):
        raise RuntimeError(
            "antechamber not on PATH; install AmberTools or switch atom_typer."
        )
    tmpdir = Path(f"/tmp/_mta_ante_{id(mol)}")
    tmpdir.mkdir(parents=True, exist_ok=True)
    src = tmpdir / "mol.mol2"
    write(mol, src)
    at = "gaff2" if ff.key == "gaff2" else "gaff"
    cmd = [
        "antechamber",
        "-i", str(src), "-fi", "mol2",
        "-o", str(tmpdir / "typed.mol2"), "-fo", "mol2",
        "-at", at, "-c", "bcc", "-s", "0",
    ]
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    typed = tmpdir / "typed.mol2"
    # parse types from mol2
    types = _read_mol2_types(typed)
    for a, t in zip(mol.atoms, types):
        a.ff_type = t


def _read_mol2_types(path: Path) -> List[str]:
    types: List[str] = []
    in_atoms = False
    with path.open() as f:
        for line in f:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue
            if line.startswith("@<TRIPOS>"):
                in_atoms = False
                continue
            if in_atoms and line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    types.append(parts[5])
    return types


def _which(exe: str) -> Optional[str]:
    import shutil as _sh
    return _sh.which(exe)
