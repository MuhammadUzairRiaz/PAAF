"""Geometry optimization wrappers around OpenBabel force fields.

OpenBabel's MMFF94 typer occasionally fails on small unsaturated or
heterocyclic molecules (e.g. ``CC=C(C)C``, epoxides, some sulfonates)
because a subset of atoms cannot be assigned an MMFF94 type. Rather than
crashing the pipeline in that case, :func:`optimize` walks a **fallback
chain** — the requested FF first, then MMFF94s, then UFF (universal, works
on any element), then Ghemical. If every FF fails, the molecule is returned
untouched with a warning so downstream steps (chain building, packing,
Moltemplate export) can still proceed.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Sequence

from .logging_utils import get_logger
from .structure import Molecule, write, load_structure

log = get_logger(__name__)

FFName = Literal["UFF", "MMFF94", "MMFF94s", "GAFF", "Ghemical"]

# Ordered list of fallbacks. UFF is deliberately last-and-cheap because it
# accepts every element and never fails to type. GAFF is skipped as a
# fallback since OpenBabel's GAFF typer is limited compared to antechamber.
_FALLBACKS: Sequence[str] = ("MMFF94s", "UFF", "Ghemical")


def _try_ff(obmol, ff_name: str):
    """Return an OBForceField instance if setup succeeds; otherwise None."""
    try:
        from openbabel import openbabel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenBabel required for optimization") from exc
    force_field = openbabel.OBForceField.FindForceField(ff_name)
    if force_field is None:
        log.debug("OpenBabel has no FF called %r", ff_name)
        return None
    if not force_field.Setup(obmol.OBMol):
        log.debug("FF %s could not type this molecule", ff_name)
        return None
    return force_field


def optimize(
    mol: Molecule,
    ff: FFName = "MMFF94",
    steps: int = 10000,
    tol: float = 1e-6,
    algorithm: Literal["cg", "sd"] = "cg",
    fallback: bool = True,
    strict: bool = False,
) -> Molecule:
    """Minimize `mol` in place with an OpenBabel force field.

    Parameters
    ----------
    ff : str
        Preferred force field (MMFF94 / MMFF94s / UFF / Ghemical / GAFF).
    fallback : bool
        If the preferred FF fails, try MMFF94s → UFF → Ghemical in order.
        Default True.
    strict : bool
        If True, raise :class:`RuntimeError` when every FF fails. If False
        (default), log a warning and return `mol` with coordinates unchanged.

    Returns the same Molecule.
    """
    try:
        from openbabel import openbabel, pybel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "OpenBabel required for optimization. Install via "
            "`conda install -c conda-forge openbabel`."
        ) from exc

    # Write once to a temp mol2 so we can retry with different FFs without
    # re-parsing the structure each time.
    tmp = Path(f"/tmp/_mta_{id(mol)}.mol2")
    write(mol, tmp)
    obmol = next(pybel.readfile("mol2", str(tmp)))

    # Build the ordered list of FF attempts (dedup while preserving order).
    tried: List[str] = []
    order: List[str] = [ff]
    if fallback:
        for f in _FALLBACKS:
            if f != ff:
                order.append(f)

    force_field = None
    used_ff = None
    for name in order:
        force_field = _try_ff(obmol, name)
        tried.append(name)
        if force_field is not None:
            used_ff = name
            break

    if force_field is None:
        msg = (f"OpenBabel could not set up any of {tried} for molecule "
               f"{mol.name!r} — returning unoptimized structure.")
        if strict:
            raise RuntimeError(msg)
        log.warning(msg)
        try:
            tmp.unlink()
        except OSError:
            pass
        return mol

    if used_ff != ff:
        log.warning("Requested FF %s failed for %s; using %s instead.",
                    ff, mol.name, used_ff)

    log.info("Optimizing %s with %s (%s, up to %d steps, tol=%g)",
             mol.name, used_ff, algorithm, steps, tol)
    try:
        e_initial = force_field.Energy()
        if algorithm == "cg":
            force_field.ConjugateGradients(steps, tol)
        else:
            force_field.SteepestDescent(steps, tol)
        force_field.GetCoordinates(obmol.OBMol)
        for a, ob_atom in zip(mol.atoms, obmol.atoms):
            a.xyz[:] = ob_atom.coords
        e_final = force_field.Energy()
        log.info(
            "Optimization done (%s):  initial E = %.4f  →  final E = %.4f  "
            "(ΔE = %+.4f)",
            used_ff, e_initial, e_final, e_final - e_initial,
        )
    except Exception as exc:
        log.warning("Optimizer (%s) errored mid-run: %s. Returning unoptimized "
                    "coordinates.", used_ff, exc)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    return mol


def optimize_file(
    path: str | Path,
    output: str | Path | None = None,
    ff: FFName = "MMFF94",
    **kwargs,
) -> Path:
    mol = load_structure(path)
    mol = optimize(mol, ff=ff, **kwargs)
    out = Path(output) if output else Path(path).with_suffix(".opt.xyz")
    write(mol, out)
    return out
