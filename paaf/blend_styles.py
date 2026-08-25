"""Force-field styles for a merged blend, taken from the components' own files.

Why this exists
---------------
A LAMMPS data file is only half a model. It carries numbers — masses, charges,
coefficient tables, connectivity — but not the *styles* that say how to read
them. ``bond_style harmonic`` is what turns "317.0 1.51" into a spring. Without
the style block, ``read_data`` fails on the first coefficient it meets.

PAAF used to write a blend input with ``units``, ``atom_style`` and
``boundary`` and nothing else, then a long list of ``pair_coeff`` lines. That
file cannot run. The styles were sitting in each component's own ``lammps.in``
the whole time; this module goes and gets them.

The hybrid problem
------------------
DL_FIELD writes every style as a *hybrid* with exactly one sub-style::

    bond_style      hybrid harmonic
    pair_style      hybrid lj/cut/coul/long 12.0

and then tags every coefficient line with the sub-style name::

    1 harmonic 317.000000 1.510000

That is valid on its own. It stops being valid once components are merged,
because each component's tables are re-numbered and concatenated, and LAMMPS
resolves hybrid sub-styles per type in a way that no longer lines up. A hybrid
of one sub-style is also just the sub-style, so the honest fix is to drop the
wrapper: ``bond_style harmonic``, and strip the token from every coefficient
line.

That conversion is what makes this a "non-hybrid" blend, and it is the reason
the reference script carries the word in its filename.

Components must agree
---------------------
Two components typed with different force fields cannot share one data file:
their coefficient tables mean different things. If the styles disagree, this
module says so and stops, rather than emitting a file that runs and is wrong.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["StyleBlock", "read_style_block", "dehybridise_style",
           "dehybridise_coeff_lines", "merge_style_blocks",
           "find_forcefield_input", "StyleMismatch"]

# DL_FIELD's own ordering, kept so a blend input reads like the single-
# component inputs it was built from: global settings, then the four bonded
# styles, then pair/kspace, then the exclusion rules.
#
# LAMMPS does not care about the order among these. People do — a file that
# matches the one the force-field generator wrote is one a user can diff.
STYLE_DIRECTIVES = (
    "units", "dimension", "atom_style", "boundary", "timestep", "dielectric",
    "bond_style", "angle_style", "dihedral_style", "improper_style",
    "pair_style", "pair_modify", "kspace_style", "special_bonds",
)

# Coefficient sections whose lines may carry a hybrid sub-style token, and the
# directive whose sub-style name appears there.
COEFF_SECTION_STYLE = {
    "Pair Coeffs": "pair_style",
    "Bond Coeffs": "bond_style",
    "Angle Coeffs": "angle_style",
    "Dihedral Coeffs": "dihedral_style",
    "Improper Coeffs": "improper_style",
}

# Styles that must agree between components for a merge to mean anything.
_MUST_MATCH = ("units", "dimension", "atom_style", "pair_style", "bond_style",
               "angle_style", "dihedral_style", "improper_style",
               "special_bonds")

#: Directives where components may legitimately differ and the safe choice is
#: the smallest value. A timestep stable for a stiff component is stable for a
#: soft one; the reverse is not true, so taking the larger would silently
#: destabilise whichever component asked for less.
_TAKE_SMALLEST = ("timestep",)


class StyleMismatch(RuntimeError):
    """Components were typed with force fields that cannot be merged."""


@dataclass
class StyleBlock:
    """The style directives read from one component's LAMMPS input."""
    source: Optional[Path] = None
    directives: Dict[str, str] = field(default_factory=dict)

    def get(self, name: str) -> str:
        return self.directives.get(name, "")

    def substyle(self, directive: str) -> str:
        """The lone sub-style name of a hybrid directive, or ``''``.

        ``pair_style hybrid lj/cut/coul/long 12.0`` -> ``lj/cut/coul/long``.
        """
        value = self.directives.get(directive, "").split("#")[0].split()
        if len(value) >= 2 and value[0] == "hybrid":
            return value[1]
        return ""

    def lines(self) -> List[str]:
        return [f"{k:<15} {self.directives[k]}"
                for k in STYLE_DIRECTIVES if k in self.directives]

    def is_empty(self) -> bool:
        return not self.directives


# =====================================================================
def find_forcefield_input(data_file: Path) -> Optional[Path]:
    """The LAMMPS input that carries the styles for ``data_file``.

    Search order, all relative to the data file:

    1. ``<stem>.in`` — moltemplate's convention.
    2. ``lammps.in`` beside it, then in ``dlf_output1/`` — DL_FIELD's.
    3. Any single remaining ``.in``.

    Files PAAF wrote itself are skipped. This matters: after one blend run the
    component folder also holds ``packed_blend.in``, and a naive "is there
    exactly one .in?" test then finds two and gives up — which is how
    minimisation came to be skipped without anyone noticing.
    """
    data_file = Path(data_file)
    folder = data_file.parent

    def _usable(p: Path) -> bool:
        if not p.is_file() or p.name in _PAAF_OUTPUT_NAMES:
            return False
        return not read_style_block(p).is_empty()

    for candidate in (folder / f"{data_file.stem}.in",
                      folder / "lammps.in",
                      folder / "dlf_output1" / "lammps.in",
                      folder.parent / "dlf_output1" / "lammps.in"):
        if _usable(candidate):
            return candidate

    remaining = [p for p in sorted(folder.glob("*.in")) if _usable(p)]
    if len(remaining) == 1:
        return remaining[0]
    if len(remaining) > 1:
        log.warning("%s: %d candidate .in files (%s); cannot choose",
                    folder, len(remaining), ", ".join(p.name for p in remaining))
    return None


_PAAF_OUTPUT_NAMES = {"packed_blend.in", "packed_box.in", "relax.in",
                      "minimise_component.in", "minimize_component.in"}


def read_style_block(path: Path) -> StyleBlock:
    """Read the style directives out of a LAMMPS input file.

    Everything after ``read_data`` is ignored: those are run commands, not the
    model. Comments are kept, because DL_FIELD's are informative.
    """
    block = StyleBlock(source=Path(path))
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return block

    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("read_data"):
            break
        if not stripped or stripped.startswith("#"):
            continue
        word, _, rest = stripped.partition(" ")
        if word in STYLE_DIRECTIVES and rest.strip():
            block.directives[word] = rest.strip()
    return block


# =====================================================================
def dehybridise_style(value: str) -> Tuple[str, str]:
    """``'hybrid harmonic'`` -> ``('harmonic', 'harmonic')``.

    Returns ``(new_value, substyle_name)``. ``substyle_name`` is empty when
    nothing was changed, either because the directive was not hybrid or
    because it wraps more than one sub-style — in which case the wrapper is
    load-bearing and must stay.
    """
    body, sep, comment = value.partition("#")
    parts = body.split()
    if len(parts) < 2 or parts[0] != "hybrid":
        return value, ""

    substyle = parts[1]
    rest = parts[2:]
    # A second bare sub-style name means a genuine hybrid: leave it alone.
    # Numeric tokens after the name are that sub-style's own arguments.
    if any(not _is_number(tok) for tok in rest):
        return value, ""

    new_body = " ".join([substyle] + rest)
    return (new_body + (" " + sep + comment if sep else "")).rstrip(), substyle


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def dehybridise_coeff_lines(lines: Sequence[str], substyle: str) -> List[str]:
    """Strip a leading sub-style token from ``<id> <substyle> <numbers…>``.

    Only the exact sub-style name is removed, and only in second position, so
    a coefficient line that never carried one is returned untouched.
    """
    if not substyle:
        return list(lines)
    out: List[str] = []
    for line in lines:
        body, sep, comment = line.partition("#")
        parts = body.split()
        if len(parts) >= 2 and parts[1] == substyle:
            parts.pop(1)
            body = " ".join(parts)
            line = body + (" " + sep + comment if sep else "")
        out.append(line.rstrip())
    return out


def strip_substyle_tokens(lines: Sequence[str],
                          expected: str = "") -> Tuple[List[str], str]:
    """Remove the sub-style name from coefficient lines, whatever it is.

    Returns ``(lines, token_removed)``.

    :func:`dehybridise_coeff_lines` needs to be told which token to remove,
    which works when the style was declared ``hybrid <name>``. It is not
    enough on its own: when the styles are *read* from the data file they come
    back as plain names (``class2``), yet the coefficient lines still carry
    the token. Stripping nothing then leaves a file whose ``Bond Coeffs`` read
    ``1 class2 1.52 …`` under a plain ``bond_style class2``, and LAMMPS tries
    to parse ``class2`` as a bond length.

    So the token is identified structurally — column 2, present, not a number
    — rather than from the declaration. ``expected`` is only used to warn on a
    mismatch, which would mean the data file and the input file disagree about
    the functional form.
    """
    token = ""
    for line in lines:
        parts = line.split("#")[0].split()
        if len(parts) >= 2 and not _is_number(parts[1]):
            token = parts[1]
            break
    if not token:
        return list(lines), ""
    if expected and token != expected:
        log.warning("Coefficient lines say %r but the style says %r; "
                    "trusting the coefficient lines", token, expected)
    return dehybridise_coeff_lines(lines, token), token


def dehybridise_pair_coeff(line: str, substyle: str) -> str:
    """Strip the sub-style from ``pair_coeff i j <substyle> eps sigma``."""
    if not substyle:
        return line
    return re.sub(rf"^(\s*pair_coeff\s+\S+\s+\S+)\s+{re.escape(substyle)}\s+",
                  r"\1 ", line)


# =====================================================================
def merge_style_blocks(blocks: Sequence[StyleBlock],
                       names: Sequence[str]) -> Tuple[StyleBlock, Dict[str, str]]:
    """One style block for the blend, plus the sub-style stripped per section.

    Raises :class:`StyleMismatch` when two components disagree on a style that
    changes what the numbers mean. Merging those would produce a file that
    runs and silently models something nobody asked for.
    """
    usable = [(n, b) for n, b in zip(names, blocks) if not b.is_empty()]
    if not usable:
        raise StyleMismatch(
            "None of the components has a readable LAMMPS input file, so the "
            "force-field styles are unknown. PAAF will not guess them — a "
            "blend written without styles cannot be read back by LAMMPS.")

    merged = StyleBlock(source=usable[0][1].source)
    for directive in STYLE_DIRECTIVES:
        values = {n: b.get(directive) for n, b in usable if b.get(directive)}
        if not values:
            continue
        normalised = {n: _normalise(v) for n, v in values.items()}
        distinct = set(normalised.values())

        # "none" is an absence, not a disagreement. A pure alkane has no
        # impropers, so DL_FIELD writes it `improper_style none`; blending it
        # with PMMA (`improper_style cvff`) was refused with a message
        # claiming the two were typed with different force fields — they were
        # not, one molecule simply has no terms of that class and contributes
        # none to the blend. Components saying "none" defer to those that
        # actually have terms; only the styles of components WITH terms have
        # to agree.
        if len(distinct) > 1 and "none" in distinct:
            values = {n: v for n, v in values.items()
                      if _normalise(v) != "none"}
            normalised = {n: _normalise(v) for n, v in values.items()}
            distinct = set(normalised.values())
            log.info("Blend: %s is 'none' for some components (no such "
                     "terms); using %s from the ones that have them.",
                     directive, ", ".join(sorted(distinct)))

        if directive in _TAKE_SMALLEST and len(distinct) > 1:
            best = min(values.items(),
                       key=lambda kv: _leading_number(kv[1], float("inf")))
            log.info("Blend: components disagree on %s (%s); taking %s",
                     directive,
                     ", ".join(f"{n}={_normalise(v)}"
                               for n, v in values.items()),
                     _normalise(best[1]))
            merged.directives[directive] = best[1]
            continue

        if len(distinct) > 1 and directive in _MUST_MATCH:
            detail = "\n".join(f"    {n}: {directive} {v}"
                               for n, v in values.items())
            raise StyleMismatch(
                f"Components disagree on {directive}, so they were typed with "
                f"different force fields and cannot share one data file:\n"
                f"{detail}\n"
                f"Re-type every component with the same force field, then "
                f"blend.")
        merged.directives[directive] = values[usable[0][0]] \
            if usable[0][0] in values else next(iter(values.values()))

    substyles: Dict[str, str] = {}
    for directive in ("pair_style", "bond_style", "angle_style",
                      "dihedral_style", "improper_style"):
        if directive not in merged.directives:
            continue
        new_value, sub = dehybridise_style(merged.directives[directive])
        merged.directives[directive] = new_value
        if sub:
            substyles[directive] = sub
            log.info("Blend: %s hybrid %s -> %s", directive, sub, sub)

    return merged, substyles


def _normalise(value: str) -> str:
    return " ".join(value.split("#")[0].split())


def _leading_number(value: str, default: float) -> float:
    tokens = value.split("#")[0].split()
    return float(tokens[0]) if tokens and _is_number(tokens[0]) else default
