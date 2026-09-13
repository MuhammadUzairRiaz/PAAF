"""The soft push-off stage shared by every LAMMPS deck PAAF writes.

Three decks resolve close contacts with ``pair_style soft`` ramped under a
displacement cap: the back-map push-off (:mod:`paaf.cell.soft_pushoff`), the
relaxation deck (:mod:`paaf.cell.relax`) and the Kremer-Grest CG deck
(:mod:`paaf.cell.cg_model`). They used to write those commands separately,
so a fix to one (``pair_style soft`` refusing PPPM, say) had to be found and
repeated in the others. The commands now come from here; each caller passes
its own numbers explicitly.
"""
from __future__ import annotations

from typing import List, Optional

__all__ = ["soft_stage_lines"]


def soft_stage_lines(cutoff, a_start, a_end, *,
                     disable_kspace: bool = False,
                     var: str = "prefactor",
                     adapt_fix: str = "push",
                     nve_limit: Optional[float] = None,
                     limit_fix: str = "lim",
                     pad: bool = True) -> List[str]:
    """The commands that switch on a ramped soft repulsion.

    ``cutoff``           where the soft potential dies (length units of the deck)
    ``a_start, a_end``   ramp of the prefactor A over the next run
    ``disable_kspace``   emit ``kspace_style none`` first: PPPM/Ewald cannot
                         run with ``pair_style soft``
    ``nve_limit``        add ``fix <limit_fix> all nve/limit <value>``
    ``pad``              align command names to 15 columns (relax/CG decks)

    ``pair_coeff * * 0.0`` is always written: the prefactor must be set
    explicitly before ``fix adapt`` ramps it, or LAMMPS reports it unset.
    """
    def cmd(name: str, rest: str) -> str:
        return f"{name:<15s} {rest}" if pad else f"{name} {rest}"

    lines: List[str] = []
    if disable_kspace:
        lines.append(cmd("kspace_style", "none"))
    lines.append(cmd("pair_style", f"soft {cutoff}"))
    lines.append(cmd("pair_coeff", "* * 0.0"))
    lines.append(cmd("variable", f"{var} equal ramp({a_start},{a_end})"))
    lines.append(cmd("fix", f"{adapt_fix} all adapt 1 pair soft a * * v_{var}"))
    if nve_limit is not None:
        lines.append(cmd("fix", f"{limit_fix} all nve/limit {nve_limit}"))
    return lines
