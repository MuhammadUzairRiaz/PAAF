"""Shipped copolymer preset library.

Loaded from ``paaf/data/copolymer_database.csv`` at import time.
Each row defines a random / alternating / block copolymer of two monomers
(with their polymerization SMILES containing ``[*]`` connection points),
the fraction of monomer B, the sequence mode, and a random seed.

Example row (from library_copolymer.csv):

    W01_P0ENR, [*]C/C=C(C)\\C[*], [*]CC1(C)OC1C[*], 0.5, random, 12345

The Builder GUI lists these alongside the 123 homopolymer entries so users
can pick a copolymer, click *Build*, and get both monomer 3D files + the
Chain page pre-populated with the correct fractions / mode / seed.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .logging_utils import get_logger

log = get_logger(__name__)

_DB_PATH = Path(__file__).resolve().parent / "data" / "copolymer_database.csv"


@dataclass(frozen=True)
class CopolymerRecipe:
    pid: str                    # e.g. "W01_P0ENR"
    smiles_a: str               # monomer A polymerization SMILES with [*]
    smiles_b: str               # monomer B polymerization SMILES with [*]
    fraction_b: float           # 0..1
    sequence_mode: str          # "random" | "alternating" | "block"
    random_seed: Optional[int]

    @property
    def fraction_a(self) -> float:
        return 1.0 - self.fraction_b

    @property
    def description(self) -> str:
        return (f"Copolymer {self.pid}: {self.sequence_mode} "
                f"({self.fraction_a:.0%} A / {self.fraction_b:.0%} B)")


def _to_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.5


def _to_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def load_copolymer_library(path: Path = _DB_PATH) -> List[CopolymerRecipe]:
    if not path.exists():
        log.warning("Copolymer library %s not found.", path)
        return []
    out: List[CopolymerRecipe] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("PID") or "").strip()
            sa = (row.get("smiles_polymer") or "").strip()
            sb = (row.get("smiles_polymer_b") or "").strip()
            if not (pid and sa and sb):
                continue
            out.append(CopolymerRecipe(
                pid=pid, smiles_a=sa, smiles_b=sb,
                fraction_b=_to_float(row.get("fraction_b", "0.5")),
                sequence_mode=(row.get("sequence_mode") or "random").strip().lower(),
                random_seed=_to_int(row.get("random_seed")),
            ))
    log.info("Loaded %d copolymer preset(s) from %s", len(out), path.name)
    return out


COPOLYMER_LIBRARY: Dict[str, CopolymerRecipe] = {
    r.pid.upper(): r for r in load_copolymer_library()
}


def list_copolymers() -> List[CopolymerRecipe]:
    return list(COPOLYMER_LIBRARY.values())


def get_copolymer(pid: str) -> CopolymerRecipe:
    key = pid.strip().upper()
    if key not in COPOLYMER_LIBRARY:
        raise KeyError(f"Unknown copolymer PID {pid!r}. "
                       f"Available: {sorted(COPOLYMER_LIBRARY)}")
    return COPOLYMER_LIBRARY[key]


# =====================================================================
# Writing back
# =====================================================================
def save_copolymer(recipe: "CopolymerRecipe",
                   path: Path = _DB_PATH) -> Path:
    """Append or replace a recipe in the shared copolymer library.

    The same CSV backs the simple Builder and the Amorphous cell tool, so a
    definition saved from either is immediately available in the other. That
    is the point of putting it here rather than in a tool-specific store.

    An existing PID is replaced rather than duplicated, because a duplicate
    would silently shadow the original depending on read order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["PID", "smiles_polymer", "smiles_polymer_b", "fraction_b",
              "sequence_mode", "random_seed"]

    rows: List[dict] = []
    if path.exists():
        with path.open(encoding="utf-8", errors="replace") as f:
            rows = [r for r in csv.DictReader(f)
                    if (r.get("PID") or "").strip().upper()
                    != recipe.pid.strip().upper()]

    rows.append({
        "PID": recipe.pid,
        "smiles_polymer": recipe.smiles_a,
        "smiles_polymer_b": recipe.smiles_b,
        "fraction_b": f"{recipe.fraction_b:.6g}",
        "sequence_mode": recipe.sequence_mode,
        "random_seed": "" if recipe.random_seed is None else str(recipe.random_seed),
    })

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in header})

    COPOLYMER_LIBRARY[recipe.pid.upper()] = recipe
    log.info("Saved copolymer %s to %s (%d total)", recipe.pid, path.name,
             len(rows))
    return path


def reload_copolymer_library(path: Path = _DB_PATH) -> None:
    """Re-read the CSV into :data:`COPOLYMER_LIBRARY` in place."""
    COPOLYMER_LIBRARY.clear()
    for r in load_copolymer_library(path):
        COPOLYMER_LIBRARY[r.pid.upper()] = r
