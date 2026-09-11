"""Column-aware search over force-field atom types (typing dialogs).

A plain substring search made ``68`` match 1068, 1168, 268 … and put the
one the user wanted somewhere in the middle. Here the user chooses the
column, and the id column is matched *exactly first*, then by prefix.
"""
from __future__ import annotations

from typing import List, Sequence

UA_TAG = "UA:"


def _bare_id(t) -> str:
    fid = str(t.ff_id)
    return fid[len(UA_TAG):] if fid.startswith(UA_TAG) else fid


def _num(q: str):
    try:
        return float(q)
    except ValueError:
        return None


def filter_types(types: Sequence, query: str, field: str = "all",
                 library: str = "all", tol: float = 0.05) -> List:
    """Return the subset of ``types`` matching ``query`` in ``field``.

    ``field``: all | id | element | key | charge | mass | params | desc.
    ``library``: all | aa | ua (rows tagged ``UA:`` are united-atom).
    Exact id matches come first, then prefix matches, then the rest.
    """
    rows = list(types)
    if library == "ua":
        rows = [t for t in rows if str(t.ff_id).startswith(UA_TAG)]
    elif library == "aa":
        rows = [t for t in rows if not str(t.ff_id).startswith(UA_TAG)]
    q = (query or "").strip()
    if not q:
        return rows
    ql = q.lower()
    qn = _num(q)

    def text_of(t, f):
        return {"element": t.element or "", "key": t.key or "",
                "desc": t.description or "", "params": t.params or ""}.get(f, "")

    if field == "id":
        exact = [t for t in rows if _bare_id(t).lower() == ql or str(t.ff_id).lower() == ql]
        prefix = [t for t in rows if t not in exact and
                  (_bare_id(t).lower().startswith(ql) or str(t.ff_id).lower().startswith(ql))]
        return exact + prefix
    if field == "mass":
        if qn is None:
            return [t for t in rows if ql in f"{t.mass:.3f}"]
        return sorted([t for t in rows if t.mass and abs(float(t.mass) - qn) <= tol],
                      key=lambda t: abs(float(t.mass) - qn))
    if field == "charge":
        if qn is None:
            return [t for t in rows if ql in f"{t.charge:+.3f}"]
        return sorted([t for t in rows if abs(float(t.charge) - qn) <= tol],
                      key=lambda t: abs(float(t.charge) - qn))
    if field in ("element", "key"):
        col = field
        exact = [t for t in rows if text_of(t, col).lower() == ql]
        rest = [t for t in rows if t not in exact and ql in text_of(t, col).lower()]
        return exact + rest
    if field in ("desc", "params"):
        return [t for t in rows if ql in text_of(t, field).lower()]
    # all fields: exact id first, then anything containing the text
    exact = [t for t in rows if _bare_id(t).lower() == ql or str(t.ff_id).lower() == ql]
    rest = [t for t in rows if t not in exact and (
        ql in str(t.ff_id).lower() or ql in (t.element or "").lower()
        or ql in (t.key or "").lower() or ql in (t.description or "").lower()
        or ql in (t.params or "").lower())]
    return exact + rest
