"""Parse the pipeline's "[stage k/N] label" progress lines (GUI-agnostic)."""
from __future__ import annotations

import re
from typing import Optional, Tuple

STAGE_RE = re.compile(r"^\[stage (\w+)/(\d+)\]\s*(.*)$")


def parse_stage(msg: str) -> Optional[Tuple[int, int, str, bool]]:
    """Return ``(percent, n_stages, status_text, finished)`` or None.

    Stage k *running* shows ``(k-1)/N`` done; the done marker shows 100 %.
    """
    m = STAGE_RE.match(msg.strip())
    if not m:
        return None
    k, n, label = m.group(1), int(m.group(2)), m.group(3)
    if k == "done":
        return 100, n, "✓ Completed — " + label, True
    ki = int(k)
    return int(100 * (ki - 1) / n), n, f"Step {ki} of {n}: {label} …", False
