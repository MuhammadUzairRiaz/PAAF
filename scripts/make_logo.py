#!/usr/bin/env python3
"""Generate the PAAF application icon.

The mark is a **polymer backbone**: a zig-zag chain of bonded nodes, which is
exactly what the tool builds. It is drawn with pure geometry (no font, no SVG
renderer) so it stays crisp at every size and has no runtime dependencies
beyond Pillow, which is only needed to *regenerate* the icons — the app itself
just loads the PNGs.

Run::

    python scripts/make_logo.py

Writes ``paaf/gui/assets/paaf_<size>.png`` for the sizes Qt asks for, plus a
Windows ``paaf.ico``. Colours come from the design tokens so the icon can never
drift from the UI.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paaf.gui import tokens as T  # noqa: E402

OUT_DIR = ROOT / "paaf" / "gui" / "assets"
SIZES = (16, 24, 32, 48, 64, 128, 256, 512)

# Supersample, then downscale — gives clean antialiasing without any blur pass.
SS = 8


def _rounded_rect(draw: ImageDraw.ImageDraw, box, radius, fill):
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def draw_icon(size: int) -> Image.Image:
    """Draw the mark at ``size`` px.

    The geometry is *simplified for small sizes*: a 5-node chain with a pendant
    group turns to mush below ~48px, so small icons drop to 3 nodes, thicker
    bonds and no pendant. Detail that cannot be resolved is noise, not richness.
    """
    small = size <= 40
    s = size * SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # --- rounded-square plate in the sidebar chrome colour
    pad = s * 0.02
    _rounded_rect(d, (pad, pad, s - pad, s - pad), radius=s * 0.22,
                  fill=T.BG_CHROME)

    # --- polymer backbone: a zig-zag of bonded nodes across the plate
    n_nodes = 3 if small else 5
    margin = s * (0.24 if small else 0.20)
    span = s - 2 * margin
    amp = s * (0.13 if small else 0.11)     # zig-zag amplitude
    mid = s * (0.50 if small else 0.52)

    pts = []
    for i in range(n_nodes):
        x = margin + span * i / (n_nodes - 1)
        y = mid + (-amp if i % 2 == 0 else amp)
        pts.append((x, y))

    bond_w = max(2.0, s * (0.085 if small else 0.055))
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        d.line([(x1, y1), (x2, y2)], fill=T.PRIMARY, width=int(bond_w))

    # --- pendant bond on the middle node: marks a functional/reactive site,
    #     which is what the reaction scheme works with. Omitted when small.
    if not small:
        mx, my = pts[len(pts) // 2]
        d.line([(mx, my), (mx, my - s * 0.20)], fill=T.MAP_TOKEN,
               width=int(bond_w))
        r_side = s * 0.052
        d.ellipse((mx - r_side, my - s * 0.20 - r_side,
                   mx + r_side, my - s * 0.20 + r_side), fill=T.MAP_TOKEN)

    # --- nodes: alternating backbone atoms, ends lighter to imply [*] growth
    r = s * (0.115 if small else 0.075)
    for i, (x, y) in enumerate(pts):
        if i in (0, n_nodes - 1):
            colour = "#93C5FD"          # open connection points
        else:
            colour = "#FFFFFF"
        d.ellipse((x - r, y - r, x + r, y + r), fill=colour)

    return img.resize((size, size), Image.LANCZOS)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    for size in SIZES:
        img = draw_icon(size)
        path = OUT_DIR / f"paaf_{size}.png"
        img.save(path)
        made.append(path)

    # A multi-resolution .ico for Windows / generic use.
    base = draw_icon(256)
    ico = OUT_DIR / "paaf.ico"
    base.save(ico, sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    made.append(ico)

    for p in made:
        print(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size:,} bytes)")
    print(f"\n{len(made)} files in {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
