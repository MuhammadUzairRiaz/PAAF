#!/usr/bin/env python3
"""Diagnose WHICH copy of PAAF your Python is loading, and whether the
redesign is present in it.

Run it exactly the way you run the app::

    conda activate mta
    cd /Users/uzair/project/PAAF
    python check_build.py

If it reports a path other than this folder, your Python is importing a
different (probably pip-installed) copy of ``paaf`` — that is why UI changes
would not appear. The fix is printed at the end.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    print("=" * 68)
    print("PAAF build check")
    print("=" * 68)
    print(f"python        : {sys.executable}")
    print(f"cwd           : {Path.cwd()}")
    print(f"this script   : {HERE}")

    try:
        import paaf
    except Exception as exc:
        print(f"\nFAILED to import paaf: {exc}")
        return 1

    pkg_dir = Path(paaf.__file__).resolve().parent
    print(f"paaf loaded   : {pkg_dir}")

    same = pkg_dir.parent == HERE
    print(f"\n{'OK   ' if same else 'WRONG'}  loading from "
          f"{'THIS folder' if same else 'a DIFFERENT folder'}")

    # --- redesign markers: each is a distinctive string from the new UI
    markers = {
        "tokens module (design tokens)":      ("paaf/gui/tokens.py", "BG_CHROME"),
        "page chrome (breadcrumb/cards)":     ("paaf/gui/page.py", "def page_header"),
        "sidebar stepper":                    ("paaf/gui/sidebar.py", "TOOL"),
        "breadcrumb wired into pages":        ("paaf/gui/main_window.py", "Pipeline  ›  Step"),
        "tabs = underlined text":             ("paaf/gui/theme.py", "QTabBar::tab:selected"),
        "reaction scheme tab":                ("paaf/gui/reaction_scheme_tab.py", "ReactionSchemeTab"),
        "dashed empty state":                 ("paaf/gui/reaction_scheme_tab.py", "emptyState"),
        "two-column validation":              ("paaf/gui/reaction_scheme_tab.py", "_bond_block"),
        "widget backgrounds paint (PE_Widget)": ("paaf/gui/sidebar.py", "PE_Widget"),
    }

    print("\nRedesign markers in the loaded package:")
    missing = 0
    for label, (rel, needle) in markers.items():
        f = pkg_dir.parent / rel
        ok = f.exists() and needle in f.read_text(errors="replace")
        if not ok:
            missing += 1
        print(f"  {'present' if ok else 'MISSING'}   {label}")

    # --- stale bytecode can shadow edited sources in odd setups
    caches = list((pkg_dir / "gui").glob("__pycache__/*.pyc"))
    if caches:
        newest_src = max((p.stat().st_mtime for p in (pkg_dir / "gui").glob("*.py")),
                         default=0)
        newest_pyc = max((p.stat().st_mtime for p in caches), default=0)
        if newest_pyc < newest_src:
            print("\nNOTE: compiled caches are older than the sources "
                  "(harmless — Python will recompile).")

    print()
    if same and not missing:
        print("RESULT: the redesigned code IS what your app will load.")
        print("        If the window still looks old, the app was not "
              "restarted — quit it fully and run:  python run_gui.py")
    elif not same:
        print("RESULT: your Python is loading a DIFFERENT paaf, so none of the")
        print("        UI changes can show up. Fix it with:")
        print(f"            pip uninstall -y paaf")
        print(f"            cd {HERE} && pip install -e .")
        print("        (or just always run from this folder)")
    else:
        print(f"RESULT: {missing} redesign marker(s) missing from the loaded copy.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
