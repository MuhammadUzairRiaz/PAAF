"""Entry point for the PyQt5 GUI."""
from __future__ import annotations

import sys


def launch() -> int:
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QLocale
    except Exception:
        try:
            from PySide2.QtWidgets import QApplication  # type: ignore
            from PySide2.QtCore import QLocale          # type: ignore
        except Exception as exc:
            print(
                "PyQt5 (or PySide2) is required for the GUI.\n"
                "Install with: pip install PyQt5",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

    # ---- Qt web engine: both of these MUST happen before the QApplication --
    #
    # Qt enforces two rules and reports neither helpfully:
    #
    #   1. QtWebEngineWidgets has to be imported before a QCoreApplication
    #      exists. Import it later — say, lazily inside a dialog — and the
    #      import itself raises, which reads exactly like "not installed".
    #   2. AA_ShareOpenGLContexts has to be set before the QApplication is
    #      constructed, or the web view renders blank.
    #
    # Doing this here costs an unused import when the 3D view is never opened,
    # which is a much better trade than a viewer that cannot be switched on.
    try:
        from PyQt5.QtCore import Qt as _Qt
        QApplication.setAttribute(_Qt.AA_ShareOpenGLContexts, True)
    except Exception:
        pass
    try:
        from PyQt5 import QtWebEngineWidgets  # noqa: F401
    except Exception as _exc:                              # pragma: no cover
        # Not fatal: the typing table works without the 3D view. Record why so
        # the dialog can say something true rather than guessing.
        import os
        os.environ.setdefault("PAAF_WEBENGINE_ERROR", str(_exc))

    from .main_window import MainWindow

    # Force English/C locale so number widgets show "1.00" not "1,00" and
    # "50" not "50,000" regardless of the user's system locale.
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))

    app = QApplication.instance() or QApplication(sys.argv)

    # Brand mark: set on the application so the dock / task switcher shows it,
    # and again on the window below for the title bar.
    try:
        from .branding import app_icon
        _icon = app_icon()
        if _icon is not None:
            app.setWindowIcon(_icon)
    except Exception:
        pass
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(launch())
