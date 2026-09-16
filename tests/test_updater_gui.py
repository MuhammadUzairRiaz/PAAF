"""Main-window wiring of the self-updater, with paaf.updater stubbed out."""
from __future__ import annotations

import json
import threading
import time

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 not installed")

from PyQt5 import QtCore  # noqa: E402
from PyQt5.QtCore import QThread  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from paaf import updater  # noqa: E402
from paaf.gui.main_window import MainWindow  # noqa: E402

_RealQSettings = QtCore.QSettings


@pytest.fixture(autouse=True)
def temp_settings(tmp_path, monkeypatch):
    """Keep QSettings("PAAF", "PAAF") out of the user's real preferences."""
    path = str(tmp_path / "paaf.ini")

    class _Settings(_RealQSettings):
        def __init__(self, *_a, **_k):
            super().__init__(path, _RealQSettings.IniFormat)

    monkeypatch.setattr(QtCore, "QSettings", _Settings)
    # No paaf_update.log in the real PAAF folder from these tests.
    monkeypatch.setattr(updater, "repo_root", lambda: None)
    yield _Settings
    updater.set_settings_backend(updater._memory_settings.get,
                                 updater._memory_settings.__setitem__)


def wait_until(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def info(remote="b" * 40, n=2):
    return updater.UpdateInfo(available=True, local="a" * 40, remote=remote,
                              commits=[f"abc{i} Commit subject {i}" for i in range(n)])


def test_banner_hidden_at_startup():
    w = MainWindow()
    assert w._update_banner.isHidden()


def test_available_update_shows_banner_with_commit_tooltip(monkeypatch):
    monkeypatch.setattr(updater, "check_for_update", lambda timeout=10: info())
    w = MainWindow()
    w._updater_check(manual=False)
    assert wait_until(lambda: not w._update_banner.isHidden())
    assert "(2 new commits)" in w._update_label.text()
    assert "Commit subject 1" in w._update_label.toolTip()
    assert not w._btn_update_now.isHidden()


def test_skipped_version_not_offered_at_startup(monkeypatch):
    monkeypatch.setattr(updater, "check_for_update", lambda timeout=10: info())
    w = MainWindow()
    updater.skip_commit("b" * 40)
    w._updater_check(manual=False)
    assert wait_until(lambda: w._update_task is None)
    assert w._update_banner.isHidden()


def test_manual_check_reports_up_to_date(monkeypatch):
    monkeypatch.setattr(updater, "check_for_update",
                        lambda timeout=10: updater.UpdateInfo(local="c" * 40,
                                                              remote="c" * 40))
    w = MainWindow()
    w._updater_check_now()
    assert wait_until(lambda: "up to date" in w._update_label.text())


def test_update_now_disabled_while_a_job_runs(monkeypatch):
    w = MainWindow()
    w._updater_show_available(info())
    assert w._btn_update_now.isEnabled()

    release = threading.Event()

    class _Job(QThread):          # stands in for a pipeline / tab worker
        def run(self):
            release.wait(5)

    blocker = _Job(w)
    blocker.start()
    try:
        w._updater_sync_buttons()
        assert not w._btn_update_now.isEnabled()
    finally:
        release.set()
        blocker.wait(5000)


def test_refusal_is_shown_and_nothing_applied(monkeypatch):
    applied = []
    monkeypatch.setattr(updater, "can_update", lambda: (False, "local changes here"))
    monkeypatch.setattr(updater, "apply_update", lambda *a: applied.append(1))
    w = MainWindow()
    w._update_info = info()
    w._updater_show_available(w._update_info)
    w._updater_update_now()
    assert wait_until(lambda: "local changes here" in w._update_label.text())
    assert applied == []


def test_successful_update_saves_session_and_relaunches(monkeypatch, temp_settings):
    relaunched = []
    monkeypatch.setattr(updater, "can_update", lambda: (True, ""))
    monkeypatch.setattr(updater, "apply_update", lambda log_fn=None: updater.UpdateResult(
        True, "Updated.", old="a" * 40, new="b" * 40))
    monkeypatch.setattr(updater, "relaunch", lambda: relaunched.append(1) or "stub")
    w = MainWindow()
    w.builder_tab.panel_solo.smiles_edit.setText("CCOCCO")
    w.builder_tab.panel_b.smiles_edit.setText("CC(C)O")
    w.output_dir.setText("/tmp/paaf-out")
    w._update_info = info()
    w._updater_update_now()

    assert wait_until(lambda: "Restarting" in w._update_label.text())
    state = json.loads(temp_settings().value(MainWindow._SESSION_KEY, "", type=str))
    assert state["smiles"]["panel_solo"] == "CCOCCO"
    assert state["smiles"]["panel_b"] == "CC(C)O"
    assert state["output_dir"] == "/tmp/paaf-out"
    assert wait_until(lambda: relaunched)


def test_session_is_restored_once_after_restart(temp_settings):
    s = temp_settings()
    s.setValue(MainWindow._SESSION_KEY, json.dumps(
        {"smiles": {"panel_solo": "c1ccccc1"}, "output_dir": "/tmp/restored", "box": [30, 31, 32]}))
    s.sync()
    w = MainWindow()
    assert w.builder_tab.panel_solo.smiles_edit.text() == "c1ccccc1"
    assert [w.box_a.value(), w.box_b.value(), w.box_c.value()] == [30, 31, 32]
    assert w.output_dir.text() == "/tmp/restored"
    assert temp_settings().value(MainWindow._SESSION_KEY, "", type=str) == ""


def test_failed_update_keeps_running_with_message(monkeypatch, temp_settings):
    relaunched = []
    monkeypatch.setattr(updater, "can_update", lambda: (True, ""))
    monkeypatch.setattr(updater, "apply_update", lambda log_fn=None: updater.UpdateResult(
        False, "Update rolled back: pip failed: boom", rolled_back=True))
    monkeypatch.setattr(updater, "relaunch", lambda: relaunched.append(1))
    w = MainWindow()
    w._update_info = info()
    w._updater_update_now()
    assert wait_until(lambda: "rolled back" in w._update_label.text())
    assert wait_until(lambda: w._update_task is None)
    QApplication.processEvents()
    assert relaunched == []
    assert w._update_applying is False
    assert temp_settings().value(MainWindow._SESSION_KEY, "", type=str) == ""
    assert "rolled back" in w.console.toPlainText()


def test_updater_exception_does_not_escape(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(updater, "can_update", boom)
    w = MainWindow()
    w._update_info = info()
    w._updater_update_now()
    assert wait_until(lambda: "failed" in w._update_label.text())
