"""The self-updater against real git repositories in a temp directory.

A bare repo plays GitHub ("origin"), a clone plays the user's install, and a
second clone plays the developer pushing new commits. pip and the import
check are the only subprocesses stubbed out.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from paaf import updater

pytestmark = pytest.mark.skipif(shutil.which("git") is None,
                                reason="git is not installed")


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(cwd), check=True,
                          capture_output=True, text=True).stdout.strip()


def commit(repo: Path, name: str, text: str, message: str) -> str:
    (repo / name).write_text(text)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


class Repos:
    def __init__(self, tmp: Path):
        self.origin = tmp / "origin.git"
        self.install = tmp / "install"
        self.dev = tmp / "dev"
        git(tmp, "init", "-q", "--bare", "-b", "main", str(self.origin))
        git(tmp, "clone", "-q", str(self.origin), str(self.dev))
        git(self.dev, "checkout", "-q", "-b", "main")
        commit(self.dev, "setup.py", "# v1\n", "Initial version")
        commit(self.dev, "paaf.py", "VERSION = 1\n", "Add module")
        git(self.dev, "push", "-q", "origin", "main")
        git(tmp, "clone", "-q", str(self.origin), str(self.install))

    def push(self, name: str, text: str, message: str) -> str:
        h = commit(self.dev, name, text, message)
        git(self.dev, "push", "-q", "origin", "main")
        return h

    def head(self) -> str:
        return git(self.install, "rev-parse", "HEAD")


class Subprocesses:
    """Records the pip / import-check commands and decides their outcome."""

    def __init__(self):
        self.calls = []
        self.pip_error = None          # list of errors, one per pip attempt
        self.import_error = None

    def __call__(self, real_run, args, cwd, timeout):
        if args[0] != sys.executable:
            return real_run(args, cwd, timeout)
        self.calls.append(list(args))
        if args[1:3] == ["-m", "pip"]:
            if self.pip_error:
                err = self.pip_error.pop(0)
                if err:
                    raise updater._StepFailed(err)
            return "Successfully installed paaf"
        if args[1] == "-c":
            if self.import_error:
                raise updater._StepFailed(self.import_error)
            return ""
        raise AssertionError(f"unexpected command {args}")

    def pip_calls(self):
        return [c for c in self.calls if c[1:3] == ["-m", "pip"]]

    def import_calls(self):
        return [c for c in self.calls if c[1] == "-c"]


@pytest.fixture
def repos(tmp_path, monkeypatch):
    # Isolate from the user's git config (signing, pull.rebase, hooks …).
    empty = tmp_path / "gitconfig"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "test@example.com")
    r = Repos(tmp_path)
    monkeypatch.setattr(updater, "repo_root", lambda: r.install)
    settings = {}
    updater.set_settings_backend(settings.get, settings.__setitem__)
    r.settings = settings
    yield r
    updater.set_settings_backend(updater._memory_settings.get,
                                 updater._memory_settings.__setitem__)


@pytest.fixture
def procs(monkeypatch):
    stub = Subprocesses()
    real_run = updater._run
    monkeypatch.setattr(updater, "_run",
                        lambda args, cwd, timeout: stub(real_run, list(args),
                                                        cwd, timeout))
    return stub


# ------------------------------------------------------------------ checks
def test_up_to_date(repos):
    info = updater.check_for_update()
    assert info.error is None
    assert info.available is False
    assert info.local == info.remote == repos.head()
    assert info.commits == []


def test_update_available_lists_new_commits(repos):
    old = repos.head()
    repos.push("a.txt", "a", "Fix packing overlap")
    new = repos.push("b.txt", "b", "Add GROMACS charge summary")

    info = updater.check_for_update()

    assert info.error is None
    assert info.available is True
    assert info.local == old and info.remote == new
    assert [c.split(" ", 1)[1] for c in info.commits] == [
        "Add GROMACS charge summary", "Fix packing overlap"]
    assert repos.head() == old          # checking never moves the install


def test_update_log_is_written(repos):
    updater.check_for_update()
    text = (repos.install / updater.LOG_NAME).read_text()
    assert "check: up to date" in text
    assert text.startswith("[20")


# ------------------------------------------------------------------ apply
def test_clean_fast_forward(repos, procs):
    old = repos.head()
    new = repos.push("paaf.py", "VERSION = 2\n", "Version 2")
    updater.check_for_update()

    lines = []
    result = updater.apply_update(lines.append)

    assert result.success, result.message
    assert (result.old, result.new) == (old, new)
    assert repos.head() == git(repos.install, "rev-parse", "origin/main") == new
    assert (repos.install / "paaf.py").read_text() == "VERSION = 2\n"
    assert procs.pip_calls() == []              # no dependency file changed
    assert len(procs.import_calls()) == 1
    assert updater.last_good_commit() == old
    assert lines


def test_dependency_change_runs_pip(repos, procs):
    repos.push("requirements.txt", "numpy\n", "Need numpy")
    updater.check_for_update()

    result = updater.apply_update()

    assert result.success, result.message
    assert procs.pip_calls() == [[sys.executable, "-m", "pip", "install",
                                  "-e", "."]]


def test_externally_managed_pip_retries_with_break_system_packages(repos, procs):
    repos.push("setup.py", "# v2\n", "Bump setup")
    updater.check_for_update()
    procs.pip_error = ["error: externally-managed-environment", None]

    result = updater.apply_update()

    assert result.success, result.message
    assert procs.pip_calls()[1][-1] == "--break-system-packages"


def test_local_modifications_refuse_and_change_nothing(repos, procs):
    old = repos.head()
    repos.push("paaf.py", "VERSION = 2\n", "Version 2")
    updater.check_for_update()
    (repos.install / "paaf.py").write_text("VERSION = 'my edit'\n")

    ok, reason = updater.can_update()
    result = updater.apply_update()

    assert ok is False
    assert "local changes" in reason and "paaf.py" in reason
    assert result.success is False
    assert repos.head() == old
    assert (repos.install / "paaf.py").read_text() == "VERSION = 'my edit'\n"
    assert procs.calls == []


def test_untracked_files_do_not_block(repos, procs):
    (repos.install / "output").mkdir()
    (repos.install / "output" / "run.data").write_text("x")
    repos.push("paaf.py", "VERSION = 2\n", "Version 2")
    updater.check_for_update()

    assert updater.can_update() == (True, "")
    assert updater.apply_update().success
    assert (repos.install / "output" / "run.data").exists()


def test_diverged_history_refuses_with_reclone_message(repos, procs):
    commit(repos.install, "local.txt", "mine", "Local experiment")
    old = repos.head()
    repos.push("paaf.py", "VERSION = 2\n", "Version 2")
    updater.check_for_update()

    ok, reason = updater.can_update()
    result = updater.apply_update()

    assert ok is False
    assert "git clone" in reason
    assert result.success is False and "git clone" in result.message
    assert repos.head() == old


def test_pip_failure_rolls_back(repos, procs):
    old = repos.head()
    repos.push("requirements.txt", "impossible-package==99\n", "Bad dep")
    updater.check_for_update()
    procs.pip_error = ["ERROR: No matching distribution for impossible-package"]

    result = updater.apply_update()

    assert result.success is False and result.rolled_back is True
    assert result.message.startswith("Update rolled back: pip failed:")
    assert "impossible-package" in result.message
    assert repos.head() == old
    assert not (repos.install / "requirements.txt").exists()
    assert procs.import_calls() == []
    assert updater.last_good_commit() is None
    assert git(repos.install, "status", "--porcelain",
               "--untracked-files=no") == ""


def test_import_failure_rolls_back(repos, procs):
    old = repos.head()
    repos.push("paaf.py", "VERSION = (\n", "Broken syntax")
    updater.check_for_update()
    procs.import_error = "SyntaxError: '(' was never closed"

    result = updater.apply_update()

    assert result.success is False and result.rolled_back is True
    assert result.message.startswith(
        "Update rolled back: new version failed to import:")
    assert "SyntaxError" in result.message
    assert repos.head() == old
    assert (repos.install / "paaf.py").read_text() == "VERSION = 1\n"


def test_failed_pull_changes_nothing(repos, procs):
    old = repos.head()
    repos.push("new.txt", "from origin", "Add new.txt")
    updater.check_for_update()
    # An untracked file the pull would overwrite makes git refuse.
    (repos.install / "new.txt").write_text("mine")

    result = updater.apply_update()

    assert result.success is False
    assert "git pull failed" in result.message
    assert repos.head() == old
    assert (repos.install / "new.txt").read_text() == "mine"


# --------------------------------------------------------------- failures
def test_no_git_folder_returns_error(repos, procs, monkeypatch):
    monkeypatch.setattr(updater, "repo_root", lambda: None)

    info = updater.check_for_update()
    ok, reason = updater.can_update()
    result = updater.apply_update()

    assert info.available is False and "git clone" in info.error
    assert ok is False and "git clone" in reason
    assert result.success is False
    assert updater.rollback("abc").success is False


def test_repo_root_without_git_dir(tmp_path, monkeypatch):
    fake = tmp_path / "paaf" / "updater.py"
    fake.parent.mkdir()
    fake.write_text("")
    monkeypatch.setattr(updater, "__file__", str(fake))
    assert updater.repo_root() is None
    (tmp_path / ".git").mkdir()
    assert updater.repo_root() == tmp_path.resolve()


def test_git_missing_from_path_returns_error(repos, procs, monkeypatch, tmp_path):
    empty = tmp_path / "empty_bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    info = updater.check_for_update()
    ok, reason = updater.can_update()
    result = updater.apply_update()

    assert info.available is False
    assert "not found" in info.error
    assert ok is False and "not found" in reason
    assert result.success is False


# ------------------------------------------------------------ skip / restore
def test_skipped_commit_is_not_reoffered(repos):
    repos.push("a.txt", "a", "Something")
    info = updater.check_for_update()
    assert updater.should_offer(info)

    updater.skip_commit(info.remote)

    assert updater.should_offer(updater.check_for_update()) is False
    repos.push("b.txt", "b", "Something newer")
    assert updater.should_offer(updater.check_for_update()) is True


def test_rollback_then_update_again(repos, procs):
    old = repos.head()
    new = repos.push("paaf.py", "VERSION = 2\n", "Version 2")
    updater.check_for_update()
    assert updater.apply_update().success

    back = updater.rollback(updater.last_good_commit())

    assert back.success, back.message
    assert repos.head() == old
    assert (repos.install / "paaf.py").read_text() == "VERSION = 1\n"
    # Detached HEAD after a restore must not strand the updater.
    info = updater.check_for_update()
    assert info.available
    assert updater.can_update() == (True, "")
    again = updater.apply_update()
    assert again.success, again.message
    assert repos.head() == new
    assert git(repos.install, "symbolic-ref", "--short", "HEAD") == "main"


def test_relaunch_execs_the_gui(monkeypatch):
    monkeypatch.setattr(updater, "repo_root", lambda: None)
    seen = []
    monkeypatch.setattr(updater.os, "execv", lambda exe, argv: seen.append((exe, argv)))
    updater.relaunch()
    assert seen == [(sys.executable,
                     [sys.executable, "-m", "paaf.cli", "gui"])]


def test_relaunch_failure_returns_message(monkeypatch):
    monkeypatch.setattr(updater, "repo_root", lambda: None)
    def boom(*_):
        raise OSError("no such file")
    monkeypatch.setattr(updater.os, "execv", boom)
    assert "could not restart" in updater.relaunch()
