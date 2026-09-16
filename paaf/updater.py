"""Self-updater for a ``git clone`` + ``pip install -e .`` install of PAAF.

The one rule: an update must never leave PAAF worse off than before the user
clicked. Every step either succeeds or puts the previous commit back and
returns a plain-language message. Nothing in here raises to the caller.

Pure Python with no Qt, so it can be tested on its own. The GUI keeps its
settings in ``QSettings("PAAF", "PAAF")``; it plugs that store in with
:func:`set_settings_backend`, so ``last_good_commit`` and ``skipped_commit``
live next to every other PAAF setting instead of in a second file.
"""
from __future__ import annotations

import datetime as _dt
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .logging_utils import get_logger

log = get_logger("paaf.updater")

BRANCH = "main"
REMOTE = "origin"
LOG_NAME = "paaf_update.log"

# If any of these change between the old and new commit, dependencies are
# reinstalled.
DEPENDENCY_FILES = ("requirements.txt", "setup.py", "pyproject.toml")

# What the new code must import before it counts as installed. paaf.cli and
# paaf.gui.app are included because relaunch() starts PAAF through them.
IMPORT_CHECK = "import paaf, paaf.pipeline, paaf.gui.main_window, paaf.cli, paaf.gui.app"

SETTING_LAST_GOOD = "updater/last_good_commit"
SETTING_SKIPPED = "updater/skipped_commit"

GIT_TIMEOUT = 120
PIP_TIMEOUT = 1800
IMPORT_TIMEOUT = 180


@dataclass
class UpdateInfo:
    available: bool = False
    local: str = ""
    remote: str = ""
    commits: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class UpdateResult:
    success: bool
    message: str
    old: str = ""
    new: str = ""
    rolled_back: bool = False


# ----------------------------------------------------------- settings store
_memory_settings: Dict[str, str] = {}
_settings_get: Callable[[str], Optional[str]] = _memory_settings.get
_settings_set: Callable[[str, Optional[str]], None] = (
    lambda k, v: _memory_settings.__setitem__(k, v))


def set_settings_backend(getter: Callable[[str], Optional[str]],
                         setter: Callable[[str, Optional[str]], None]) -> None:
    """Route setting reads/writes to the application's store (QSettings)."""
    global _settings_get, _settings_set
    _settings_get, _settings_set = getter, setter


def _get_setting(key: str) -> Optional[str]:
    try:
        value = _settings_get(key)
        return str(value) if value else None
    except Exception:
        log.debug("updater: reading setting %s failed", key, exc_info=True)
        return None


def _set_setting(key: str, value: Optional[str]) -> None:
    try:
        _settings_set(key, value or "")
    except Exception:
        log.debug("updater: writing setting %s failed", key, exc_info=True)


def last_good_commit() -> Optional[str]:
    return _get_setting(SETTING_LAST_GOOD)


def skipped_commit() -> Optional[str]:
    return _get_setting(SETTING_SKIPPED)


def skip_commit(remote_hash: str) -> None:
    _set_setting(SETTING_SKIPPED, remote_hash)
    _write_log(f"skipped version {remote_hash[:10]}")


def should_offer(info: UpdateInfo) -> bool:
    """True when ``info`` is an update the user has not chosen to skip."""
    return bool(info.available and not info.error
                and info.remote and info.remote != skipped_commit())


# ----------------------------------------------------------------- helpers
def repo_root() -> Optional[Path]:
    """The PAAF install folder if it is a git checkout, else ``None``."""
    try:
        root = Path(__file__).resolve().parent.parent
        return root if (root / ".git").exists() else None
    except Exception:
        return None


def _write_log(message: str) -> None:
    """Append a timestamped line to ``paaf_update.log`` in the repo root."""
    try:
        root = repo_root()
        if root is None:
            return
        stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(root / LOG_NAME, "a", encoding="utf-8") as fh:
            for line in str(message).splitlines() or [""]:
                fh.write(f"[{stamp}] {line}\n")
    except Exception:
        log.debug("updater: could not write %s", LOG_NAME, exc_info=True)


class _StepFailed(Exception):
    """A command could not run or exited non-zero."""


def _env() -> Dict[str, str]:
    env = dict(os.environ)
    # Never sit waiting for a password prompt nobody can see.
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault("GIT_SSH_COMMAND", "ssh -oBatchMode=yes")
    return env


def _run(args: Sequence[str], cwd: Path, timeout: float) -> str:
    """Run a command; return its combined output or raise :class:`_StepFailed`."""
    try:
        proc = subprocess.run(
            list(args), cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout, stdin=subprocess.DEVNULL,
            env=_env(),
        )
    except FileNotFoundError:
        raise _StepFailed(f"'{args[0]}' was not found. Is it installed and on PATH?")
    except subprocess.TimeoutExpired:
        raise _StepFailed(f"'{' '.join(args[:3])}' did not finish within "
                          f"{int(timeout)} s.")
    except OSError as exc:
        raise _StepFailed(f"could not run '{args[0]}': {exc}")
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        raise _StepFailed(out or f"'{' '.join(args[:3])}' exited with "
                                 f"code {proc.returncode}")
    return out


def _git(root: Path, *args: str, timeout: float = GIT_TIMEOUT) -> str:
    return _run(["git", *args], root, timeout)


def _no_repo_message() -> str:
    return ("This PAAF folder is not a git checkout, so it cannot update "
            "itself. Re-install with 'git clone' to get automatic updates.")


def _short(h: str) -> str:
    return (h or "")[:10]


# --------------------------------------------------------------- the checks
def check_for_update(timeout: float = 10) -> UpdateInfo:
    """Fetch ``origin/main`` and compare it with the running commit. Never raises."""
    info = UpdateInfo()
    try:
        root = repo_root()
        if root is None:
            info.error = _no_repo_message()
        else:
            _git(root, "fetch", REMOTE, BRANCH, timeout=timeout)
            info.local = _git(root, "rev-parse", "HEAD")
            info.remote = _git(root, "rev-parse", f"{REMOTE}/{BRANCH}")
            if info.local != info.remote:
                out = _git(root, "log", f"HEAD..{REMOTE}/{BRANCH}", "--oneline")
                info.commits = [ln for ln in out.splitlines() if ln.strip()]
                # A checkout with local commits differs from origin
                # but may have nothing new to pull.
                info.available = bool(info.commits)
    except _StepFailed as exc:
        info.error = f"Could not check for updates: {exc}"
    except Exception as exc:
        info.error = f"Could not check for updates: {exc}"
    if info.error:
        info.available = False
        log.info("update check failed: %s", info.error)
        _write_log(f"check: error: {info.error}")
    elif info.available:
        _write_log(f"check: update available {_short(info.local)} -> "
                   f"{_short(info.remote)} ({len(info.commits)} commits)")
    else:
        _write_log(f"check: up to date at {_short(info.local)}")
    return info


def can_update() -> Tuple[bool, str]:
    """Whether :func:`apply_update` may run, with the reason when it may not."""
    try:
        root = repo_root()
        if root is None:
            return False, _no_repo_message()
        # Tracked files only: output/ and other untracked files inside the
        # folder are left alone by pull and reset, so they do not block.
        dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
        if dirty.strip():
            files = [ln.strip().split(None, 1)[-1]
                     for ln in dirty.splitlines() if ln.strip()]
            shown = ", ".join(files[:5]) + (" …" if len(files) > 5 else "")
            return False, ("PAAF can't update because files in the PAAF folder "
                           f"have local changes ({shown}). Commit or stash "
                           "them ('git stash'), then try again.")
        branch = _current_branch(root)
        if branch not in (None, BRANCH):
            return False, (f"PAAF is on branch '{branch}', not '{BRANCH}'. "
                           f"Switch with 'git checkout {BRANCH}' to use the "
                           "updater.")
        try:
            _git(root, "rev-parse", "--verify", "--quiet", f"{REMOTE}/{BRANCH}")
        except _StepFailed:
            return False, ("PAAF hasn't fetched the latest version yet. "
                           "Use Help → Check for updates first.")
        try:
            _git(root, "merge-base", "--is-ancestor", "HEAD", f"{REMOTE}/{BRANCH}")
        except _StepFailed:
            return False, ("Your copy of PAAF has history that isn't on GitHub, "
                           "so it can't be updated safely in place. Make a "
                           "fresh copy with 'git clone "
                           "https://github.com/MuhammadUzairRiaz/PAAF.git' "
                           "(keep your old folder until the new one works).")
        return True, ""
    except _StepFailed as exc:
        return False, f"PAAF can't check whether it is safe to update: {exc}"
    except Exception as exc:
        return False, f"PAAF can't check whether it is safe to update: {exc}"


def _current_branch(root: Path) -> Optional[str]:
    """The checked-out branch name, or ``None`` for a detached HEAD."""
    try:
        return _git(root, "symbolic-ref", "--short", "-q", "HEAD") or None
    except _StepFailed:
        return None


# ------------------------------------------------------------ the update
def _pip_install(root: Path, log_fn: Callable[[str], None]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    try:
        _run(cmd, root, PIP_TIMEOUT)
    except _StepFailed as exc:
        if "externally-managed-environment" not in str(exc) \
                and "externally managed" not in str(exc):
            raise
        log_fn("pip refused (externally managed environment); retrying with "
               "--break-system-packages …")
        _run(cmd + ["--break-system-packages"], root, PIP_TIMEOUT)


def _import_check(root: Path) -> None:
    _run([sys.executable, "-c", IMPORT_CHECK], root, IMPORT_TIMEOUT)


def _tail(text: str, lines: int = 15) -> str:
    return "\n".join(str(text).strip().splitlines()[-lines:])


def apply_update(log_fn: Optional[Callable[[str], None]] = None) -> UpdateResult:
    """Fast-forward to ``origin/main``, reinstall if needed, verify, or roll back."""

    def say(msg: str) -> None:
        _write_log(msg)
        try:
            if log_fn is not None:
                log_fn(msg)
        except Exception:
            log.debug("updater: log_fn failed", exc_info=True)

    def finish(result: UpdateResult) -> UpdateResult:
        say(("update OK: " if result.success else "update FAILED: ")
            + result.message)
        return result

    old = ""
    try:
        root = repo_root()
        if root is None:
            return finish(UpdateResult(False, _no_repo_message()))
        # Re-checked here, not only in the GUI: the rollback below uses
        # 'git reset --hard', which would destroy uncommitted edits.
        ok, reason = can_update()
        if not ok:
            return finish(UpdateResult(False, reason))

        old = _git(root, "rev-parse", "HEAD")
        say(f"Updating PAAF from {_short(old)} …")

        # After "Restore previous version" HEAD is detached. Put the branch
        # on the running commit (the files do not change) so the pull
        # fast-forwards from exactly what is running now.
        if _current_branch(root) is None:
            try:
                _git(root, "checkout", "-B", BRANCH, old)
            except _StepFailed as exc:
                back = _restore(root, old, say)
                return finish(UpdateResult(
                    False, f"Update not applied: could not switch to "
                           f"'{BRANCH}': {_tail(exc)}" + back, old=old))

        say(f"git pull --ff-only {REMOTE} {BRANCH} …")
        try:
            _git(root, "pull", "--ff-only", REMOTE, BRANCH)
        except _StepFailed as exc:
            # A failed ff-only pull leaves the tree as it was; make sure.
            _restore(root, old, say)
            return finish(UpdateResult(
                False, f"Update not applied: git pull failed:\n{_tail(exc)}",
                old=old))

        new = _git(root, "rev-parse", "HEAD")
        if new == old:
            return finish(UpdateResult(True, "PAAF is already up to date.",
                                       old=old, new=new))

        changed = _git(root, "diff", "--name-only", old, new).splitlines()
        if any(name.strip() in DEPENDENCY_FILES for name in changed):
            say("Dependencies changed; running pip install -e . "
                "(this can take a few minutes) …")
            try:
                _pip_install(root, say)
            except _StepFailed as exc:
                back = _restore(root, old, say)
                return finish(UpdateResult(
                    False, f"Update rolled back: pip failed: {_tail(exc)}"
                           + back, old=old, new=new, rolled_back=True))

        say("Checking that the new version imports …")
        try:
            _import_check(root)
        except _StepFailed as exc:
            back = _restore(root, old, say)
            return finish(UpdateResult(
                False, "Update rolled back: new version failed to import: "
                       f"{_tail(exc)}" + back,
                old=old, new=new, rolled_back=True))

        _set_setting(SETTING_LAST_GOOD, old)
        return finish(UpdateResult(
            True, f"Updated to {_short(new)}.", old=old, new=new))
    except Exception as exc:
        # Unexpected: put the old commit back if we know it and moved away.
        try:
            root = repo_root()
            if root is not None and old and _git(root, "rev-parse", "HEAD") != old:
                _restore(root, old, say)
        except Exception:
            log.debug("updater: emergency restore failed", exc_info=True)
        return finish(UpdateResult(False, f"Update not applied: {exc}"))


def _restore(root: Path, old: str, say: Callable[[str], None]) -> str:
    """Hard-reset to ``old``. Returns extra text for the user when that fails."""
    try:
        if _git(root, "rev-parse", "HEAD") != old:
            say(f"Rolling back to {_short(old)} …")
            _git(root, "reset", "--hard", old)
        return ""
    except _StepFailed as exc:
        msg = (f"\n\nPAAF could not restore the previous version automatically "
               f"({_tail(exc, 3)}). Run 'git reset --hard {old}' in "
               f"{root} to restore it.")
        say(msg.strip())
        return msg


def rollback(to_hash: str) -> UpdateResult:
    """Check out ``to_hash`` (Help → Restore previous version)."""
    result: UpdateResult
    try:
        root = repo_root()
        if root is None:
            result = UpdateResult(False, _no_repo_message())
        elif not to_hash:
            result = UpdateResult(False, "There is no previous version to restore.")
        else:
            dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
            if dirty.strip():
                result = UpdateResult(
                    False, "PAAF can't restore the previous version because "
                           "files in the PAAF folder have local changes. "
                           "Commit or stash them first.")
            else:
                old = _git(root, "rev-parse", "HEAD")
                branch = _current_branch(root)
                _git(root, "checkout", to_hash)
                try:
                    _import_check(root)
                except _StepFailed as exc:
                    _git(root, "checkout", branch or old)
                    result = UpdateResult(
                        False, "Previous version was not restored because it "
                               f"failed to import: {_tail(exc)}", old=old)
                else:
                    result = UpdateResult(
                        True, f"Restored version {_short(to_hash)}.",
                        old=old, new=_git(root, "rev-parse", "HEAD"))
    except _StepFailed as exc:
        result = UpdateResult(False, f"Could not restore the previous version: {exc}")
    except Exception as exc:
        result = UpdateResult(False, f"Could not restore the previous version: {exc}")
    _write_log(("rollback OK: " if result.success else "rollback FAILED: ")
               + result.message)
    return result


def relaunch() -> str:
    """Replace this process with a fresh PAAF GUI.

    Only returns on failure, with a message saying why.
    """
    # There is no paaf/__main__.py, so 'python -m paaf gui' would not start;
    # paaf.cli is the module that owns the 'gui' subcommand.
    args = [sys.executable, "-m", "paaf.cli", "gui"]
    _write_log("relaunching: " + " ".join(args))
    try:
        sys.stdout.flush(); sys.stderr.flush()
    except Exception:
        pass
    try:
        os.execv(sys.executable, args)
    except Exception as exc:
        msg = f"PAAF could not restart itself ({exc}). Close and reopen PAAF."
        _write_log(msg)
        return msg
    return ""  # pragma: no cover - execv does not return
