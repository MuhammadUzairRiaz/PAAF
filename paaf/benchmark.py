"""Timing benchmarks written with every export.

Every export — the main pipeline, the Blend / Layering / CG tools, the
amorphous cell builder, reaction export — runs inside :func:`benchmark`. When
it ends (finished, failed or cancelled) three things are written:

* ``benchmark.json`` in the export folder — the full, machine-readable record;
* ``benchmark.txt`` next to it — the same numbers as a readable table;
* one row appended to ``~/.paaf/benchmark_history.csv`` (override with
  ``PAAF_BENCH_HISTORY``), so runs can be compared across projects, tools,
  PAAF versions and machines.

What is measured, and how, so the numbers mean the same thing every time:

* wall time with :func:`time.perf_counter` (monotonic, sub-microsecond);
* time spent in external programs (dl_field, moltemplate.sh, packmol, LAMMPS,
  gmx) — every child process is timed where it is started, so "in PAAF" is
  the wall time *minus* the time spent waiting on those programs;
* CPU seconds of the PAAF process and, on macOS/Linux, of the child programs
  it ran (``os.times``), which gives the average number of cores kept busy;
* peak resident memory, sampled every 0.2 s (PAAF alone and PAAF + child
  programs) when ``psutil`` is installed, else the process's lifetime peak;
* the size of the job — atoms counted from the file that was actually
  written, and a ``workload_id`` hashed from the inputs that set the cost, so
  two rows with the same id did the same work and can be compared directly;
* the machine and software versions, so a slower row can be told apart from
  a slower computer.

A benchmark must never break an export: every failure inside this module is
logged and swallowed.
"""
from __future__ import annotations

import contextlib
import csv
import datetime as _dt
import functools
import hashlib
import inspect
import json
import logging
import os
import platform
import re
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

log = logging.getLogger("paaf.benchmark")

BENCH_JSON = "benchmark.json"
BENCH_TXT = "benchmark.txt"
HISTORY_ENV = "PAAF_BENCH_HISTORY"
SCHEMA_VERSION = 1

HISTORY_COLUMNS = [
    "started", "tool", "status", "workload_id", "atoms", "chains",
    "wall_s", "paaf_s", "external_s", "cpu_paaf_s", "cpu_children_s",
    "cpu_cores_busy", "peak_rss_mb", "peak_rss_total_mb", "atoms_per_s",
    "s_per_1000_atoms", "files_written", "mb_written", "paaf_version",
    "python", "os", "cpu_model", "logical_cores", "folder",
]

_local = threading.local()
_active_lock = threading.Lock()
_active: List["Recorder"] = []


# ------------------------------------------------------------------ helpers
def history_path() -> Path:
    override = os.environ.get(HISTORY_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".paaf" / "benchmark_history.csv"


def _children_cpu() -> Optional[float]:
    """CPU seconds of reaped child processes (None on Windows)."""
    if os.name == "nt":
        return None
    t = os.times()
    return t.children_user + t.children_system


def count_atoms(path) -> Optional[int]:
    """Atoms in a written structure file (.data, .gro, .xyz, .pdb)."""
    try:
        p = Path(path)
        if not p.is_file():
            return None
        suf = p.suffix.lower()
        with p.open("r", errors="ignore") as fh:
            if suf == ".gro":
                fh.readline()
                return int(fh.readline().split()[0])
            if suf == ".xyz":
                return int(fh.readline().split()[0])
            if suf == ".pdb":
                return sum(1 for ln in fh if ln.startswith(("ATOM", "HETATM")))
            for i, ln in enumerate(fh):          # LAMMPS data header
                m = re.match(r"^\s*(\d+)\s+atoms\s*$", ln)
                if m:
                    return int(m.group(1))
                if i > 60:
                    break
    except (OSError, ValueError, IndexError):
        return None
    return None


@functools.lru_cache(maxsize=1)
def environment() -> Dict[str, object]:
    """Machine and software fingerprint, stored with every benchmark."""
    import subprocess
    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True,
                                 timeout=5).stdout.strip() or cpu
        elif sys.platform.startswith("linux"):
            for ln in Path("/proc/cpuinfo").read_text().splitlines():
                if ln.lower().startswith("model name"):
                    cpu = ln.split(":", 1)[1].strip()
                    break
    except Exception as exc:
        log.debug("CPU model not read: %s", exc)
    env: Dict[str, object] = {
        "paaf_version": _paaf_version(),
        "python": platform.python_version(),
        "os": platform.platform(terse=True),
        "cpu_model": cpu,
        "logical_cores": os.cpu_count(),
        "physical_cores": None,
        "ram_gb": None,
    }
    try:
        import psutil
        env["physical_cores"] = psutil.cpu_count(logical=False)
        env["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception as exc:
        log.debug("psutil unavailable for core/RAM counts: %s", exc)
    libs = {}
    for mod in ("numpy", "scipy", "rdkit", "networkx"):
        m = sys.modules.get(mod)
        if m is None:
            try:
                m = __import__(mod)
            except Exception:
                continue
        libs[mod] = getattr(m, "__version__", "?")
    env["libraries"] = libs
    return env


def _paaf_version() -> str:
    try:
        from .__version__ import __version__
        return __version__
    except Exception:
        return "?"


class _MemorySampler:
    """Peak RSS of this process (and of it plus its children) over a run."""

    def __init__(self, interval: float = 0.2):
        self.peak_self = 0
        self.peak_total = 0
        self.method = "none"
        self._stop = threading.Event()
        self._thread = None
        try:
            import psutil
            self._proc = psutil.Process()
            self.method = "psutil (sampled every %.1f s)" % interval
            self._interval = interval
            self._sample()
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="paaf-bench-mem")
            self._thread.start()
        except Exception:
            self._proc = None

    def _sample(self) -> None:
        try:
            own = self._proc.memory_info().rss
            total = own
            for ch in self._proc.children(recursive=True):
                try:
                    total += ch.memory_info().rss
                except Exception:       # the child exited between calls
                    continue
            self.peak_self = max(self.peak_self, own)
            self.peak_total = max(self.peak_total, total)
        except Exception as exc:
            log.debug("memory sample failed: %s", exc)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2)
            self._sample()
            return
        try:                      # no psutil: the process's lifetime peak
            import resource
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.peak_self = r if sys.platform == "darwin" else r * 1024
            self.peak_total = 0
            self.method = "getrusage (lifetime peak of the process)"
        except Exception as exc:
            log.debug("peak memory not available: %s", exc)


class _Phase:
    __slots__ = ("name", "depth", "t0", "t1", "external_s")

    def __init__(self, name: str, depth: int, t0: float):
        self.name, self.depth, self.t0 = name, depth, t0
        self.t1: Optional[float] = None
        self.external_s = 0.0


# ----------------------------------------------------------------- recorder
class Recorder:
    """Collects one export's timings. Use through :func:`benchmark`."""

    def __init__(self, tool: str, out_dir, workload: Optional[dict],
                 emit: Optional[Callable[[str], None]]):
        self.tool = tool
        self.folder: Optional[Path] = Path(out_dir) if out_dir else None
        self.workload: Dict[str, object] = dict(workload or {})
        self.metrics: Dict[str, object] = {}
        self.emit = emit
        self.started = _dt.datetime.now().astimezone()
        self._epoch0 = time.time()
        self.t0 = time.perf_counter()
        self.cpu0 = time.process_time()
        self.children0 = _children_cpu()
        self.phases: List[_Phase] = []
        self._open: List[Optional[_Phase]] = [None]   # open phase per depth
        self._prefix: List[str] = []
        self._auto: List[bool] = []
        self.external: Dict[str, Dict[str, float]] = {}
        self._size_files: List[Path] = []
        self._mem = _MemorySampler()
        self.report: Optional[dict] = None
        self.json_file: Optional[Path] = None
        self.txt_file: Optional[Path] = None

    # -- phases -----------------------------------------------------------
    def phase(self, name: str) -> None:
        """End the running phase at this level and start ``name``."""
        now = time.perf_counter()
        depth = len(self._prefix)
        cur = self._open[depth]
        if cur is not None:
            cur.t1 = now
        full = " / ".join(self._prefix + [name])
        ph = _Phase(full, depth, now)
        self.phases.append(ph)
        self._open[depth] = ph

    def end_phase(self) -> None:
        depth = len(self._prefix)
        cur = self._open[depth]
        if cur is not None:
            cur.t1 = time.perf_counter()
            self._open[depth] = None

    def _enter_nested(self, tool: str, out_dir, workload) -> None:
        if self.folder is None and out_dir:
            self.folder = Path(out_dir)
        for k, v in (workload or {}).items():
            self.workload.setdefault(f"{tool}.{k}", v)
        # Called outside any phase: the nested run is a phase of its own.
        opened = self._open[len(self._prefix)] is None
        if opened:
            self.phase(tool)
        self._auto.append(opened)
        self._prefix.append(tool)
        self._open.append(None)

    def _exit_nested(self) -> None:
        self.end_phase()
        self._prefix.pop()
        self._open.pop()
        if self._auto.pop():
            self.end_phase()

    # -- measurements -----------------------------------------------------
    def metric(self, **kw) -> None:
        """Record size metrics, e.g. ``atoms=``, ``chains=``."""
        for k, v in kw.items():
            if v is not None:
                self.metrics[k] = v

    def size_from(self, path) -> None:
        """Count atoms from this output file if nothing else sets them."""
        if path:
            self._size_files.append(Path(path))

    def _add_external(self, program: str, dt: float) -> None:
        e = self.external.setdefault(program, {"seconds": 0.0, "calls": 0})
        e["seconds"] += dt
        e["calls"] += 1
        for ph in self._open:
            if ph is not None:
                ph.external_s += dt

    # -- report -----------------------------------------------------------
    def _output_stats(self):
        if self.folder is None or not self.folder.is_dir():
            return 0, 0
        n = size = 0
        cutoff = self._epoch0 - 1.0
        for p in self.folder.rglob("*"):
            try:
                if p.is_file() and p.name not in (BENCH_JSON, BENCH_TXT):
                    st = p.stat()
                    if st.st_mtime >= cutoff:
                        n += 1
                        size += st.st_size
            except OSError:
                continue
        return n, size

    def finish(self, status: str, error: Optional[str]) -> dict:
        t_end = time.perf_counter()
        while len(self._prefix):
            self._exit_nested()
        self.end_phase()
        self._mem.stop()
        wall = t_end - self.t0
        cpu_paaf = time.process_time() - self.cpu0
        ch1 = _children_cpu()
        cpu_children = (ch1 - self.children0
                        if ch1 is not None and self.children0 is not None
                        else None)
        ext_total = sum(e["seconds"] for e in self.external.values())

        atoms = self.metrics.get("atoms")
        if atoms is None:
            for f in self._size_files:
                atoms = count_atoms(f)
                if atoms:
                    self.metrics["atoms"] = atoms
                    break
        atoms = self.metrics.get("atoms")
        n_files, n_bytes = self._output_stats()

        cpu_all = cpu_paaf + (cpu_children or 0.0)
        wl = {"tool": self.tool, **self.workload}
        wid = hashlib.sha1(json.dumps(wl, sort_keys=True, default=str)
                           .encode()).hexdigest()[:10]
        phases = [{
            "name": ph.name, "depth": ph.depth,
            "start_s": round(ph.t0 - self.t0, 4),
            "wall_s": round((ph.t1 or t_end) - ph.t0, 4),
            "external_s": round(ph.external_s, 4),
        } for ph in self.phases]
        top = sum(p["wall_s"] for p in phases if p["depth"] == 0)
        if phases and wall - top > max(0.01, 0.01 * wall):
            phases.append({"name": "(outside named phases)", "depth": 0,
                           "start_s": None, "wall_s": round(wall - top, 4),
                           "external_s": None})
        self.report = {
            "schema": SCHEMA_VERSION,
            "tool": self.tool,
            "status": status,
            "error": error,
            "started": self.started.isoformat(timespec="seconds"),
            "folder": str(self.folder) if self.folder else None,
            "workload_id": wid,
            "workload": self.workload,
            "size": dict(self.metrics),
            "totals": {
                "wall_s": round(wall, 4),
                "paaf_s": round(wall - ext_total, 4),
                "external_s": round(ext_total, 4),
                "cpu_paaf_s": round(cpu_paaf, 4),
                "cpu_children_s": (round(cpu_children, 4)
                                   if cpu_children is not None else None),
                "cpu_cores_busy": round(cpu_all / wall, 2) if wall > 0 else None,
                "peak_rss_mb": round(self._mem.peak_self / 1024 ** 2, 1)
                if self._mem.peak_self else None,
                "peak_rss_total_mb": round(self._mem.peak_total / 1024 ** 2, 1)
                if self._mem.peak_total else None,
                "memory_method": self._mem.method,
                "atoms_per_s": round(atoms / wall, 2)
                if atoms and wall > 0 else None,
                "s_per_1000_atoms": round(1e3 * wall / atoms, 4)
                if atoms else None,
                "files_written": n_files,
                "mb_written": round(n_bytes / 1024 ** 2, 3),
            },
            "external_programs": {
                k: {"seconds": round(v["seconds"], 4), "calls": int(v["calls"])}
                for k, v in sorted(self.external.items(),
                                   key=lambda kv: -kv[1]["seconds"])},
            "phases": phases,
            "environment": environment(),
            "timer": "time.perf_counter (wall), time.process_time + os.times "
                     "(CPU)",
        }
        self._write()
        self._announce()
        return self.report

    def _write(self) -> None:
        r = self.report
        if self.folder is not None:
            try:
                self.folder.mkdir(parents=True, exist_ok=True)
                self.json_file = self.folder / BENCH_JSON
                self.json_file.write_text(json.dumps(r, indent=2, default=str))
                self.txt_file = self.folder / BENCH_TXT
                self.txt_file.write_text(format_report(r))
            except OSError as exc:
                log.warning("Benchmark files not written in %s: %s",
                            self.folder, exc)
        try:
            hp = history_path()
            hp.parent.mkdir(parents=True, exist_ok=True)
            new = not hp.exists() or hp.stat().st_size == 0
            t, env = r["totals"], r["environment"]
            row = {**{k: t.get(k) for k in HISTORY_COLUMNS if k in t},
                   "started": r["started"], "tool": r["tool"],
                   "status": r["status"], "workload_id": r["workload_id"],
                   "atoms": r["size"].get("atoms"),
                   "chains": r["size"].get("chains"),
                   "paaf_version": env.get("paaf_version"),
                   "python": env.get("python"), "os": env.get("os"),
                   "cpu_model": env.get("cpu_model"),
                   "logical_cores": env.get("logical_cores"),
                   "folder": r["folder"]}
            with hp.open("a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=HISTORY_COLUMNS,
                                   extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(row)
        except OSError as exc:
            log.warning("Benchmark history not updated: %s", exc)

    def _announce(self) -> None:
        t = self.report["totals"]
        lines = [f"[benchmark] {self.tool}: {t['wall_s']:.2f} s wall "
                 f"({t['paaf_s']:.2f} s PAAF, {t['external_s']:.2f} s "
                 f"external programs), {self.report['status']}"]
        if t["atoms_per_s"]:
            lines.append(f"[benchmark] {self.report['size']['atoms']:,} atoms "
                         f"-> {t['atoms_per_s']:,.1f} atoms/s")
        if self.txt_file:
            lines.append(f"[benchmark] report: {self.txt_file}")
        # A cancelled run still gets its report and history row, but nothing
        # more is printed at someone who just pressed Cancel.
        show = self.emit if self.report["status"] != "cancelled" else None
        for ln in lines:
            log.info(ln)
            if show:
                try:
                    show(ln)
                except Exception as exc:
                    # e.g. a cancel check raising in the callback
                    log.debug("benchmark summary not shown: %s", exc)


def format_report(r: dict) -> str:
    """Render a benchmark record as a plain-text table."""
    t, env = r["totals"], r["environment"]
    wall = t["wall_s"] or 0.0

    def pct(x):
        return f"{100 * x / wall:5.1f} %" if wall and x is not None else "      "

    title = f"PAAF benchmark — {r['tool']}"
    out = [title, "=" * len(title),
           f"Started        {r['started']}",
           f"Status         {r['status']}" + (f" ({r['error']})" if r["error"] else ""),
           f"Folder         {r['folder'] or '-'}",
           f"Workload id    {r['workload_id']}   (same id = same inputs, "
           f"directly comparable)"]
    if r["size"]:
        out.append("Size           " + ", ".join(
            f"{k} {v:,}" if isinstance(v, int) else f"{k} {v}"
            for k, v in r["size"].items()))
    cores = f"{env.get('logical_cores')} logical"
    if env.get("physical_cores"):
        cores += f" / {env['physical_cores']} physical"
    ram = f" · {env['ram_gb']} GB RAM" if env.get("ram_gb") else ""
    out.append(f"Machine        {env.get('cpu_model')} · {cores} cores{ram}")
    libs = " · ".join(f"{k} {v}" for k, v in env.get("libraries", {}).items())
    out.append(f"Software       PAAF {env.get('paaf_version')} · Python "
               f"{env.get('python')} · {env.get('os')}")
    if libs:
        out.append(f"               {libs}")
    out += ["", "Totals"]
    ext = ", ".join(f"{k} {v['seconds']:.2f} s ×{v['calls']}"
                    for k, v in r["external_programs"].items())
    out.append(f"  wall time              {wall:12.3f} s")
    out.append(f"  in PAAF (Python)       {t['paaf_s']:12.3f} s  {pct(t['paaf_s'])}")
    out.append(f"  external programs      {t['external_s']:12.3f} s  "
               f"{pct(t['external_s'])}" + (f"   ({ext})" if ext else ""))
    out.append(f"  CPU, PAAF process      {t['cpu_paaf_s']:12.3f} s")
    if t["cpu_children_s"] is not None:
        out.append(f"  CPU, child programs    {t['cpu_children_s']:12.3f} s")
    if t["cpu_cores_busy"] is not None:
        out.append(f"  cores kept busy        {t['cpu_cores_busy']:12.2f}")
    if t["peak_rss_mb"]:
        mem = f"  peak memory            {t['peak_rss_mb']:10.1f} MB PAAF"
        if t["peak_rss_total_mb"]:
            mem += f", {t['peak_rss_total_mb']:.1f} MB incl. child programs"
        out.append(mem + f"   [{t['memory_method']}]")
    if t["atoms_per_s"]:
        out.append(f"  throughput             {t['atoms_per_s']:12,.1f} atoms/s"
                   f"   ({t['s_per_1000_atoms']:,.3f} s per 1000 atoms)")
    out.append(f"  output                 {t['files_written']:12d} files, "
               f"{t['mb_written']:.2f} MB")
    if r["phases"]:
        out += ["", f"Phases{'':46s}   wall s       %   external s"]
        for p in r["phases"]:
            # Indentation shows nesting; the JSON keeps the full path.
            name = ("  " * p["depth"] + p["name"].split(" / ")[-1])[:50]
            ext_s = f"{p['external_s']:10.3f}" if p["external_s"] is not None else ""
            out.append(f"  {name:<50s} {p['wall_s']:10.3f} {pct(p['wall_s'])} {ext_s}")
    if r["workload"]:
        out += ["", "Workload (inputs that set the cost)"]
        for k, v in r["workload"].items():
            out.append(f"  {k:<28s} {v}")
    out += ["", "Wall time: time.perf_counter. 'in PAAF' = wall time minus time "
            "spent waiting on external programs.",
            "Every run is also appended to " + str(history_path()) + "."]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------- public API
def current() -> Optional[Recorder]:
    return getattr(_local, "rec", None)


def _is_cancel(exc: BaseException) -> bool:
    return isinstance(exc, KeyboardInterrupt) or "Cancel" in type(exc).__name__


@contextlib.contextmanager
def benchmark(tool: str, out_dir=None, *, workload: Optional[dict] = None,
              emit: Optional[Callable[[str], None]] = None) -> Iterator[Recorder]:
    """Time one export. Nested calls become phases of the outer benchmark.

    ``out_dir`` is the export folder the report goes into (None: only the
    history row and the log line). ``workload`` holds the inputs that set
    the cost — they are hashed into ``workload_id``.
    """
    rec = current()
    if rec is not None:
        rec._enter_nested(tool, out_dir, workload)
        try:
            yield rec
        finally:
            rec._exit_nested()
        return
    rec = Recorder(tool, out_dir, workload, emit)
    _local.rec = rec
    with _active_lock:
        _active.append(rec)
    status, error = "completed", None
    try:
        yield rec
    except BaseException as exc:
        status = "cancelled" if _is_cancel(exc) else "failed"
        error = f"{type(exc).__name__}: {exc}"[:300]
        raise
    finally:
        _local.rec = None
        with _active_lock:
            if rec in _active:
                _active.remove(rec)
        try:
            rec.finish(status, error)
        except Exception as exc:                      # pragma: no cover
            log.warning("Benchmark for %s not recorded: %s", tool, exc)


def phase(name: str) -> None:
    """Start a named phase in the running benchmark (no-op outside one)."""
    rec = current()
    if rec is not None:
        rec.phase(name)


def end_phase() -> None:
    """Close the running phase without starting another."""
    rec = current()
    if rec is not None:
        rec.end_phase()


def metric(**kw) -> None:
    rec = current()
    if rec is not None:
        rec.metric(**kw)


def size_from(path) -> None:
    rec = current()
    if rec is not None:
        rec.size_from(path)


@contextlib.contextmanager
def external(program) -> Iterator[None]:
    """Time a child program (``program`` may be a command list or a path)."""
    if isinstance(program, (list, tuple)):
        program = program[0] if program else "?"
    name = Path(str(program)).stem or str(program)
    rec = current()
    if rec is None:
        # Started from a helper thread: credit the one running benchmark.
        with _active_lock:
            rec = _active[0] if len(_active) == 1 else None
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if rec is not None:
            try:
                rec._add_external(name, time.perf_counter() - t0)
            except Exception as exc:                  # pragma: no cover
                log.debug("external time for %s not recorded: %s", name, exc)


def benchmarked(tool: str, *, folder: Optional[Callable] = None,
                workload: Optional[Callable] = None,
                after: Optional[Callable] = None):
    """Decorator form of :func:`benchmark` for export entry points.

    ``folder(args)`` and ``workload(args)`` receive the bound arguments as a
    dict; ``after(rec, args, result)`` can record metrics or the folder once
    the result is known. ``progress=`` of the wrapped call receives the
    one-line summary.
    """
    def deco(fn):
        sig = inspect.signature(fn)

        def _safe(f, *a):
            if f is None:
                return None
            try:
                return f(*a)
            except Exception as exc:                  # pragma: no cover
                log.debug("benchmark hook for %s failed: %s", tool, exc)
                return None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                ba = sig.bind_partial(*args, **kwargs)
                ba.apply_defaults()
                a = dict(ba.arguments)
            except TypeError:
                return fn(*args, **kwargs)
            with benchmark(tool, _safe(folder, a),
                           workload=_safe(workload, a) or {},
                           emit=a.get("progress")) as rec:
                result = fn(*args, **kwargs)
                _safe(after, rec, a, result)
                return result
        return wrapper
    return deco
