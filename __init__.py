"""hermes-rlm — persistent Python kernel + subagents-as-function-calls.

Implements the two useful halves of Prime Agent's RLM model on top of
Hermes' own tool stack:

  1. A persistent Python namespace per session. Load a dataset once, then
     ask many questions about it without pushing raw data through the
     conversation context.
  2. `rlm(...)` / `rlm_many(...)` inside that namespace, so delegation is a
     function call instead of a separate agent turn.

Deliberately NOT a replacement for `execute_code`: that stays the safe,
ephemeral, secret-scrubbed lane. This is the stateful lane you opt into.

Trust model: the kernel runs model-generated Python with the user's own
permissions. Durable control environment, not a security sandbox.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _load_local_module(name: str, filename: str):
    """Load a sibling module even when Hermes imports this plugin by file path."""
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_skill_loader = _load_local_module("hermes_rlm_skill_loader", "skill_loader.py")
_checkpoint = _load_local_module("hermes_rlm_checkpoint", "checkpoint.py")
_fast_path = _load_local_module("hermes_rlm_fast_path", "fast_path.py")
_harness = _load_local_module("hermes_rlm_harness", "harness.py")

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_KERNEL_SERVER = _HERE / "kernel_server.py"
_BRIDGE_MODULE = _HERE / "rlm_bridge.py"

# Tools reachable from inside the kernel. `delegate_task` is NOT here: it is
# an agent-loop tool that needs the live parent agent object, which plugin
# handlers never receive. `rlm()` spawns a real Hermes CLI subagent instead
# (see _run_subagent) rather than forking Hermes core.
ALLOWED_TOOLS = frozenset({
    "web_search", "web_extract", "read_file", "write_file",
    "search_files", "patch", "terminal",
})

# Sentinel the bridge sends for delegation; handled locally, never dispatched.
DELEGATE_SENTINEL = "__rlm_delegate__"

def _env(name: str, default: str) -> str:
    """HERMES_RLM_* knob: process env first, then ~/.hermes/.env (HERMES_HOME
    aware). The gateway does not bridge .env into os.environ, so operator
    knobs set there must still reach the plugin."""
    val = os.environ.get(name)
    if val is not None:
        return val
    try:
        return _fast_path._read_env_key(name) or default
    except Exception:  # noqa: BLE001
        return default


# Feature flags: "0" disables a capability COMPLETELY — the tool is not
# registered, kernel helpers are not injected, and the internal bridge
# refuses the call. This is what makes a minimal pilot (exec/vars/reset
# only) actually enforceable rather than a README suggestion.
FEATURE_CHECKPOINT = _env("HERMES_RLM_ENABLE_CHECKPOINT", "1") != "0"
FEATURE_REFINE = _env("HERMES_RLM_ENABLE_REFINE", "1") != "0"
FEATURE_SUBAGENTS = _env("HERMES_RLM_ENABLE_SUBAGENTS", "1") != "0"
FEATURE_PYTHON_SKILLS = _env("HERMES_RLM_ENABLE_PYTHON_SKILLS", "1") != "0"

DEFAULT_EXEC_TIMEOUT = 300
SUBAGENT_TIMEOUT = 900
IDLE_SHUTDOWN_SECONDS = int(_env("HERMES_RLM_IDLE_SECONDS", "3600"))

# Resource ceilings. Without these one session can allocate until the machine
# swaps, and a busy gateway can accumulate kernels until it runs out of file
# descriptors. Measured baseline: ~17 MB per idle kernel.
MAX_KERNELS = int(_env("HERMES_RLM_MAX_KERNELS", "12"))
MAX_KERNEL_RSS_MB = int(_env("HERMES_RLM_MAX_RSS_MB", "2048"))
# "warn" (default) surfaces a remedy; "stop" autosaves and kills the kernel —
# the right setting for shared/fleet machines.
RSS_POLICY = _env("HERMES_RLM_RSS_POLICY", "warn")
RSS_CHECK_EVERY = 10  # execs between resource checks — ps is not free

# Depth guard: a subagent inheriting this plugin must not recurse forever.
_DEPTH_ENV = "HERMES_RLM_DEPTH"
MAX_RLM_DEPTH = 2

# Matches a typical delegation.max_concurrent_children setting. A subagent
# costs ~30s wall-clock, so parallelism is what makes rlm_many worth using.
MAX_PARALLEL_SUBAGENTS = 9

_kernels: dict[str, "KernelHandle"] = {}
_kernels_lock = threading.Lock()


def _hermes_home() -> Path:
    """Hermes home with HERMES_HOME respected (fleet/profile isolation)."""
    env = os.environ.get("HERMES_HOME")
    return Path(env).expanduser() if env else Path.home() / ".hermes"


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _hermes_cli() -> str:
    """Absolute path to the hermes CLI that owns this install."""
    candidate = Path(sys.executable).parent / "hermes"
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("hermes")
    if found:
        return found
    raise RuntimeError("hermes CLI not found next to sys.executable or on PATH")


def _subagent_flags() -> list[str]:
    """Extra CLI flags for child agents: think less, cap the loop.

    Children inherit the parent's model (no pin), but run at low reasoning
    effort with a tighter turn cap — legwork, not synthesis. Profile-level
    agent settings do not reliably survive the config merge, so these are
    passed as explicit CLI flags. Override via HERMES_RLM_CHILD_REASONING /
    HERMES_RLM_CHILD_MAX_TURNS (set either to "0" to omit the flag).
    """
    flags: list[str] = []
    # AGENTS.md/SOUL.md injection is ON by default: fleet rules, branding and
    # secret policy must reach children unless the operator explicitly says
    # a bare leaf is acceptable (~30% faster startup).
    if _env("HERMES_RLM_CHILD_IGNORE_RULES", "0") == "1":
        flags.append("--ignore-rules")
    reasoning = _env("HERMES_RLM_CHILD_REASONING", "low")
    if reasoning and reasoning != "0":
        flags += ["--reasoning", reasoning]
    max_turns = _env("HERMES_RLM_CHILD_MAX_TURNS", "25")
    if max_turns and max_turns != "0":
        flags += ["--max-turns", max_turns]
    return flags


def _subagent_env(depth: int) -> dict:
    """Environment for a spawned subagent CLI process.

    Marks the child as a delegated context (blocks kanban mutations) and
    routes it to the minimal ``rlm-leaf`` profile when one exists: no MCP
    spawns, no memory/hindsight, no plugin load — leaf work needs none of it
    and every skipped subsystem is startup time. Opt out or rename via
    ``HERMES_RLM_LEAF_PROFILE`` (set to ``0`` to inherit the parent profile).
    """
    env = dict(os.environ)
    env[_DEPTH_ENV] = str(depth + 1)
    env["HERMES_DELEGATED_CHILD_CONTEXT"] = "1"
    leaf = _env("HERMES_RLM_LEAF_PROFILE", "rlm-leaf")
    if leaf and leaf != "0" and (_hermes_home() / "profiles" / leaf).is_dir():
        env["HERMES_PROFILE"] = leaf
    return env


def _run_subagent(goal: str, context: str = "", timeout: int = SUBAGENT_TIMEOUT,
                  fast: bool = False) -> dict:
    """Run ONE real Hermes subagent to completion and return its summary.

    Uses the CLI's non-interactive mode in a throwaway session so the child
    gets its own context, its own tools and its own transcript — the same
    isolation `delegate_task` gives — without needing the parent agent object
    that plugin handlers cannot see.

    With `fast=True` a tool-free task is answered by one direct model call
    instead (~3s vs ~17s). Falls back to the full agent on any doubt.
    """
    if not goal or not goal.strip():
        return {"ok": False, "error": "goal is required"}

    depth = int(os.environ.get(_DEPTH_ENV, "0"))
    if depth >= MAX_RLM_DEPTH:
        return {
            "ok": False,
            "error": f"RLM delegation depth limit reached (depth={depth}, "
                     f"max={MAX_RLM_DEPTH})",
        }

    # Fast path: skip ~14s of agent-loop startup when no tools are needed.
    # Any failure falls through to the full agent rather than degrading.
    if fast and _fast_path.eligible(goal, context):
        try:
            return _fast_path.run(goal, context)
        except _fast_path.FastPathUnavailable as exc:
            logger.debug("fast path unavailable, using full agent: %s", exc)

    prompt = f"{goal}\n\nContext:\n{context}" if context.strip() else goal
    env = _subagent_env(depth)

    started = time.monotonic()
    try:
        # --ignore-rules skips AGENTS.md/SOUL.md injection: a leaf subagent
        # gets its goal in the prompt and needs no persona. Measured ~30%
        # faster startup (12.4s -> 8.6s); the rest is agent-loop assembly.
        proc = subprocess.run(
            [_hermes_cli(), "chat", "-q", prompt, "-Q", *_subagent_flags()],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"subagent exceeded {timeout}s", "goal": goal}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"subagent launch failed: {exc}", "goal": goal}

    summary = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    # The CLI emits a trailing "session_id: ..." line — on stdout in some
    # modes, on stderr in others. Keep it out of the summary the model reads,
    # but hand it back so the child's transcript stays reachable.
    session_id = None

    def _split_session(text: str) -> tuple[str, str | None]:
        lines = text.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("session_id:"):
                sid = lines[i].split(":", 1)[1].strip()
                return "\n".join(lines[:i] + lines[i + 1:]).strip(), sid
        return text, None

    summary, session_id = _split_session(summary)
    if session_id is None:
        stderr, session_id = _split_session(stderr)

    return {
        "ok": proc.returncode == 0,
        "goal": goal,
        "summary": summary,
        "session_id": session_id,
        "exit_code": proc.returncode,
        "duration_seconds": round(time.monotonic() - started, 1),
        "stderr": stderr[-2000:] or None,
    }


# --- Handle-only (async) subagents ------------------------------------------
# rlm_spawn() returns a handle immediately; children run detached
# (start_new_session) with output to a file, and a registry JSON on disk
# survives kernel restarts and even a gateway restart. Mirrors prime-agent's
# child-registry invariant without needing a daemon.

SUBAGENT_STATE_DIR = _hermes_home() / "state" / "rlm" / "subagents"
SUBAGENT_KEEP = 200  # newest child dirs kept; older pruned on spawn
_TERMINAL_STATUSES = {"done", "error", "finished-unverified"}


def _write_child_meta(child_dir: Path, meta: dict) -> None:
    tmp = child_dir / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, default=str))
    tmp.replace(child_dir / "meta.json")


def _summary_from_output(out_path: Path) -> tuple[str, str | None]:
    """Summary text + child session id from a finished child's output file."""
    try:
        text = out_path.read_text(errors="replace")
    except OSError:
        return "", None
    session_id = None
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("session_id:"):
            session_id = lines[i].split(":", 1)[1].strip()
            lines = lines[:i] + lines[i + 1:]
            break
    return "\n".join(lines).strip()[-4000:], session_id


def _prune_children() -> None:
    try:
        dirs = sorted(SUBAGENT_STATE_DIR.iterdir(), key=lambda p: p.stat().st_mtime)
        for old in dirs[:-SUBAGENT_KEEP]:
            shutil.rmtree(old, ignore_errors=True)
    except OSError:
        pass


def _spawn_subagent(goal: str, context: str = "") -> dict:
    """Start a detached subagent and return a handle immediately."""
    if not goal or not goal.strip():
        return {"ok": False, "error": "goal is required"}
    depth = int(os.environ.get(_DEPTH_ENV, "0"))
    if depth >= MAX_RLM_DEPTH:
        return {"ok": False,
                "error": f"RLM delegation depth limit reached (depth={depth}, "
                         f"max={MAX_RLM_DEPTH})"}

    child_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    child_dir = SUBAGENT_STATE_DIR / child_id
    child_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    _prune_children()
    out_path = child_dir / "output.txt"

    prompt = f"{goal}\n\nContext:\n{context}" if context.strip() else goal
    try:
        out_fh = open(out_path, "w")
        proc = subprocess.Popen(
            [_hermes_cli(), "chat", "-q", prompt, "-Q", *_subagent_flags()],
            stdout=out_fh, stderr=subprocess.STDOUT, text=True,
            env=_subagent_env(depth), start_new_session=True,
        )
        out_fh.close()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"spawn failed: {exc}", "goal": goal}

    meta = {"id": child_id, "goal": goal[:400], "status": "running",
            "pid": proc.pid, "started_at": time.time(),
            "output_path": str(out_path)}
    _write_child_meta(child_dir, meta)

    def _reap() -> None:
        code = proc.wait()
        summary, child_session = _summary_from_output(out_path)
        meta.update(status="done" if code == 0 else "error", exit_code=code,
                    ended_at=time.time(), summary=summary,
                    session_id=child_session)
        _write_child_meta(child_dir, meta)

    threading.Thread(target=_reap, name=f"rlm-reap-{child_id}", daemon=True).start()
    return {"ok": True, "id": child_id, "status": "running", "pid": proc.pid,
            "goal": goal[:200], "output_path": str(out_path)}


def _child_meta(child_id: str) -> dict | None:
    child_dir = SUBAGENT_STATE_DIR / child_id
    try:
        meta = json.loads((child_dir / "meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("status") == "running":
        # Reaper thread may be gone (gateway restarted). If the pid is dead,
        # finalise from the output file — visibly unverified, never silent.
        pid = meta.get("pid")
        try:
            os.kill(int(pid), 0)
        except (OSError, TypeError, ValueError):
            summary, child_session = _summary_from_output(Path(meta["output_path"]))
            meta.update(status="finished-unverified", exit_code=None,
                        ended_at=time.time(), summary=summary,
                        session_id=child_session)
            _write_child_meta(child_dir, meta)
    return meta


def _wait_subagents(ids: list, timeout: int = 600) -> dict:
    """Block until the given children reach a terminal status (or timeout)."""
    ids = [str(i) for i in (ids or []) if i]
    if not ids:
        return {"ok": False, "error": "ids is required (from rlm_spawn handles)"}
    deadline = time.monotonic() + max(1, int(timeout))
    pending = set(ids)
    results: dict[str, dict] = {}
    while pending and time.monotonic() < deadline:
        for cid in list(pending):
            meta = _child_meta(cid)
            if meta is None:
                results[cid] = {"id": cid, "status": "unknown",
                                "error": "no such child in registry"}
                pending.discard(cid)
            elif meta.get("status") in _TERMINAL_STATUSES:
                results[cid] = meta
                pending.discard(cid)
        if pending:
            time.sleep(1.0)
    return {"ok": not pending,
            "results": [results.get(c) or {"id": c, "status": "running"} for c in ids],
            "timed_out": sorted(pending)}


def _list_subagents(limit: int = 20) -> dict:
    try:
        dirs = sorted(SUBAGENT_STATE_DIR.iterdir(),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return {"ok": True, "children": []}
    children = []
    for d in dirs[:max(1, int(limit))]:
        meta = _child_meta(d.name)
        if meta:
            children.append({k: meta.get(k) for k in
                             ("id", "status", "goal", "started_at", "ended_at",
                              "exit_code", "session_id")})
    return {"ok": True, "children": children}


def _handle_delegate(args: dict) -> dict:
    """Serve `rlm()` / `rlm_many()` / spawn / wait / list from the kernel."""
    mode = str(args.get("mode") or "")
    if mode == "spawn":
        tasks = args.get("tasks")
        if isinstance(tasks, list) and tasks:
            return {"children": [_spawn_subagent(t.get("goal", ""),
                                                 t.get("context", "") or "")
                                 for t in tasks]}
        return _spawn_subagent(args.get("goal", ""), args.get("context", "") or "")
    if mode == "wait":
        return _wait_subagents(args.get("ids") or [],
                               int(args.get("timeout") or 600))
    if mode == "list":
        return _list_subagents(int(args.get("limit") or 20))

    fast = bool(args.get("fast"))
    tasks = args.get("tasks")
    if isinstance(tasks, list) and tasks:
        # A subagent costs ~30s wall-clock, so a serial loop over N goals is
        # N*30s. Fan out instead — this is the difference between rlm_many
        # being useful and being a footgun.
        workers = min(len(tasks), MAX_PARALLEL_SUBAGENTS)
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_run_subagent, t.get("goal", ""),
                            t.get("context", "") or "", SUBAGENT_TIMEOUT,
                            bool(t.get("fast", fast)))
                for t in tasks
            ]
            results = [f.result() for f in futures]
        return {
            "results": results,
            "parallel_workers": workers,
            "wall_seconds": round(time.monotonic() - started, 1),
        }
    return _run_subagent(args.get("goal", ""), args.get("context", "") or "",
                         SUBAGENT_TIMEOUT, fast)


class KernelHandle:
    """One long-lived interpreter process plus its tool-bridge RPC server."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.started_at = time.time()
        self.last_used = time.time()
        self.exec_count = 0
        self._io_lock = threading.Lock()
        self._stop = threading.Event()

        self.rpc_dir = tempfile.mkdtemp(prefix="hermes_rlm_")
        os.chmod(self.rpc_dir, 0o700)
        self.socket_path = os.path.join(self.rpc_dir, "rpc.sock")
        self.token = secrets.token_urlsafe(32)

        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        self._server_sock.listen(8)
        self._rpc_thread = threading.Thread(
            target=self._serve_rpc, name=f"rlm-rpc-{session_id[:8]}", daemon=True
        )
        self._rpc_thread.start()

        env = dict(os.environ)
        env["HERMES_RLM_RPC_SOCKET"] = self.socket_path
        env["HERMES_RLM_RPC_TOKEN"] = self.token
        env["HERMES_RLM_SESSION_ID"] = session_id
        disabled = [n for n, on in (("subagents", FEATURE_SUBAGENTS),
                                     ("refine", FEATURE_REFINE),
                                     ("checkpoint", FEATURE_CHECKPOINT)) if not on]
        env["HERMES_RLM_DISABLED"] = ",".join(disabled)
        env["PYTHONUNBUFFERED"] = "1"
        # Plugin dir first, then every Python-backed skill package parent so
        # `import <skill>` works inside the kernel with no install step.
        path_parts = [str(_HERE)]
        if FEATURE_PYTHON_SKILLS:
            try:
                path_parts.extend(_skill_loader.sys_path_entries())
            except Exception:  # noqa: BLE001 - skills are optional, kernel is not
                pass
        if env.get("PYTHONPATH"):
            path_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(path_parts)

        self.proc = subprocess.Popen(
            [sys.executable, str(_KERNEL_SERVER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=True, bufsize=1,
        )
        # Make the bridge importable without an explicit import step.
        self.execute("from rlm_bridge import *  # noqa: F401,F403", timeout=30)

    # --- tool bridge ------------------------------------------------------

    def _serve_rpc(self) -> None:
        from model_tools import handle_function_call

        while not self._stop.is_set():
            try:
                self._server_sock.settimeout(0.5)
                conn, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                # shutdown() closes the listening socket out from under this
                # thread; that races with the settimeout/accept above and
                # surfaces as EBADF. Expected during teardown, not an error.
                break
            try:
                conn.settimeout(600)
                buf = b""
                while b"\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                if not buf:
                    continue
                request = json.loads(buf.split(b"\n", 1)[0].decode())

                if not secrets.compare_digest(
                    str(request.get("token") or "").encode(), self.token.encode()
                ):
                    conn.sendall(b'{"__rlm_error__": "unauthorized"}\n')
                    continue

                tool = request.get("tool", "")

                # Delegation is served locally: it spawns a real Hermes
                # subagent instead of going through handle_function_call,
                # which cannot reach the live parent agent from a plugin.
                if tool == DELEGATE_SENTINEL:
                    if not FEATURE_SUBAGENTS:
                        conn.sendall((json.dumps({"__rlm_error__":
                            "subagents are disabled by the operator "
                            "(HERMES_RLM_ENABLE_SUBAGENTS=0)"}) + "\n").encode())
                        continue
                    reply = _handle_delegate(request.get("args", {}))
                    conn.sendall((json.dumps(reply, default=str) + "\n").encode())
                    continue

                if tool not in ALLOWED_TOOLS:
                    msg = f"tool {tool!r} not available in the RLM kernel"
                    conn.sendall((json.dumps({"__rlm_error__": msg}) + "\n").encode())
                    continue

                real_out, real_err = sys.stdout, sys.stderr
                devnull = open(os.devnull, "w", encoding="utf-8")
                try:
                    sys.stdout = sys.stderr = devnull
                    result = handle_function_call(
                        tool, request.get("args", {}), task_id=self.session_id
                    )
                finally:
                    sys.stdout, sys.stderr = real_out, real_err
                    devnull.close()

                if not isinstance(result, str):
                    result = json.dumps(result, default=str)
                # The wire protocol is newline-delimited, so a tool result
                # containing raw newlines must be JSON-escaped, not flattened
                # with str.replace — flattening corrupts multi-line output.
                if "\n" in result:
                    result = json.dumps(result)
                conn.sendall((result + "\n").encode())
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller
                try:
                    conn.sendall(
                        (json.dumps({"__rlm_error__": str(exc)}) + "\n").encode()
                    )
                except OSError:
                    pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    # --- kernel I/O -------------------------------------------------------

    def alive(self) -> bool:
        return self.proc.poll() is None

    def execute(self, code: str, op: str = "exec", timeout: int = DEFAULT_EXEC_TIMEOUT,
                extra: dict | None = None) -> dict:
        if not self.alive():
            return {"ok": False, "error": "kernel process is not running"}
        req = {"id": uuid.uuid4().hex[:8], "op": op, "code": code}
        if extra:
            req.update(extra)
        with self._io_lock:
            try:
                self.proc.stdin.write(json.dumps(req) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError) as exc:
                return {"ok": False, "error": f"kernel stdin closed: {exc}"}

            result: dict = {}
            done = threading.Event()

            def _read() -> None:
                nonlocal result
                try:
                    line = self.proc.stdout.readline()
                    result = json.loads(line) if line else {
                        "ok": False, "error": "kernel closed stdout"
                    }
                except Exception as exc:  # noqa: BLE001
                    result = {"ok": False, "error": f"kernel read failed: {exc}"}
                finally:
                    done.set()

            reader = threading.Thread(target=_read, daemon=True)
            reader.start()
            if not done.wait(timeout):
                # Last chance to save the namespace before the kernel dies:
                # a timeout is exactly when expensive state is most costly to
                # lose. Best-effort and short — the kernel is unresponsive, so
                # this usually fails, but when the hang is in a worker thread
                # rather than the main loop it succeeds and saves the load.
                salvaged = None
                try:
                    if not FEATURE_CHECKPOINT:
                        raise RuntimeError("checkpoint feature disabled")
                    self.proc.stdin.write(json.dumps({
                        "id": "salvage", "op": "checkpoint", "code": "",
                        "session": self.session_id, "name": "autosave",
                    }) + "\n")
                    self.proc.stdin.flush()
                    if done.wait(3):
                        salvaged = "autosave"
                except Exception:  # noqa: BLE001
                    pass

                self.shutdown()
                message = (
                    f"execution exceeded {timeout}s; kernel was killed and "
                    "all in-memory state was lost"
                )
                if salvaged:
                    message = (
                        f"execution exceeded {timeout}s; kernel was killed. "
                        "State was salvaged to checkpoint 'autosave' — recover "
                        "it with rlm_checkpoint(action='restore', name='autosave')"
                    )
                return {"ok": False, "error": message}

        self.last_used = time.time()
        if op == "exec":
            self.exec_count += 1
        return result

    def shutdown(self) -> None:
        # Signal first, then close: the serve loop checks _stop before
        # touching the socket, so it exits cleanly instead of racing into
        # EBADF on a descriptor closed under it.
        self._stop.set()
        thread = getattr(self, "_rpc_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        try:
            self._server_sock.close()
        except OSError:
            pass
        if self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self.proc.kill()
        shutil.rmtree(self.rpc_dir, ignore_errors=True)


def _rss_mb(pid: int) -> float:
    """Resident set size in MB, or -1 when it cannot be read."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out) / 1024 if out else -1.0
    except Exception:  # noqa: BLE001
        return -1.0


def _autosave(handle: "KernelHandle") -> bool:
    """Best-effort namespace autosave before a managed shutdown.

    Resource management (LRU eviction, idle reap) must not cost state: the
    next rlm_exec for this session restores the autosave automatically.
    """
    if not handle.alive() or not FEATURE_CHECKPOINT:
        return False
    try:
        result = handle.execute("", op="checkpoint", timeout=20,
                                extra={"session": handle.session_id,
                                       "name": "autosave"})
        return bool(result.get("ok"))
    except Exception:  # noqa: BLE001 - shutdown proceeds regardless
        return False


def _reap_idle() -> None:
    cutoff = time.time() - IDLE_SHUTDOWN_SECONDS
    for key, handle in list(_kernels.items()):
        if handle.last_used < cutoff or not handle.alive():
            if handle.last_used < cutoff:
                _autosave(handle)
            handle.shutdown()
            _kernels.pop(key, None)


def _evict_oldest(exclude: str) -> str | None:
    """Shut down the least-recently-used kernel to make room. Returns its id."""
    candidates = [(handle.last_used, key) for key, handle in _kernels.items()
                  if key != exclude]
    if not candidates:
        return None
    _, victim = min(candidates)
    handle = _kernels.pop(victim, None)
    if handle is not None:
        _autosave(handle)
        handle.shutdown()
    return victim


def _get_kernel(session_id: str, create: bool = True) -> KernelHandle | None:
    with _kernels_lock:
        _reap_idle()
        handle = _kernels.get(session_id)
        if handle is not None and handle.alive():
            return handle
        if handle is not None:
            handle.shutdown()
            _kernels.pop(session_id, None)
        if not create:
            return None
        # Bound the population: evict least-recently-used rather than refuse,
        # so a busy gateway degrades by losing cold state instead of failing
        # the call the user is waiting on.
        while len(_kernels) >= MAX_KERNELS:
            if _evict_oldest(exclude=session_id) is None:
                break
        handle = KernelHandle(session_id)
        # Close the autosave loop: if this session was evicted or reaped
        # earlier, bring its state straight back so the model never notices
        # beyond a one-line note on the next reply.
        try:
            autosave_file = _checkpoint._session_dir(session_id) / "autosave.pkl"
            if FEATURE_CHECKPOINT and autosave_file.exists():
                result = handle.execute("", op="restore", timeout=30,
                                        extra={"session": session_id,
                                               "name": "autosave"})
                if result.get("ok"):
                    handle.autosave_note = (
                        "state auto-restored from checkpoint 'autosave' "
                        "(kernel was evicted or idle-reaped earlier)"
                    )
        except Exception:  # noqa: BLE001 - a fresh kernel is still usable
            pass
        _kernels[session_id] = handle
        return handle


# --- Tool handlers ----------------------------------------------------------

EXEC_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "Python to run in the persistent kernel. Variables, "
                           "imports and loaded data persist across calls. If the "
                           "last line is an expression its repr is returned.",
        },
        "timeout": {
            "type": "integer",
            "description": f"Seconds before the kernel is killed (default {DEFAULT_EXEC_TIMEOUT}). "
                           "A timeout destroys all in-memory state.",
        },
    },
    "required": ["code"],
}

VARS_SCHEMA = {"type": "object", "properties": {}}
RESET_SCHEMA = {"type": "object", "properties": {}}
SKILLS_SCHEMA = {"type": "object", "properties": {}}


CHECKPOINT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["save", "restore", "list", "delete"],
            "description": "save: persist the namespace. restore: merge a saved "
                           "namespace back in. list: show checkpoints. delete: remove one.",
        },
        "name": {
            "type": "string",
            "description": "Checkpoint name (default 'latest'). Use distinct names "
                           "to keep several restore points.",
        },
        "allow_cross_session": {
            "type": "boolean",
            "description": "restore only: allow borrowing the newest same-named "
                           "checkpoint from ANOTHER session when this session has "
                           "none. Default false — cross-session data mixing must "
                           "be an explicit choice.",
        },
    },
    "required": ["action"],
}


def _exec_handler(args: dict, **kwargs) -> str:
    session_id = str(kwargs.get("task_id") or "default")
    code = args.get("code", "")
    if not code.strip():
        return _json({"ok": False, "error": "code is required"})
    timeout = int(args.get("timeout") or DEFAULT_EXEC_TIMEOUT)

    handle = _get_kernel(session_id)
    result = handle.execute(code, timeout=timeout)

    # Resource guard. A kernel that has ballooned will push the machine into
    # swap and take the whole gateway down with it, so warn the model with a
    # concrete remedy while the session is still usable.
    warning = None
    if handle.alive() and handle.exec_count % RSS_CHECK_EVERY == 0:
        rss = _rss_mb(handle.proc.pid)
        if rss > MAX_KERNEL_RSS_MB:
            if RSS_POLICY == "stop":
                _autosave(handle)
                handle.shutdown()
                _kernels.pop(session_id, None)
                return _json({"ok": False, "error": (
                    f"kernel exceeded the hard RSS limit ({rss:.0f} MB > "
                    f"{MAX_KERNEL_RSS_MB} MB) and was stopped. State was "
                    "autosaved — restore with rlm_checkpoint(action='restore', "
                    "name='autosave') and load less at once.")})
            warning = (
                f"kernel is using {rss:.0f} MB (limit {MAX_KERNEL_RSS_MB} MB). "
                "Free what you no longer need (`del big_var; import gc; gc.collect()`), "
                "or checkpoint the parts worth keeping and call rlm_reset."
            )

    # Keep the reply lean: every byte here is conversation context on EVERY
    # call, and a chatty envelope wipes out the saving the kernel exists to
    # deliver. Emit only fields that carry information for this call.
    payload: dict[str, Any] = {"ok": result.get("ok", False)}
    for key in ("value", "stdout", "stderr", "error"):
        val = result.get(key)
        if val:
            payload[key] = val
    duration = result.get("duration") or 0
    if duration >= 1:
        payload["duration"] = duration
    if warning:
        payload["warning"] = warning
    note = getattr(handle, "autosave_note", None)
    if note:
        payload["note"] = note
        handle.autosave_note = None
    if not result.get("ok"):
        # Diagnostics matter only when something went wrong.
        payload["kernel"] = {
            "session": session_id,
            "exec_count": handle.exec_count,
            "uptime_seconds": round(time.time() - handle.started_at, 1),
        }
    return json.dumps(payload, ensure_ascii=False)


def _vars_handler(args: dict, **kwargs) -> str:
    del args
    session_id = str(kwargs.get("task_id") or "default")
    handle = _get_kernel(session_id, create=False)
    if handle is None:
        return _json({"ok": True, "kernel": "not started", "variables": []})
    result = handle.execute("", op="vars")
    try:
        variables = json.loads(result.get("value") or "[]")
    except json.JSONDecodeError:
        variables = []
    return _json({
        "ok": result.get("ok", False),
        "exec_count": handle.exec_count,
        "uptime_seconds": round(time.time() - handle.started_at, 1),
        "variables": variables,
    })


def _reset_handler(args: dict, **kwargs) -> str:
    del args
    session_id = str(kwargs.get("task_id") or "default")
    with _kernels_lock:
        handle = _kernels.pop(session_id, None)
    if handle is None:
        return _json({"ok": True, "message": "no kernel was running"})
    handle.shutdown()
    return _json({"ok": True, "message": "kernel stopped; all state discarded"})


def _skills_handler(args: dict, **kwargs) -> str:
    del args, kwargs
    try:
        return _skill_loader.catalog_json()
    except Exception as exc:  # noqa: BLE001
        return _json({"ok": False, "error": f"skill discovery failed: {exc}"})


def _checkpoint_handler(args: dict, **kwargs) -> str:
    session_id = str(kwargs.get("task_id") or "default")
    action = str(args.get("action") or "").strip()
    name = str(args.get("name") or "latest")

    if action == "list":
        return _json({"ok": True, "checkpoints": _checkpoint.listing()})
    if action == "delete":
        return _json(_checkpoint.delete(session_id, name))
    if action not in {"save", "restore"}:
        return _json({"ok": False, "error": "action must be save/restore/list/delete"})

    handle = _get_kernel(session_id)
    op = "checkpoint" if action == "save" else "restore"
    extra = {"session": session_id, "name": name}
    if action == "restore" and args.get("allow_cross_session"):
        extra["allow_cross_session"] = True
    result = handle.execute("", op=op, extra=extra)
    if not result.get("ok"):
        return _json({"ok": False, "error": result.get("error") or "checkpoint failed"})
    try:
        return _json(json.loads(result.get("value") or "{}"))
    except json.JSONDecodeError:
        return _json({"ok": True, "raw": result.get("value")})


# --- Continual-harness-lite --------------------------------------------------

REFINE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["refine", "add", "list", "delete", "rollback"],
            "description": "refine: propose evidence-backed harness edits from "
                           "the recent transcript via one model call. add: store "
                           "one entry directly. list: entries + recent "
                           "refinements. delete: remove an entry by id. "
                           "rollback: invert a refinement by id.",
        },
        "scope": {"type": "string", "enum": ["session", "global"],
                  "description": "Where entries live (default session). Global "
                                 "entries reach every session."},
        "kind": {"type": "string", "enum": list(_harness.KINDS),
                 "description": "For add: prompt (standing guidance), memory "
                                "(fact/decision/failure), skill (how-to), "
                                "subagent (reusable delegation recipe)."},
        "title": {"type": "string"},
        "content": {"type": "string"},
        "id": {"type": "string", "description": "Entry id (delete) or "
                                                "refinement id (rollback)."},
        "instruction": {"type": "string",
                        "description": "For refine: what to distil, e.g. 'capture "
                                       "the debugging lesson about X'."},
    },
    "required": ["action"],
}

_REFINE_PROMPT = """You maintain an agent's durable harness: small, reusable
entries (kinds: prompt, memory, skill, subagent) that persist across sessions.
Study the transcript and propose at most 4 edits that would genuinely help
future sessions — durable lessons, decisions, failure modes, working recipes.
No one-off noise, no transient tool output, no secrets or credentials.

Current entries (id | kind | title):
{entries}

{instruction}

Answer with ONLY a JSON object, no prose:
{{"edits": [{{"op": "create", "kind": "memory", "title": "...", "content": "..."}},
            {{"op": "update", "id": "...", "content": "..."}},
            {{"op": "delete", "id": "..."}}]}}
Return {{"edits": []}} if nothing is worth keeping."""


def _recent_transcript(session_id: str, max_chars: int = 40_000) -> str:
    """Recent turns for this session from the Hermes session store, oldest first."""
    import sqlite3
    db = _hermes_home() / "state.db"
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        rows = con.execute(
            "SELECT role, COALESCE(content,''), COALESCE(tool_name,'') "
            "FROM messages WHERE session_id=? AND active=1 "
            "ORDER BY id DESC LIMIT 150", (session_id,)).fetchall()
        con.close()
    except Exception:  # noqa: BLE001 - refine simply has less evidence
        return ""
    lines, total = [], 0
    for role, content, tool in rows:
        line = f"{role}{'/' + tool if tool else ''}: {content[:1500]}"
        total += len(line)
        lines.append(line)
        if total > max_chars:
            break
    return "\n".join(reversed(lines))


def _refine_handler(args: dict, **kwargs) -> str:
    session_id = str(kwargs.get("task_id") or "default")
    action = args.get("action") or "refine"
    scope = args.get("scope") or "session"
    store = _harness.HarnessStore("global" if scope == "global" else session_id)
    try:
        if action == "add":
            entry = store.create(args.get("kind") or "memory",
                                 args.get("title") or "", args.get("content") or "")
            return _json({"ok": True, "created": entry["id"]})
        if action == "list":
            entries = [{k: e.get(k) for k in ("id", "kind", "title", "version")}
                       for e in store.entries()]
            return _json({"ok": True, "scope": scope, "entries": entries,
                          "refinements": store.refinements()})
        if action == "delete":
            store.delete(args.get("id") or "")
            return _json({"ok": True, "deleted": args.get("id")})
        if action == "rollback":
            return _json({"ok": True, **store.rollback(args.get("id") or "")})

        # action == "refine": evidence -> one model call -> validated apply.
        transcript = _recent_transcript(session_id)
        if not transcript:
            return _json({"ok": False,
                          "error": "no transcript found for this session; use "
                                   "action='add' to store an entry directly"})
        entries_txt = "\n".join(
            f"{e['id']} | {e['kind']} | {e['title']}" for e in store.entries()
        ) or "(none yet)"
        instruction = args.get("instruction") or ""
        goal = _REFINE_PROMPT.format(entries=entries_txt,
                                     instruction=instruction and
                                     f"Focus: {instruction}")
        result = _fast_path.run(goal, context=f"Transcript:\n{transcript}",
                                max_tokens=1500)
        raw = result.get("summary", "")
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return _json({"ok": False, "error": "model returned no JSON proposal",
                          "raw": raw[:400]})
        proposal = json.loads(raw[start:end + 1])
        if not proposal.get("edits"):
            return _json({"ok": True, "applied": [],
                          "note": "model found nothing durable to keep"})
        outcome = store.apply_proposal(
            proposal, evidence_note=f"refine session={session_id} "
                                    f"instruction={instruction[:120]}")
        return _json({"ok": True, **outcome,
                      "hint": "undo with action='rollback', "
                              f"id='{outcome['refinement_id']}'"})
    except (ValueError, KeyError) as exc:
        return _json({"ok": False, "error": str(exc)})
    except _fast_path.FastPathUnavailable as exc:
        return _json({"ok": False, "error": f"model call unavailable: {exc}"})


def _harness_pre_llm_call(user_message: str, conversation_history: list,
                          is_first_turn: bool, model: str, platform: str,
                          session_id: str = "", **kwargs: Any) -> dict | None:
    """Inject the compact harness overview; never break the LLM call."""
    if _env("HERMES_RLM_HARNESS_INJECT", "1") == "0":
        return None
    try:
        text = _harness.overview(session_id or None)
        return {"context": text} if text else None
    except Exception:  # noqa: BLE001
        logger.warning("harness overview injection failed", exc_info=True)
        return None


def register(ctx) -> None:
    ctx.register_tool(
        name="rlm_exec",
        toolset="rlm",
        schema=EXEC_SCHEMA,
        handler=_exec_handler,
        description=(
            "PREFERRED for data analysis, multi-step investigation and any task "
            "where you will ask more than one question about the same data. Runs "
            "Python in a PERSISTENT kernel scoped to this session: variables, "
            "imports and loaded datasets survive between calls and across context "
            "compaction, so you load once and then query many times without the "
            "raw data ever entering the conversation. Measured ~48% less context "
            "than re-running a stateless script per question. Hermes tools "
            "(read_file, write_file, search_files, patch, terminal, web_search, "
            "web_extract) are preloaded, plus rlm(goal, context) and "
            "rlm_many([...]) to run subagents as function calls, and any "
            "Python-backed skill via a plain import (see rlm_skills). "
            "DISCIPLINE: always assign results to named variables (df = ..., "
            "hits = ...) and print only the small summary you need — everything "
            "you print becomes permanent conversation context, while variables "
            "stay queryable for free. Oversized output is tail-truncated and "
            "spilled to a file whose path appears in the reply. "
            "Use execute_code ONLY for a genuine one-shot script that needs no "
            "memory between runs; if you might ask a follow-up, start here."
        ),
    )
    ctx.register_tool(
        name="rlm_vars",
        toolset="rlm",
        schema=VARS_SCHEMA,
        handler=_vars_handler,
        description=(
            "List what currently lives in the persistent RLM kernel: variable "
            "names, types, sizes/shapes and short previews. Cheap — never dumps "
            "full objects. Use it to recall state after compaction."
        ),
    )
    ctx.register_tool(
        name="rlm_reset",
        toolset="rlm",
        schema=RESET_SCHEMA,
        handler=_reset_handler,
        description=(
            "Stop this session's persistent kernel and discard all in-memory "
            "state. Use when memory grows too large or state became inconsistent."
        ),
    )
    # Optional capabilities: a disabled feature is absent, not hidden —
    # no tool schema, no kernel helper, and the bridge refuses the call.
    if FEATURE_CHECKPOINT:
        ctx.register_tool(
            name="rlm_checkpoint",
            toolset="rlm",
            schema=CHECKPOINT_SCHEMA,
            handler=_checkpoint_handler,
            description=(
                "Persist or restore the RLM kernel namespace so expensive state "
                "survives a crash, a timeout kill or a machine restart. "
                "action='save' after a costly load; action='restore' to bring it "
                "back into a fresh kernel; 'list' and 'delete' to manage them. "
                "Checkpoints are deliberately partial — objects that cannot be "
                "pickled (open connections, sockets, lambdas) are reported by name "
                "rather than silently dropped, so you know what to rebuild."
            ),
        )
    if FEATURE_PYTHON_SKILLS:
        ctx.register_tool(
            name="rlm_skills",
            toolset="rlm",
            schema=SKILLS_SCHEMA,
            handler=_skills_handler,
            description=(
                "List Python-backed skills importable inside the RLM kernel. These are "
                "skills that ship a python/ package next to SKILL.md, so a proven "
                "workflow is a typed callable instead of code the model rewrites each "
                "run. Returns names, import names and one-line summaries only."
            ),
        )
    if FEATURE_REFINE:
        ctx.register_tool(
            name="rlm_refine",
            toolset="rlm",
            schema=REFINE_SCHEMA,
            handler=_refine_handler,
            description=(
                "Durable harness: keep lessons, decisions, failure modes and "
                "delegation recipes across sessions. action='refine' distils the "
                "recent transcript into at most 4 evidence-backed entries via one "
                "model call (validated, snapshotted, reversible with rollback); "
                "action='add' stores one entry directly. Entries are injected "
                "compactly into future system prompts. Use after finishing "
                "non-trivial work whose lessons should outlive this session."
            ),
        )
        if hasattr(ctx, "register_hook"):
            ctx.register_hook("pre_llm_call", _harness_pre_llm_call)
