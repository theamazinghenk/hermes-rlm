"""Tool bridge injected into the persistent RLM kernel's namespace.

Gives model-written Python direct access to Hermes tools *and* to
`delegate_task`, so subagents become an ordinary function call from inside
the kernel instead of a separate conversation turn (RLM invariant 2).

Transport: one line-delimited JSON request per call over a Unix domain
socket owned by the parent Hermes process. The parent dispatches through
the normal `handle_function_call` path, so approvals, policy and hooks all
still apply. A shared secret guards the socket.
"""

from __future__ import annotations

import json
import os
import socket
import threading

_SOCKET_PATH = os.environ.get("HERMES_RLM_RPC_SOCKET", "")
_TOKEN = os.environ.get("HERMES_RLM_RPC_TOKEN", "")
_DISABLED = set(filter(None, os.environ.get("HERMES_RLM_DISABLED", "").split(",")))
_lock = threading.Lock()


def _require(feature: str) -> None:
    if feature in _DISABLED:
        raise RlmBridgeError(
            f"{feature} are disabled by the operator "
            f"(HERMES_RLM_ENABLE_{feature.upper()}=0)")


class RlmBridgeError(RuntimeError):
    pass


def _call(tool: str, args: dict):
    if not _SOCKET_PATH:
        raise RlmBridgeError("RLM tool bridge is not configured for this kernel.")
    payload = json.dumps({"token": _TOKEN, "tool": tool, "args": args}) + "\n"
    with _lock:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(600)
        try:
            sock.connect(_SOCKET_PATH)
            sock.sendall(payload.encode())
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
        finally:
            sock.close()
    raw = buf.split(b"\n", 1)[0].decode()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(result, dict) and result.get("__rlm_error__"):
        raise RlmBridgeError(result["__rlm_error__"])
    return result


# --- Hermes tools -----------------------------------------------------------

def read_file(path: str, offset: int = 1, limit: int = 2000):
    """Read a text file with line numbers."""
    return _call("read_file", {"path": path, "offset": offset, "limit": limit})


def write_file(path: str, content: str):
    """Write a file, replacing existing content."""
    return _call("write_file", {"path": path, "content": content})


def search_files(pattern: str, target: str = "content", path: str = ".",
                 file_glob: str | None = None, limit: int = 50,
                 output_mode: str = "content"):
    """Search file contents or find files by name."""
    args = {"pattern": pattern, "target": target, "path": path,
            "limit": limit, "output_mode": output_mode}
    if file_glob:
        args["file_glob"] = file_glob
    return _call("search_files", args)


def patch(path: str, old_string: str, new_string: str, replace_all: bool = False):
    """Targeted find-and-replace in a file."""
    return _call("patch", {"mode": "replace", "path": path,
                           "old_string": old_string, "new_string": new_string,
                           "replace_all": replace_all})


def terminal(command: str, timeout: int | None = None, workdir: str | None = None):
    """Run a foreground shell command."""
    args: dict = {"command": command}
    if timeout is not None:
        args["timeout"] = timeout
    if workdir is not None:
        args["workdir"] = workdir
    return _call("terminal", args)


def web_search(query: str, limit: int = 5):
    """Search the web."""
    return _call("web_search", {"query": query, "limit": limit})


def web_extract(urls: list, char_limit: int | None = None):
    """Extract content from web pages."""
    args: dict = {"urls": urls}
    if char_limit is not None:
        args["char_limit"] = char_limit
    return _call("web_extract", args)


# --- Subagents as function calls (RLM invariant 2) ---------------------------

def rlm(goal: str, context: str = "", role: str = "leaf", fast: bool = False):
    """Spawn ONE subagent and return its result dict.

    This is the RLM delegation primitive: a subagent becomes an ordinary
    function call inside the kernel rather than a separate agent turn.
    Blocks until the child finishes. Keys: ok, goal, summary, session_id,
    exit_code, duration_seconds.

    `fast=True` answers tool-free tasks (summarise, classify, rephrase,
    reason about supplied text) with a direct model call: ~3s instead of
    ~17s. Anything mentioning files, commands, repos or the network is
    detected and routed to the full agent anyway, and any failure falls
    back automatically — so `fast=True` is safe but only pays off on
    genuinely tool-free work.
    """
    del role  # reserved; the CLI subagent inherits the parent's toolsets
    _require("subagents")
    return _call("__rlm_delegate__", {"goal": goal, "context": context, "fast": fast})


def rlm_many(tasks: list, fast: bool = False):
    """Spawn several subagents IN PARALLEL (up to 9 at once).

    Strongly preferred over a loop of `rlm()` calls: each subagent costs
    ~30s wall-clock, so `[rlm(g) for g in goals]` takes N*30s while
    `rlm_many([...])` takes roughly 30s regardless of N (up to the worker
    limit).

    `tasks` is a list of dicts with keys: goal, context (optional),
    fast (optional, overrides the batch default).
    Returns {"results": [...], "parallel_workers": int, "wall_seconds": float}.
    """
    _require("subagents")
    return _call("__rlm_delegate__", {"tasks": tasks, "fast": fast})


def rlm_spawn(goal, context: str = ""):
    """Start a subagent WITHOUT waiting and return a handle immediately.

    The child runs detached; its output goes to a file, never into this
    conversation. `goal` may also be a list of dicts ({goal, context}) to
    spawn several at once. Collect results later with
    `rlm_wait([handle["id"], ...])` or inspect `rlm_children()`.
    Children and their registry survive kernel — and even gateway — restarts.
    Prefer this over `rlm()` whenever you have other work to do meanwhile,
    or when the child's full output does not belong in your context.
    """
    _require("subagents")
    if isinstance(goal, list):
        return _call("__rlm_delegate__", {"mode": "spawn", "tasks": goal})
    return _call("__rlm_delegate__",
                 {"mode": "spawn", "goal": goal, "context": context})


def rlm_wait(ids: list, timeout: int = 600):
    """Wait for spawned children (by id) to finish; returns their metas.

    Each result carries status (done | error | finished-unverified), a
    summary tail and output_path for the full transcript. `timed_out` lists
    ids still running when the timeout hit — they keep running; call again.
    """
    _require("subagents")
    if isinstance(ids, str):
        ids = [ids]
    return _call("__rlm_delegate__",
                 {"mode": "wait", "ids": ids, "timeout": timeout})


def rlm_children(limit: int = 20):
    """List recent spawned subagents (newest first) from the disk registry."""
    _require("subagents")
    return _call("__rlm_delegate__", {"mode": "list", "limit": limit})


def coder(goal: str, repo: str, test_cmd: str = "", context: str = ""):
    """Delegate a repo coding task to Hermes itself, with fixed discipline.

    Runs in an ISOLATED git worktree of `repo`: a full Hermes agent (high
    reasoning, own tools/skills/rules) implements the goal; when `test_cmd`
    is given it gates the result and one bounded retry feeds the failure
    back. Returns {ok, attempts, worktree, diff, tests_ok, test_output,
    note} — the diff is YOURS to review; nothing is merged automatically.
    Prefer this over hand-rolled terminal calls for any substantive repo
    change: the worktree keeps the repo safe and the test gate keeps you
    honest.
    """
    _require("subagents")
    return _call("__rlm_delegate__",
                 {"mode": "coder", "goal": goal, "repo": repo,
                  "test_cmd": test_cmd, "context": context})


def harness_store(scope: str = "session"):
    """Direct access to the durable harness from inside the kernel.

    Returns a HarnessStore (create/update/delete/entries/apply_proposal/
    rollback). scope='global' reaches the store shared by every session.
    File-level mtime sync keeps kernel and host writers consistent.
    """
    _require("refine")
    from harness import HarnessStore
    if scope == "global":
        return HarnessStore("global")
    sid = os.environ.get("HERMES_RLM_SESSION_ID") or "default"
    return HarnessStore(sid)


__all__ = [
    "read_file", "write_file", "search_files", "patch", "terminal",
    "web_search", "web_extract", "rlm", "rlm_many", "rlm_spawn", "rlm_wait",
    "rlm_children", "coder", "harness_store", "RlmBridgeError",
]
