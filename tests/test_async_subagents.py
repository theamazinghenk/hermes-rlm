"""Async (handle-only) subagents + autosave/auto-restore.

Covers: spawn returns a handle immediately; the registry finalises a child
whose reaper thread is gone (simulated gateway restart); wait collects
results; eviction autosaves and a fresh kernel auto-restores.

Run: ~/.hermes/hermes-agent/venv/bin/python tests/test_async_subagents.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import types
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

# The plugin imports model_tools lazily inside the RPC thread only, so a
# plain import of the plugin module works without the Hermes runtime.
import importlib.util

spec = importlib.util.spec_from_file_location("hermes_rlm_plugin", HERE / "__init__.py")
plugin = importlib.util.module_from_spec(spec)
sys.modules["hermes_rlm_plugin"] = plugin
spec.loader.exec_module(plugin)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok" if cond else "FAIL"
    print(f"  {status}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# --- 1. spawn returns a handle immediately (use /bin/echo as fake CLI) ------

real_cli = plugin._hermes_cli
plugin._hermes_cli = lambda: "/bin/echo"  # child exits instantly, code 0
t0 = time.monotonic()
handle = plugin._spawn_subagent("test goal alfa", "wat context")
spawn_latency = time.monotonic() - t0
check("spawn returns ok handle", handle.get("ok") is True, str(handle))
check("spawn is immediate (<2s)", spawn_latency < 2.0, f"{spawn_latency:.2f}s")
check("handle has id and output_path", bool(handle.get("id")) and bool(handle.get("output_path")))

# --- 2. wait collects the finished child ------------------------------------

res = plugin._wait_subagents([handle["id"]], timeout=15)
check("wait resolves child", res.get("ok") is True, str(res))
meta = res["results"][0]
check("child status terminal", meta.get("status") in plugin._TERMINAL_STATUSES, str(meta.get("status")))
check("child exit_code recorded", meta.get("exit_code") == 0, str(meta.get("exit_code")))

# --- 3. registry finalises orphaned child (reaper thread gone) --------------

child_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
child_dir = plugin.SUBAGENT_STATE_DIR / child_id
child_dir.mkdir(parents=True, exist_ok=True)
out = child_dir / "output.txt"
out.write_text("resultaat regel\nsession_id: 20260101_000000_abc\n")
plugin._write_child_meta(child_dir, {
    "id": child_id, "goal": "orphan", "status": "running",
    "pid": 99999999, "started_at": time.time(), "output_path": str(out),
})
meta = plugin._child_meta(child_id)
check("orphan finalised as finished-unverified", meta.get("status") == "finished-unverified", str(meta.get("status")))
check("orphan summary parsed", meta.get("summary") == "resultaat regel", repr(meta.get("summary")))
check("orphan child session extracted", meta.get("session_id") == "20260101_000000_abc")

# --- 4. wait on unknown id is explicit, not silent --------------------------

res = plugin._wait_subagents(["does-not-exist"], timeout=2)
check("unknown id reported", res["results"][0].get("status") == "unknown")

# --- 5. depth limit refuses spawn -------------------------------------------

os.environ[plugin._DEPTH_ENV] = str(plugin.MAX_RLM_DEPTH)
res = plugin._spawn_subagent("te diep")
check("depth limit refused", res.get("ok") is False and "depth" in res.get("error", ""))
os.environ.pop(plugin._DEPTH_ENV, None)
plugin._hermes_cli = real_cli

# --- 6. autosave on eviction + auto-restore on next kernel ------------------

sess = f"test-autorestore-{uuid.uuid4().hex[:6]}"
k = plugin._get_kernel(sess)
r = k.execute("survivor = 12345")
check("kernel exec ok", r.get("ok") is True, str(r))
saved = plugin._autosave(k)
check("autosave succeeds", saved is True)
k.shutdown()
plugin._kernels.pop(sess, None)

k2 = plugin._get_kernel(sess)
check("auto-restore note set", bool(getattr(k2, "autosave_note", None)),
      str(getattr(k2, "autosave_note", None)))
r = k2.execute("survivor")
check("restored variable intact", r.get("value") == "12345", str(r.get("value")))
k2.shutdown()
plugin._kernels.pop(sess, None)
try:
    plugin._checkpoint.delete(sess, "autosave")
except Exception:
    pass

# --- 7. cleanup spawn test dirs --------------------------------------------

import shutil as _shutil
for d in (plugin.SUBAGENT_STATE_DIR / handle["id"], child_dir):
    _shutil.rmtree(d, ignore_errors=True)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} — {FAILURES}")
    sys.exit(1)
print("ok")
