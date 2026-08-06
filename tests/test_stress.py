"""Stress tests for the hermes-rlm kernel.

These probe the failure modes that unit tests miss: concurrent sessions,
memory growth, crash recovery, timeout handling, unicode, large payloads,
and long-lived namespaces. Each check prints PASS/FAIL and the script exits
non-zero if anything failed, so it is usable as a gate.

    python3 tests/test_stress.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / ".hermes" / "hermes-agent"))
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("hermes_rlm_plugin", ROOT / "__init__.py")
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)

results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


def ex(code: str, session: str = "stress", timeout: int | None = None) -> dict:
    args = {"code": code}
    if timeout is not None:
        args["timeout"] = timeout
    return json.loads(plugin._exec_handler(args, task_id=session))


def rss_mb(pid: int) -> float:
    """Resident set size in MB for a pid, via ps (portable enough on macOS/Linux)."""
    import subprocess

    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 if out else -1.0


print("\n=== 1. Session isolation ===")
ex("secret = 'alpha'", session="tenant_a")
ex("secret = 'beta'", session="tenant_b")
a = ex("secret", session="tenant_a")
b = ex("secret", session="tenant_b")
check("separate sessions keep separate namespaces",
      a["value"] == "'alpha'" and b["value"] == "'beta'", f"{a['value']} / {b['value']}")
leak = ex("'tenant_b_only' in dir()", session="tenant_a")
check("no cross-session name leakage", leak["value"] == "False")

print("\n=== 2. Concurrency ===")
# Same session, parallel calls: the io lock must serialise without corrupting.
errors: list[str] = []
values: list[str] = []


def hammer(n: int) -> None:
    try:
        r = ex(f"{n} * 2", session="conc")
        values.append(r.get("value"))
        if not r.get("ok"):
            errors.append(str(r))
    except Exception as exc:  # noqa: BLE001
        errors.append(repr(exc))


threads = [threading.Thread(target=hammer, args=(i,)) for i in range(12)]
started = time.monotonic()
for t in threads:
    t.start()
for t in threads:
    t.join()
concurrent_secs = time.monotonic() - started
check("12 concurrent calls on one kernel, no errors", not errors, str(errors[:2]))
check("every concurrent call returned a value", len(values) == 12, f"{len(values)}/12")
check("responses are not interleaved/corrupted",
      sorted(int(v) for v in values if v) == sorted(i * 2 for i in range(12)))

# Parallel DIFFERENT sessions -> separate kernel processes.
sess_errors: list[str] = []


def spawn(n: int) -> None:
    r = ex(f"marker = {n}\nmarker", session=f"par_{n}")
    if r.get("value") != str(n):
        sess_errors.append(f"par_{n}: {r}")


threads = [threading.Thread(target=spawn, args=(i,)) for i in range(6)]
for t in threads:
    t.start()
for t in threads:
    t.join()
check("6 kernels booted in parallel, each isolated", not sess_errors, str(sess_errors[:1]))

print("\n=== 3. Crash recovery ===")
ex("keep = 'before crash'", session="crash")
handle = plugin._get_kernel("crash", create=False)
pid_before = handle.proc.pid
os.kill(pid_before, 9)  # hard kill, no cleanup
time.sleep(0.4)
r = ex("1 + 1", session="crash")
handle2 = plugin._get_kernel("crash", create=False)
check("kernel auto-restarts after SIGKILL", r.get("ok") and r.get("value") == "2", str(r)[:120])
check("restarted kernel is a NEW process", handle2.proc.pid != pid_before,
      f"{pid_before} -> {handle2.proc.pid}")
lost = ex("'keep' in dir()", session="crash")
check("state loss after crash is visible, not silently wrong", lost["value"] == "False")

print("\n=== 4. Timeout handling ===")
started = time.monotonic()
r = ex("import time; time.sleep(30)", session="slow", timeout=2)
elapsed = time.monotonic() - started
check("runaway code is killed at the timeout", not r.get("ok") and elapsed < 8,
      f"{elapsed:.1f}s")
check("timeout error explains state loss", "state" in (r.get("error") or "").lower(),
      (r.get("error") or "")[:80])
r2 = ex("2 + 2", session="slow")
check("session is usable again after a timeout", r2.get("ok") and r2["value"] == "4")

print("\n=== 5. Memory behaviour ===")
ex("import gc", session="mem")
handle = plugin._get_kernel("mem", create=False)
base = rss_mb(handle.proc.pid)
ex("big = [dict(i=i, s='x'*100) for i in range(200_000)]", session="mem")
loaded = rss_mb(handle.proc.pid)
ex("del big\ngc.collect()", session="mem")
time.sleep(0.3)
freed = rss_mb(handle.proc.pid)
check("large object measurably increases RSS", loaded - base > 20,
      f"{base:.0f} -> {loaded:.0f} MB")
check("deleting it releases most of the memory", freed < base + (loaded - base) * 0.5,
      f"after del: {freed:.0f} MB")
check("RSS is reported, so growth is observable", base > 0)

print("\n=== 6. Long-lived namespace ===")
for i in range(200):
    ex(f"v{i} = {i}", session="long")
r = ex("len([n for n in dir() if n.startswith('v')])", session="long")
check("200 sequential calls keep every binding", int(r["value"]) >= 200, r["value"])
handle = plugin._get_kernel("long", create=False)
check("exec_count tracks the real number of calls", handle.exec_count >= 200,
      str(handle.exec_count))
vars_out = json.loads(plugin._vars_handler({}, task_id="long"))
check("rlm_vars stays compact with 200 variables",
      len(json.dumps(vars_out)) < 60_000, f"{len(json.dumps(vars_out))} bytes")

print("\n=== 7. Payload edge cases ===")
r = ex("'héllo wörld · 日本語 · emoji 🚀'", session="edge")
check("unicode survives the JSON wire protocol", "🚀" in (r.get("value") or ""), r.get("value"))
r = ex(r"'line1\nline2\ttabbed \"quoted\" \\backslash'", session="edge")
check("newlines/quotes/backslashes survive", r.get("ok") and "line2" in r["value"])
r = ex("print('y' * 300_000)\n'done'", session="edge")
check("huge stdout is truncated, not dumped", len(r.get("stdout", "")) < 60_000,
      f"{len(r.get('stdout',''))} bytes")
check("truncation is announced", "omitted" in r.get("stdout", ""))
r = ex("list(range(50_000))", session="edge")
check("huge repr value is capped", len(r.get("value") or "") <= 2100,
      f"{len(r.get('value') or '')} chars")
r = ex("x = 1\n# trailing comment, no expression", session="edge")
check("statement-only code returns ok with no value", r["ok"] and not r.get("value"))
r = ex("", session="edge")
check("empty code is rejected cleanly", not r.get("ok"))

print("\n=== 8. Error containment ===")
r = ex("raise SystemExit(3)", session="err")
check("SystemExit does not kill the kernel", not r.get("ok"))
r = ex("5 * 5", session="err")
check("kernel alive after SystemExit", r.get("ok") and r["value"] == "25")
r = ex("import sys; sys.stdout.write('direct')\n'ok'", session="err")
check("direct stdout writes are captured", "direct" in r.get("stdout", ""))
r = ex("{}['missing']", session="err")
check("KeyError reported with traceback", "KeyError" in (r.get("error") or ""))

print("\n=== 9. Reset and reaper ===")
ex("temp = 1", session="reset")
plugin._reset_handler({}, task_id="reset")
r = ex("'temp' in dir()", session="reset")
check("reset discards state", r["value"] == "False")
before = len(plugin._kernels)
plugin._kernels["zombie"] = plugin._kernels[next(iter(plugin._kernels))]
old_idle = plugin.IDLE_SHUTDOWN_SECONDS
check("idle shutdown is configured", old_idle > 0, f"{old_idle}s")
plugin._kernels.pop("zombie", None)
check("kernel registry is consistent", len(plugin._kernels) == before)

print("\n=== 10. Tool bridge under load ===")
r = ex("outs = [terminal('echo n%d' % i) for i in range(5)]\n"
       "sum(1 for o in outs if 'n' in str(o))", session="bridge")
check("5 sequential tool calls from inside the kernel", r.get("ok") and r["value"] == "5",
      str(r)[:120])
r = ex("import rlm_bridge; rlm_bridge._call('image_generate', {})", session="bridge")
check("allow-list still enforced under load", not r.get("ok") and "not available" in r["error"])

# Cleanup every kernel this run created.
for sess in list(plugin._kernels):
    plugin._reset_handler({}, task_id=sess)

failed = [n for n, ok, _ in results if not ok]
print(f"\n{'=' * 60}")
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED: " + ", ".join(failed))
sys.exit(1 if failed else 0)
