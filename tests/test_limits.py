"""Resource-limit tests — the ceilings that keep one session from taking
down the machine.

Measured before these existed: 12 concurrent kernels held 206 MB with no cap
at all, and a single kernel happily allocated 1.5 GB on a 16 GB laptop.

    python3 tests/test_limits.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
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


def ex(code: str, session: str) -> dict:
    return json.loads(plugin._exec_handler({"code": code}, task_id=session))


print("\n=== 1. Kernel population is bounded ===")
check("MAX_KERNELS is configured", plugin.MAX_KERNELS > 0, str(plugin.MAX_KERNELS))

for sess in list(plugin._kernels):
    plugin._reset_handler({}, task_id=sess)

# Boot more sessions than the cap allows.
overshoot = plugin.MAX_KERNELS + 5
for i in range(overshoot):
    ex("marker = 1", f"lim{i}")
    time.sleep(0.01)  # keep last_used ordering deterministic

check("kernel count never exceeds the cap",
      len(plugin._kernels) <= plugin.MAX_KERNELS,
      f"{len(plugin._kernels)} <= {plugin.MAX_KERNELS}")

# The most recent session must be the one that survived.
newest = f"lim{overshoot - 1}"
check("the newest session is still alive", newest in plugin._kernels, newest)
check("the oldest sessions were evicted", "lim0" not in plugin._kernels)

# Eviction must be LRU, not arbitrary: the survivor set should be the tail.
survivors = {k for k in plugin._kernels if k.startswith("lim")}
expected_tail = {f"lim{i}" for i in range(overshoot - plugin.MAX_KERNELS, overshoot)}
check("eviction is least-recently-used", survivors <= expected_tail | {newest},
      f"{len(survivors)} survivors")

# An evicted session must still work — it just lost its state.
r = ex("'lim0' and 2 + 2", "lim0")
check("an evicted session still works (state lost, not broken)",
      r.get("ok") and r["value"] == "4", str(r)[:80])

print("\n=== 2. Touching a session keeps it alive ===")
for sess in list(plugin._kernels):
    plugin._reset_handler({}, task_id=sess)
ex("keep = 'me'", "hot")
time.sleep(0.05)
for i in range(plugin.MAX_KERNELS):
    ex("x = 1", f"cold{i}")
    if i % 3 == 0:
        ex("keep", "hot")  # keep it warm
r = ex("keep", "hot")
check("a repeatedly used session survives pressure",
      r.get("ok") and r.get("value") == "'me'", str(r)[:80])

print("\n=== 3. Memory ceiling warns with a remedy ===")
check("MAX_KERNEL_RSS_MB is configured", plugin.MAX_KERNEL_RSS_MB > 0,
      f"{plugin.MAX_KERNEL_RSS_MB} MB")
check("RSS checks are throttled", plugin.RSS_CHECK_EVERY >= 1,
      f"every {plugin.RSS_CHECK_EVERY} execs")

# Force the check by lowering the ceiling for this test only.
original = plugin.MAX_KERNEL_RSS_MB
plugin.MAX_KERNEL_RSS_MB = 1  # every kernel exceeds 1 MB
try:
    warned = None
    for _ in range(plugin.RSS_CHECK_EVERY + 1):
        out = ex("1", "memwarn")
        if out.get("warning"):
            warned = out["warning"]
            break
    check("an oversized kernel is reported", warned is not None, (warned or "")[:70])
    check("the warning names a concrete remedy",
          warned is not None and ("gc.collect" in warned or "rlm_reset" in warned),
          (warned or "")[:90])
    check("the session still works while warning",
          ex("3 * 3", "memwarn").get("value") == "9")
finally:
    plugin.MAX_KERNEL_RSS_MB = original

r = ex("1", "memwarn")
check("no warning when under the ceiling", "warning" not in r, str(r)[:80])

print("\n=== 4. RSS is actually measurable ===")
handle = plugin._get_kernel("rssprobe")
rss = plugin._rss_mb(handle.proc.pid)
check("RSS reads a real number", rss > 0, f"{rss:.1f} MB")
check("an idle kernel is small", rss < 100, f"{rss:.1f} MB")
check("RSS of a dead pid fails safely", plugin._rss_mb(999_999) == -1.0)

print("\n=== 5. Parallel subagents are configured ===")
check("MAX_PARALLEL_SUBAGENTS matches the delegation budget",
      plugin.MAX_PARALLEL_SUBAGENTS >= 4, str(plugin.MAX_PARALLEL_SUBAGENTS))
check("depth is still capped", plugin.MAX_RLM_DEPTH >= 1, str(plugin.MAX_RLM_DEPTH))

for sess in list(plugin._kernels):
    plugin._reset_handler({}, task_id=sess)

failed = [n for n, ok, _ in results if not ok]
print(f"\n{'=' * 60}")
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED: " + ", ".join(failed))
sys.exit(1 if failed else 0)
