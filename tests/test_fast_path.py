"""Tests for the tool-free subagent fast path.

Measured motivation: `hermes chat -q` takes ~17s of which only ~3s is the
model. The remaining ~14s is agent-loop startup, which is pure waste when the
task needs no tools.

The risk is misrouting: answering a task that needed tools with a plain model
call would produce a confident wrong answer. These checks are mostly about
the eligibility gate being conservative and the fallback being real.

    python3 tests/test_fast_path.py
"""

from __future__ import annotations

import importlib.util
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

fp = plugin._fast_path
results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


print("\n=== 1. Eligibility gate refuses anything tool-shaped ===")
NEEDS_AGENT = [
    "Read tools/calc.py and summarise it",
    "Run the test suite and report failures",
    "Check whether the API returns 200",
    "Search the repo for TODO comments",
    "Fix the bug in the auth middleware",
    "Download the dataset and count rows",
    "Audit this repository for security issues",
    "Verify the database schema",
    "Investigate why the deploy failed",
    "Commit the change and push it",
    "Analyse /path/to/file.csv",
    "Open the website and read the pricing",
]
missed = [g for g in NEEDS_AGENT if fp.eligible(g)]
check("no tool-shaped task is treated as tool-free", not missed, str(missed[:2]))

print("\n=== 2. Genuinely tool-free work is allowed ===")
TOOL_FREE = [
    "Reply with exactly OK",
    "Summarise the following paragraph in one sentence",
    "Classify this sentiment as positive or negative",
    "Rephrase this text to be more concise",
    "Translate this sentence into Dutch",
    "Which of these two options is more persuasive?",
]
blocked = [g for g in TOOL_FREE if not fp.eligible(g)]
check("plain reasoning tasks are eligible", not blocked, str(blocked[:2]))

print("\n=== 3. Structural guards ===")
check("empty goal is refused", not fp.eligible(""))
check("whitespace goal is refused", not fp.eligible("   "))
check("a long brief is refused", not fp.eligible("Summarise: " + "x" * 5000))
check("context is inspected too, not only the goal",
      not fp.eligible("Summarise this", "the file at /etc/passwd"))

print("\n=== 4. Failure falls back rather than degrading ===")
original = fp._read_env_key
try:
    fp._read_env_key = lambda name: None  # simulate a missing credential
    raised = False
    try:
        fp.run("Reply with exactly OK")
    except fp.FastPathUnavailable:
        raised = True
    check("missing credential raises FastPathUnavailable", raised)

    # _run_subagent must swallow that and use the full agent instead.
    started = time.monotonic()
    result = plugin._run_subagent("Reply with exactly FALLBACK_OK", "", timeout=180, fast=True)
    elapsed = time.monotonic() - started
    check("subagent still succeeds via the full agent", result.get("ok"), str(result)[:90])
    check("fallback did NOT claim the fast path", not result.get("fast_path"))
    check("fallback took full-agent time (proving it really fell back)",
          elapsed > 5, f"{elapsed:.1f}s")
finally:
    fp._read_env_key = original

print("\n=== 5. Fast path is opt-in, never automatic ===")
started = time.monotonic()
slow = plugin._run_subagent("Reply with exactly SLOW_OK", "", timeout=180)
slow_secs = time.monotonic() - started
check("default call does not use the fast path", not slow.get("fast_path"))
check("default call still works", slow.get("ok"), str(slow)[:80])

print("\n=== 6. Fast path actually is faster (live) ===")
started = time.monotonic()
fast = plugin._run_subagent("Reply with exactly FAST_OK", "", timeout=180, fast=True)
fast_secs = time.monotonic() - started
check("fast call succeeds", fast.get("ok"), str(fast)[:80])
check("fast call is marked as such", fast.get("fast_path") is True)
check("fast call beats the full agent", fast_secs < slow_secs,
      f"{fast_secs:.1f}s vs {slow_secs:.1f}s")
check("speedup is material (>2x)", slow_secs / max(fast_secs, 0.01) > 2,
      f"{slow_secs / max(fast_secs, 0.01):.1f}x")

failed = [n for n, ok, _ in results if not ok]
print(f"\n{'=' * 60}")
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED: " + ", ".join(failed))
sys.exit(1 if failed else 0)
