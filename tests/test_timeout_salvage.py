"""A timeout must not silently destroy the namespace.

The old salvage path asked the kernel for an autosave checkpoint while the
kernel was still stuck inside the runaway user code, then waited 3s for an
answer that could not come — so every real timeout reported "all in-memory
state was lost". This proves the fixed path: interrupt first, then
checkpoint, and only claim salvage when the file is actually on disk.

Deterministic: a blocking time.sleep against a 3s timeout. No model calls,
no network.

    python3 tests/test_timeout_salvage.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Isolate checkpoints from the operator's real store — CHECKPOINT_ROOT is
# resolved at import time, so this must be set before the plugin loads.
HOME = Path(tempfile.mkdtemp(prefix="rlm-salvage-"))
os.environ["HERMES_HOME"] = str(HOME)
os.environ["HERMES_RLM_SALVAGE_SECONDS"] = "20"

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


def ex(code: str, session: str, timeout: int | None = None) -> dict:
    args: dict = {"code": code}
    if timeout is not None:
        args["timeout"] = timeout
    return json.loads(plugin._exec_handler(args, task_id=session))


SESSION = "salvage-test"
AUTOSAVE = plugin._checkpoint._session_dir(SESSION) / "autosave.pkl"

try:
    check("salvage budget is configurable, not a hardcoded 3s",
          plugin.SALVAGE_SECONDS == 20, str(plugin.SALVAGE_SECONDS))

    print("\n=== 1. Expensive state, then a hang ===")
    r = ex("precious = list(range(50_000))\nlabel = 'expensive to rebuild'\n"
           "total = sum(precious)\ntotal", SESSION)
    check("setup ran", r.get("ok") and r["value"] == "1249975000", str(r)[:120])
    check("no autosave exists yet", not AUTOSAVE.exists(), str(AUTOSAVE))

    # The kernel main loop is blocked inside user code here — exactly the
    # case the old 3s best-effort request could never serve.
    started = time.monotonic()
    r = ex("import time\ntime.sleep(120)", SESSION, timeout=3)
    elapsed = time.monotonic() - started

    print("\n=== 2. Timeout is still a failure, and bounded ===")
    check("timeout reported as failure", not r.get("ok"), str(r)[:120])
    check("salvage stays bounded (no second hang)", elapsed < 45, f"{elapsed:.1f}s")

    print("\n=== 3. Salvage actually happened ===")
    check("autosave checkpoint EXISTS on disk", AUTOSAVE.exists(), str(AUTOSAVE))
    error = r.get("error") or ""
    check("message claims salvage only because it is true",
          ("salvaged" in error) == AUTOSAVE.exists(), error[:160])
    if AUTOSAVE.exists():
        check("message names the recovery command",
              "rlm_checkpoint" in error and "autosave" in error, error[:160])
    else:
        check("honest 'state was lost' wording plus advice",
              "state was lost" in error and "rlm_checkpoint" in error, error[:160])

    print("\n=== 4. A fresh kernel gets the state back ===")
    handle = plugin._get_kernel(SESSION, create=False)
    check("dead kernel was dropped or replaced", handle is None or handle.alive(),
          repr(handle))

    r = ex("(len(precious), label, total)", SESSION)
    check("variable created before the hang is restored",
          r.get("ok") and r.get("value") == "(50000, 'expensive to rebuild', 1249975000)",
          str(r)[:160])

    print("\n=== 5. Salvage never claims more than it did ===")
    # No checkpoint feature -> no salvage, and the message must say so.
    S2 = "salvage-off"
    ex("throwaway = 1", S2)
    plugin.FEATURE_CHECKPOINT = False
    try:
        r = ex("import time\ntime.sleep(120)", S2, timeout=3)
    finally:
        plugin.FEATURE_CHECKPOINT = True
    check("no false salvage claim when checkpointing is off",
          "salvaged" not in (r.get("error") or "") and "state was lost" in (r.get("error") or ""),
          (r.get("error") or "")[:160])
finally:
    for sess in list(plugin._kernels):
        plugin._reset_handler({}, task_id=sess)
    shutil.rmtree(HOME, ignore_errors=True)

failed = [n for n, ok, _ in results if not ok]
print(f"\n{'=' * 60}")
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED: " + ", ".join(failed))
sys.exit(1 if failed else 0)
