"""Tests for checkpoint / restore — state that survives kernel death.

The kernel's core weakness is that everything lives in one process. These
checks prove a checkpoint genuinely rescues expensive state across a hard
kill, and — just as important — that a partial checkpoint is honest about
what it could not save.

    python3 tests/test_checkpoint.py
"""

from __future__ import annotations

import importlib.util
import json
import os
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


def ex(code: str, session: str, timeout: int | None = None) -> dict:
    args = {"code": code}
    if timeout is not None:
        args["timeout"] = timeout
    return json.loads(plugin._exec_handler(args, task_id=session))


def cp(action: str, session: str, name: str = "latest") -> dict:
    return json.loads(plugin._checkpoint_handler({"action": action, "name": name},
                                                 task_id=session))


SESSION = "cp-test"

print("\n=== 1. Save and restore across a hard kill ===")
ex("dataset = [{'i': i, 'v': i * 2} for i in range(50_000)]\n"
   "label = 'expensive load'\n"
   "total = sum(d['v'] for d in dataset)", SESSION)
before = ex("total", SESSION)["value"]

saved = cp("save", SESSION)
check("save reports success", saved.get("ok"), str(saved)[:100])
check("save lists what it stored", {"dataset", "label", "total"} <= set(saved.get("saved", [])),
      str(saved.get("saved"))[:80])
check("checkpoint file has real size", saved.get("size_bytes", 0) > 1000,
      f"{saved.get('size_bytes')} bytes")

handle = plugin._get_kernel(SESSION, create=False)
os.kill(handle.proc.pid, 9)
time.sleep(0.4)

gone = ex("'dataset' in dir()", SESSION)
check("state really is lost after SIGKILL", gone["value"] == "False")

restored = cp("restore", SESSION)
check("restore reports success", restored.get("ok"), str(restored)[:100])
after = ex("total", SESSION)
check("restored value matches the pre-crash value", after["value"] == before,
      f"{before} -> {after['value']}")
size = ex("len(dataset)", SESSION)
check("large object survived intact", size["value"] == "50000", size["value"])

print("\n=== 2. Honesty about what cannot be saved ===")
S2 = "cp-partial"
ex("import sqlite3\n"
   "good = {'a': 1}\n"
   "conn = sqlite3.connect(':memory:')\n"
   "func = lambda x: x + 1", S2)
saved = cp("save", S2)
check("picklable binding is saved", "good" in saved.get("saved", []))
skipped = {s["name"]: s for s in saved.get("skipped", [])}
check("open connection is reported, not dropped silently", "conn" in skipped, str(list(skipped)))
check("lambda is reported too", "func" in skipped, str(list(skipped)))
check("skip reasons are explanatory",
      all("reason" in s for s in saved.get("skipped", [])),
      str(list(skipped.values()))[:100])
check("modules are skipped with guidance",
      any("module" in (s.get("reason") or "") for s in saved.get("skipped", [])))

print("\n=== 3. Restore merges, never wipes ===")
S3 = "cp-merge"
ex("kept = 'from checkpoint'", S3)
cp("save", S3)
ex("added_later = 'not in checkpoint'", S3)
cp("restore", S3)
r = ex("(kept, added_later)", S3)
check("restore keeps bindings created after the save",
      "not in checkpoint" in (r.get("value") or ""), r.get("value"))

print("\n=== 4. Named checkpoints ===")
S4 = "cp-named"
ex("stage = 'one'", S4)
cp("save", S4, "stage1")
ex("stage = 'two'", S4)
cp("save", S4, "stage2")
cp("restore", S4, "stage1")
r = ex("stage", S4)
check("named checkpoints are independent", r["value"] == "'one'", r["value"])
listed = cp("list", S4)
names = {c["name"] for c in listed.get("checkpoints", [])}
check("list shows both checkpoints", {"stage1", "stage2"} <= names, str(names))

print("\n=== 5. Failure handling ===")
missing = cp("restore", "no-such-session", "nope")
check("restoring a missing checkpoint fails cleanly", not missing.get("ok"),
      str(missing)[:80])
bad = json.loads(plugin._checkpoint_handler({"action": "wat"}, task_id=SESSION))
check("invalid action is rejected", not bad.get("ok"), str(bad)[:80])
deleted = cp("delete", S4, "stage2")
check("delete works", deleted.get("ok"))
again = cp("delete", S4, "stage2")
check("deleting twice fails cleanly", not again.get("ok"))

print("\n=== 6. On-disk safety ===")
root = plugin._checkpoint.CHECKPOINT_ROOT
sample = next(root.rglob("*.pkl"), None)
check("checkpoint files exist on disk", sample is not None, str(sample))
if sample:
    check("checkpoint file is owner-only (0600)",
          (sample.stat().st_mode & 0o777) == 0o600, oct(sample.stat().st_mode & 0o777))
    check("checkpoint dir is owner-only (0700)",
          (sample.parent.stat().st_mode & 0o777) == 0o700,
          oct(sample.parent.stat().st_mode & 0o777))

print("\n=== 6b. Cross-session recovery ===")
# The session that most needs a checkpoint is a NEW one, after the old
# kernel died with its session id. Keying strictly on session would make
# the feature useless exactly then.
S6 = "cp-origin"
ex("rescued = 'from a dead session'", S6)
cp("save", S6, "handover")
plugin._reset_handler({}, task_id=S6)
# Cross-session borrowing is opt-in since the fleet-isolation hardening:
# without the flag this restore must fail (covered in test_isolation.py).
fresh = json.loads(plugin._checkpoint_handler(
    {"action": "restore", "name": "handover", "allow_cross_session": True},
    task_id="cp-brand-new-session"))
check("a fresh session can recover another session's checkpoint (opt-in)",
      fresh.get("ok"), str(fresh)[:120])
check("the borrow is reported, never silent",
      fresh.get("borrowed_from_session") == "cp-origin",
      str(fresh.get("borrowed_from_session")))
r = ex("rescued", "cp-brand-new-session")
check("borrowed state is actually usable",
      r.get("value") == "'from a dead session'", r.get("value"))
direct = plugin._checkpoint.load("cp-origin", "handover", allow_cross_session=False)
check("same-session load still prefers its own file",
      direct.get("ok") and "borrowed_from_session" not in direct)

print("\n=== 6c. Store is bounded ===")
before_count = len(plugin._checkpoint.listing())
old = plugin._checkpoint._session_dir("cp-ancient")
old.mkdir(parents=True, exist_ok=True)
stale = old / "latest.pkl"
stale.write_bytes(b"\x80\x04}\x94.")  # valid empty pickle
ancient = time.time() - 40 * 86_400
os.utime(stale, (ancient, ancient))
pruned = plugin._checkpoint.prune()
check("stale checkpoints are pruned", any("cp-ancient" in r for r in pruned["removed"]),
      str(pruned["removed"])[:120])
check("prune removes the file from disk", not stale.exists())
check("prune leaves fresh checkpoints alone",
      len(plugin._checkpoint.listing()) >= before_count - 1,
      f"{before_count} -> {len(plugin._checkpoint.listing())}")
check("age and size budgets are configured",
      plugin._checkpoint.MAX_AGE_DAYS > 0 and plugin._checkpoint.MAX_TOTAL_MB > 0,
      f"{plugin._checkpoint.MAX_AGE_DAYS}d / {plugin._checkpoint.MAX_TOTAL_MB}MB")
saved_now = cp("save", "cp-prune-hook")
check("save runs pruning automatically", saved_now.get("ok"))

print("\n=== 7. Timeout salvage path ===")
S7 = "cp-timeout"
ex("import threading, time\n"
   "precious = list(range(10_000))\n"
   "def hog():\n"
   "    time.sleep(60)\n"
   "threading.Thread(target=hog, daemon=True).start()", S7)
r = ex("import time; time.sleep(30)", S7, timeout=2)
check("timeout still reported as failure", not r.get("ok"))
check("timeout message mentions state outcome",
      "state" in (r.get("error") or "").lower() or "salvage" in (r.get("error") or "").lower(),
      (r.get("error") or "")[:100])

for sess in list(plugin._kernels):
    plugin._reset_handler({}, task_id=sess)
for sess in (SESSION, S2, S3, S4, S7, "cp-origin", "cp-brand-new-session",
             "cp-ancient", "cp-prune-hook"):
    for nm in ("latest", "stage1", "stage2", "autosave", "handover"):
        plugin._checkpoint.delete(sess, nm)

failed = [n for n, ok, _ in results if not ok]
print(f"\n{'=' * 60}")
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED: " + ", ".join(failed))
sys.exit(1 if failed else 0)
