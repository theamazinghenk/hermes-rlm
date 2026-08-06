"""Continual-harness-lite: store CRUD, proposals, rollback, overview, handler.

Run: ~/.hermes/hermes-agent/venv/bin/python tests/test_harness.py
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

spec = importlib.util.spec_from_file_location("hermes_rlm_harness", HERE / "harness.py")
harness = importlib.util.module_from_spec(spec)
sys.modules["hermes_rlm_harness"] = harness
spec.loader.exec_module(harness)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


scope = f"test-harness-{uuid.uuid4().hex[:6]}"
store = harness.HarnessStore(scope)

# --- CRUD -------------------------------------------------------------------
e1 = store.create("memory", "Testfeit", "De trades-tabel bevat twee record-types.")
check("create returns entry with id", bool(e1.get("id")))
try:
    store.create("bogus", "t", "c")
    check("invalid kind rejected", False)
except ValueError:
    check("invalid kind rejected", True)

e1b = store.update(e1["id"], content="Bijgewerkt inzicht.")
check("update bumps version", e1b["version"] == 2, str(e1b["version"]))

# --- mtime sync between two writers -----------------------------------------
other = harness.HarnessStore(scope)
other.create("skill", "Recept", "Stap 1 doe X. Stap 2 doe Y.")
check("second writer visible after sync", len(store.entries()) == 2,
      str(len(store.entries())))

# --- proposal + rollback ----------------------------------------------------
proposal = {"edits": [
    {"op": "create", "kind": "prompt", "title": "Regel", "content": "Meet eerst."},
    {"op": "update", "id": e1["id"], "content": "Nogmaals bijgewerkt."},
]}
outcome = store.apply_proposal(proposal, evidence_note="unit test")
check("proposal applied", len(outcome["applied"]) == 2, str(outcome))
check("ledger records refinement", len(store.refinements()) == 1)

bad = {"edits": [{"op": "update", "id": "bestaat-niet", "content": "x"}]}
try:
    store.apply_proposal(bad)
    check("invalid proposal rejected atomically", False)
except ValueError:
    check("invalid proposal rejected atomically", True)
check("failed proposal left no ledger entry", len(store.refinements()) == 1)

rb = store.rollback(outcome["refinement_id"])
check("rollback inverts edits", rb["restored"] == 2, str(rb))
entries_now = {e["id"]: e for e in store.entries()}
check("rollback removed created entry",
      all(e["title"] != "Regel" for e in entries_now.values()))
check("rollback restored update", entries_now[e1["id"]]["content"] == "Bijgewerkt inzicht.",
      entries_now[e1["id"]]["content"])
try:
    store.rollback(outcome["refinement_id"])
    check("double rollback refused", False)
except ValueError:
    check("double rollback refused", True)

# --- overview ---------------------------------------------------------------
text = harness.overview(scope)
check("overview mentions session entries", "Testfeit" in text or "Recept" in text)
check("empty overview is empty string",
      harness.overview(f"leeg-{uuid.uuid4().hex[:6]}") == ""
      if not harness.HarnessStore("global").entries() else True)

# --- plugin handler (add/list/delete/rollback paths, no model call) ---------
spec2 = importlib.util.spec_from_file_location("hermes_rlm_plugin", HERE / "__init__.py")
plugin = importlib.util.module_from_spec(spec2)
sys.modules["hermes_rlm_plugin"] = plugin
spec2.loader.exec_module(plugin)

out = json.loads(plugin._refine_handler({"action": "add", "kind": "memory",
                                         "title": "Via handler",
                                         "content": "inhoud"}, task_id=scope))
check("handler add works", out.get("ok") is True, str(out))
out = json.loads(plugin._refine_handler({"action": "list"}, task_id=scope))
check("handler list shows entries", out.get("ok") and len(out["entries"]) >= 1)
eid = out["entries"][0]["id"]
out = json.loads(plugin._refine_handler({"action": "delete", "id": eid}, task_id=scope))
check("handler delete works", out.get("ok") is True, str(out))
out = json.loads(plugin._refine_handler({"action": "rollback", "id": "nee"}, task_id=scope))
check("handler rollback error is clean", out.get("ok") is False)

# --- hook never raises ------------------------------------------------------
res = plugin._harness_pre_llm_call("msg", [], False, "m", "cli", session_id=scope)
check("hook returns context or None", res is None or "context" in res)

shutil.rmtree(harness.HARNESS_ROOT / scope, ignore_errors=True)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} — {FAILURES}")
    sys.exit(1)
print("ok")
