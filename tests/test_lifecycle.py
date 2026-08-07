"""Eval-gated staged refinement lifecycle."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
home = Path(tempfile.mkdtemp(prefix="rlm-lifecycle-"))
os.environ["HERMES_HOME"] = str(home)
spec = importlib.util.spec_from_file_location("hermes_rlm_harness", HERE / "harness.py")
harness = importlib.util.module_from_spec(spec)
sys.modules["hermes_rlm_harness"] = harness
spec.loader.exec_module(harness)

store = harness.HarnessStore("suite")
proposal = {"edits": [{"op": "create", "kind": "memory", "title": "candidate", "content": "safe note"}]}
passing = {"cases": [{"baseline": .6, "candidate": .8}, {"baseline": .7, "candidate": .75}],
           "thresholds": {"min_candidate_mean": .7, "min_mean_improvement": .1,
                          "max_case_regression": .05}}
failing = {"cases": [{"baseline": .8, "candidate": .5}],
           "thresholds": {"min_candidate_mean": .7, "min_mean_improvement": 0,
                          "max_case_regression": .05}}

candidate = store.stage(proposal, "fixture")
assert store.entries() == [], "stage mutated active entries"
assert store.candidates(candidate["id"])["proposal"] == proposal
assert harness.HarnessStore("suite").candidates(candidate["id"])["status"] == "staged"
assert store.evaluate(candidate["id"], passing)["passed"]
promoted = store.promote(candidate["id"])
assert promoted["status"] == "promoted" and len(store.entries()) == 1

failed_candidate = store.stage(proposal)
assert not store.evaluate(failed_candidate["id"], failing)["passed"]
try:
    store.promote(failed_candidate["id"])
    raise AssertionError("failed candidate promoted")
except ValueError:
    pass
assert len(store.entries()) == 1

rejected = store.stage(proposal)
assert store.reject(rejected["id"], "not useful")["status"] == "rejected"
try:
    store.promote(rejected["id"])
    raise AssertionError("rejected candidate promoted")
except ValueError:
    pass

regression = store.evaluate(candidate["id"], failing)
assert regression["rolled_back"] and store.entries() == []
assert store.candidates(candidate["id"])["status"] == "rolled_back"
assert len(store._state["evaluations"]) == 3
assert harness.HarnessStore("suite").candidates(candidate["id"])["status"] == "rolled_back"

for invalid in ({}, {"cases": [], "thresholds": {}},
                {"cases": [{"baseline": "x", "candidate": 1}], "thresholds": passing["thresholds"]}):
    fresh = store.stage(proposal)
    try:
        store.evaluate(fresh["id"], invalid)
        raise AssertionError(f"invalid suite accepted: {invalid}")
    except (ValueError, TypeError):
        pass

mode = (harness.HARNESS_ROOT / "suite" / "harness_state.json").stat().st_mode & 0o777
assert mode == 0o600, oct(mode)
json.loads((harness.HARNESS_ROOT / "suite" / "harness_state.json").read_text())
shutil.rmtree(home)
print("ok")
