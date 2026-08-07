"""Continual-harness-lite: durable, evidence-backed agent state.

A small port of prime-agent's continual harness onto the hermes-rlm plugin.
Entries (prompt-notes, memories, skills, subagent-specs) live in a JSON file
per session plus one global file. Edits arrive as validated proposals with
before/after snapshots in a refinements ledger, so every change can be
rolled back. Stdlib-only, atomic writes, mode 0600 in 0700 dirs.

Layout:
    ~/.hermes/state/rlm/harness/global/harness_state.json
    ~/.hermes/state/rlm/harness/<session_id>/harness_state.json
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

def _hermes_home() -> Path:
    """Hermes home with HERMES_HOME respected (fleet/profile isolation)."""
    env = os.environ.get("HERMES_HOME")
    return Path(env).expanduser() if env else Path.home() / ".hermes"


HARNESS_ROOT = _hermes_home() / "state" / "rlm" / "harness"
KINDS = ("prompt", "memory", "skill", "subagent")
MAX_ENTRY_CHARS = 4000
MAX_LEDGER = 100
MAX_CANDIDATES = 100
MAX_EVALUATIONS = 1_000
MAX_PROPOSAL_CHARS = 20_000
MAX_EVAL_SUITE_CHARS = 100_000
OVERVIEW_PER_KIND = 6


def _now() -> float:
    return time.time()


class HarnessStore:
    """One harness_state.json: entries + refinements ledger, mtime-synced."""

    def __init__(self, scope_dir: str) -> None:
        self.dir = HARNESS_ROOT / scope_dir
        self.path = self.dir / "harness_state.json"
        self._mtime: tuple[int, int] = (-1, -1)
        self._state: dict = {"schema": 2, "entries": {}, "refinements": [],
                             "candidates": [], "evaluations": []}
        self._sync()

    # --- persistence ------------------------------------------------------

    def _sync(self) -> None:
        """Reload from disk when another writer (host or kernel) changed it.

        Identity is (mtime_ns, size), not mtime: on filesystems with coarse
        timestamp resolution several consecutive writes share one mtime, so
        a second writer's changes would be invisible. Reported by a fleet
        operator who reproduced it with five writes in a row.
        """
        try:
            st = self.path.stat()
            token = (st.st_mtime_ns, st.st_size)
        except OSError:
            return
        if token != self._mtime:
            try:
                self._state = json.loads(self.path.read_text())
                self._mtime = token
            except (OSError, json.JSONDecodeError):
                pass  # keep in-memory state; next write repairs the file

    def _write(self) -> None:
        self.dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._state, indent=1, default=str))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)
        try:
            st = self.path.stat()
            self._mtime = (st.st_mtime_ns, st.st_size)
        except OSError:
            pass

    # --- CRUD -------------------------------------------------------------

    def create(self, kind: str, title: str, content: str) -> dict:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not title or not str(content).strip():
            raise ValueError("title and content are required")
        entry = {
            "id": uuid.uuid4().hex[:12], "kind": kind, "title": str(title)[:120],
            "content": str(content)[:MAX_ENTRY_CHARS], "version": 1,
            "created_at": _now(), "updated_at": _now(),
        }
        self._sync()
        self._state["entries"][entry["id"]] = entry
        self._write()
        return entry

    def update(self, entry_id: str, title: str | None = None,
               content: str | None = None) -> dict:
        self._sync()
        entry = self._state["entries"].get(entry_id)
        if entry is None:
            raise KeyError(f"no entry {entry_id!r}")
        if title:
            entry["title"] = str(title)[:120]
        if content is not None:
            entry["content"] = str(content)[:MAX_ENTRY_CHARS]
        entry["version"] += 1
        entry["updated_at"] = _now()
        self._write()
        return entry

    def delete(self, entry_id: str) -> dict:
        self._sync()
        entry = self._state["entries"].pop(entry_id, None)
        if entry is None:
            raise KeyError(f"no entry {entry_id!r}")
        self._write()
        return entry

    def entries(self, kind: str | None = None) -> list[dict]:
        self._sync()
        found = list(self._state["entries"].values())
        if kind:
            found = [e for e in found if e.get("kind") == kind]
        return sorted(found, key=lambda e: e.get("updated_at", 0), reverse=True)

    # --- proposals + ledger ----------------------------------------------

    def apply_proposal(self, proposal: dict, evidence_note: str = "") -> dict:
        """Apply {edits: [{op, kind?, id?, title?, content?}]} atomically-ish.

        Every edit is validated first; the whole proposal is rejected on the
        first invalid edit. Applied proposals land in the refinements ledger
        with before/after snapshots so rollback() can invert them.
        """
        edits = proposal.get("edits")
        if not isinstance(edits, list) or not edits:
            raise ValueError("proposal.edits must be a non-empty list")
        if len(edits) > 12:
            raise ValueError("proposal too large (max 12 edits)")
        self._sync()

        # Validate all before applying any.
        for e in edits:
            op = e.get("op")
            if op == "create":
                if e.get("kind") not in KINDS:
                    raise ValueError(f"create needs kind in {KINDS}")
                if not e.get("title") or not str(e.get("content", "")).strip():
                    raise ValueError("create needs title and content")
            elif op in ("update", "delete"):
                if e.get("id") not in self._state["entries"]:
                    raise ValueError(f"{op} references unknown id {e.get('id')!r}")
            else:
                raise ValueError(f"unknown op {op!r}")

        snapshots: list[dict] = []
        applied: list[dict] = []
        for e in edits:
            op = e["op"]
            if op == "create":
                entry = self.create(e["kind"], e["title"], e["content"])
                snapshots.append({"op": op, "before": None, "after": dict(entry)})
                applied.append({"op": op, "id": entry["id"]})
            elif op == "update":
                before = dict(self._state["entries"][e["id"]])
                entry = self.update(e["id"], e.get("title"), e.get("content"))
                snapshots.append({"op": op, "before": before, "after": dict(entry)})
                applied.append({"op": op, "id": e["id"]})
            else:
                before = self.delete(e["id"])
                snapshots.append({"op": op, "before": dict(before), "after": None})
                applied.append({"op": op, "id": e["id"]})

        refinement = {
            "id": uuid.uuid4().hex[:12], "created_at": _now(),
            "evidence": str(evidence_note)[:500], "edits": applied,
            "snapshots": snapshots, "rolled_back": False,
        }
        self._state["refinements"].append(refinement)
        self._state["refinements"] = self._state["refinements"][-MAX_LEDGER:]
        self._write()
        return {"refinement_id": refinement["id"], "applied": applied}

    def rollback(self, refinement_id: str) -> dict:
        self._sync()
        target = next((r for r in self._state["refinements"]
                       if r["id"] == refinement_id), None)
        if target is None:
            raise KeyError(f"no refinement {refinement_id!r}")
        if target.get("rolled_back"):
            raise ValueError(f"refinement {refinement_id!r} already rolled back")
        for snap in reversed(target["snapshots"]):
            before, after = snap.get("before"), snap.get("after")
            if before is None and after is not None:        # was create → remove
                self._state["entries"].pop(after["id"], None)
            elif before is not None:                        # update/delete → restore
                self._state["entries"][before["id"]] = before
        target["rolled_back"] = True
        self._write()
        return {"rolled_back": refinement_id,
                "restored": len(target["snapshots"])}

    def refinements(self, limit: int = 10) -> list[dict]:
        self._sync()
        return [{k: r.get(k) for k in ("id", "created_at", "evidence",
                                       "edits", "rolled_back")}
                for r in self._state["refinements"][-limit:]][::-1]

    # --- eval-gated staged lifecycle -------------------------------------

    def _validate_proposal(self, proposal: dict) -> None:
        try:
            encoded = json.dumps(proposal, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"proposal must be JSON: {exc}") from exc
        if len(encoded) > MAX_PROPOSAL_CHARS:
            raise ValueError(f"proposal too large (max {MAX_PROPOSAL_CHARS} chars)")
        edits = proposal.get("edits")
        if not isinstance(edits, list) or not edits or len(edits) > 12:
            raise ValueError("proposal.edits must contain 1-12 edits")
        for edit in edits:
            if not isinstance(edit, dict):
                raise ValueError("each edit must be an object")
            op = edit.get("op")
            if op == "create":
                if edit.get("kind") not in KINDS or not edit.get("title") or not str(edit.get("content", "")).strip():
                    raise ValueError("create needs valid kind, title and content")
            elif op in ("update", "delete"):
                if edit.get("id") not in self._state["entries"]:
                    raise ValueError(f"{op} references unknown id {edit.get('id')!r}")
            else:
                raise ValueError(f"unknown op {op!r}")

    def stage(self, proposal: dict, evidence_note: str = "") -> dict:
        """Persist a bounded proposal without changing active entries."""
        self._sync()
        self._validate_proposal(proposal)
        candidate = {"id": uuid.uuid4().hex[:12], "created_at": _now(),
                     "status": "staged", "proposal": proposal,
                     "evidence": str(evidence_note)[:500], "evaluation_ids": []}
        self._state.setdefault("candidates", []).append(candidate)
        self._state["candidates"] = self._state["candidates"][-MAX_CANDIDATES:]
        self._write()
        return {k: candidate[k] for k in ("id", "created_at", "status", "evidence")}

    def candidates(self, candidate_id: str | None = None) -> list[dict] | dict:
        self._sync()
        found = self._state.setdefault("candidates", [])
        if candidate_id:
            candidate = next((c for c in found if c["id"] == candidate_id), None)
            if candidate is None:
                raise KeyError(f"no candidate {candidate_id!r}")
            return json.loads(json.dumps(candidate))
        return [{k: c.get(k) for k in ("id", "created_at", "status", "evidence",
                                        "evaluation_ids", "refinement_id")}
                for c in reversed(found)]

    @staticmethod
    def _score_suite(suite: dict) -> tuple[dict, bool]:
        if not isinstance(suite, dict):
            raise ValueError("suite must be a JSON object")
        encoded = json.dumps(suite, allow_nan=False)
        if len(encoded) > MAX_EVAL_SUITE_CHARS:
            raise ValueError(f"eval suite too large (max {MAX_EVAL_SUITE_CHARS} chars)")
        cases, thresholds = suite.get("cases"), suite.get("thresholds")
        if not isinstance(cases, list) or not cases or len(cases) > 1000:
            raise ValueError("suite.cases must contain 1-1000 cases")
        if not isinstance(thresholds, dict):
            raise ValueError("suite.thresholds is required")
        required = ("min_candidate_mean", "min_mean_improvement", "max_case_regression")
        if any(not isinstance(thresholds.get(k), (int, float)) for k in required):
            raise ValueError(f"thresholds require numeric {required}")
        baseline, candidate = [], []
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("baseline"), (int, float)) or not isinstance(case.get("candidate"), (int, float)):
                raise ValueError("each case requires numeric baseline and candidate")
            baseline.append(float(case["baseline"]))
            candidate.append(float(case["candidate"]))
        b_mean, c_mean = sum(baseline) / len(baseline), sum(candidate) / len(candidate)
        worst_regression = max((b - c for b, c in zip(baseline, candidate)), default=0.0)
        metrics = {"cases": len(cases), "baseline_mean": b_mean,
                   "candidate_mean": c_mean, "mean_improvement": c_mean - b_mean,
                   "worst_case_regression": worst_regression,
                   "thresholds": {k: float(thresholds[k]) for k in required}}
        passed = (c_mean >= thresholds["min_candidate_mean"] and
                  c_mean - b_mean >= thresholds["min_mean_improvement"] and
                  worst_regression <= thresholds["max_case_regression"])
        return metrics, passed

    def evaluate(self, candidate_id: str, suite: dict) -> dict:
        """Append an immutable evaluation; regressions roll back promoted edits."""
        self._sync()
        candidate = next((c for c in self._state.setdefault("candidates", [])
                          if c["id"] == candidate_id), None)
        if candidate is None:
            raise KeyError(f"no candidate {candidate_id!r}")
        if candidate["status"] in ("rejected", "rolled_back"):
            raise ValueError(f"candidate is {candidate['status']}")
        metrics, passed = self._score_suite(suite)
        record = {"id": uuid.uuid4().hex[:12], "candidate_id": candidate_id,
                  "created_at": _now(), "passed": passed, "metrics": metrics}
        evaluations = self._state.setdefault("evaluations", [])
        evaluations.append(record)
        if len(evaluations) > MAX_EVALUATIONS:
            referenced = {eid for c in self._state.get("candidates", [])
                          for eid in c.get("evaluation_ids", [])}
            removable = [e for e in evaluations if e["id"] not in referenced]
            remove_ids = {e["id"] for e in removable[:len(evaluations) - MAX_EVALUATIONS]}
            self._state["evaluations"] = [e for e in evaluations if e["id"] not in remove_ids]
        live = next(c for c in self._state["candidates"] if c["id"] == candidate_id)
        live.setdefault("evaluation_ids", []).append(record["id"])
        rolled_back = False
        if live["status"] == "promoted" and not passed:
            self.rollback(live["refinement_id"])
            live = next(c for c in self._state["candidates"] if c["id"] == candidate_id)
            live["status"] = "rolled_back"
            live["rolled_back_at"] = _now()
            rolled_back = True
        self._write()
        return {"evaluation_id": record["id"], "candidate_id": candidate_id,
                "passed": passed, "metrics": metrics, "rolled_back": rolled_back}

    def promote(self, candidate_id: str) -> dict:
        self._sync()
        candidate = next((c for c in self._state.setdefault("candidates", [])
                          if c["id"] == candidate_id), None)
        if candidate is None:
            raise KeyError(f"no candidate {candidate_id!r}")
        if candidate["status"] != "staged":
            raise ValueError(f"candidate is {candidate['status']}")
        evaluations = self._state.setdefault("evaluations", [])
        latest = next((e for e in reversed(evaluations)
                       if e["candidate_id"] == candidate_id), None)
        if latest is None or not latest["passed"]:
            raise ValueError("candidate needs a passing evaluation before promotion")
        outcome = self.apply_proposal(candidate["proposal"], candidate["evidence"])
        candidate = next(c for c in self._state["candidates"] if c["id"] == candidate_id)
        candidate.update(status="promoted", promoted_at=_now(),
                         refinement_id=outcome["refinement_id"])
        self._write()
        return {"candidate_id": candidate_id, "status": "promoted", **outcome}

    def reject(self, candidate_id: str, reason: str = "") -> dict:
        self._sync()
        candidate = next((c for c in self._state.setdefault("candidates", [])
                          if c["id"] == candidate_id), None)
        if candidate is None:
            raise KeyError(f"no candidate {candidate_id!r}")
        if candidate["status"] != "staged":
            raise ValueError(f"candidate is {candidate['status']}")
        candidate.update(status="rejected", rejected_at=_now(),
                         rejection_reason=str(reason)[:500])
        self._write()
        return {"candidate_id": candidate_id, "status": "rejected"}


def overview(session_id: str | None = None,
             per_kind: int = OVERVIEW_PER_KIND) -> str:
    """Compact text overview of global + session entries for prompt injection.

    Empty string when there is nothing — callers can inject conditionally.
    """
    parts: list[str] = []
    stores = [("global", HarnessStore("global"))]
    if session_id:
        stores.append(("session", HarnessStore(session_id)))
    for scope, store in stores:
        for kind in KINDS:
            for entry in store.entries(kind)[:per_kind]:
                head = entry["content"].split("\n", 1)[0][:160]
                parts.append(f"- [{scope}/{kind}] {entry['title']}: {head}")
    if not parts:
        return ""
    return ("Durable harness notes (earned from earlier work — follow unless "
            "clearly outdated; manage via rlm_refine):\n" + "\n".join(parts))
