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
OVERVIEW_PER_KIND = 6


def _now() -> float:
    return time.time()


class HarnessStore:
    """One harness_state.json: entries + refinements ledger, mtime-synced."""

    def __init__(self, scope_dir: str) -> None:
        self.dir = HARNESS_ROOT / scope_dir
        self.path = self.dir / "harness_state.json"
        self._mtime: tuple[int, int] = (-1, -1)
        self._state: dict = {"schema": 1, "entries": {}, "refinements": []}
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
