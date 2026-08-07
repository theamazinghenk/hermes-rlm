"""Update-resistant, deterministic self-improvement controller.

All mutable data lives below $HERMES_HOME/state/rlm/improvement.  The plugin
checkout is code only: reinstalling it cannot erase observations or decisions.
"""
from __future__ import annotations

import hashlib
import copy
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

SCHEMA_VERSION = 2
PLUGIN_VERSION = "0.8.0"
STATE_RELATIVE = Path("state/rlm/improvement")
PROTECTED = (
    ".github/workflows/", "security/", "deploy/", "deployment/", "release/",
    "approvals/", "evaluator", "evaluation/holdout", "holdout/", "SOUL.md",
    "AGENTS.md", "plugin.yaml", "improvement.py",
)
AUTO_EXTENSIONS = {".md", ".rst", ".txt"}
AUTO_DIRS = ("docs/", "skills/", "tests/")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=repo, capture_output=True, text=True,
                          timeout=120)


def _changed_paths(diff: str, untracked: str = "") -> list[str]:
    paths = set(re.findall(r"^(?:\+\+\+ b/|--- a/)(.+)$", diff, re.MULTILINE))
    paths.update(line[3:].strip() for line in untracked.splitlines()
                 if len(line) > 3)
    clean = set()
    for path in paths:
        path = path.replace("\\", "/").lstrip("/")
        if path and path != "/dev/null" and ".." not in Path(path).parts:
            clean.add(path)
    return sorted(clean)


def protected_paths(paths: list[str]) -> list[str]:
    blocked = []
    for path in paths:
        normalized = path.replace("\\", "/").lstrip("/")
        if ".." in Path(normalized).parts or any(
                normalized == rule.rstrip("/") or normalized.startswith(rule)
                for rule in PROTECTED):
            blocked.append(path)
    return sorted(blocked)


class IntegrityError(RuntimeError):
    """Persistent state or import failed cryptographic integrity checks."""


class ImprovementController:
    """Durable controller. `coder` is the existing isolated-worktree primitive."""

    def __init__(self, home: str | Path | None = None,
                 coder: Callable[[dict], dict] | None = None):
        base = Path(home).expanduser() if home else Path(
            os.environ.get("HERMES_HOME", Path.home() / ".hermes")).expanduser()
        self.root = base / STATE_RELATIVE
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.state_path = self.root / "state.json"
        self.audit_path = self.root / "audit.jsonl"
        self.snapshots = self.root / "snapshots"
        self.snapshots.mkdir(mode=0o700, exist_ok=True)
        os.chmod(self.snapshots, 0o700)
        self.coder = coder
        self.state = self._load()

    @staticmethod
    def _empty() -> dict:
        return {"schema_version": SCHEMA_VERSION, "revision": 0,
                "observations": {}, "candidates": {}, "events": [],
                "compatibility": []}

    @staticmethod
    def migrate(raw: dict) -> dict:
        version = int(raw.get("schema_version", 1))
        if version < 1 or version > SCHEMA_VERSION:
            raise IntegrityError(f"unsupported schema version {version}")
        if version == 1:
            raw.setdefault("compatibility", [])
            raw.setdefault("events", [])
            raw["schema_version"] = 2
            version = 2
        for key, default in (("observations", {}), ("candidates", {}),
                             ("events", []), ("compatibility", [])):
            raw.setdefault(key, default)
        raw.setdefault("revision", 0)
        return raw

    def _load(self) -> dict:
        if not self.state_path.exists():
            return self._empty()
        try:
            envelope = json.loads(self.state_path.read_text())
            payload = envelope["payload"]
            if envelope.get("sha256") != _hash(payload):
                raise IntegrityError("state hash mismatch")
            old_version = payload.get("schema_version", 1)
            migrated = self.migrate(copy.deepcopy(payload))
            if migrated != payload:
                self.state = migrated
                self._save("migration", {"from": old_version,
                                          "to": SCHEMA_VERSION})
            return migrated
        except IntegrityError:
            raise
        except Exception as exc:
            raise IntegrityError(f"invalid state: {exc}") from exc

    def _atomic(self, path: Path, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            os.replace(tmp, path)
            os.chmod(path, 0o600)
            dfd = os.open(path.parent, os.O_RDONLY)
            try: os.fsync(dfd)
            finally: os.close(dfd)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    def _audit(self, event: dict) -> None:
        line = _canonical(event) + b"\n"
        fd = os.open(self.audit_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line); os.fsync(fd)
        finally: os.close(fd)
        os.chmod(self.audit_path, 0o600)

    def _save(self, kind: str, detail: dict) -> dict:
        self.state["revision"] += 1
        event = {"id": uuid.uuid4().hex, "at": _now(), "kind": kind,
                 "detail": detail, "revision": self.state["revision"]}
        self.state["events"].append(event)
        payload = self.state
        self._atomic(self.state_path, _canonical(
            {"sha256": _hash(payload), "payload": payload}))
        snap = self.snapshots / f"{self.state['revision']:08d}-{kind}.json"
        self._atomic(snap, _canonical({"sha256": _hash(payload), "payload": payload}))
        self._audit(event)
        return event

    def observe(self, text: str, evidence: object = None,
                observation_only: bool = True, **extra) -> dict:
        if not text.strip(): raise ValueError("observation text is required")
        item = {"id": uuid.uuid4().hex, "created_at": _now(), "text": text,
                "evidence": evidence, "observation_only": observation_only, **extra}
        self.state["observations"][item["id"]] = item
        self._save("observation", {"id": item["id"]})
        return item

    def list(self, kind: str = "candidates") -> list[dict]:
        if kind not in ("candidates", "observations", "events", "compatibility"):
            raise ValueError("kind must be candidates/observations/events/compatibility")
        value = self.state[kind]
        return list(value.values()) if isinstance(value, dict) else list(value)

    def get(self, item_id: str) -> dict:
        for key in ("candidates", "observations"):
            if item_id in self.state[key]: return self.state[key][item_id]
        raise KeyError(item_id)

    def classify(self, observation_id: str = "", text: str = "",
                 evidence: object = None) -> dict:
        if observation_id:
            observation = self.get(observation_id)
            text, evidence = observation["text"], observation.get("evidence")
        value = text.lower()
        # Deterministic precedence: executable/source symptoms beat prose hints.
        if re.search(r"\b(traceback|exception|bug|source|function|class|api|\.py|\.ts|\.rs)\b", value):
            kind = "code"
        elif re.search(r"\b(eval|benchmark|metric|threshold|holdout|regression|test gate)\b", value):
            kind = "harness"
        elif re.search(r"\b(how to|steps|workflow|procedure|recipe|repeat)\b", value):
            kind = "skill"
        else:
            kind = "memory"
        candidate = {"id": uuid.uuid4().hex, "created_at": _now(),
                     "observation_id": observation_id or None, "text": text,
                     "evidence": evidence, "kind": kind, "status": "classified",
                     "build": None, "evaluations": [], "promotion": None}
        self.state["candidates"][candidate["id"]] = candidate
        self._save("classification", {"id": candidate["id"], "kind": kind})
        return candidate

    def build(self, candidate_id: str, repo: str, goal: str = "",
              test_cmd: str = "", context: str = "") -> dict:
        candidate = self.get(candidate_id)
        if candidate["kind"] != "code":
            raise ValueError("build is only valid for code candidates")
        path = Path(repo).expanduser().resolve()
        dirty = _run(path, "git", "status", "--porcelain")
        if dirty.returncode or dirty.stdout.strip():
            raise ValueError("source repo is dirty; only observation-only actions are allowed")
        baseline = _run(path, "git", "rev-parse", "HEAD")
        if baseline.returncode: raise ValueError("source repo is not a git repository")
        if not self.coder: raise RuntimeError("coder primitive unavailable")
        result = self.coder({"goal": goal or candidate["text"], "repo": str(path),
                             "test_cmd": test_cmd, "context": context})
        diff = result.get("diff") or ""
        paths = _changed_paths(diff, result.get("untracked") or "")
        blocked = protected_paths(paths)
        build = {"at": _now(), "repo": str(path), "baseline_commit": baseline.stdout.strip(),
                 "worktree": result.get("worktree"), "diff": diff,
                 "diff_hash": hashlib.sha256(diff.encode()).hexdigest(),
                 "paths": paths, "protected_paths": blocked,
                 "test_cmd": result.get("test_cmd") or test_cmd or None,
                 "test_output": result.get("test_output"),
                 "tests_ok": result.get("tests_ok"), "coder_ok": bool(result.get("ok")),
                 "review": result.get("review"), "error": result.get("error")}
        candidate["build"] = build
        candidate["status"] = "built" if result.get("ok") and not blocked else "build_failed"
        self._save("build", {"id": candidate_id, "status": candidate["status"],
                             "diff_hash": build["diff_hash"], "protected_paths": blocked})
        return build

    def evaluate(self, candidate_id: str, manifest: dict, canary_command: str = "") -> dict:
        candidate, build = self.get(candidate_id), self.get(candidate_id).get("build")
        if not build: raise ValueError("candidate has no build")
        supplied_hash = manifest.get("sha256")
        body = {k: v for k, v in manifest.items() if k != "sha256"}
        manifest_hash = _hash(body)
        if supplied_hash and supplied_hash != manifest_hash:
            raise IntegrityError("evaluation manifest hash mismatch")
        thresholds = body.get("thresholds") or {}
        baseline = body.get("baseline") or {}
        metrics = body.get("candidate") or {}
        if not isinstance(thresholds, dict) or not isinstance(baseline, dict) or not isinstance(metrics, dict):
            raise ValueError("manifest thresholds/baseline/candidate must be objects")
        regressions = {}
        for name, limit in thresholds.get("max_regression", {}).items():
            if name not in baseline or name not in metrics:
                raise ValueError(f"evaluation metric {name!r} missing from baseline or candidate")
            regressions[name] = float(baseline[name]) - float(metrics[name]) > float(limit)
        max_lines = int(thresholds.get("max_diff_lines", 500))
        review = str(build.get("review") or "LGTM")
        review_ok = review.strip().upper().startswith("LGTM")
        canary = None
        if canary_command:
            worktree = Path(str(build.get("worktree") or "")).expanduser().resolve()
            if not worktree.is_dir():
                raise ValueError("candidate worktree is unavailable")
            isolated = Path(tempfile.mkdtemp(prefix="rlm-improve-canary-"))
            env = {k: v for k, v in os.environ.items()
                   if not any(word in k.upper() for word in
                              ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY",
                               "PRIVATE_KEY", "CREDENTIAL", "AUTH", "COOKIE"))}
            env["HERMES_HOME"] = str(isolated)
            proc = subprocess.run(["/bin/sh", "-c", canary_command],
                                  cwd=worktree, env=env,
                                  capture_output=True, text=True, timeout=120)
            canary = {"command": canary_command, "exit_code": proc.returncode,
                      "output": (proc.stdout + proc.stderr)[-4000:],
                      "hermes_home": str(isolated), "messaging_tokens_removed": True}
        gates = {"manifest": True, "tests": build.get("tests_ok") is True,
                 "coder": build.get("coder_ok") is True,
                 "review": review_ok, "protected_paths": not build["protected_paths"],
                 "diff_size": len(build["diff"].splitlines()) <= max_lines,
                 "regressions": not any(regressions.values()),
                 "canary": canary is None or canary["exit_code"] == 0}
        result = {"id": uuid.uuid4().hex, "at": _now(), "manifest": body,
                  "manifest_hash": manifest_hash, "baseline": baseline,
                  "candidate_metrics": metrics, "regressions": regressions,
                  "gates": gates, "canary": canary, "passed": all(gates.values())}
        candidate["evaluations"].append(result)
        candidate["status"] = "evaluated_pass" if result["passed"] else "evaluated_fail"
        self._save("evaluation", {"candidate_id": candidate_id,
                                  "evaluation_id": result["id"], "passed": result["passed"]})
        return result

    def promote(self, candidate_id: str, automatic: bool = False) -> dict:
        candidate = self.get(candidate_id)
        evaluation = candidate["evaluations"][-1] if candidate["evaluations"] else None
        if not evaluation or not evaluation["passed"]: raise ValueError("all evaluation gates must pass")
        build = candidate["build"]
        repo, wt, base = build["repo"], build["worktree"], build["baseline_commit"]
        commands = [f"git -C {json.dumps(repo)} fetch --all --prune",
                    f"git -C {json.dumps(repo)} worktree add --detach <review-worktree> {base}",
                    f"git -C <review-worktree> apply --check <candidate.diff>",
                    f"git -C <review-worktree> apply <candidate.diff>",
                    f"git -C <review-worktree> {build.get('test_cmd') or '<run deterministic tests>'}",
                    "# review, commit on a topic branch, then open/merge manually"]
        paths = build["paths"]
        allowlisted = bool(paths) and all(
            (Path(p).suffix in AUTO_EXTENSIONS and p.startswith(AUTO_DIRS)) or
            p.startswith("tests/") for p in paths)
        auto_approved = bool(automatic and allowlisted and not build["protected_paths"])
        plan = {"at": _now(), "mode": "automatic-approval" if auto_approved else "manual",
                "approved": auto_approved, "commands": commands, "source_worktree": wt,
                "diff_hash": build["diff_hash"],
                "note": "approval record only; no merge, push, publish, restart, or active-plugin edit performed"}
        candidate["promotion"], candidate["status"] = plan, "promotion_approved" if auto_approved else "promotion_planned"
        self._save("promotion", {"id": candidate_id, "approved": auto_approved})
        return plan

    def reject(self, candidate_id: str, reason: str = "") -> dict:
        candidate = self.get(candidate_id); candidate["status"] = "rejected"
        candidate["rejection_reason"] = reason
        self._save("rejection", {"id": candidate_id, "reason": reason})
        return candidate

    def rollback(self, candidate_id: str, commit: str = "", reason: str = "") -> dict:
        candidate = self.get(candidate_id)
        snapshot = self.snapshots / f"rollback-{uuid.uuid4().hex}.json"
        self._atomic(snapshot, _canonical({"sha256": _hash(self.state), "payload": self.state}))
        repo = (candidate.get("build") or {}).get("repo", "<repo>")
        target = commit or (candidate.get("build") or {}).get("baseline_commit", "<known-good-commit>")
        record = {"at": _now(), "commit": target, "state_snapshot": str(snapshot),
                  "reason": reason,
                  "commands": [f"git -C {json.dumps(repo)} checkout -b rollback/{candidate_id[:8]} {target}",
                               "# test and review this rollback branch; do not reset main"]}
        candidate["rollback"] = record; candidate["status"] = "rollback_planned"
        self._save("rollback", {"id": candidate_id, "commit": target,
                                "state_snapshot": str(snapshot)})
        return record

    def export_state(self, destination: str) -> dict:
        dest = Path(destination).expanduser().resolve()
        if dest == self.state_path or self.root in dest.parents:
            raise ValueError("export destination must be outside improvement state")
        self._save("export", {"path": str(dest)})
        self.verify()
        manifest = {"schema_version": SCHEMA_VERSION, "created_at": _now(),
                    "state_sha256": hashlib.sha256(self.state_path.read_bytes()).hexdigest(),
                    "audit_sha256": hashlib.sha256(self.audit_path.read_bytes()).hexdigest()
                    if self.audit_path.exists() else None}
        with tarfile.open(dest, "w:gz") as archive:
            archive.add(self.state_path, arcname="state.json")
            if self.audit_path.exists(): archive.add(self.audit_path, arcname="audit.jsonl")
            data = _canonical(manifest)
            info = tarfile.TarInfo("manifest.json"); info.size = len(data); info.mode = 0o600
            archive.addfile(info, io.BytesIO(data))
        os.chmod(dest, 0o600)
        return {"path": str(dest), "manifest": manifest}

    def import_state(self, source: str) -> dict:
        source_path = Path(source).expanduser().resolve()
        with tempfile.TemporaryDirectory(prefix="rlm-improve-import-") as tmp:
            with tarfile.open(source_path, "r:gz") as archive:
                members = archive.getmembers()
                names = {member.name for member in members}
                if not names <= {"state.json", "audit.jsonl", "manifest.json"} or "state.json" not in names:
                    raise IntegrityError("unsafe or incomplete import archive")
                if "manifest.json" not in names or any(not member.isfile() for member in members):
                    raise IntegrityError("unsafe or incomplete import archive")
                blobs = {}
                for member in members:
                    stream = archive.extractfile(member)
                    if stream is None: raise IntegrityError("unreadable import member")
                    blobs[member.name] = stream.read()
            manifest = json.loads(blobs["manifest.json"])
            state_bytes = blobs["state.json"]
            if hashlib.sha256(state_bytes).hexdigest() != manifest["state_sha256"]:
                raise IntegrityError("import state hash mismatch")
            envelope = json.loads(state_bytes); payload = envelope["payload"]
            if envelope.get("sha256") != _hash(payload): raise IntegrityError("import payload hash mismatch")
            audit_bytes = blobs.get("audit.jsonl")
            expected_audit = manifest.get("audit_sha256")
            if expected_audit != (hashlib.sha256(audit_bytes).hexdigest() if audit_bytes is not None else None):
                raise IntegrityError("import audit hash mismatch")
            if audit_bytes is not None:
                for line in audit_bytes.splitlines(): json.loads(line)
            imported = self.migrate(payload)
            self.state = imported
            if audit_bytes is not None: self._atomic(self.audit_path, audit_bytes)
            elif self.audit_path.exists(): self.audit_path.unlink()
            self._save("import", {"source": str(source_path)})
        return {"imported": True, "revision": self.state["revision"]}

    def verify(self) -> dict:
        envelope = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "payload": self.state, "sha256": _hash(self.state)}
        if envelope.get("sha256") != _hash(envelope.get("payload")):
            raise IntegrityError("state hash mismatch")
        audit_count = 0
        if self.audit_path.exists():
            for line in self.audit_path.read_text().splitlines():
                json.loads(line); audit_count += 1
        return {"ok": True, "schema_version": self.state["schema_version"],
                "revision": self.state["revision"], "state_hash": envelope["sha256"],
                "audit_records": audit_count, "root": str(self.root)}

    def compatibility(self, hermes_version: str, plugin_version: str = PLUGIN_VERSION,
                      import_ok: bool = True, tool_registered: bool = True,
                      detail: str = "") -> dict:
        record = {"at": _now(), "hermes_version": hermes_version,
                  "plugin_version": plugin_version, "import_ok": bool(import_ok),
                  "tool_registered": bool(tool_registered), "detail": detail,
                  "compatible": bool(import_ok and tool_registered)}
        self.state["compatibility"].append(record)
        self._save("compatibility", record)
        return record

    def dispatch(self, action: str, **args) -> object:
        aliases = {"export": "export_state", "import": "import_state"}
        name = aliases.get(action, action)
        if name == "list": return self.list(args.get("kind", "candidates"))
        if name == "get": return self.get(args.get("id", ""))
        method = getattr(self, name, None)
        if not method or name.startswith("_"): raise ValueError(f"unknown action {action}")
        return method(**args)


__all__ = ["ImprovementController", "IntegrityError", "protected_paths",
           "SCHEMA_VERSION", "PLUGIN_VERSION"]
