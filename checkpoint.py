"""Checkpoint / restore for the RLM kernel.

The kernel's weakness is that state lives only in one process: a crash, a
timeout kill, or a machine restart loses everything. Prime Agent's RLM model
explicitly assumes state outlives a turn; PR #46428's disk corpus survives a
restart. This closes that gap without giving up the live-object model.

Approach: pickle what can be pickled, name what cannot. A checkpoint is
deliberately partial and says so — silently dropping half a namespace and
calling it a restore would be worse than not having the feature.

Checkpoints live under ~/.hermes/cache/rlm_checkpoints/<session>/<name>.pkl
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

CHECKPOINT_ROOT = Path.home() / ".hermes" / "cache" / "rlm_checkpoints"

# Never worth persisting: modules re-import, and the bridge is re-injected on
# boot with a fresh socket path and token.
_SKIP_NAMES = {
    "read_file", "write_file", "search_files", "patch", "terminal",
    "web_search", "web_extract", "rlm", "rlm_many", "RlmBridgeError",
    "rlm_bridge", "_",
}


def _session_dir(session: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session)[:80]
    return CHECKPOINT_ROOT / (safe or "default")


def save(namespace: dict, session: str, name: str = "latest") -> dict:
    """Persist every picklable binding. Returns a report, never raises."""
    import types

    saved: dict = {}
    skipped: list[dict] = []

    for key, value in namespace.items():
        if key.startswith("__") or key in _SKIP_NAMES:
            continue
        if isinstance(value, types.ModuleType):
            skipped.append({"name": key, "reason": "module (re-import instead)"})
            continue
        try:
            pickle.dumps(value)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"name": key, "type": type(value).__name__,
                            "reason": f"not picklable: {type(exc).__name__}"})
            continue
        saved[key] = value

    target_dir = _session_dir(session)
    target_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(target_dir, 0o700)
    path = target_dir / f"{name}.pkl"

    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as handle:
        pickle.dump(saved, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)  # atomic: a crash mid-write cannot corrupt the last good one

    report = {
        "ok": True,
        "checkpoint": str(path),
        "saved": sorted(saved),
        "saved_count": len(saved),
        "skipped": skipped,
        "size_bytes": path.stat().st_size,
    }

    # Nothing else ever removes these, so bound the store on every write.
    try:
        pruned = prune()
        if pruned["removed"]:
            report["pruned"] = pruned["removed"]
    except Exception:  # noqa: BLE001 - pruning must never fail a save
        pass
    return report


def load(session: str, name: str = "latest", allow_cross_session: bool = True) -> dict:
    """Read a checkpoint. Returns {"ok", "namespace"|"error"}.

    Sessions are keyed by a per-session id, so the session that needs a
    checkpoint most — a NEW one, after the old kernel died with its session —
    would never find it. When the current session has no such checkpoint we
    fall back to the newest one with that name from any session, and report
    where it came from so the borrow is never silent.
    """
    path = _session_dir(session) / f"{name}.pkl"
    borrowed_from = None

    if not path.is_file() and allow_cross_session:
        candidates = [entry for entry in listing() if entry["name"] == name]
        if candidates:
            newest = candidates[0]  # listing() is newest-first
            path = _session_dir(newest["session"]) / f"{name}.pkl"
            borrowed_from = newest["session"]

    if not path.is_file():
        return {"ok": False, "error": f"no checkpoint {name!r} for session {session!r}"}
    try:
        with open(path, "rb") as handle:
            namespace = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"checkpoint unreadable: {exc}"}
    if not isinstance(namespace, dict):
        return {"ok": False, "error": "checkpoint is not a namespace mapping"}

    result = {"ok": True, "namespace": namespace, "path": str(path),
              "restored": sorted(namespace), "restored_count": len(namespace)}
    if borrowed_from:
        result["borrowed_from_session"] = borrowed_from
    return result


def listing(session: str | None = None) -> list[dict]:
    """Available checkpoints, newest first."""
    roots = [_session_dir(session)] if session else (
        sorted(CHECKPOINT_ROOT.iterdir()) if CHECKPOINT_ROOT.is_dir() else []
    )
    found: list[dict] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.glob("*.pkl"):
            stat = path.stat()
            found.append({
                "session": root.name,
                "name": path.stem,
                "size_bytes": stat.st_size,
                "modified": int(stat.st_mtime),
            })
    return sorted(found, key=lambda entry: entry["modified"], reverse=True)


def delete(session: str, name: str = "latest") -> dict:
    path = _session_dir(session) / f"{name}.pkl"
    if not path.is_file():
        return {"ok": False, "error": "checkpoint not found"}
    path.unlink()
    return {"ok": True, "deleted": str(path)}


# Checkpoints are pickles of whole namespaces, so they get large fast and
# nothing else ever removes them. Prune on every save: bounded by age first
# (stale state is worse than none) and then by total size.
MAX_AGE_DAYS = 14
MAX_TOTAL_MB = 500


def prune(max_age_days: int = MAX_AGE_DAYS, max_total_mb: int = MAX_TOTAL_MB) -> dict:
    """Remove stale and excess checkpoints. Best-effort, never raises."""
    import time

    removed: list[str] = []
    freed = 0
    try:
        entries = listing()
    except Exception:  # noqa: BLE001
        return {"removed": [], "freed_bytes": 0}

    cutoff = time.time() - max_age_days * 86_400
    keep: list[dict] = []
    for entry in entries:
        path = _session_dir(entry["session"]) / f"{entry['name']}.pkl"
        if entry["modified"] < cutoff:
            try:
                path.unlink()
                removed.append(f"{entry['session']}/{entry['name']} (stale)")
                freed += entry["size_bytes"]
            except OSError:
                pass
        else:
            keep.append(entry)

    # keep is newest-first, so dropping from the tail removes the oldest.
    budget = max_total_mb * 1024 * 1024
    running = 0
    for entry in keep:
        running += entry["size_bytes"]
        if running > budget:
            path = _session_dir(entry["session"]) / f"{entry['name']}.pkl"
            try:
                path.unlink()
                removed.append(f"{entry['session']}/{entry['name']} (over budget)")
                freed += entry["size_bytes"]
            except OSError:
                pass

    # Drop directories left empty by the pruning above.
    if CHECKPOINT_ROOT.is_dir():
        for directory in CHECKPOINT_ROOT.iterdir():
            if directory.is_dir() and not any(directory.iterdir()):
                try:
                    directory.rmdir()
                except OSError:
                    pass

    return {"removed": removed, "freed_bytes": freed}
