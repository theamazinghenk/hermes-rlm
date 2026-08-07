"""Deterministic self-check for the durable improvement controller."""
import hashlib
import io
import json
import os
import stat
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from improvement import ImprovementController, IntegrityError, _canonical, _hash, protected_paths


def envelope(payload):
    return _canonical({"sha256": _hash(payload), "payload": payload})


with tempfile.TemporaryDirectory() as tmp:
    home = Path(tmp) / "home"
    state_dir = home / "state/rlm/improvement"
    state_dir.mkdir(parents=True)
    old = {"schema_version": 1, "revision": 2, "observations": {}, "candidates": {}}
    (state_dir / "state.json").write_bytes(envelope(old))
    controller = ImprovementController(home)
    assert controller.state["schema_version"] == 2
    assert controller.state["revision"] == 3
    assert controller.state["events"][-1]["detail"] == {"from": 1, "to": 2}
    assert stat.S_IMODE(controller.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(controller.state_path.stat().st_mode) == 0o600

    obs = controller.observe("Traceback in api.py", {"ticket": 7})
    candidate = controller.classify(obs["id"])
    assert candidate["kind"] == "code"
    assert protected_paths(["docs/guide.md", "security/policy.py", "foo/evaluator.py", "../escape"])[1:]

    repo = Path(tmp) / "repo"
    repo.mkdir()
    os.system(f"git -C {repo} init -q && git -C {repo} config user.email t@e && git -C {repo} config user.name t && touch {repo}/x && git -C {repo} add x && git -C {repo} commit -qm init")
    wt = Path(tmp) / "worktree"; wt.mkdir()
    calls = []
    def coder(args):
        calls.append(args)
        return {"ok": True, "worktree": str(wt), "diff": "--- a/docs/a.md\n+++ b/docs/a.md\n@@ -0,0 +1 @@\n+x\n", "tests_ok": True, "review": "LGTM"}
    controller.coder = coder
    build = controller.build(candidate["id"], str(repo), test_cmd="python -m test")
    assert build["paths"] == ["docs/a.md"] and calls

    os.environ["MY_PASSWORD"] = "must-not-leak"
    manifest = {"baseline": {"score": 1}, "candidate": {"score": 1.1},
                "thresholds": {"max_regression": {"score": 0.0}}}
    result = controller.evaluate(candidate["id"], manifest,
        "test -z \"$MY_PASSWORD\" && test \"$HERMES_HOME\" != \"" + str(home) + "\"")
    assert result["passed"] and result["canary"]["exit_code"] == 0
    plan = controller.promote(candidate["id"], automatic=True)
    assert plan["approved"] and "no merge, push" in plan["note"]
    assert all(not line.strip().startswith(("git merge", "git push")) for line in plan["commands"])

    try:
        controller.evaluate(candidate["id"], {"baseline": {}, "candidate": {},
            "thresholds": {"max_regression": {"missing": 0}}})
        raise AssertionError("missing metric accepted")
    except ValueError:
        pass

    archive = Path(tmp) / "state.tgz"
    before = controller.state["revision"]
    exported = controller.export_state(str(archive))
    with tarfile.open(archive) as tf:
        imported_payload = json.loads(tf.extractfile("state.json").read())["payload"]
    assert imported_payload["revision"] == before + 1
    restored = ImprovementController(Path(tmp) / "restored")
    restored.import_state(str(archive))
    assert restored.verify()["ok"]

    evil = Path(tmp) / "evil.tgz"
    with tarfile.open(evil, "w:gz") as tf:
        info = tarfile.TarInfo("state.json"); info.type = tarfile.SYMTYPE; info.linkname = "../x"; tf.addfile(info)
        data = b"{}"; info = tarfile.TarInfo("manifest.json"); info.size = len(data); tf.addfile(info, io.BytesIO(data))
    try:
        restored.import_state(str(evil))
        raise AssertionError("unsafe tar accepted")
    except IntegrityError:
        pass

    # Survives a plugin reinstall: only $HERMES_HOME state matters.
    reopened = ImprovementController(home)
    assert reopened.get(candidate["id"])["status"] == "promotion_approved"
    assert reopened.verify()["revision"] == controller.state["revision"]
    assert reopened.compatibility("v0.20.0")["compatible"]

    # A dirty source repo may not be built from.
    (repo / "scratch").write_text("dirty")
    try:
        reopened.build(reopened.classify(text="Traceback in api.py")["id"], str(repo))
        raise AssertionError("dirty repo accepted")
    except ValueError:
        pass
    (repo / "scratch").unlink()

    # Protected paths block promotion even when the coder reports success.
    blocked_candidate = reopened.classify(text="Traceback in api.py")
    reopened.coder = lambda args: {"ok": True, "worktree": str(wt), "tests_ok": True,
                                   "review": "LGTM",
                                   "diff": "--- a/security/policy.py\n+++ b/security/policy.py\n@@ -0,0 +1 @@\n+x\n"}
    blocked_build = reopened.build(blocked_candidate["id"], str(repo))
    assert blocked_build["protected_paths"] == ["security/policy.py"]
    assert not reopened.evaluate(blocked_candidate["id"],
        {"baseline": {}, "candidate": {}, "thresholds": {}})["gates"]["protected_paths"]
    try:
        reopened.promote(blocked_candidate["id"])
        raise AssertionError("protected-path candidate promoted")
    except ValueError:
        pass

    # Corrupted state fails closed rather than silently resetting history.
    corrupt = json.loads(reopened.state_path.read_text())
    corrupt["payload"]["revision"] = 999999
    reopened.state_path.write_text(json.dumps(corrupt))
    try:
        ImprovementController(home)
        raise AssertionError("corrupt state accepted")
    except IntegrityError:
        pass

with tempfile.TemporaryDirectory() as tmp:
    # Auto-approval covers prose only: a candidate may never auto-approve a
    # change to its own tests, or it can "improve" by deleting assertions.
    home = Path(tmp) / "home"
    ctl = ImprovementController(home)
    repo = Path(tmp) / "repo"; repo.mkdir()
    os.system(f"git -C {repo} init -q && git -C {repo} config user.email t@e && "
              f"git -C {repo} config user.name t && touch {repo}/x && "
              f"git -C {repo} add x && git -C {repo} commit -qm init")
    wt = Path(tmp) / "wt"; wt.mkdir()

    def build_for(path: str):
        cand = ctl.classify(text="Traceback in api.py")
        ctl.coder = lambda args, p=path: {
            "ok": True, "worktree": str(wt), "tests_ok": True, "review": "LGTM",
            "diff": f"--- a/{p}\n+++ b/{p}\n@@ -0,0 +1 @@\n+x\n"}
        ctl.build(cand["id"], str(repo))
        ctl.evaluate(cand["id"], {"baseline": {}, "candidate": {}, "thresholds": {}})
        return ctl.promote(cand["id"], automatic=True)

    assert build_for("docs/guide.md")["approved"], "prose should auto-approve"
    assert not build_for("tests/test_kernel.py")["approved"], "tests must not auto-approve"
    assert not build_for("kernel_server.py")["approved"], "source must not auto-approve"

print("ok")
