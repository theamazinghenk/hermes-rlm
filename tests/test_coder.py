"""coder(): disciplined repo delegation — worktree, test gate, retry, diff.

Uses a scripted fake worker via HERMES_RLM_CODER_CMD, so no live model runs.

Run: ~/.hermes/hermes-agent/venv/bin/python tests/test_coder.py
"""
from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

spec = importlib.util.spec_from_file_location("hermes_rlm_plugin", HERE / "__init__.py")
plugin = importlib.util.module_from_spec(spec)
sys.modules["hermes_rlm_plugin"] = plugin
spec.loader.exec_module(plugin)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# --- fixture: a real git repo + a scripted worker ----------------------------

tmp = Path(tempfile.mkdtemp(prefix="rlm_coder_test_"))
repo = tmp / "repo"
repo.mkdir()
subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c",
                "user.name=t", "commit", "-q", "--allow-empty", "-m", "root"],
               check=True)
(repo / "app.py").write_text("VALUE = 1\n")
subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c",
                "user.name=t", "commit", "-q", "-m", "init"], check=True)

# Worker: first run writes VALUE = 2 (tests want 3); second run fixes it.
worker = tmp / "worker.sh"
worker.write_text("""#!/bin/sh
WD="$1"
if [ -f "$WD/.attempt2" ]; then
  printf 'VALUE = 3\\n' > "$WD/app.py"
else
  touch "$WD/.attempt2"
  printf 'VALUE = 2\\n' > "$WD/app.py"
fi
""")
worker.chmod(worker.stat().st_mode | stat.S_IEXEC)

os.environ["HERMES_RLM_CODER_CMD"] = f"{worker} {{workdir}} {{prompt}}"
os.environ["HERMES_RLM_CODER_REVIEW"] = "0"  # review needs a live model; off here

# --- 1. happy path with retry: fail once, pass on attempt 2 ------------------

r = plugin._run_coder({"goal": "zet VALUE op 3", "repo": str(repo),
                       "test_cmd": "grep -q 'VALUE = 3' app.py"})
check("test gate passes after retry", r.get("ok") is True, str(r)[:150])
check("two attempts recorded", r.get("attempts") == 2, str(r.get("attempts")))
check("diff shows the change", "VALUE = 3" in r.get("diff", ""), r.get("diff", "")[:80])
check("worktree reported and kept", Path(r.get("worktree", "/nope")).is_dir())
check("original repo untouched", (repo / "app.py").read_text() == "VALUE = 1\n")
check("note explains merge policy", "orchestrator" in r.get("note", ""))
subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force",
                r["worktree"]], capture_output=True)

# --- 2. persistent test failure is honest ------------------------------------

r = plugin._run_coder({"goal": "onmogelijk", "repo": str(repo),
                       "test_cmd": "false"})
check("impossible test reports ok=False", r.get("ok") is False)
check("bounded attempts", r.get("attempts") == plugin.CODER_RETRIES + 1)
subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force",
                r.get("worktree", "")], capture_output=True)

# --- 3. guards ---------------------------------------------------------------

r = plugin._run_coder({"goal": "x", "repo": str(tmp / "geen-repo")})
check("non-repo refused", not r.get("ok") and "git repo" in r.get("error", ""))
r = plugin._run_coder({"goal": "", "repo": str(repo)})
check("empty goal refused", not r.get("ok"))

os.environ[plugin._DEPTH_ENV] = str(plugin.MAX_RLM_DEPTH)
r = plugin._run_coder({"goal": "x", "repo": str(repo)})
check("depth limit enforced", not r.get("ok") and "depth" in r.get("error", ""))
os.environ.pop(plugin._DEPTH_ENV, None)

plugin.FEATURE_CODER = False
r = plugin._run_coder({"goal": "x", "repo": str(repo)})
check("feature flag disables coder", not r.get("ok") and "disabled" in r.get("error", ""))
plugin.FEATURE_CODER = True

# --- 4. swarm: parallel independent tasks in separate worktrees --------------

plugin.FEATURE_CODER = True
os.environ.pop(plugin._DEPTH_ENV, None)
# Two independent repos so the parallel workers never share edit scope.
repo2 = tmp / "repo2"
repo2.mkdir()
subprocess.run(["git", "-C", str(repo2), "init", "-q"], check=True)
(repo2 / "app.py").write_text("VALUE = 1\n")
subprocess.run(["git", "-C", str(repo2), "add", "-A"], check=True)
subprocess.run(["git", "-C", str(repo2), "-c", "user.email=t@t", "-c",
                "user.name=t", "commit", "-q", "-m", "init"], check=True)

batch = plugin._handle_delegate({"mode": "coder", "tasks": [
    {"goal": "a", "repo": str(repo), "test_cmd": "grep -q 'VALUE = 3' app.py"},
    {"goal": "b", "repo": str(repo2), "test_cmd": "grep -q 'VALUE = 3' app.py"},
]})
check("swarm returns per-task results", len(batch.get("results", [])) == 2, str(batch)[:120])
check("swarm ran in parallel", batch.get("parallel_workers") == 2)
check("both worktrees isolated from origins",
      (repo / "app.py").read_text() == "VALUE = 1\n"
      and (repo2 / "app.py").read_text() == "VALUE = 1\n")
for res in batch.get("results", []):
    wt = res.get("worktree")
    if wt:
        origin = str(repo) if "repo2" not in wt else str(repo2)
        subprocess.run(["git", "-C", origin, "worktree", "remove", "--force", wt],
                       capture_output=True)

import shutil
shutil.rmtree(tmp, ignore_errors=True)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} — {FAILURES}")
    sys.exit(1)
print("ok")
