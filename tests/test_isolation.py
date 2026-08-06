"""Fleet-isolation guarantees: HERMES_HOME resolution + no silent cross-session.

Covers the two hard blockers from the 2026-08-06 fleet review:
1. Every state path follows HERMES_HOME — two homes stay provably separate.
2. Checkpoint restore never borrows another session's data unless the caller
   explicitly passes allow_cross_session=True.

Run: ~/.hermes/hermes-agent/venv/bin/python tests/test_isolation.py
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# --- 1. HERMES_HOME resolution in every module (fresh interpreter per home) --

PROBE = (
    "import importlib.util, sys, json\n"
    f"sys.path.insert(0, {str(HERE)!r})\n"
    "paths = {}\n"
    "for mod, attr in [('harness', 'HARNESS_ROOT'), ('checkpoint', 'CHECKPOINT_ROOT'),\n"
    "                  ('kernel_server', 'SPILL_DIR'), ('fast_path', 'HERMES_ENV'),\n"
    "                  ('skill_loader', 'SKILL_ROOTS')]:\n"
    f"    spec = importlib.util.spec_from_file_location(mod, {str(HERE)!r} + '/' + mod + '.py')\n"
    "    m = importlib.util.module_from_spec(spec); sys.modules[mod] = m; spec.loader.exec_module(m)\n"
    "    v = getattr(m, attr)\n"
    "    paths[mod] = [str(x) for x in v] if isinstance(v, list) else str(v)\n"
    "print(json.dumps(paths))\n"
)

homes = [tempfile.mkdtemp(prefix="rlm_homeA_"), tempfile.mkdtemp(prefix="rlm_homeB_")]
results = []
for home in homes:
    out = subprocess.run([sys.executable, "-c", PROBE],
                         capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "HERMES_HOME": home})
    try:
        results.append(json.loads(out.stdout.strip()))
    except json.JSONDecodeError:
        print(out.stdout, out.stderr)
        results.append({})

for mod in ("harness", "checkpoint", "kernel_server", "fast_path", "skill_loader"):
    a, b = json.dumps(results[0].get(mod)), json.dumps(results[1].get(mod))
    check(f"{mod} follows HERMES_HOME", homes[0] in a and homes[1] in b,
          f"A={a} B={b}")
    check(f"{mod} homes are disjoint", a != b)

# --- 2. cross-session restore is opt-in ------------------------------------

spec = importlib.util.spec_from_file_location("checkpoint", HERE / "checkpoint.py")
cp = importlib.util.module_from_spec(spec)
sys.modules["checkpoint_isolation_test"] = cp
spec.loader.exec_module(cp)

import uuid
sess_a = f"iso-a-{uuid.uuid4().hex[:6]}"
sess_b = f"iso-b-{uuid.uuid4().hex[:6]}"
saved = cp.save({"customer_data": [1, 2, 3]}, sess_a, "latest")
check("save works", saved.get("ok") is True, str(saved))

blocked = cp.load(sess_b, "latest")
check("other session CANNOT restore by default", not blocked.get("ok"),
      str(blocked)[:120])

allowed = cp.load(sess_b, "latest", allow_cross_session=True)
check("explicit opt-in CAN restore", allowed.get("ok") is True, str(allowed)[:120])
check("opt-in restore reports the borrow", "session" in json.dumps(allowed).lower())

own = cp.load(sess_a, "latest")
check("own session restores without flag", own.get("ok") is True, str(own)[:120])

cp.delete(sess_a, "latest")
import shutil
for home in homes:
    shutil.rmtree(home, ignore_errors=True)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} — {FAILURES}")
    sys.exit(1)
print("ok")
