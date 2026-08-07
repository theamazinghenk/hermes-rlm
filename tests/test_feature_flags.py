"""Feature flags: a disabled capability is ABSENT, not hidden.

With HERMES_RLM_ENABLE_{CHECKPOINT,REFINE,SUBAGENTS,PYTHON_SKILLS}=0:
- the tool is not registered at all;
- kernel helpers refuse with an operator message;
- skills paths are not injected into the kernel;
- autosave (a pickle path) becomes a no-op.

This is what makes a minimal pilot (exec/vars/reset only) enforceable.

Run: ~/.hermes/hermes-agent/venv/bin/python tests/test_feature_flags.py
"""
from __future__ import annotations

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


PROBE = f"""
import importlib.util, json, sys
sys.path.insert(0, {str(HERE)!r})
spec = importlib.util.spec_from_file_location("plugin", {str(HERE)!r} + "/__init__.py")
p = importlib.util.module_from_spec(spec); sys.modules["plugin"] = p
spec.loader.exec_module(p)

class Ctx:
    def __init__(self):
        self.tools = []
        self.hooks = []
    def register_tool(self, name, **kw):
        self.tools.append(name)
    def register_hook(self, event, fn):
        self.hooks.append(event)

ctx = Ctx()
p.register(ctx)

# Kernel: helpers must refuse, skills must be absent from PYTHONPATH.
k = p._get_kernel("ff-test")
refused = {{}}
for fn in ("rlm('x')", "rlm_spawn('x')", "rlm_children()", "harness_store()"):
    r = k.execute("try:\\n    " + fn + "\\n    out='ALLOWED'\\nexcept RlmBridgeError as e:\\n    out='refused: ' + str(e)\\nout")
    refused[fn] = r.get("value") or r.get("error") or ""
pythonpath = k.execute("import os; os.environ.get('PYTHONPATH','')").get("value")
autosave = p._autosave(k)
k.shutdown(); p._kernels.pop("ff-test", None)

print(json.dumps({{"tools": ctx.tools, "hooks": ctx.hooks,
                   "refused": refused, "pythonpath": pythonpath,
                   "autosave": autosave}}))
"""

env_off = {
    "PATH": "/usr/bin:/bin",
    "HOME": str(Path.home()),
    "HERMES_HOME": tempfile.mkdtemp(prefix="rlm_ff_home_"),
    "HERMES_RLM_ENABLE_CHECKPOINT": "0",
    "HERMES_RLM_ENABLE_REFINE": "0",
    "HERMES_RLM_ENABLE_SUBAGENTS": "0",
    "HERMES_RLM_ENABLE_PYTHON_SKILLS": "0",
}
out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True,
                     text=True, env=env_off, timeout=120)
try:
    d = json.loads(out.stdout.strip().splitlines()[-1])
except (json.JSONDecodeError, IndexError):
    print(out.stdout, out.stderr)
    sys.exit(1)

check("only exec/vars/reset registered",
      sorted(d["tools"]) == ["rlm_exec", "rlm_reset", "rlm_vars"], str(d["tools"]))
check("no pre_llm_call hook registered", d["hooks"] == [], str(d["hooks"]))
for fn, res in d["refused"].items():
    check(f"kernel refuses {fn}", "refused" in str(res) and "operator" in str(res),
          str(res)[:90])
check("skills paths not injected",
      "skills" not in (d["pythonpath"] or ""), str(d["pythonpath"])[:120])
check("autosave is a no-op when checkpoints are off", d["autosave"] is False)

# --- default mode: everything present ---------------------------------------

env_on = dict(env_off)
for key in list(env_on):
    if key.startswith("HERMES_RLM_ENABLE"):
        env_on.pop(key)
out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True,
                     text=True, env=env_on, timeout=120)
d = json.loads(out.stdout.strip().splitlines()[-1])
check("default registers all seven tools", len(d["tools"]) == 7, str(d["tools"]))
check("default registers the harness hook", d["hooks"] == ["pre_llm_call"])
check("default kernel does not refuse rlm_children",
      "refused" not in str(d["refused"].get("rlm_children()")),
      str(d["refused"].get("rlm_children()"))[:90])

import shutil
shutil.rmtree(env_off["HERMES_HOME"], ignore_errors=True)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} — {FAILURES}")
    sys.exit(1)
print("ok")
