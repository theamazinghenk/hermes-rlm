"""End-to-end check: load the plugin the way Hermes does and drive the tools.

Unlike the kernel/bridge unit checks, this exercises the real handlers,
the real Unix-socket RPC server and a real dispatch through Hermes'
`handle_function_call`. Proves the persistent-state property survives the
full stack, not just the kernel subprocess.

    python3 tests/test_integration.py   ->  prints "ok" and exits 0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERMES = Path.home() / ".hermes" / "hermes-agent"
sys.path.insert(0, str(HERMES))
sys.path.insert(0, str(ROOT.parent))

import importlib.util

spec = importlib.util.spec_from_file_location("hermes_rlm_plugin", ROOT / "__init__.py")
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)

SESSION = "integration-test"


def call(handler, args=None):
    return json.loads(handler(args or {}, task_id=SESSION))


try:
    # 1. First exec boots the kernel and preloads the bridge.
    r = call(plugin._exec_handler, {"code": "total = sum(range(1_000_000))\ntotal"})
    assert r["ok"], r
    assert r["value"] == "499999500000", r

    # 2. A SEPARATE tool call still sees the variable — the whole point.
    r = call(plugin._exec_handler, {"code": "total // 1_000_000"})
    assert r["ok"] and r["value"] == "499999", r

    # 2b. A successful reply carries no diagnostic envelope: those bytes are
    #     conversation context on every call and would erase the saving.
    assert "kernel" not in r, r
    assert set(r) <= {"ok", "value", "stdout", "stderr", "duration"}, r

    # 2c. Failures DO carry diagnostics, where they actually help.
    bad = call(plugin._exec_handler, {"code": "1 / 0"})
    assert not bad["ok"] and "kernel" in bad, bad
    assert bad["kernel"]["exec_count"] >= 3, bad

    # 3. Hermes tools dispatch for real through handle_function_call.
    r = call(plugin._exec_handler,
             {"code": "res = terminal('echo RLM_TOOLBRIDGE_OK')\n"
                      "'RLM_TOOLBRIDGE_OK' in str(res)"})
    assert r["ok"] and r["value"] == "True", r

    # 4. rlm_vars summarises state without dumping it into context.
    r = call(plugin._vars_handler)
    names = {v["name"] for v in r["variables"]}
    assert {"total", "res"} <= names, names
    assert len(json.dumps(r)) < 20_000, "vars output must stay compact"

    # 5. Tools outside the allow-list are refused at the parent.
    r = call(plugin._exec_handler, {"code": "import rlm_bridge; rlm_bridge._call('image_generate', {})"})
    assert not r["ok"] and "not available" in r["error"], r

    # 5b. delegate_task cannot be reached directly — it is an agent-loop tool.
    #     rlm() must go through the local subagent path instead.
    r = call(plugin._exec_handler,
             {"code": "import rlm_bridge; rlm_bridge._call('delegate_task', {'goal': 'x'})"})
    assert not r["ok"] and "not available" in r["error"], r

    # 5c. Delegation depth is capped so a subagent cannot recurse forever.
    import os as _os
    _prev = _os.environ.get(plugin._DEPTH_ENV)
    _os.environ[plugin._DEPTH_ENV] = str(plugin.MAX_RLM_DEPTH)
    try:
        capped = plugin._run_subagent("should never run")
        assert not capped["ok"] and "depth limit" in capped["error"], capped
    finally:
        if _prev is None:
            _os.environ.pop(plugin._DEPTH_ENV, None)
        else:
            _os.environ[plugin._DEPTH_ENV] = _prev

    # 5d. An empty goal is rejected before any process is spawned.
    assert not plugin._run_subagent("")["ok"]

    # 5e. The CLI the subagent would launch actually exists.
    assert Path(plugin._hermes_cli()).is_file()

    # 6. Reset destroys state; a later call starts from a clean namespace.
    call(plugin._reset_handler)
    r = call(plugin._exec_handler, {"code": "'total' in dir()"})
    assert r["ok"] and r["value"] == "False", r

    print("ok")
finally:
    try:
        call(plugin._reset_handler)
    except Exception:
        pass
