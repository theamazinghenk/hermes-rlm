"""RLM workers ignore dedicated warm routes and inherit the main worker."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("hermes_rlm_plugin", HERE / "__init__.py")
plugin = importlib.util.module_from_spec(spec)
sys.modules["hermes_rlm_plugin"] = plugin
spec.loader.exec_module(plugin)

os.environ["HERMES_RLM_WARM_URL"] = "http://127.0.0.1:1"
os.environ["HERMES_PROFILE"] = "stale-profile"

env = plugin._subagent_env(0)
assert env["HERMES_RLM_DEPTH"] == "1"
assert env["HERMES_DELEGATED_CHILD_CONTEXT"] == "1"
assert "HERMES_PROFILE" not in env

calls = 0
def inherited_cli():
    global calls
    calls += 1
    return "/bin/echo"

real_cli = plugin._hermes_cli
plugin._hermes_cli = inherited_cli
try:
    result = plugin._run_subagent("inherit main", "")
finally:
    plugin._hermes_cli = real_cli

assert calls > 0, result
assert not result.get("warm"), result
assert not result.get("fast_path"), result
print("ok")
