"""Warm subagent lane: resident leaf gateway with CLI fallback.

Uses a mock OpenAI-compatible server — no Hermes install needed.

Run: ~/.hermes/hermes-agent/venv/bin/python tests/test_warm.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
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


class MockGateway(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MockGateway.last_request = {"path": self.path,
                                    "auth": self.headers.get("Authorization"),
                                    "body": body}
        reply = json.dumps({"choices": [{"message": {
            "role": "assistant", "content": "WARM_ANSWER"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *a):  # noqa: D102 - silence
        pass


server = HTTPServer(("127.0.0.1", 0), MockGateway)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()

# --- 1. warm lane answers without any CLI spawn ------------------------------

os.environ["HERMES_RLM_WARM_URL"] = f"http://127.0.0.1:{port}"
os.environ["HERMES_RLM_FAST_API_KEY"] = "x"  # unused; keep fast path quiet
os.environ["API_SERVER_KEY"] = "testkey-1234567890"

plugin._hermes_cli_calls = 0
real_cli = plugin._hermes_cli
def _counting_cli():
    plugin._hermes_cli_calls += 1
    return "/bin/echo"
plugin._hermes_cli = _counting_cli

r = plugin._run_subagent("zeg iets terug", "wat context")
check("warm lane answers", r.get("ok") and r.get("summary") == "WARM_ANSWER", str(r)[:100])
check("warm result is marked", r.get("warm") is True)
check("no CLI spawn used", plugin._hermes_cli_calls == 0)
req = MockGateway.last_request
check("hits chat completions", req["path"] == "/v1/chat/completions")
check("sends bearer auth", str(req["auth"]).startswith("Bearer "))
check("goal and context in prompt",
      "zeg iets terug" in req["body"]["messages"][0]["content"]
      and "wat context" in req["body"]["messages"][0]["content"])

# --- 2. unreachable warm gateway falls back to the CLI -----------------------

os.environ["HERMES_RLM_WARM_URL"] = "http://127.0.0.1:1"  # nothing listens
r = plugin._run_subagent("fallback test", "")
check("dead warm gateway falls back to CLI", plugin._hermes_cli_calls > 0, str(r)[:80])

# --- 3. unset knob means: warm lane fully off --------------------------------

os.environ["HERMES_RLM_WARM_URL"] = ""
before = plugin._hermes_cli_calls
r = plugin._run_subagent("no warm configured", "")
check("unset knob skips warm lane", plugin._hermes_cli_calls == before + 1)

plugin._hermes_cli = real_cli
server.shutdown()

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} — {FAILURES}")
    sys.exit(1)
print("ok")
