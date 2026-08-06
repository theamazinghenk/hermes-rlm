"""Self-check for the hermes-rlm persistent kernel.

Runs the kernel process directly over its stdin/stdout JSON protocol —
no Hermes gateway, no tool bridge, no network. Asserts the property that
justifies the whole plugin: state survives between calls.

    python3 tests/test_kernel.py   ->  prints "ok" and exits 0
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "kernel_server.py"


class KernelClient:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

    def send(self, **req) -> dict:
        req.setdefault("id", "t")
        req.setdefault("op", "exec")
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=5)


k = KernelClient()
try:
    # 1. Liveness.
    assert k.send(op="ping")["value"] == "pong"

    # 2. THE point of the plugin: state survives across separate calls.
    k.send(code="dataset = list(range(10_000))")
    r = k.send(code="len(dataset)")
    assert r["ok"] and r["value"] == "10000", r

    # 3. Imports persist too.
    k.send(code="import json as _j")
    r = k.send(code="_j.dumps({'a': 1})")
    assert r["value"] == "'{\"a\": 1}'", r

    # 4. Trailing expression is returned; stdout is captured separately.
    r = k.send(code="print('side effect')\n21 * 2")
    assert r["stdout"].strip() == "side effect", r
    assert r["value"] == "42", r

    # 5. Errors are reported without killing the kernel, and state is intact.
    r = k.send(code="1 / 0")
    assert not r["ok"] and "ZeroDivisionError" in r["error"], r
    r = k.send(code="len(dataset)")
    assert r["ok"] and r["value"] == "10000", "kernel lost state after an error"

    # 6. Syntax errors are handled the same way.
    r = k.send(code="def broken(")
    assert not r["ok"] and "SyntaxError" in r["error"], r

    # 7. `vars` summarises without dumping the object itself.
    r = k.send(op="vars", code="")
    entries = {v["name"]: v for v in json.loads(r["value"])}
    assert entries["dataset"]["type"] == "list", entries
    assert entries["dataset"]["len"] == 10_000, entries
    assert len(entries["dataset"]["preview"]) <= 200, "preview must stay small"

    # 8. Large stdout is truncated, not streamed whole into context.
    r = k.send(code="print('x' * 200_000)")
    assert len(r["stdout"]) < 60_000, len(r["stdout"])
    assert "omitted" in r["stdout"]

    print("ok")
finally:
    k.close()
