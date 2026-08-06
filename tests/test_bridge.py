"""Self-check for the hermes-rlm tool bridge (RLM invariant 2).

Starts a fake parent RPC server on a Unix socket, boots the kernel with the
bridge wired to it, and asserts that model-written Python can call Hermes
tools — including `rlm(...)` for subagents — as ordinary functions.

No Hermes gateway and no real subagents are involved: the fake server
records what was dispatched and replies with canned results.

    python3 tests/test_bridge.py   ->  prints "ok" and exits 0
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "kernel_server.py"

TOKEN = secrets.token_urlsafe(16)
tmpdir = tempfile.mkdtemp(prefix="rlm_bridge_test_")
sock_path = os.path.join(tmpdir, "rpc.sock")
dispatched: list[dict] = []
ALLOWED = {"read_file", "terminal", "__rlm_delegate__"}


def serve(server_sock: socket.socket, stop: threading.Event) -> None:
    server_sock.settimeout(0.3)
    while not stop.is_set():
        try:
            conn, _ = server_sock.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            buf = b""
            while b"\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
            if not buf:
                continue
            req = json.loads(buf.split(b"\n", 1)[0].decode())

            if req.get("token") != TOKEN:
                conn.sendall(b'{"__rlm_error__": "unauthorized"}\n')
                continue
            tool = req.get("tool")
            if tool not in ALLOWED:
                conn.sendall(
                    (json.dumps({"__rlm_error__": f"tool {tool!r} not available"}) + "\n").encode()
                )
                continue

            dispatched.append(req)
            if tool == "__rlm_delegate__":
                args = req.get("args", {})
                if "tasks" in args:
                    reply = {"results": [{"goal": t["goal"], "summary": "done", "ok": True}
                                         for t in args["tasks"]]}
                else:
                    reply = {"ok": True, "summary": f"handled: {args.get('goal')}"}
            else:
                reply = {"echo": req.get("args")}
            conn.sendall((json.dumps(reply) + "\n").encode())
        finally:
            conn.close()


server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server_sock.bind(sock_path)
server_sock.listen(8)
stop = threading.Event()
threading.Thread(target=serve, args=(server_sock, stop), daemon=True).start()

env = dict(os.environ)
env["HERMES_RLM_RPC_SOCKET"] = sock_path
env["HERMES_RLM_RPC_TOKEN"] = TOKEN
env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
env["PYTHONUNBUFFERED"] = "1"

proc = subprocess.Popen(
    [sys.executable, str(SERVER)],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    env=env, text=True, bufsize=1,
)


def run(code: str) -> dict:
    proc.stdin.write(json.dumps({"id": "t", "op": "exec", "code": code}) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


try:
    r = run("from rlm_bridge import *")
    assert r["ok"], r

    # 1. A Hermes tool is callable as a plain function.
    r = run("terminal('echo hi')['echo']['command']")
    assert r["value"] == "'echo hi'", r

    # 2. rlm(...) — delegation as a function call, the RLM primitive.
    r = run("rlm('review the auth flow', context='repo x')['summary']")
    assert r["value"] == "'handled: review the auth flow'", r
    call = dispatched[-1]
    assert call["tool"] == "__rlm_delegate__"
    assert call["args"]["context"] == "repo x", call

    # 3. rlm_many(...) — parallel batch in one call.
    r = run("len(rlm_many([{'goal': 'a'}, {'goal': 'b'}])['results'])")
    assert r["value"] == "2", r
    assert dispatched[-1]["args"]["tasks"][1]["goal"] == "b"

    # 4. Results are ordinary Python objects, so they compose in the kernel.
    run("findings = [rlm(f'check {n}')['summary'] for n in ('x', 'y')]")
    r = run("len(findings)")
    assert r["value"] == "2", r

    # 5. Tools outside the allow-list are refused by the parent, not the kernel.
    r = run("write_file('/tmp/nope', 'x')")
    assert not r["ok"] and "not available" in r["error"], r

    # 6. A bad token cannot be forged from inside the kernel.
    r = run("import rlm_bridge; rlm_bridge._TOKEN = 'wrong'; terminal('echo x')")
    assert not r["ok"] and "unauthorized" in r["error"], r

    print("ok")
finally:
    proc.terminate()
    proc.wait(timeout=5)
    stop.set()
    server_sock.close()
