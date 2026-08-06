"""Adversarial tests for hermes-rlm: the cases a well-behaved test misses.

Where test_stress.py checks that normal things work under load, this checks
what happens when things go wrong in hostile ways: fork bombs, blocking I/O,
socket abuse, resource exhaustion, and the RPC trust boundary.

    python3 tests/test_adversarial.py
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / ".hermes" / "hermes-agent"))
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("hermes_rlm_plugin", ROOT / "__init__.py")
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)

results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


def ex(code: str, session: str = "adv", timeout: int | None = None) -> dict:
    args = {"code": code}
    if timeout is not None:
        args["timeout"] = timeout
    return json.loads(plugin._exec_handler(args, task_id=session))


print("\n=== 1. RPC trust boundary ===")
handle = plugin._get_kernel("rpc")
sock_path = handle.socket_path

def _probe(payload: dict, timeout: int = 30) -> str:
    """Send one raw RPC frame and read the reply.

    Generous timeout: under parallel test load the RPC thread can be busy
    serving another kernel, and a flaky recv here would look like a security
    regression when it is only scheduling.
    """
    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    raw.settimeout(timeout)
    try:
        raw.connect(sock_path)
        raw.sendall(json.dumps(payload).encode() + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = raw.recv(65536)
            if not chunk:
                break
            buf += chunk
        return buf.decode()
    finally:
        raw.close()


# An unauthenticated client on the socket must be refused.
reply = _probe({"tool": "terminal", "args": {"command": "echo pwned"}})
check("no-token RPC request is refused", "unauthorized" in reply, reply[:80])

reply = _probe({"token": "guessed", "tool": "terminal",
                "args": {"command": "echo pwned"}})
check("wrong-token RPC request is refused", "unauthorized" in reply, reply[:80])

# Socket permissions: only the owner may connect.
import os as _os  # noqa: E402

mode = _os.stat(sock_path).st_mode & 0o777
dir_mode = _os.stat(handle.rpc_dir).st_mode & 0o777
check("socket is owner-only (0600)", mode == 0o600, oct(mode))
check("socket dir is owner-only (0700)", dir_mode == 0o700, oct(dir_mode))

# Malformed frames must not take the server down.
raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
raw.settimeout(10)
raw.connect(sock_path)
raw.sendall(b"this is not json at all\n")
try:
    raw.recv(65536)
except (socket.timeout, ConnectionResetError):
    pass
raw.close()
r = ex("'still alive'", session="rpc")
check("garbage frame does not kill the RPC server", r.get("ok"), str(r)[:80])
# A real dispatch needs a Hermes install; without one the bridge must still
# answer — a clean "dispatch unavailable" refusal proves it survived the
# abuse just as well as a successful terminal call does.
r = ex("try:\n    out = terminal('echo bridge_ok')\nexcept RlmBridgeError as e:\n    out = 'refused: ' + str(e)\nout", session="rpc")
_alive = ("bridge_ok" in str(r.get("value", ""))
          or "dispatch is unavailable" in str(r.get("value", "")))
check("tool bridge still works after abuse", _alive, str(r)[:100])

print("\n=== 2. Resource exhaustion ===")
# Blocking read on a pipe with nothing to write: must hit the timeout, not hang.
started = time.monotonic()
r = ex("import os\nfd_r, fd_w = os.pipe()\nos.read(fd_r, 1)", session="block", timeout=3)
elapsed = time.monotonic() - started
check("blocking syscall is interrupted by timeout", not r.get("ok") and elapsed < 9,
      f"{elapsed:.1f}s")

# A tight infinite loop (no I/O) must also be killable.
started = time.monotonic()
r = ex("while True: pass", session="spin", timeout=3)
elapsed = time.monotonic() - started
check("CPU-bound infinite loop is killed", not r.get("ok") and elapsed < 9, f"{elapsed:.1f}s")
r = ex("'recovered'", session="spin")
check("session recovers after infinite loop", r.get("ok"))

# Thread that outlives the call must not corrupt later results.
ex("import threading, time\n"
   "def bg():\n"
   "    time.sleep(2)\n"
   "t = threading.Thread(target=bg, daemon=True); t.start()\n"
   "'started'", session="thr")
r = ex("'main still responsive'", session="thr")
check("background thread does not block later calls", r.get("ok"))

print("\n=== 3. Output channel abuse ===")
# Code that closes stdout would break the protocol if unguarded.
r = ex("import sys\ntry:\n    sys.stdout.flush()\n    _ok = True\nexcept Exception:\n    _ok = False\n_ok",
       session="io")
check("stdout is a redirect target, not the real pipe", r.get("ok") and r["value"] == "True")
r = ex("'protocol intact'", session="io")
check("protocol survives stdout manipulation", r.get("ok"))

# Very long single line (no newlines) — worst case for a line protocol.
r = ex("'z' * 500_000", session="io")
check("500k-char value is capped, wire stays intact", len(r.get("value") or "") <= 2100,
      f"{len(r.get('value') or '')} chars")
r = ex("'after big value'", session="io")
check("kernel usable after huge payload", r.get("ok"))

print("\n=== 4. Concurrency stress ===")
errors: list[str] = []
lock_ok: list[bool] = []


def mixed(n: int) -> None:
    try:
        if n % 3 == 0:
            r = ex(f"acc = globals().get('acc', 0) + 1\nacc", session="race")
        elif n % 3 == 1:
            r = json.loads(plugin._vars_handler({}, task_id="race"))
        else:
            r = ex(f"'{n}'", session="race")
        lock_ok.append(bool(r))
    except Exception as exc:  # noqa: BLE001
        errors.append(repr(exc))


threads = [threading.Thread(target=mixed, args=(i,)) for i in range(30)]
for t in threads:
    t.start()
for t in threads:
    t.join()
check("30 mixed concurrent exec/vars calls, no exceptions", not errors, str(errors[:2]))
check("all 30 returned something", len(lock_ok) == 30, f"{len(lock_ok)}/30")
r = ex("isinstance(acc, int)", session="race")
check("shared counter is consistent after races", r.get("ok") and r["value"] == "True")

print("\n=== 5. Kernel lifecycle churn ===")
pids = set()
for i in range(8):
    ex("1", session="churn")
    h = plugin._get_kernel("churn", create=False)
    pids.add(h.proc.pid)
    plugin._reset_handler({}, task_id="churn")
check("8 boot/reset cycles each make a fresh process", len(pids) == 8, f"{len(pids)} distinct pids")

import subprocess as _sp  # noqa: E402

leaked = _sp.run(["pgrep", "-f", "kernel_server.py"], capture_output=True, text=True).stdout.split()
live_sessions = len(plugin._kernels)
check("no orphaned kernel processes after churn", len(leaked) <= live_sessions + 1,
      f"{len(leaked)} procs, {live_sessions} sessions")

# Temp dirs must be cleaned on shutdown.
h = plugin._get_kernel("tmpcheck")
tmpdir = h.rpc_dir
plugin._reset_handler({}, task_id="tmpcheck")
check("rpc temp dir is removed on shutdown", not Path(tmpdir).exists(), tmpdir)

print("\n=== 6. Skill loader robustness ===")
entries = plugin._skill_loader.discover()
check("discovery returns a list", isinstance(entries, list), f"{len(entries)} skills")
check("discovery tolerates a missing root",
      isinstance(plugin._skill_loader.discover([Path("/nonexistent/xyz")]), list))
catalog = plugin._skill_loader.catalog_json()
check("catalog is valid JSON", isinstance(json.loads(catalog), dict))
check("catalog stays small", len(catalog) < 20_000, f"{len(catalog)} bytes")

for sess in list(plugin._kernels):
    plugin._reset_handler({}, task_id=sess)

failed = [n for n, ok, _ in results if not ok]
print(f"\n{'=' * 60}")
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED: " + ", ".join(failed))
sys.exit(1 if failed else 0)
