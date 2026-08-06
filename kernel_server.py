"""Persistent Python kernel process for the Hermes RLM plugin.

Runs as a child process. Reads newline-delimited JSON requests on stdin,
executes each `code` block against ONE long-lived globals namespace, and
writes newline-delimited JSON responses on stdout.

State (variables, imports, loaded DataFrames) survives across requests, so
the model can load a large dataset once and then ask many questions about
it without ever putting the raw data into the conversation context.

Protocol
--------
request : {"id": str, "op": "exec"|"vars"|"ping", "code": str, "timeout": int}
response: {"id": str, "ok": bool, "stdout": str, "stderr": str,
           "error": str|None, "value": str|None, "duration": float}

Trust model: this executes model-generated Python with the user's own
permissions. It is a durable control environment, NOT a security sandbox.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import traceback

MAX_STREAM_BYTES = 50_000
MAX_REPR_CHARS = 2_000

# Oversized output spills to disk instead of being lost: the reply keeps the
# tail (recent lines are usually the informative ones) plus the spill path,
# so the model can read the full text back with open()/read_file on demand.
# Mirrors prime-agent's output-accumulator (tail buffer + fullOutputPath).
SPILL_DIR = os.path.join(
    os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes"),
    "state", "rlm", "spill",
)
SPILL_KEEP = 40  # newest spill files kept; older ones pruned on each spill


def _spill(text: str) -> str | None:
    """Write full oversized output to a private spill file; return its path."""
    try:
        os.makedirs(SPILL_DIR, mode=0o700, exist_ok=True)
        fd, path = tempfile.mkstemp(
            prefix=time.strftime("%Y%m%d-%H%M%S-"), suffix=".txt", dir=SPILL_DIR
        )
        with os.fdopen(fd, "w", errors="replace") as fh:
            fh.write(text)
        os.chmod(path, 0o600)
        spills = sorted(
            (os.path.join(SPILL_DIR, n) for n in os.listdir(SPILL_DIR)),
            key=os.path.getmtime,
        )
        for old in spills[:-SPILL_KEEP]:
            try:
                os.unlink(old)
            except OSError:
                pass
        return path
    except OSError:
        return None


def _truncate(text: str, limit: int = MAX_STREAM_BYTES) -> str:
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    omitted = len(text) - len(tail)
    path = _spill(text)
    if path:
        note = (f"... [{omitted} chars omitted — full output saved to {path}; "
                f"open() it in this kernel if you need the rest] ...\n")
    else:
        note = f"... [{omitted} chars omitted] ...\n"
    return note + tail


def _describe(name: str, value: object) -> dict:
    """Cheap, safe summary of a namespace entry — never the full object."""
    info: dict = {"name": name, "type": type(value).__name__}
    try:
        if hasattr(value, "shape"):
            info["shape"] = str(getattr(value, "shape"))
        elif isinstance(value, (list, tuple, set, dict, str, bytes)):
            info["len"] = len(value)
    except Exception:
        pass
    try:
        preview = repr(value)
        info["preview"] = preview[:200]
    except Exception:
        info["preview"] = "<unreprable>"
    return info


class Kernel:
    def __init__(self) -> None:
        self.ns: dict = {"__name__": "__hermes_rlm__", "__builtins__": __builtins__}
        self.exec_count = 0

    def _run(self, code: str) -> tuple[str, str, str | None, str | None]:
        """Exec `code`; if the final statement is an expression, capture its repr."""
        out, err = io.StringIO(), io.StringIO()
        error: str | None = None
        value: str | None = None
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError:
            return "", "", traceback.format_exc(limit=2), None

        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body.pop().value)

        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                if tree.body:
                    exec(compile(tree, "<rlm>", "exec"), self.ns)
                if last_expr is not None:
                    result = eval(compile(last_expr, "<rlm>", "eval"), self.ns)
                    if result is not None:
                        self.ns["_"] = result
                        value = repr(result)[:MAX_REPR_CHARS]
        except BaseException:
            error = traceback.format_exc(limit=6)

        return out.getvalue(), err.getvalue(), error, value

    def _handle_checkpoint(self, op: str, req: dict, started: float) -> dict:
        """Persist or reload the namespace so state survives a kernel death.

        Runs inside the kernel because only this process holds the live
        objects. Import is lazy and failure is reported, never raised: a
        broken checkpoint must not take down a working kernel.
        """
        try:
            import checkpoint as _cp
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "stdout": "", "stderr": "", "value": None,
                    "error": f"checkpoint module unavailable: {exc}",
                    "duration": round(time.monotonic() - started, 3)}

        session = req.get("session") or "default"
        name = req.get("name") or "latest"
        try:
            if op == "checkpoint":
                result = _cp.save(self.ns, session, name)
            else:
                result = _cp.load(session, name,
                                  allow_cross_session=bool(req.get("allow_cross_session")))
                if result.get("ok"):
                    # Merge rather than replace: a restore must not wipe
                    # bindings the caller created since the checkpoint.
                    self.ns.update(result.pop("namespace"))
        except Exception:
            return {"ok": False, "stdout": "", "stderr": "", "value": None,
                    "error": traceback.format_exc(limit=4),
                    "duration": round(time.monotonic() - started, 3)}

        return {"ok": bool(result.get("ok")), "stdout": "", "stderr": "",
                "error": result.get("error"),
                "value": json.dumps(result, default=str),
                "duration": round(time.monotonic() - started, 3),
                "exec_count": self.exec_count}

    def handle(self, req: dict) -> dict:
        op = req.get("op", "exec")
        started = time.monotonic()

        if op == "ping":
            return {"ok": True, "stdout": "", "stderr": "", "error": None,
                    "value": "pong", "duration": 0.0}

        if op in {"checkpoint", "restore"}:
            return self._handle_checkpoint(op, req, started)

        if op == "vars":
            names = [
                n for n in self.ns
                if not n.startswith("__") and n not in {"_"}
            ]
            described = [_describe(n, self.ns[n]) for n in sorted(names)]
            return {"ok": True, "stdout": "", "stderr": "", "error": None,
                    "value": json.dumps(described), "duration": 0.0,
                    "exec_count": self.exec_count}

        stdout, stderr, error, value = self._run(req.get("code", ""))
        self.exec_count += 1
        return {
            "ok": error is None,
            "stdout": _truncate(stdout),
            "stderr": _truncate(stderr),
            "error": error,
            "value": value,
            "duration": round(time.monotonic() - started, 3),
            "exec_count": self.exec_count,
        }


def main() -> None:
    kernel = Kernel()
    # Unbuffered line protocol on stdout; stderr stays free for crash traces.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            resp = {"id": None, "ok": False, "error": f"bad request: {exc}"}
        else:
            try:
                resp = kernel.handle(req)
            except BaseException:
                resp = {"ok": False, "error": traceback.format_exc(limit=4)}
            resp["id"] = req.get("id")
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
