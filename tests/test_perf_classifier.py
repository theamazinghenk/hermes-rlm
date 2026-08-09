"""Deterministic checks for the exit-code-driven terminal classifier."""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("hermes_rlm_perf_classifier", HERE / "perf_classifier.py")
pc = importlib.util.module_from_spec(spec)
sys.modules["hermes_rlm_perf_classifier"] = pc
spec.loader.exec_module(pc)

checks = []


def check(name, cond, detail=""):
    checks.append((name, bool(cond), detail))
    print(f"  {'ok' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))


def res(exit_code, output="", error=""):
    return json.dumps({"exit_code": exit_code, "output": output, "error": error})


# exit 0 is success even when output contains the word 'error'
check("exit-0 met 'error' in output = ok",
      pc.classify(pc.parse_result(res(0, "total errors: 0\nAll good"))) == "ok")
check("exit-0 met 'failed' in output = ok",
      pc.classify(pc.parse_result(res(0, "0 failed, 10 passed"))) == "ok")

# Real failure buckets
check("embedded null byte",
      pc.classify(pc.parse_result(res(-1, "", "Failed to execute command: embedded null byte"))) == "embedded-null-byte")
check("gateway restart blocked",
      pc.classify(pc.parse_result(res(1, "Blocked: command or referenced script cannot restart or stop the gateway"))) == "blocked-commando")
check("command timeout",
      pc.classify(pc.parse_result(res(124, "[Command timed out after 200s]"))) == "command-timeout")
check("python traceback",
      pc.classify(pc.parse_result(res(1, "Traceback (most recent call last):\n  File \"x.py\""))) == "python-traceback")
check("pad niet gevonden",
      pc.classify(pc.parse_result(res(127, "bash: scripts/x.sh: No such file or directory"))) == "command-not-found")
check("qm schema",
      pc.classify(pc.parse_result(res(1, "Error: in prepare, no such column: exit_timestamp"))) == "qm-schema")
check("pathspec",
      pc.classify(pc.parse_result(res(128, "fatal: pathspec 'worker/x.ts' did not match any files"))) == "git-pathspec")
check("netwerk",
      pc.classify(pc.parse_result(res(7, "curl: (7) Failed to connect to 127.0.0.1 port 3000"))) == "netwerk-ssh")
check("background-gebruik",
      pc.classify(pc.parse_result(res(-1, "Foreground command uses '&' backgrounding. Re-send"))) == "background-gebruik")

# Non-zero but legitimate outputs are NOT failures
check("solc-check non-zero = geen fout",
      pc.classify(pc.parse_result(res(1, "optimizer = true optimizer_runs = 200"))) == "non-zero-ok-solc-check")
check("test-F-output non-zero = geen fout",
      pc.classify(pc.parse_result(res(1, ".........F..FFFF....F.F....."))) == "non-zero-ok-test-output")
check("vitest-run = geen fout",
      pc.classify(pc.parse_result(res(1, "> vitest run server/live-submit-path.test.ts"))) == "non-zero-ok-test-run")
check("locales-output = geen fout",
      pc.classify(pc.parse_result(res(1, '[ { "n": 5, "label": "SEMAINE SURF ESSENTIELLE"'))) == "non-zero-ok-locales")
check("git-status = geen fout",
      pc.classify(pc.parse_result(res(1, "## swarm/qm-live-reconcile-20260704\n?? docs/x.md"))) == "non-zero-ok-git-status")

# bucket_counts counts correctly
rows = [res(0, "ok"), res(1, "Traceback"), res(124, "timed out"), "garbage"]
counts = pc.bucket_counts(rows)
check("bucket_counts sommeert", counts.get("ok") == 1 and counts.get("python-traceback") == 1
      and counts.get("command-timeout") == 1 and counts.get("unparseable") == 1, str(counts))

failed = [n for n, ok, _ in checks if not ok]
print()
if failed:
    print(f"FAILED: {len(failed)} — {failed}")
    sys.exit(1)
print(f"{len(checks)}/{len(checks)} checks passed")
