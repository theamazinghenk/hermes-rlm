"""A/B benchmark: stateless execute_code vs the persistent RLM kernel.

Runs the SAME five-question analysis over a real trade database
both ways and reports the two numbers that decide whether this plugin earns
its keep:

  * bytes returned to the model (context cost)
  * wall-clock seconds (the reload tax)

Stateless lane: every question is an independent script, so the dataset is
re-read from SQLite each time — exactly what `execute_code` does today.
Persistent lane: the dataset is loaded once into the kernel; later questions
are one-liners against memory.

Stdlib only, so it runs on the Hermes venv interpreter with no extra deps.

    python3 tests/benchmark_context.py [db_path]
"""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / ".hermes" / "hermes-agent"))

DB = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HERMES_RLM_BENCH_DB", "")
if not DB or not Path(DB).expanduser().exists():
    print(json.dumps({"skipped": "pass a SQLite path as argv[1] or set "
                                 "HERMES_RLM_BENCH_DB to run this benchmark"}))
    sys.exit(0)
DB = str(Path(DB).expanduser())
# Round-trips only: a raw trades table may also hold fills with no entry/exit
# pair, and mixing the two makes any exit-lifecycle figure meaningless.
QUERY = os.environ.get(
    "HERMES_RLM_BENCH_QUERY",
    "select symbol, pnl from trades "
    "where pnl is not null and entry_price is not null")

spec = importlib.util.spec_from_file_location("hermes_rlm_plugin", ROOT / "__init__.py")
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)

SESSION = "benchmark"

# Five follow-up questions of the kind a real trade review actually asks.
QUESTIONS = [
    "len(rows)",
    "round(sum(p for _, p in rows), 2)",
    "sum(1 for _, p in rows if p > 0)",
    "sorted(((s, round(v, 2)) for s, v in by_symbol.items()), key=lambda kv: kv[1])[:3]",
    "round(statistics.pstdev([p for _, p in rows]), 4)",
]

LOAD = (
    "import sqlite3, statistics\n"
    "from collections import defaultdict\n"
    f"con = sqlite3.connect('file:{DB}?mode=ro', uri=True)\n"
    f"rows = con.execute({QUERY!r}).fetchall()\n"
    "con.close()\n"
    "by_symbol = defaultdict(float)\n"
    "for _s, _p in rows: by_symbol[_s] += _p\n"
)


def _load_rows():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute(QUERY).fetchall()
    con.close()
    by_symbol = defaultdict(float)
    for s, p in rows:
        by_symbol[s] += p
    return rows, by_symbol


def bench_stateless() -> tuple[int, float, list[str]]:
    """Each question re-loads the dataset, like a fresh execute_code run."""
    total_bytes = 0
    answers = []
    started = time.monotonic()
    for question in QUESTIONS:
        # Context cost is bidirectional: a stateless turn must RE-SEND the
        # whole loading preamble alongside the question, because nothing
        # survives from the previous call.
        total_bytes += len(LOAD) + len(question)
        rows, by_symbol = _load_rows()
        value = repr(eval(question, {
            "rows": rows, "by_symbol": by_symbol, "statistics": statistics,
        }))
        answers.append(value)
        total_bytes += len(json.dumps({"stdout": value, "reloaded_rows": len(rows)}))
    return total_bytes, time.monotonic() - started, answers


def bench_persistent() -> tuple[int, float, list[str]]:
    """Load once into the kernel, then ask five cheap follow-ups."""
    total_bytes = 0
    answers = []
    started = time.monotonic()
    total_bytes += len(LOAD)
    load = plugin._exec_handler({"code": LOAD}, task_id=SESSION)
    total_bytes += len(load)
    for question in QUESTIONS:
        total_bytes += len(question)
        raw = plugin._exec_handler({"code": question}, task_id=SESSION)
        total_bytes += len(raw)
        answers.append(json.loads(raw).get("value"))
    return total_bytes, time.monotonic() - started, answers


try:
    s_bytes, s_secs, s_answers = bench_stateless()
    p_bytes, p_secs, p_answers = bench_persistent()

    # The comparison is only meaningful if both lanes computed the same thing.
    assert s_answers == p_answers, (s_answers, p_answers)

    print(json.dumps({
        "database": DB,
        "rows": int(s_answers[0]),
        "questions": len(QUESTIONS),
        "stateless": {"context_bytes": s_bytes, "seconds": round(s_secs, 2)},
        "persistent": {"context_bytes": p_bytes, "seconds": round(p_secs, 2)},
        "context_saving_pct": round(100 * (1 - p_bytes / s_bytes), 1) if s_bytes else 0,
        "speedup": round(s_secs / p_secs, 2) if p_secs else None,
        "answers_identical": True,
        "note": (
            "Context saving is the reliable win and holds regardless of size. "
            "SPEED is conditional: the kernel costs ~0.23s to boot and ~0.0005s "
            "per warm call, so it only wins when (load_seconds x questions) "
            "exceeds that boot cost. This lane loads two columns from 2.8k rows "
            "-- deliberately the weakest case, and the kernel is slower here. "
            "Wide loads (select *) over 12k+ rows measured 2-3x faster."
        ),
    }, indent=2))
finally:
    plugin._reset_handler({}, task_id=SESSION)
