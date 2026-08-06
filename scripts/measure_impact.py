#!/usr/bin/env python3
"""Measure the real-world impact of the persistent kernel.

Replays a realistic multi-question analysis both ways and reports the two
numbers that decide whether the plugin earns its place: context bytes the
model must carry, and wall-clock seconds.

Unlike the earlier micro-benchmark this uses the ACTUAL question chain a
trade review produces, including the expensive load, and reports per-question
marginal cost so the crossover point is visible rather than asserted.

Requires a Python-backed skill and its dataset. Point it at yours:

    HERMES_RLM_IMPACT_SKILL_PY=~/.hermes/skills/<cat>/<skill>/python \\
    HERMES_RLM_IMPACT_MODULE=<module> \\
    HERMES_RLM_IMPACT_DB=/path/to/data.db \\
    python3 scripts/measure_impact.py

Exits 0 with {"skipped": ...} when not configured, so CI stays green.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_PY = os.environ.get("HERMES_RLM_IMPACT_SKILL_PY", "")
MODULE = os.environ.get("HERMES_RLM_IMPACT_MODULE", "")
DB = os.environ.get("HERMES_RLM_IMPACT_DB", "")
if not SKILL_PY or not MODULE or not DB or not Path(DB).expanduser().exists():
    print(json.dumps({"skipped": "set HERMES_RLM_IMPACT_SKILL_PY, "
                                 "HERMES_RLM_IMPACT_MODULE and "
                                 "HERMES_RLM_IMPACT_DB to run this benchmark"}))
    sys.exit(0)
DB = str(Path(DB).expanduser())
sys.path.insert(0, str(Path(SKILL_PY).expanduser()))

spec = importlib.util.spec_from_file_location("hermes_rlm_plugin", ROOT / "__init__.py")
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)

q = importlib.import_module(MODULE)

LOAD = (f"import {MODULE} as q\n"
        f"trips = q.load({DB!r})\n")

# The question chain an actual trade review produces.
QUESTIONS = [
    "len(trips)",
    "q.summarise(trips)['net_pnl']",
    "q.by_field(trips, 'exit_reason')[:2]",
    "len({t['symbol'] for t in trips})",
    "round(sum(t['pnl'] for t in trips if 'tsl' in (t.get('exit_reason') or '')), 2)",
    "round(sum(t['pnl'] for t in trips if (t.get('exit_reason') or '').startswith('fixed_sl')), 2)",
    "c = q.compare_halves(trips, 'exit_reason'); (c['early']['n'], c['late']['n'])",
    "sorted(((s, round(v, 2)) for s, v in __import__('collections').Counter().items()), key=str)[:1] or 'noop'",
]


def stateless() -> dict:
    """Each question is an independent script: reload, compute, discard."""
    total_bytes = 0
    seconds = 0.0
    per_question = []
    for question in QUESTIONS:
        started = time.monotonic()
        # A stateless turn must re-send the preamble AND re-run the load.
        sent = len(LOAD) + len(question)
        trips = q.load(DB)
        try:
            value = repr(eval(question, {"trips": trips, "q": q}))
        except Exception as exc:  # noqa: BLE001
            value = f"error: {exc}"
        received = len(json.dumps({"ok": True, "value": value[:2000]}))
        elapsed = time.monotonic() - started
        total_bytes += sent + received
        seconds += elapsed
        per_question.append({"bytes": sent + received, "secs": round(elapsed, 3)})
    return {"context_bytes": total_bytes, "seconds": round(seconds, 2),
            "per_question": per_question}


def persistent() -> dict:
    """Load once into the kernel, then ask follow-ups against live state."""
    session = "impact"
    total_bytes = len(LOAD)
    started = time.monotonic()
    total_bytes += len(plugin._exec_handler({"code": LOAD}, task_id=session))
    load_secs = time.monotonic() - started

    seconds = load_secs
    per_question = []
    for question in QUESTIONS:
        t0 = time.monotonic()
        raw = plugin._exec_handler({"code": question}, task_id=session)
        elapsed = time.monotonic() - t0
        total_bytes += len(question) + len(raw)
        seconds += elapsed
        per_question.append({"bytes": len(question) + len(raw), "secs": round(elapsed, 4)})
    plugin._reset_handler({}, task_id=session)
    return {"context_bytes": total_bytes, "seconds": round(seconds, 2),
            "boot_plus_load_secs": round(load_secs, 3), "per_question": per_question}


def raw_sqlite_baseline() -> dict:
    """What the agent did BEFORE routing: terminal + sqlite3 per question."""
    total_bytes = 0
    seconds = 0.0
    sql = [
        "select count(*) from trades where entry_price is not null",
        "select round(sum(pnl),2) from trades where entry_price is not null",
        "select exit_reason, count(*), round(sum(pnl),2) from trades "
        "where entry_price is not null group by 1 order by 3 limit 2",
        "select count(distinct symbol) from trades where entry_price is not null",
    ]
    for statement in sql:
        started = time.monotonic()
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        rows = con.execute(statement).fetchall()
        con.close()
        # A terminal call carries the full command plus the raw stdout.
        total_bytes += len(f'sqlite3 "{DB}" "{statement}"') + len(str(rows))
        seconds += time.monotonic() - started
    return {"context_bytes": total_bytes, "seconds": round(seconds, 2),
            "questions": len(sql), "note": "only 4 of 8 questions expressible in plain SQL"}


s = stateless()
p = persistent()
b = raw_sqlite_baseline()

marginal_stateless = sum(x["secs"] for x in s["per_question"][1:]) / max(len(QUESTIONS) - 1, 1)
marginal_persistent = sum(x["secs"] for x in p["per_question"][1:]) / max(len(QUESTIONS) - 1, 1)

print(json.dumps({
    "database": DB,
    "questions": len(QUESTIONS),
    "stateless_execute_code": {k: s[k] for k in ("context_bytes", "seconds")},
    "persistent_rlm_exec": {k: p[k] for k in ("context_bytes", "seconds", "boot_plus_load_secs")},
    "raw_sqlite_baseline": b,
    "context_saving_pct": round(100 * (1 - p["context_bytes"] / s["context_bytes"]), 1),
    "speedup": round(s["seconds"] / p["seconds"], 2) if p["seconds"] else None,
    "marginal_cost_per_followup": {
        "stateless_secs": round(marginal_stateless, 4),
        "persistent_secs": round(marginal_persistent, 4),
        "ratio": round(marginal_stateless / marginal_persistent, 1) if marginal_persistent else None,
    },
}, indent=2))
