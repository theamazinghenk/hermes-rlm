#!/usr/bin/env python3
"""Exit-code-driven terminal failure classifier for self-improvement.

Replaces the naive substring matching (which counted 4.882 "suspects" when only
~489 real non-zero exits existed, and 526 exit=0 calls as false positives).
Given the raw terminal tool result JSON, returns a stable, testable bucket.

Used by improve_perf_signals.py (which lives OUTSIDE the repo by design) —
this module lives IN the repo so candidates can improve it through the normal
build/eval gate, and the deployed script imports it.
"""
from __future__ import annotations

import json
import re
from typing import Any


def parse_result(content: str) -> dict | None:
    """First JSON object of a tool result; None if not parseable."""
    try:
        obj, _ = json.JSONDecoder().raw_decode(content)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


def classify(result: dict) -> str:
    """Bucket a terminal tool result into a stable failure class.

    Returns 'ok' for exit 0 (even when output contains the word 'error'),
    a concrete failure bucket for real failures, and 'non-zero-ok' for
    commands that legitimately exit non-zero (grep no-match, test runs,
    solc/forge checks, tool output).
    """
    if not isinstance(result, dict):
        return "unparseable"
    ec = result.get("exit_code")
    out = str(result.get("output") or "")
    err = str(result.get("error") or "")
    text = out + "\n" + err

    # exit 0 is success, no matter what the output says.
    if ec == 0:
        return "ok"

    if not isinstance(ec, int):
        return "no-exit-code"

    if ec == -1:
        if "null byte" in text:
            return "embedded-null-byte"
        if "backgrounding" in text or "long-lived" in text:
            return "background-gebruik"
        if "Could not determine home directory" in text:
            return "home-not-determinable"
        return "exit--1-ander"

    if ec == 124:
        return "command-timeout"

    if "Blocked:" in text or "BLOCKED" in text:
        return "blocked-commando"

    if "Traceback" in text:
        return "python-traceback"

    if "timed out" in text:
        return "command-timeout"

    if "not found" in text.lower() or "no such file" in text.lower() \
            or "does not exist" in text.lower():
        if ec in (127, 126):
            return "command-not-found"
        return "pad-bestand-niet-gevonden"

    if "no such column" in text:
        return "qm-schema"

    if "better-sqlite3" in text or "collector.mjs" in text:
        return "qm-sqlite"

    if "unused import" in text:
        return "rust-warning"

    if "wrangler" in text or "TS5058" in text:
        return "wrangler-tsconfig"

    if "prime-agent" in text.lower():
        return "prime-agent-cli"

    if "pathspec" in text:
        return "git-pathspec"

    if "not a git repository" in text or "gitdir" in text:
        return "git-worktree"

    if ec in (127, 126):
        return "command-not-found"

    if "syntaxerror" in text.lower() or "error compiling" in text.lower():
        return "python-syntax"

    if "curl" in text.lower() or "ssh" in text.lower() or "connect" in text.lower():
        return "netwerk-ssh"

    if "pm2" in text.lower() or "ec2-user" in text:
        return "vps-pm2"

    if "docker" in text.lower() or "colima" in text.lower():
        return "docker-colima"

    if "parser limit or malformed" in text:
        return "command-parser-blocked"

    # Legitimate non-zero outputs that are NOT failures:
    if "optimizer = true" in text or "pragma solidity" in text or "solc" in text.lower():
        return "non-zero-ok-solc-check"
    if re.fullmatch(r"[.F\s-]+", out[:80].strip()) and "F" in out[:80]:
        return "non-zero-ok-test-output"
    if "vitest" in text or "test session" in text:
        return "non-zero-ok-test-run"
    if "SEMAINE" in text:
        return "non-zero-ok-locales"
    if "[sitemap]" in text:
        return "non-zero-ok-sitemap"
    if "swarm/qm-live-reconcile" in text:
        return "non-zero-ok-git-status"
    if "interrupted" in text:
        return "interrupted"

    if ec in (128, 130):
        return "signal-killed"
    if ec in (2,):
        return "usage-error"

    if text.strip() == "":
        return "leeg"

    return "overig"


def bucket_counts(rows: list[str]) -> dict[str, int]:
    """Count buckets across raw tool-result strings."""
    counts: dict[str, int] = {}
    for row in rows:
        parsed = parse_result(row)
        bucket = classify(parsed) if parsed else "unparseable"
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def real_failure_buckets() -> set[str]:
    """Buckets that count as genuine failures (vs 'ok' or 'non-zero-ok')."""
    return {
        "embedded-null-byte", "background-gebruik", "home-not-determinable",
        "command-timeout", "blocked-commando", "python-traceback",
        "pad-bestand-niet-gevonden", "qm-schema", "qm-sqlite", "rust-warning",
        "wrangler-tsconfig", "prime-agent-cli", "git-pathspec", "git-worktree",
        "python-syntax", "netwerk-ssh", "vps-pm2", "docker-colima",
        "command-parser-blocked", "command-not-found", "signal-killed",
        "usage-error", "exit--1-ander", "no-exit-code", "unparseable", "leeg",
    }


__all__ = ["parse_result", "classify", "bucket_counts", "real_failure_buckets"]
