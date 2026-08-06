"""Fast path for tool-free subagent calls.

Measured: `hermes chat -q` costs ~17s of which only ~2-3s is the model. The
remaining ~14s is agent-loop startup — system prompt assembly, skill index,
memory injection, tool registry, session persistence. That overhead is worth
paying when a subagent needs tools; it is pure waste when the task is
"summarise this text" or "classify these strings".

This module answers such calls with one direct chat-completions request
against the same provider the agent uses, and falls back to the full agent
whenever anything is uncertain. Prime Agent solves the same problem with a
resident daemon; we cannot keep Hermes warm from a plugin, so we skip the
cold start instead.

The fast path is opt-in per call and defaults OFF: a wrong routing decision
that silently drops tool access would be much worse than 14 extra seconds.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

HERMES_ENV = (Path(os.environ["HERMES_HOME"]).expanduser()
              if os.environ.get("HERMES_HOME") else Path.home() / ".hermes") / ".env"

# Signals that a task genuinely needs the agent loop: filesystem, shell,
# network, or any multi-step work. Matching is deliberately eager — a false
# "needs tools" costs 14s, a false "tool-free" produces a wrong answer.
_NEEDS_AGENT = re.compile(
    r"\b(read|write|edit|patch|create|delete|run|execute|install|build|test|"
    r"search|find|grep|fetch|download|curl|browse|open|commit|push|deploy|"
    r"check|verify|inspect|audit|analyse|analyze|investigate|debug|fix|"
    r"file|repo|repository|directory|folder|path|url|website|api|database|"
    r"command|script|terminal|shell|git)\b",
    re.IGNORECASE,
)


class FastPathUnavailable(RuntimeError):
    """Raised when the fast path cannot be used; caller should fall back."""


def _read_env_key(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        for line in HERMES_ENV.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return None


def eligible(goal: str, context: str = "") -> bool:
    """True when a task looks answerable without tools.

    Conservative by construction: anything mentioning files, commands, the
    network, or investigation goes to the full agent.
    """
    if not goal or not goal.strip():
        return False
    text = f"{goal} {context}"
    if len(text) > 4000:
        return False  # long briefs are usually real work
    return not _NEEDS_AGENT.search(text)


def run(goal: str, context: str = "", model: str | None = None,
        base_url: str | None = None,
        timeout: int = 120, max_tokens: int = 2048) -> dict:
    """Answer a tool-free task with one direct model call.

    Provider comes from the environment (or ~/.hermes/.env):
      HERMES_RLM_FAST_BASE_URL  — OpenAI-compatible endpoint (required)
      HERMES_RLM_FAST_MODEL     — model id (required)
      HERMES_RLM_FAST_KEY_ENV   — name of the env var holding the API key
                                  (default HERMES_RLM_FAST_API_KEY, falling
                                  back to OPENAI_API_KEY)

    Raises FastPathUnavailable when the provider is not configured or
    unreachable, so the caller falls back to the full agent rather than
    returning a degraded answer.
    """
    base_url = base_url or _read_env_key("HERMES_RLM_FAST_BASE_URL")
    model = model or _read_env_key("HERMES_RLM_FAST_MODEL")
    if not base_url or not model:
        raise FastPathUnavailable(
            "fast path not configured (set HERMES_RLM_FAST_BASE_URL and "
            "HERMES_RLM_FAST_MODEL)")
    key_env = _read_env_key("HERMES_RLM_FAST_KEY_ENV") or "HERMES_RLM_FAST_API_KEY"
    key = _read_env_key(key_env) or _read_env_key("OPENAI_API_KEY")
    if not key:
        raise FastPathUnavailable(f"no API key available in ${key_env}")

    prompt = f"{goal}\n\nContext:\n{context}" if context.strip() else goal
    payload = json.dumps({
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode()

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise FastPathUnavailable(f"provider call failed: {exc}") from exc

    try:
        summary = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise FastPathUnavailable(f"unexpected response shape: {exc}") from exc

    if not isinstance(summary, str) or not summary.strip():
        raise FastPathUnavailable("empty response")

    return {
        "ok": True,
        "goal": goal,
        "summary": summary.strip(),
        "session_id": None,
        "exit_code": 0,
        "fast_path": True,
    }
