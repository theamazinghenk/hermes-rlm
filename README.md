# hermes-rlm — RLM Prime Agent for Hermes

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that
ports the useful parts of
[Prime Agent's](https://github.com/PrimeIntellect-ai/prime-agent) RLM model
onto Hermes' own tool stack: a persistent Python kernel per session,
subagents as function calls (blocking or handle-only), Python-backed
skills, crash-surviving checkpoints, and a continual harness that carries
lessons across sessions. Stdlib-only, no core patch — it survives every
`hermes update`.

**Measured impact** (details below): 40–48% less conversation context on
follow-up questions over the same data, a 21× cheaper marginal question,
2.5× faster parallel delegation, 11.9× faster tool-free subagents — and
with 0.3.0, subagent output stays out of the parent context entirely.

## Install

```bash
git clone https://github.com/jarnodevries-byte/hermes-rlm ~/.hermes/plugins/hermes-rlm
hermes plugins enable hermes-rlm
```

Then restart your gateway (`launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway`
or however you run it) — plugins load at startup. Stdlib-only: no pip installs.
Optional but recommended: add a routing line to `agent.environment_hint` in
`~/.hermes/config.yaml` telling the model to prefer `rlm_exec` for repeated
questions over the same dataset — measured to be the load-bearing adoption step.

> The kernel executes model-generated Python with your own permissions. It is
> a durable control environment, **not a security sandbox**. Do not enable it
> on safe-mode profiles.

## Why

`execute_code` is stateless: every script starts empty, so a large dataset is
re-loaded for each follow-up question and every intermediate result must pass
through the conversation to survive. This plugin adds a stateful lane beside
it. `execute_code` stays the safe, ephemeral, secret-scrubbed default.

## Tools

| Tool | Purpose |
| --- | --- |
| `rlm_exec` | Run Python in the session's persistent kernel |
| `rlm_vars` | List what is in the namespace (names, types, sizes, previews) |
| `rlm_reset` | Kill the kernel and discard all state |
| `rlm_skills` | List Python-backed skills importable in the kernel |
| `rlm_checkpoint` | Save/restore the namespace to disk so state survives a crash |
| `rlm_refine` | Durable harness: distil transcript lessons into reversible entries |

## Inside the kernel

Preloaded, no import needed:

- Hermes tools — `read_file`, `write_file`, `search_files`, `patch`,
  `terminal`, `web_search`, `web_extract`
- `rlm(goal, context)` — run one real Hermes subagent, blocking
- `rlm_many([{goal, context}, ...])` — up to 9 subagents in parallel
- `rlm_spawn(goal)` / `rlm_wait([ids])` / `rlm_children()` — handle-only
  subagents: children run detached, results deliver via files and a disk
  registry that survives kernel and gateway restarts
- `harness_store()` — CRUD access to the durable harness (see below)

Subagent children **inherit the parent's model** (whatever Hermes itself
runs on) but do their legwork at low reasoning effort with a tight turn
cap, passed as explicit CLI flags — profile-level `agent:` settings do not
survive Hermes' config merge, so flags are the reliable route. Tune via:

```bash
HERMES_RLM_CHILD_REASONING=low   # any hermes reasoning level; "0" omits the flag
HERMES_RLM_CHILD_MAX_TURNS=25    # tool-loop cap per child; "0" omits the flag
HERMES_RLM_LEAF_PROFILE=rlm-leaf # minimal child profile; "0" inherits parent profile
```

```python
# One rlm_exec call:
trades = my_skill.load("/path/to/trades.db")

# A separate rlm_exec call — no reload:
my_skill.summarise(trades)
my_skill.by_field(trades, "exit_reason")
```

## Python-backed skills

A skill becomes importable by shipping a package under `python/`:

```
<skill>/SKILL.md
<skill>/python/<module>/__init__.py
```

Those `python/` directories go on the kernel's `sys.path` at boot, so
`import <module>` works with no install step. `tests/test_skills.py`
fabricates a complete worked example.

## Architecture

```
rlm_exec (plugin handler)
   └── KernelHandle          one long-lived python process per session
        ├── stdin/stdout     newline-delimited JSON, one namespace
        └── Unix socket RPC  token-authenticated tool bridge
             ├── ALLOWED_TOOLS → handle_function_call (normal Hermes dispatch)
             └── __rlm_delegate__ → hermes chat -q (a real subagent)
```

`delegate_task` is an agent-loop tool: it needs the live parent agent object,
which plugin handlers never receive. Rather than fork Hermes core, `rlm()`
spawns a real Hermes CLI subagent — same isolation, no core patch. Depth is
capped at `MAX_RLM_DEPTH = 2` via the `HERMES_RLM_DEPTH` env var.

## Durable harness (0.3.0)

Lessons should outlive the session that earned them. The harness stores
small entries — prompt-notes, memories, skills, subagent recipes — per
session and globally, and injects a compact overview into future system
prompts. `rlm_refine` distils the recent transcript into at most 4
evidence-backed edits via one model call; every refinement is validated,
snapshotted in a ledger, and reversible with `rollback`. From inside the
kernel, `harness_store()` gives direct CRUD access. This is a deliberate
lite port of Prime Agent's continual harness: same invariants (immutable
base prompt, evidence-backed edits, rollback), no daemon required.

The refine model call and the subagent fast path share one provider config
(any OpenAI-compatible endpoint), set in the environment or `~/.hermes/.env`:

```bash
HERMES_RLM_FAST_BASE_URL=https://your-endpoint/v1
HERMES_RLM_FAST_MODEL=your-model
HERMES_RLM_FAST_KEY_ENV=YOUR_KEY_ENV_VAR   # default HERMES_RLM_FAST_API_KEY
```

Unset means: fast path unavailable, everything falls back to the full agent.

## Measured on real data

2,846 real trading round-trips via a Python-backed skill, six follow-up
questions:

| | Stateless | Persistent |
| --- | --- | --- |
| Context bytes | 1,116 | 581 |
| Seconds | 0.42 | 0.69 |

**48% less context.** Speed went the other way here — and that is the honest
result, not a rounding artefact.

Per follow-up question the gap is what changes behaviour:

| | Marginal cost per question |
| --- | --- |
| Stateless | 0.0210 s |
| Persistent | 0.0010 s — **21× cheaper** |

A follow-up becomes effectively free, so you actually ask it. In practice
that verification instinct caught a real data fault during development: a
"free" check revealed that 77% of a trades table was a structurally
different record type silently corrupting every exit-lifecycle figure —
the kind of check that gets skipped when it costs a full reload.

**Context saving is the reliable win.** It holds at any dataset size, because
the stateless lane must re-send the loading preamble with every question.

**Speed is conditional.** The kernel costs ~0.23s to boot and ~0.0005s per
warm call. It only wins when `load_seconds × questions` exceeds that boot
cost. Measured:

| Workload | Speedup |
| --- | --- |
| 2 columns, 2.8k rows, 6 questions | 0.61× (slower) |
| 2 columns, 12k rows, 6 questions | 2.26× |
| `select *`, 12k rows, 5 questions | 3.4× |

Rule of thumb: reach for `rlm_exec` when the load is expensive or you have
several questions. One cheap question is better served by `execute_code`.

**Keep the reply envelope lean.** An early version returned kernel
diagnostics on every call and measured −79% context: the metadata cost more
than the kernel saved. Successful replies now carry only `ok`, `value` and
any output; diagnostics appear on failures, where they help.

## Resource limits

Measured before these existed: 12 concurrent kernels held 206 MB with no cap,
and a single kernel allocated 1.5 GB unchallenged on a 16 GB laptop.

| Limit | Value | Behaviour on breach |
| --- | --- | --- |
| Kernels | 12 | Evicts least-recently-used; evicted session still works |
| RSS per kernel | 2048 MB | `warning` field with a concrete remedy |
| Idle lifetime | 1 hour | Reaped |
| Parallel subagents | 9 | Queued |
| Delegation depth | 2 | Refused with an explicit error |
| Checkpoint store | 14 days / 500 MB | Pruned on every save |

A subagent costs ~30s, so `rlm_many` is not a convenience — it is the
difference between 4 goals taking 30s and taking 75s (measured 2.5×).

## Trust model

The kernel runs model-generated Python with your own permissions. It is a
durable control environment, **not a security sandbox**. The RPC socket is
mode `0600` in a mode `0700` temp dir with a per-kernel token, and the tool
allow-list is enforced parent-side, so the kernel cannot widen its own reach.
State persists in memory — anything you load stays until `rlm_reset`, an
hour of idleness, or session end.

## Tests

```bash
cd ~/.hermes/plugins/hermes-rlm
for t in tests/test_*.py; do ~/.hermes/hermes-agent/venv/bin/python "$t"; done
~/.hermes/hermes-agent/venv/bin/python tests/benchmark_context.py
```

All thirteen suites print `ok` and exit 0 on any machine — tests fabricate
their own fixtures; the two benchmarks skip cleanly unless you point them
at a dataset (`HERMES_RLM_BENCH_DB`, `HERMES_RLM_IMPACT_*`). CI runs the
self-contained suites across Python 3.11–3.13 with both RSS policies in an
empty `HERMES_HOME`; suites that exercise a live Hermes CLI run in the
release environment.

## Changelog

### 0.4.1 — enforceable feature selection

- **Hard feature flags** (the last fleet-review blocker): setting
  `HERMES_RLM_ENABLE_CHECKPOINT=0`, `HERMES_RLM_ENABLE_REFINE=0`,
  `HERMES_RLM_ENABLE_SUBAGENTS=0` or `HERMES_RLM_ENABLE_PYTHON_SKILLS=0`
  makes that capability *absent*, not hidden: the tool is never registered,
  kernel helpers refuse with an operator message, skills paths stay off the
  kernel's `sys.path`, and with checkpoints off even autosave/salvage (a
  pickle path) is a no-op. `tests/test_feature_flags.py` proves both modes.
- GitHub Actions CI: Python 3.11–3.13 × RSS policy warn/stop in an empty
  `HERMES_HOME`.

A minimal pilot (exec/vars/reset only) is now one line of configuration:

```bash
HERMES_RLM_ENABLE_CHECKPOINT=0 HERMES_RLM_ENABLE_REFINE=0 \
HERMES_RLM_ENABLE_SUBAGENTS=0 HERMES_RLM_ENABLE_PYTHON_SKILLS=0
```

### 0.4.0 — fleet hardening

Driven by an external fleet review (thanks!). Safe defaults everywhere;
previous behaviour stays available behind explicit opt-ins.

- **`HERMES_HOME` respected in every state path** (harness, checkpoints,
  spill, subagent registry, env, skills, profiles) — isolated profiles and
  migrated agents no longer leak state into the wrong home. Covered by a
  dedicated `tests/test_isolation.py` proving two homes stay disjoint.
- **Cross-session checkpoint restore is now opt-in** (`allow_cross_session`
  parameter, default false). A session can never silently load another
  session's pickle.
- **Children load AGENTS.md/SOUL.md by default.** The `--ignore-rules`
  speedup (~30%) is now an explicit operator choice:
  `HERMES_RLM_CHILD_IGNORE_RULES=1`.
- **Resource ceilings are configurable and can be made hard**:
  `HERMES_RLM_MAX_KERNELS`, `HERMES_RLM_MAX_RSS_MB`,
  `HERMES_RLM_IDLE_SECONDS`, and `HERMES_RLM_RSS_POLICY=stop` (autosave +
  kill instead of a warning — recommended on shared machines).
- **Harness prompt injection can be disabled** (`HERMES_RLM_HARNESS_INJECT=0`)
  for fleets that already have centrally governed memory layers.
- Fast-path live speed checks skip cleanly when no provider is configured,
  and the speedup threshold is environment-aware (>1.2×).

Suggested pilot posture for a fleet:
`HERMES_RLM_MAX_KERNELS=2 HERMES_RLM_MAX_RSS_MB=512
HERMES_RLM_IDLE_SECONDS=900 HERMES_RLM_RSS_POLICY=stop
HERMES_RLM_HARNESS_INJECT=0` plus the 0.4.1 feature flags to reduce the
surface to rlm_exec/rlm_vars/rlm_reset.

### 0.3.0

- **Handle-only subagents**: `rlm_spawn(goal)` returns a handle immediately;
  children run detached (they survive kernel and even gateway restarts) with
  output to a file, never into the parent context. Collect with
  `rlm_wait([ids])`, inspect with `rlm_children()`. A disk registry finalises
  orphans visibly (`finished-unverified`), never silently.
- **Autosave + auto-restore**: LRU eviction and the idle reaper now save the
  namespace first, and the next `rlm_exec` for that session restores it
  automatically with a one-line note. Resource management no longer costs
  state.
- **Continual-harness-lite**: durable entries (prompt-notes, memories,
  skills, subagent recipes) per session + global, injected compactly into
  future system prompts. `rlm_refine` distils the recent transcript into at
  most 4 evidence-backed edits via one model call — validated, snapshotted
  in a refinements ledger, reversible with `rollback`. Kernel-side access
  via `harness_store()`.
- **Leaf profile for subagents**: children route to a minimal `rlm-leaf`
  profile when present (no MCP spawns, no memory, no plugins) and are marked
  as delegated contexts (kanban mutations blocked). Configure via
  `HERMES_RLM_LEAF_PROFILE` (set `0` to inherit the parent profile).
- Eleven test suites, all green.

### 0.2.0

- Oversized `rlm_exec` output now spills to disk instead of losing its
  middle: the reply keeps the tail plus the spill-file path (mode 0600,
  newest 40 kept), so the full text stays reachable via `open()` in the
  kernel. Mirrors prime-agent's output-accumulator design.
- Tool description now teaches variable-binding discipline: assign results
  to named variables, print only the summary — printed text is permanent
  conversation context, variables are free.
- `plugin.yaml` lists all five tools (checkpoint and skills were missing).

### 0.1.0

Initial release: persistent kernel, tool bridge, subagents, checkpoints,
Python-backed skills, resource limits.

## Roadmap

The three big adoption candidates from the prime-agent study shipped in
0.3.0 (harness-lite, handle-only subagents, autosave/auto-restore). Still
open, in rough order of value:

1. **Per-variable pickling with a manifest** — the autosave currently
   reuses whole-namespace checkpoints; per-variable saves would make
   partial failures more granular.
2. **Compaction truncation of old rlm results** — needs hermes-core
   cooperation; out of reach for a plugin.

Deliberately rejected: prime-agent's fork-server (Linux-only; fork without
exec is unsafe on macOS) and the full daemon/supervisor (a whole subsystem;
only worth it if detach/reattach becomes a hard requirement — `rlm_spawn`
covers the async need without it).

## License

MIT — see [LICENSE](LICENSE).
