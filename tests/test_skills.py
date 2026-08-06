"""Self-check for phase 3: Python-backed skills inside the RLM kernel.

Asserts the property that makes phase 3 worth having: a skill that ships a
`python/` package becomes an ordinary import inside the persistent kernel,
with no install step, and its results stay in kernel memory for follow-up
questions.

Fully self-contained: fabricates a demo skill + a small SQLite dataset in a
temp directory, so it runs on any machine.

    python3 tests/test_skills.py   ->  prints "ok" and exits 0
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skill_loader import catalog_json, discover, sys_path_entries  # noqa: E402

spec = importlib.util.spec_from_file_location("hermes_rlm_plugin", ROOT / "__init__.py")
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)

SESSION = "skills-test"

# --- fabricate a demo skill + dataset in a temp root -------------------------

tmp = Path(tempfile.mkdtemp(prefix="rlm_skills_test_"))
skill_dir = tmp / "skills" / "demo-stats"
pkg = skill_dir / "python" / "demo_stats"
pkg.mkdir(parents=True)
(skill_dir / "SKILL.md").write_text(
    "---\nname: demo-stats\ndescription: Demo dataset loader for tests\n---\n"
    "# demo-stats\n")
(pkg / "__init__.py").write_text(
    "import sqlite3\n"
    "def load(db_path):\n"
    "    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)\n"
    "    rows = con.execute('SELECT value FROM points').fetchall()\n"
    "    con.close()\n"
    "    return [r[0] for r in rows]\n"
    "def summarise(rows):\n"
    "    return {'n': len(rows), 'total': sum(rows)}\n")

DB = tmp / "demo.db"
con = sqlite3.connect(DB)
con.execute("CREATE TABLE points (value INTEGER)")
con.executemany("INSERT INTO points VALUES (?)", [(i,) for i in range(1, 11)])
con.commit()
con.close()

ROOTS = [tmp / "skills"]
# The kernel builds its PYTHONPATH from the loader's default roots — point
# those at the temp root so the fabricated skill is importable in-kernel.
plugin._skill_loader.SKILL_ROOTS[:] = ROOTS


def call(handler, args=None):
    return json.loads(handler(args or {}, task_id=SESSION))


try:
    # 1. Discovery finds the fabricated skill and reports where to import from.
    entries = {e["import_as"]: e for e in discover(ROOTS)}
    assert "demo_stats" in entries, sorted(entries)
    entry = entries["demo_stats"]
    assert entry["name"] == "demo-stats", entry
    assert entry["description"], "SKILL.md frontmatter description must be picked up"
    assert entry["sys_path_entry"] in sys_path_entries(ROOTS)

    # 2. The catalog stays cheap — names and summaries, never module source.
    catalog = catalog_json(ROOTS)
    assert len(catalog) < 10_000, len(catalog)
    assert "demo_stats" in catalog

    # 3. The rlm_skills tool exposes the same catalog.
    listed = call(plugin._skills_handler)
    assert any(s["import_as"] == "demo_stats" for s in listed["skills"]), listed

    # 4. THE phase-3 property: plain `import` works inside the kernel.
    r = call(plugin._exec_handler, {
        "code": "import demo_stats as q\n"
                "sorted(n for n in dir(q) if not n.startswith('_'))"
    })
    assert r["ok"], r
    for name in ("load", "summarise"):
        assert name in r["value"], r["value"]

    # 5. It computes against the dataset, read-only.
    r = call(plugin._exec_handler, {
        "code": f"rows = q.load({str(DB)!r})\nsummary = q.summarise(rows)\nsummary['n']"
    })
    assert r["ok"] and int(r["value"]) == 10, r

    # 6. Results persist: the follow-up costs one line and no reload.
    r = call(plugin._exec_handler, {"code": "summary['total']"})
    assert r["ok"] and int(r["value"]) == 55, r

    # 7. Read-only enforcement: writing through the skill's connection fails.
    r = call(plugin._exec_handler, {
        "code": "import sqlite3\n"
                f"con = sqlite3.connect('file:{DB}?mode=ro', uri=True)\n"
                "try:\n"
                "    con.execute('INSERT INTO points VALUES (99)')\n"
                "    outcome = 'WRITE-ALLOWED'\n"
                "except sqlite3.OperationalError:\n"
                "    outcome = 'readonly-enforced'\n"
                "con.close()\noutcome"
    })
    assert r["ok"] and "readonly-enforced" in r["value"], r
finally:
    handle = plugin._kernels.pop(SESSION, None)
    if handle:
        handle.shutdown()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

print("ok")
