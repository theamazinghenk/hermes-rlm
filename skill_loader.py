"""Python-backed skills for the RLM kernel (phase 3).

A normal Hermes skill is markdown: the model reads it and then writes the
code itself, every time. A Python-backed skill also ships an importable
package, so a proven workflow becomes a typed callable:

    report = trade_review(db_path="...", days=7)

instead of thirty lines the model re-derives on each run.

Discovery is filesystem-based and lazy: only names and one-line summaries
are cheap to list; the body is loaded when the model actually imports it.
A skill directory qualifies when it contains a `python/` subdirectory:

    <skill>/SKILL.md          instructions (unchanged, still markdown)
    <skill>/python/__init__.py   importable package, same name as the skill

The kernel gets those `python/` parents on `sys.path`, so `import <name>`
works with no install step and no dependency on the Hermes skill loader.
"""

from __future__ import annotations

import json
import re
import os
from pathlib import Path

def _hermes_home() -> Path:
    """Hermes home with HERMES_HOME respected (fleet/profile isolation)."""
    env = os.environ.get("HERMES_HOME")
    return Path(env).expanduser() if env else Path.home() / ".hermes"


SKILL_ROOTS = [
    _hermes_home() / "skills",
    _hermes_home() / "plugins",
]

_NAME_RE = re.compile(r"^name:\s*(.+?)\s*$", re.MULTILINE)
_DESC_RE = re.compile(r"^description:\s*(.+?)\s*$", re.MULTILINE)


def _frontmatter(skill_md: Path) -> tuple[str, str]:
    """Pull name/description without a YAML dependency."""
    try:
        head = skill_md.read_text(encoding="utf-8", errors="ignore")[:2000]
    except OSError:
        return skill_md.parent.name, ""
    name = _NAME_RE.search(head)
    desc = _DESC_RE.search(head)
    return (
        name.group(1) if name else skill_md.parent.name,
        (desc.group(1) if desc else "")[:200],
    )


def discover(roots: list[Path] | None = None) -> list[dict]:
    """Find skills that expose an importable package under `python/`.

    Layout: `<skill>/python/<module>/__init__.py`. The `python/` directory is
    what goes on sys.path, so `import <module>` works with no install step.
    """
    found: list[dict] = []
    seen: set[str] = set()
    for root in roots or SKILL_ROOTS:
        if not root.is_dir():
            continue
        for pkg_init in sorted(root.glob("**/python/*/__init__.py")):
            python_dir = pkg_init.parent.parent
            skill_dir = python_dir.parent
            module = pkg_init.parent.name
            skill_md = skill_dir / "SKILL.md"
            name, description = (
                _frontmatter(skill_md) if skill_md.is_file() else (skill_dir.name, "")
            )
            if module in seen:
                continue
            seen.add(module)
            found.append({
                "name": name,
                "import_as": module,
                "description": description,
                "path": str(skill_dir),
                "sys_path_entry": str(python_dir),
            })
    return found


def sys_path_entries(roots: list[Path] | None = None) -> list[str]:
    """Directories to add to the kernel's sys.path for `import <skill>`."""
    return sorted({entry["sys_path_entry"] for entry in discover(roots)})


def catalog_json(roots: list[Path] | None = None) -> str:
    entries = discover(roots)
    return json.dumps({
        "count": len(entries),
        "skills": [
            {k: entry[k] for k in ("name", "import_as", "description", "path")}
            for entry in entries
        ],
        "usage": "Inside rlm_exec: `import <import_as>` then call its documented API.",
    }, ensure_ascii=False, indent=2)
