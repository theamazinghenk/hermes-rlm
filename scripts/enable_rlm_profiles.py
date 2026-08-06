#!/usr/bin/env python3
"""Enable the hermes-rlm plugin across Hermes profiles.

Every profile carries its own `plugins.enabled` list, so a plugin enabled in
the main config does NOT reach them. This adds `hermes-rlm` to each profile
that should have it, idempotently, with a timestamped backup per file.

Profiles with `safe_mode: true` in their config are skipped automatically:
the RLM kernel executes arbitrary model-written Python with user
permissions, which is the opposite of safe mode. Add more exclusions with
--exclude.

Usage:
    python3 enable_rlm_profiles.py --dry-run          # show what would change
    python3 enable_rlm_profiles.py                    # apply
    python3 enable_rlm_profiles.py --exclude a,b      # extra profiles to skip
    python3 enable_rlm_profiles.py --revert TS        # undo via backup stamp
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import sys
from pathlib import Path

PROFILES_DIR = Path.home() / ".hermes" / "profiles"
PLUGIN = "hermes-rlm"
EXCLUDE: set[str] = set()  # extended via --exclude; safe_mode is auto-detected

STAMP = dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def profile_configs() -> list[Path]:
    if not PROFILES_DIR.is_dir():
        return []
    return sorted(p / "config.yaml" for p in PROFILES_DIR.iterdir()
                  if (p / "config.yaml").is_file())


def add_plugin(text: str) -> tuple[str, str]:
    """Return (new_text, status). Status is added / already / no-block."""
    if re.search(rf"^\s*-\s*{re.escape(PLUGIN)}\s*$", text, re.MULTILINE):
        return text, "already"

    # Find `plugins:` then its `enabled:` child, and append one list item
    # using the indentation of the existing entries.
    match = re.search(
        r"(^plugins:\n(?:[ \t]+.*\n)*?[ \t]+enabled:\n)((?:[ \t]+-[ \t]*.*\n)*)",
        text, re.MULTILINE,
    )
    if not match:
        return text, "no-block"

    header, items = match.group(1), match.group(2)
    indent = "    "
    first = re.match(r"([ \t]+)-", items)
    if first:
        indent = first.group(1)
    if not items.endswith("\n") and items:
        items += "\n"
    new_items = items + f"{indent}- {PLUGIN}\n"
    return text[:match.start()] + header + new_items + text[match.end():], "added"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exclude", default="",
                    help="comma-separated profile names to skip")
    ap.add_argument("--revert", metavar="STAMP")
    args = ap.parse_args()

    configs = profile_configs()
    if not configs:
        print("no profile configs found")
        return 1

    if args.revert:
        restored = 0
        for cfg in configs:
            backup = cfg.with_suffix(f".yaml.bak-rlm-{args.revert}")
            if backup.is_file():
                shutil.copy2(backup, cfg)
                restored += 1
                print(f"  restored {cfg.parent.name}")
        print(f"\nrestored {restored} file(s) from stamp {args.revert}")
        return 0

    counts = {"added": 0, "already": 0, "no-block": 0, "skipped": 0}
    for cfg in configs:
        name = cfg.parent.name
        if name in EXCLUDE or (args.exclude and name in args.exclude.split(",")):
            counts["skipped"] = counts.get("skipped", 0) + 1
            print(f"  SKIP     {name}  (excluded)")
            continue
        if re.search(r"^\s*safe_mode:\s*true", cfg.read_text(), re.MULTILINE):
            counts["skipped"] += 1
            print(f"  SKIP     {name}  (safe_mode profile)")
            continue

        text = cfg.read_text(encoding="utf-8")
        new_text, status = add_plugin(text)
        counts[status] += 1

        if status == "added" and not args.dry_run:
            shutil.copy2(cfg, cfg.with_suffix(f".yaml.bak-rlm-{STAMP}"))
            cfg.write_text(new_text, encoding="utf-8")

        label = {"added": "ADD" if not args.dry_run else "WOULD-ADD",
                 "already": "OK", "no-block": "WARN"}[status]
        print(f"  {label:9s} {name}" + ("  (no plugins.enabled block)" if status == "no-block" else ""))

    print(f"\n{counts['added']} added, {counts['already']} already had it, "
          f"{counts['skipped']} skipped, {counts['no-block']} without a plugins block")
    if counts["added"] and not args.dry_run:
        print(f"backup stamp: {STAMP}   (revert: --revert {STAMP})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
