#!/usr/bin/env python3
"""Assemble the static Pages artifact before a deploy.

`frontend/data/` is gitignored, so a manual `wrangler pages deploy` after a
pipeline run can silently ship stale JSON unless it is refreshed first. This
script mirrors the current predictions/reports, regenerates the team pages and
fails if the artifact is incomplete or the mirrored meta is stale.

Usage:
    .venv/bin/python scripts/prepare_site.py
    npx wrangler pages deploy frontend --project-name nfl-ml-data --commit-dirty=true
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"


def mirror(source: Path, target: Path) -> int:
    target.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("*.json"))
    for path in files:
        shutil.copy2(path, target / path.name)
    return len(files)


def main() -> None:
    predictions = mirror(ROOT / "predictions", FRONTEND / "data" / "predictions")
    reports = mirror(ROOT / "reports", FRONTEND / "data" / "reports")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_team_pages.py")],
        cwd=ROOT,
        check=True,
    )

    source_meta = json.loads((ROOT / "predictions" / "meta.json").read_text())
    mirror_meta = json.loads(
        (FRONTEND / "data" / "predictions" / "meta.json").read_text()
    )
    if source_meta.get("generated_at") != mirror_meta.get("generated_at"):
        raise SystemExit(
            "frontend/data/predictions/meta.json is stale: "
            f"{mirror_meta.get('generated_at')} != {source_meta.get('generated_at')}"
        )

    pages = sorted((FRONTEND / "teams").glob("*.html"))
    if len(pages) < 33:
        raise SystemExit(f"only {len(pages)} team pages generated; expected at least 33")

    sitemap = (FRONTEND / "sitemap.xml").read_text()
    urls = sitemap.count("<loc>")
    if urls < 34:
        raise SystemExit(f"sitemap has {urls} URLs; expected at least 34")

    print(
        f"Prepared frontend: {predictions} prediction files, {reports} reports, "
        f"{len(pages)} team pages, {urls} sitemap URLs; "
        f"meta {source_meta.get('generated_at')}"
    )


if __name__ == "__main__":
    main()
