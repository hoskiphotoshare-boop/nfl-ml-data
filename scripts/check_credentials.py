#!/usr/bin/env python3
"""Fail when a GitHub credential is present in tracked files or .git/config.

Credentials belong in a credential helper, repository secrets, or Apps Script
properties -- never in a remote URL or a committed file. This check is wired
into the test suite and into the optional pre-commit hook installed by
``scripts/install_hooks.py``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

PATTERNS = [
    ("classic PAT", re.compile(r"ghp_[A-Za-z0-9]{20,}")),
    ("fine-grained PAT", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    ("oauth token", re.compile(r"gho_[A-Za-z0-9]{20,}")),
    ("server token", re.compile(r"ghs_[A-Za-z0-9]{20,}")),
    ("user token", re.compile(r"ghu_[A-Za-z0-9]{20,}")),
    (
        "credential in URL",
        re.compile(r"https://[^\s/@]+:[^\s/@]+@(?:github\.com|api\.github\.com)"),
    ),
]

SKIP_SUFFIXES = {
    ".parquet",
    ".joblib",
    ".pkl",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".zip",
    ".gz",
    ".woff",
    ".woff2",
    ".ttf",
    ".pdf",
    ".mp4",
    ".webm",
}

MAX_BYTES = 2_000_000


def mask(value: str) -> str:
    """Redact anything that looks like a credential before printing it."""
    value = re.sub(r"gh[pousr]_[A-Za-z0-9_]{8,}", "***TOKEN***", value)
    value = re.sub(r"github_pat_[A-Za-z0-9_]{8,}", "***TOKEN***", value)
    return value


def scan_text(path: str, text: str) -> list[tuple[str, int, str, str]]:
    hits: list[tuple[str, int, str, str]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for kind, pattern in PATTERNS:
            if pattern.search(line):
                hits.append((path, line_no, kind, mask(line.strip())[:200]))
                break
    return hits


def tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [p for p in out.decode("utf-8", "replace").split("\0") if p]


def git_config_paths() -> list[str]:
    paths = []
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--git-path", "config"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        out = ""
    if out and os.path.exists(out):
        paths.append(out)
    return paths


def scan_file(path: str) -> list[tuple[str, int, str, str]]:
    if os.path.splitext(path)[1].lower() in SKIP_SUFFIXES:
        return []
    try:
        if os.path.getsize(path) > MAX_BYTES:
            return []
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return scan_text(path, handle.read())
    except OSError:
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only print failures",
    )
    args = parser.parse_args()

    hits: list[tuple[str, int, str, str]] = []
    for path in tracked_files():
        hits.extend(scan_file(path))
    for path in git_config_paths():
        hits.extend(scan_file(path))

    if not hits:
        if not args.quiet:
            print("No GitHub credentials found in tracked files or .git/config.")
        return 0

    print("GitHub credentials detected:", file=sys.stderr)
    for path, line_no, kind, evidence in hits:
        print(f"  {path}:{line_no}: {kind}: {evidence}", file=sys.stderr)
    print(
        "\nRemove the credential, rotate it, and move it to a credential helper "
        "or repository secret.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
