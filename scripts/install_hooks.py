#!/usr/bin/env python3
"""Point Git at the versioned .githooks directory (run once per clone)."""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(["git", "config", "core.hooksPath", ".githooks"], cwd=root, check=True)
    hook = os.path.join(root, ".githooks", "pre-commit")
    if os.path.exists(hook):
        os.chmod(hook, 0o755)
    print("Git hooks enabled: core.hooksPath -> .githooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
