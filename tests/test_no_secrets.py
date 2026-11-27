from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_credentials as cc  # noqa: E402


def test_scanner_flags_sample_tokens():
    sample = "token = " + "ghp_" + "A" * 30
    hits = cc.scan_text("example.py", sample)
    assert hits and hits[0][2] == "classic PAT"


def test_scanner_flags_credential_in_remote_url():
    sample = "https://user:" + "secret" + "@github.com/org/repo.git"
    hits = cc.scan_text(".git/config", sample)
    assert hits and hits[0][2] == "credential in URL"


def test_scanner_passes_clean_text():
    assert cc.scan_text("README.md", "Use a credential helper, never a URL token.") == []


def test_repo_has_no_tracked_credentials():
    result = subprocess.run(
        [sys.executable, "scripts/check_credentials.py", "--quiet"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
