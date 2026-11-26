import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

import build_team_pages


def test_meter_renders_decimal_suffix():
    html = build_team_pages.meter(0.5)
    assert "50%" in html
    assert 'class="dec"' in html
    assert "2.00" in html


def test_page_head_has_pwa_tags():
    html = build_team_pages.page(
        "Test Team",
        "Test description",
        "<p>body</p>",
        "https://nfl-ml-data.pages.dev/teams/TEST",
    )
    assert 'rel="manifest" href="/manifest.webmanifest"' in html
    assert 'rel="apple-touch-icon"' in html
    assert 'src="/pwa.js"' in html
    assert 'name="apple-mobile-web-app-status-bar-style" content="default"' in html
