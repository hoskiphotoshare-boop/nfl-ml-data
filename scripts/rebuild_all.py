#!/usr/bin/env python3
"""Local full rebuild in the correct order. Do not skip steps."""
import subprocess, sys

steps = [
    ["python", "scripts/build_features.py"],
    ["python", "scripts/add_match_context.py"],
    ["python", "scripts/add_player_context.py"],
    ["python", "scripts/train_models.py"],
    ["python", "scripts/component_models.py", "--train"],
    ["python", "scripts/predict_upcoming.py"],
    ["python", "scripts/adjust_player_weather.py"],
    ["python", "scripts/add_player_matchups.py"],
    ["python", "scripts/compare_market.py"],
    ["python", "scripts/update_calls.py"],
    ["python", "scripts/explain_models.py"],
    ["python", "scripts/prepare_site.py"],
]
for step in steps:
    print(">>", " ".join(step))
    subprocess.run(step, check=True)
print("Rebuild complete.")
