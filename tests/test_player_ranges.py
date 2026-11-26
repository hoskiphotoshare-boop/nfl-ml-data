import json
from pathlib import Path


def test_player_ranges_bracket_projection():
    path = Path(__file__).resolve().parent.parent / "predictions" / "players_week.json"
    players = json.loads(path.read_text(encoding="utf-8"))
    bad = []
    for player in players:
        floor = player.get("floor_fantasy_points")
        proj = player.get("projected_fantasy_points")
        ceiling = player.get("ceiling_fantasy_points")
        if floor is None or proj is None or ceiling is None:
            bad.append((player.get("player_id"), "missing"))
            continue
        if floor > proj + 0.01 or proj > ceiling + 0.01:
            bad.append((player.get("player_id"), floor, proj, ceiling))
    assert not bad, f"floor/ceiling does not bracket projection: {bad[:10]}"
