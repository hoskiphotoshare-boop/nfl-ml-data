-- D1 schema for the NFL picks Worker
CREATE TABLE IF NOT EXISTS picks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  game_id INTEGER NOT NULL,
  pick TEXT NOT NULL CHECK (pick IN ('home', 'away')),
  confidence REAL DEFAULT 0,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(user_id, game_id)
);

CREATE INDEX IF NOT EXISTS idx_picks_user ON picks(user_id);
CREATE INDEX IF NOT EXISTS idx_picks_game ON picks(game_id);
