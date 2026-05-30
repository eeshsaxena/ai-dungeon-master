"""
Disk-backed save/load system for AI Dungeon Master.
Saves complete session state (player stats + story memory key beats) to JSON.
Save files live in backend/saves/ (gitignored).
"""
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

SAVE_DIR = Path(__file__).parent / "saves"
MAX_SAVES = 30  # cap on files kept per player


class SaveManager:
    def __init__(self):
        SAVE_DIR.mkdir(exist_ok=True)

    # ── Write ──────────────────────────────────────────────────────────────

    def save(self, session_id: str, player_data: dict, memory_data: dict) -> str:
        """Persist a session to disk. Returns the save filename."""
        player_name = player_data.get("name", "unknown").replace(" ", "_")[:20]
        timestamp = int(time.time())
        filename = f"{player_name}_{timestamp}.json"
        path = SAVE_DIR / filename
        payload: Dict[str, Any] = {
            "version": 1,
            "saved_at": timestamp,
            "session_id": session_id,
            "player": player_data,
            "memory": memory_data,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self._prune_old_saves(player_name)
        return filename

    def _prune_old_saves(self, player_name: str) -> None:
        """Keep only the most recent MAX_SAVES files for a given player name."""
        prefix = player_name + "_"
        files = sorted(
            [p for p in SAVE_DIR.glob("*.json") if p.name.startswith(prefix)],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in files[MAX_SAVES:]:
            try:
                old.unlink()
            except OSError:
                pass

    # ── Read ───────────────────────────────────────────────────────────────

    def list_saves(self) -> List[Dict[str, Any]]:
        """Return metadata for all save files, newest first."""
        saves = []
        for p in sorted(SAVE_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                player = data.get("player", {})
                saves.append({
                    "filename": p.name,
                    "player_name": player.get("name", "Unknown"),
                    "level": player.get("level", 1),
                    "title": player.get("title", "First-Year"),
                    "turns": player.get("turns_played", 0),
                    "location": player.get("current_location", ""),
                    "house": player.get("house", "Unselected"),
                    "xp": player.get("xp", 0),
                    "saved_at": data.get("saved_at", 0),
                })
            except Exception:
                pass
        return saves[:50]

    def load(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a save file by filename. Returns the raw payload dict or None."""
        path = SAVE_DIR / filename
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ── Delete ─────────────────────────────────────────────────────────────

    def delete(self, filename: str) -> bool:
        path = SAVE_DIR / filename
        if path.exists():
            path.unlink()
            return True
        return False

    def stats(self) -> Dict[str, Any]:
        files = list(SAVE_DIR.glob("*.json"))
        return {"save_count": len(files), "save_dir": str(SAVE_DIR)}


save_manager = SaveManager()
