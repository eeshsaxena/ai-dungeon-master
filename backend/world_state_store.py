"""
World State Persistence (Phase 9).

Tracks mutable world state that changes during play and must survive server
restarts: quest completion, location discoveries, faction reputation changes,
and world events triggered by player choices.

The WizardingWorldGraph (knowledge_graph.py) is loaded fresh from JSON on
every startup — it has no session-level mutations. This module overlays a
persistent mutation layer on top of it.

Storage: backend/world_state/<session_id>.json (gitignored)
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Set

STATE_DIR = Path(__file__).parent / "world_state"
STATE_DIR.mkdir(exist_ok=True)


class WorldStateStore:
    """
    Per-session persistent world state overlay.
    Stores only the *delta* from the static JSON data — never the whole graph.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._path = STATE_DIR / f"{session_id}.json"

        # Quest state: quest_id → "active"|"completed"|"failed"|"available"
        self.quest_statuses: Dict[str, str] = {}
        # Objective completion: quest_id → set of objective ids completed
        self.objectives_completed: Dict[str, List[str]] = {}
        # Locations the player has visited
        self.visited_locations: Set[str] = set()
        # World events that have been triggered (string tags)
        self.world_events: List[str] = []
        # Reputation delta on top of defaults: faction_id → delta int
        self.reputation_delta: Dict[str, int] = {}
        # NPC states: npc_id → dict of mutable attrs (e.g. {"alive": False})
        self.npc_states: Dict[str, Dict[str, Any]] = {}
        # Free-form world flags: arbitrary bool flags set by story logic
        self.flags: Dict[str, bool] = {}

        self._load()

    # ── Quest API ─────────────────────────────────────────────────────────────

    def set_quest_status(self, quest_id: str, status: str) -> None:
        self.quest_statuses[quest_id] = status
        self._save()

    def complete_objective(self, quest_id: str, objective_id: str) -> None:
        self.objectives_completed.setdefault(quest_id, [])
        if objective_id not in self.objectives_completed[quest_id]:
            self.objectives_completed[quest_id].append(objective_id)
        self._save()

    def get_quest_status(self, quest_id: str, default: str = "locked") -> str:
        return self.quest_statuses.get(quest_id, default)

    def completed_quests(self) -> List[str]:
        return [q for q, s in self.quest_statuses.items() if s == "completed"]

    # ── Location API ──────────────────────────────────────────────────────────

    def visit_location(self, location_id: str) -> bool:
        """Mark location visited. Returns True if this is the first visit."""
        first = location_id not in self.visited_locations
        self.visited_locations.add(location_id)
        if first:
            self._save()
        return first

    def has_visited(self, location_id: str) -> bool:
        return location_id in self.visited_locations

    # ── World events & flags ──────────────────────────────────────────────────

    def add_event(self, event: str) -> None:
        self.world_events.append(event)
        if len(self.world_events) > 200:
            self.world_events = self.world_events[-200:]
        self._save()

    def set_flag(self, flag: str, value: bool = True) -> None:
        self.flags[flag] = value
        self._save()

    def get_flag(self, flag: str, default: bool = False) -> bool:
        return self.flags.get(flag, default)

    # ── Reputation ────────────────────────────────────────────────────────────

    def adjust_reputation(self, faction: str, delta: int) -> None:
        self.reputation_delta[faction] = self.reputation_delta.get(faction, 0) + delta
        self._save()

    def get_reputation_deltas(self) -> Dict[str, int]:
        return dict(self.reputation_delta)

    # ── NPC states ────────────────────────────────────────────────────────────

    def set_npc_state(self, npc_id: str, key: str, value: Any) -> None:
        self.npc_states.setdefault(npc_id, {})[key] = value
        self._save()

    def get_npc_state(self, npc_id: str) -> Dict[str, Any]:
        return self.npc_states.get(npc_id, {})

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "saved_at": time.time(),
            "quest_statuses": self.quest_statuses,
            "objectives_completed": self.objectives_completed,
            "visited_locations": sorted(self.visited_locations),
            "world_events": self.world_events[-50:],  # last 50 only
            "reputation_delta": self.reputation_delta,
            "npc_states": self.npc_states,
            "flags": self.flags,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        self.quest_statuses = data.get("quest_statuses", {})
        self.objectives_completed = data.get("objectives_completed", {})
        self.visited_locations = set(data.get("visited_locations", []))
        self.world_events = data.get("world_events", [])
        self.reputation_delta = data.get("reputation_delta", {})
        self.npc_states = data.get("npc_states", {})
        self.flags = data.get("flags", {})

    def _load(self) -> None:
        if self._path.exists():
            try:
                self.from_dict(json.loads(self._path.read_text(encoding="utf-8")))
            except Exception as e:
                print(f"[WorldState] Could not load {self._path}: {e}")

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self.to_dict(), indent=2), encoding="utf-8"
            )
        except Exception as e:
            print(f"[WorldState] Could not save: {e}")

    def delete(self) -> None:
        if self._path.exists():
            self._path.unlink()

    def summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "quests_active": sum(1 for s in self.quest_statuses.values() if s == "active"),
            "quests_completed": len(self.completed_quests()),
            "locations_visited": len(self.visited_locations),
            "world_events": len(self.world_events),
            "flags": self.flags,
        }


# ── Global store registry ─────────────────────────────────────────────────────

_stores: Dict[str, WorldStateStore] = {}


def get_world_state(session_id: str) -> WorldStateStore:
    """Get (or lazily create) the world state for a session."""
    if session_id not in _stores:
        _stores[session_id] = WorldStateStore(session_id)
    return _stores[session_id]


def drop_world_state(session_id: str) -> None:
    _stores.pop(session_id, None)
