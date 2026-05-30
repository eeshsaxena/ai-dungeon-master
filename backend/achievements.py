"""
Achievement System (Phase 11).

30 in-game achievements tracking player progress. Persists per session.
Achievement unlocks fire a toast notification in the frontend.

Hook points (call check_achievements with relevant event types):
  - "combat_won"       — defeated an enemy
  - "level_up"         — reached a new level
  - "quest_completed"  — finished a quest
  - "item_used"        — consumed/used an item
  - "location_visited" — discovered a new location
  - "dialogue_started" — spoke with an NPC
  - "spell_learned"    — added a spell to spellbook
  - "save_loaded"      — restored a save file
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Set

ACHIEVEMENT_DIR = Path(__file__).parent / "achievements_data"
ACHIEVEMENT_DIR.mkdir(exist_ok=True)


ACHIEVEMENTS: Dict[str, Dict[str, Any]] = {
    # ── First-time / progression ────────────────────────────────────────────
    "first_steps":      {"name": "First Steps",      "emoji": "👣", "desc": "Take your first action in the wizarding world", "secret": False},
    "first_blood":      {"name": "First Blood",      "emoji": "⚔️", "desc": "Win your first combat",                          "secret": False},
    "level_5":          {"name": "Adept Witch/Wizard","emoji": "✨","desc": "Reach Level 5",                                   "secret": False},
    "level_10":         {"name": "Auror Trainee",    "emoji": "🛡️", "desc": "Reach Level 10",                                  "secret": False},
    "level_15":         {"name": "Senior Auror",     "emoji": "⭐", "desc": "Reach Level 15",                                  "secret": False},
    "level_20":         {"name": "Legendary",        "emoji": "👑", "desc": "Reach maximum Level 20",                          "secret": False},

    # ── Combat ──────────────────────────────────────────────────────────────
    "death_eater_slayer":   {"name": "Death Eater Slayer",   "emoji": "💀", "desc": "Defeat 5 Death Eaters",                  "secret": False, "counter": "death_eater_kills", "threshold": 5},
    "dementor_repellent":   {"name": "Dementor Repellent",   "emoji": "🌫️", "desc": "Defeat 3 Dementors",                     "secret": False, "counter": "dementor_kills", "threshold": 3},
    "boggart_buster":       {"name": "Boggart Buster",       "emoji": "🃏", "desc": "Defeat a Boggart",                       "secret": False},
    "hollow_mage_defeated": {"name": "Hollow Mage Defeated", "emoji": "🌑", "desc": "Defeat the Hollow Mage",                 "secret": True},
    "untouchable":          {"name": "Untouchable",          "emoji": "💎", "desc": "Win a combat without taking damage",     "secret": False},

    # ── Exploration ─────────────────────────────────────────────────────────
    "wanderer":         {"name": "Wanderer",        "emoji": "🗺️", "desc": "Visit 5 different locations",                  "secret": False, "counter": "locations_visited", "threshold": 5},
    "world_walker":     {"name": "World Walker",    "emoji": "🌍", "desc": "Visit all 7 main locations",                     "secret": False, "counter": "locations_visited", "threshold": 7},
    "night_owl":        {"name": "Night Owl",       "emoji": "🦉", "desc": "Survive 10 turns at night",                      "secret": False, "counter": "night_turns", "threshold": 10},

    # ── Quests ──────────────────────────────────────────────────────────────
    "quest_starter":    {"name": "Quest Starter",   "emoji": "📜", "desc": "Complete your first quest",                      "secret": False},
    "quest_hunter":     {"name": "Quest Hunter",    "emoji": "🏹", "desc": "Complete 5 quests",                              "secret": False, "counter": "quests_completed", "threshold": 5},
    "story_complete":   {"name": "The Hollow Saga", "emoji": "📚", "desc": "Complete the main story arc",                    "secret": True},

    # ── Items ───────────────────────────────────────────────────────────────
    "first_potion":     {"name": "First Potion",    "emoji": "🧪", "desc": "Use your first potion",                          "secret": False},
    "collector":        {"name": "Collector",       "emoji": "🎒", "desc": "Have 10+ items in your inventory at once",       "secret": False},
    "legendary_owner":  {"name": "Legendary Owner", "emoji": "🪄", "desc": "Obtain a legendary item",                        "secret": False},
    "brewer":           {"name": "Master Brewer",   "emoji": "⚗️", "desc": "Successfully brew a potion",                     "secret": False},

    # ── Dialogue / NPC ──────────────────────────────────────────────────────
    "social_butterfly": {"name": "Social Butterfly","emoji": "🦋", "desc": "Speak with 5 different NPCs",                    "secret": False, "counter": "npcs_spoken_to", "threshold": 5},
    "remembered":       {"name": "Remembered",      "emoji": "💭", "desc": "An NPC recalls your past actions",                "secret": False},

    # ── Spells ──────────────────────────────────────────────────────────────
    "spell_scholar":    {"name": "Spell Scholar",   "emoji": "📖", "desc": "Learn 10 different spells",                      "secret": False, "counter": "spells_known", "threshold": 10},
    "patronus_caster":  {"name": "Patronus Caster", "emoji": "🦌", "desc": "Cast Expecto Patronum",                          "secret": False},

    # ── Achievement / meta ──────────────────────────────────────────────────
    "save_scummer":     {"name": "Save Scummer",    "emoji": "💾", "desc": "Load a saved game",                              "secret": True},
    "time_traveler":    {"name": "Time Traveler",   "emoji": "⏳", "desc": "Use a Time-Turner Fragment",                     "secret": True},
    "the_chosen":       {"name": "The Chosen One",  "emoji": "⚡", "desc": "Unlock all other achievements",                  "secret": True},

    # ── Difficulty ──────────────────────────────────────────────────────────
    "no_mercy":         {"name": "No Mercy",        "emoji": "🔥", "desc": "Win a combat on Legendary difficulty",           "secret": False},
    "veteran":          {"name": "Veteran",         "emoji": "🎖️", "desc": "Play 100 turns",                                 "secret": False, "counter": "turns_played", "threshold": 100},
}


class AchievementTracker:
    """Per-session achievement progress."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.unlocked: Set[str] = set()
        self.counters: Dict[str, int] = {}
        self.unlocked_at: Dict[str, float] = {}
        self._path = ACHIEVEMENT_DIR / f"{session_id}.json"
        self._load()

    # ── Public ────────────────────────────────────────────────────────────

    def unlock(self, achievement_id: str) -> bool:
        """Unlock an achievement. Returns True if newly unlocked."""
        if achievement_id in self.unlocked or achievement_id not in ACHIEVEMENTS:
            return False
        self.unlocked.add(achievement_id)
        self.unlocked_at[achievement_id] = time.time()
        self._save()
        self._check_meta()
        return True

    def increment(self, counter: str, amount: int = 1) -> List[str]:
        """Bump a counter. Returns list of newly-unlocked achievement IDs."""
        self.counters[counter] = self.counters.get(counter, 0) + amount
        newly_unlocked = []
        for aid, ach in ACHIEVEMENTS.items():
            if ach.get("counter") == counter and aid not in self.unlocked:
                if self.counters[counter] >= ach.get("threshold", 1):
                    if self.unlock(aid):
                        newly_unlocked.append(aid)
        if newly_unlocked or counter:
            self._save()
        return newly_unlocked

    def is_unlocked(self, achievement_id: str) -> bool:
        return achievement_id in self.unlocked

    def progress(self) -> Dict[str, Any]:
        """Return achievement progress for UI."""
        out = []
        for aid, ach in ACHIEVEMENTS.items():
            entry = {
                "id": aid,
                "name": ach["name"],
                "emoji": ach["emoji"],
                "description": ach["desc"],
                "secret": ach.get("secret", False),
                "unlocked": aid in self.unlocked,
                "unlocked_at": self.unlocked_at.get(aid),
            }
            if "counter" in ach:
                entry["progress"] = self.counters.get(ach["counter"], 0)
                entry["threshold"] = ach.get("threshold", 1)
            # Hide secret achievements that aren't unlocked
            if entry["secret"] and not entry["unlocked"]:
                entry["name"] = "???"
                entry["description"] = "Secret achievement"
            out.append(entry)
        return {
            "total": len(ACHIEVEMENTS),
            "unlocked": len(self.unlocked),
            "achievements": out,
        }

    # ── Event handlers (call these from main.py) ──────────────────────────

    def on_event(self, event: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Dispatch an event and return list of newly-unlocked achievement dicts.
        Events: combat_won, level_up, quest_completed, item_used,
                location_visited, dialogue_started, spell_learned, save_loaded.
        """
        newly: List[str] = []

        if event == "first_action":
            if self.unlock("first_steps"): newly.append("first_steps")

        elif event == "combat_won":
            if self.unlock("first_blood"): newly.append("first_blood")
            archetype = payload.get("archetype", "")
            if archetype == "Death Eater":  newly.extend(self.increment("death_eater_kills"))
            elif archetype == "Dementor":   newly.extend(self.increment("dementor_kills"))
            elif archetype == "Boggart":    self.unlock("boggart_buster") and newly.append("boggart_buster")
            elif archetype == "Hollow Mage":
                self.unlock("hollow_mage_defeated") and newly.append("hollow_mage_defeated")
                self.unlock("story_complete") and newly.append("story_complete")
            if payload.get("took_damage", True) is False:
                self.unlock("untouchable") and newly.append("untouchable")
            if payload.get("difficulty") == "legendary":
                self.unlock("no_mercy") and newly.append("no_mercy")

        elif event == "level_up":
            lvl = payload.get("level", 1)
            for threshold, aid in [(5, "level_5"), (10, "level_10"), (15, "level_15"), (20, "level_20")]:
                if lvl >= threshold and self.unlock(aid):
                    newly.append(aid)

        elif event == "quest_completed":
            if self.unlock("quest_starter"): newly.append("quest_starter")
            newly.extend(self.increment("quests_completed"))

        elif event == "item_used":
            item = payload.get("item", {})
            if item.get("type") == "consumable":
                self.unlock("first_potion") and newly.append("first_potion")
            if item.get("rarity") == "legendary":
                self.unlock("legendary_owner") and newly.append("legendary_owner")
            if item.get("id") == "time_turner":
                self.unlock("time_traveler") and newly.append("time_traveler")

        elif event == "inventory_update":
            if payload.get("count", 0) >= 10:
                self.unlock("collector") and newly.append("collector")

        elif event == "location_visited":
            if payload.get("first_time"):
                visited = payload.get("total_visited", 1)
                self.counters["locations_visited"] = visited
                for threshold, aid in [(5, "wanderer"), (7, "world_walker")]:
                    if visited >= threshold and self.unlock(aid):
                        newly.append(aid)
                self._save()

        elif event == "dialogue_started":
            newly.extend(self.increment("npcs_spoken_to"))
            if payload.get("npc_recalled_memory"):
                self.unlock("remembered") and newly.append("remembered")

        elif event == "spell_learned":
            spell = payload.get("spell", "")
            if spell == "Expecto Patronum":
                self.unlock("patronus_caster") and newly.append("patronus_caster")
            count = payload.get("total_spells", 1)
            self.counters["spells_known"] = max(self.counters.get("spells_known", 0), count)
            if count >= 10:
                self.unlock("spell_scholar") and newly.append("spell_scholar")
                self._save()

        elif event == "save_loaded":
            self.unlock("save_scummer") and newly.append("save_scummer")

        elif event == "potion_brewed":
            self.unlock("brewer") and newly.append("brewer")

        elif event == "turn_played":
            night = payload.get("is_night", False)
            self.increment("turns_played")
            if night:
                newly.extend(self.increment("night_turns"))
            if self.counters.get("turns_played", 0) >= 100:
                self.unlock("veteran") and newly.append("veteran")

        # Convert IDs to full dicts for frontend
        return [
            {"id": aid, **ACHIEVEMENTS[aid]}
            for aid in set(newly) if aid in ACHIEVEMENTS
        ]

    # ── Internal ──────────────────────────────────────────────────────────

    def _check_meta(self) -> None:
        """Check if all non-meta achievements are unlocked → unlock 'the_chosen'."""
        non_meta = {aid for aid in ACHIEVEMENTS if aid != "the_chosen"}
        if non_meta.issubset(self.unlocked):
            self.unlock("the_chosen")

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self.unlocked = set(data.get("unlocked", []))
                self.counters = data.get("counters", {})
                self.unlocked_at = data.get("unlocked_at", {})
            except Exception:
                pass

    def _save(self) -> None:
        try:
            self._path.write_text(json.dumps({
                "unlocked": sorted(self.unlocked),
                "counters": self.counters,
                "unlocked_at": self.unlocked_at,
            }, indent=2), encoding="utf-8")
        except Exception:
            pass


# Global registry: session_id → tracker
_trackers: Dict[str, AchievementTracker] = {}


def get_tracker(session_id: str) -> AchievementTracker:
    if session_id not in _trackers:
        _trackers[session_id] = AchievementTracker(session_id)
    return _trackers[session_id]
