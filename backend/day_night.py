"""
Day/Night Cycle (Phase 11).

Tracks in-game time of day per session. Each turn advances ~30 minutes.
Affects:
  - Encounter rates (night = +50% danger)
  - Narrator system prompt (time-of-day flavor)
  - NPC availability hints (some NPCs only present at night)
  - Scene atmosphere overrides
"""
from typing import Dict, Tuple

# Turn-to-time mapping: turn 1 = 8:00 AM, each turn = 30 min
START_HOUR = 8       # 8 AM start
MINUTES_PER_TURN = 30
TURNS_PER_DAY = 48   # 24 hours × 60 min / 30 min per turn

# Time-of-day phases: (start_hour, end_hour, phase_name, emoji)
PHASES: list = [
    (5,  7,  "dawn",     "🌅", "The first grey light of morning"),
    (7,  11, "morning",  "🌄", "The morning sun rises high"),
    (11, 14, "midday",   "☀️", "The sun stands high overhead"),
    (14, 17, "afternoon","🌤️", "Long golden afternoon shadows"),
    (17, 19, "dusk",     "🌆", "The sky turns purple and orange"),
    (19, 22, "evening",  "🌃", "Lanterns flicker on as twilight settles"),
    (22, 24, "night",    "🌙", "Deep night cloaks everything"),
    (0,  5,  "midnight", "🌑", "The darkest watches of the night"),
]


def turn_to_time(turn: int) -> Tuple[int, int]:
    """Convert turn number to (hour, minute) tuple in 24h format."""
    total_minutes = (turn - 1) * MINUTES_PER_TURN + START_HOUR * 60
    total_minutes %= 24 * 60
    return total_minutes // 60, total_minutes % 60


def get_phase(turn: int) -> Dict:
    """Return the time-of-day phase for this turn."""
    hour, minute = turn_to_time(turn)
    for start, end, name, emoji, description in PHASES:
        if start <= hour < end:
            return {
                "phase": name,
                "emoji": emoji,
                "description": description,
                "hour": hour,
                "minute": minute,
                "display": f"{hour:02d}:{minute:02d}",
                "is_night": name in ("evening", "night", "midnight"),
                "is_dawn_dusk": name in ("dawn", "dusk"),
            }
    # Fallback for midnight (0:00-5:00 wraps)
    return {
        "phase": "midnight", "emoji": "🌑",
        "description": "The darkest watches of the night",
        "hour": hour, "minute": minute,
        "display": f"{hour:02d}:{minute:02d}",
        "is_night": True, "is_dawn_dusk": False,
    }


def time_prompt(turn: int) -> str:
    """Get a narrator system-prompt addendum for the current time of day."""
    p = get_phase(turn)
    return (
        f"\n[TIME OF DAY: {p['display']} — {p['description']}. "
        f"Weave subtle references to {p['phase']} into your narration.]"
    )


def night_encounter_multiplier(turn: int) -> float:
    """Encounter probability multiplier (1.0 = normal, 1.5 = dangerous night)."""
    p = get_phase(turn)
    if p["is_night"]:
        return 1.5
    if p["is_dawn_dusk"]:
        return 1.2
    return 1.0
