"""
Random Encounter Engine (Phase 11).

Triggers atmospheric or combat encounters when the player travels between
locations or stays in a dangerous area. Each location has a danger level
(0.0 safe ↔ 1.0 perilous) that controls encounter probability + severity.

Encounter types:
  - "flavor"   — atmospheric DM beat (no combat)
  - "npc"      — random NPC interaction
  - "combat"   — enemy spawns
  - "discovery"— find an item or lore clue

Time-of-day modifier: nighttime doubles dangerous encounter rate.
"""
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Per-location danger profile + flavor pool
LOCATION_PROFILES: Dict[str, Dict[str, Any]] = {
    "loc_001": {  # Three Broomsticks
        "danger": 0.05,
        "enemies": [],
        "flavor": [
            "The fire in the hearth flares green for a moment — someone is Floo-ing through, though no one arrives.",
            "A traveling musician strikes up a quiet ballad on a battered lute in the corner.",
            "Two students from Hogwarts hurry in, cheeks red, and order Butterbeers to warm up.",
            "Rosmerta catches your eye across the bar and gives the smallest meaningful nod.",
        ],
    },
    "loc_002": {  # Hogwarts
        "danger": 0.20,
        "enemies": ["Boggart"],
        "flavor": [
            "A staircase pivots underfoot, redirecting you to a corridor you haven't walked before.",
            "A portrait of a Renaissance wizard winks at you and whispers, 'Mind the third arch.'",
            "Peeves swoops past, cackling, and drops a water balloon that misses by inches.",
            "The air smells faintly of incense and old paper — someone is in the library after hours.",
        ],
    },
    "loc_003": {  # Knockturn Alley
        "danger": 0.55,
        "enemies": ["Death Eater", "Dark Wizard"],
        "flavor": [
            "A hooded figure crosses the alley ahead and ducks into a doorway that wasn't there moments before.",
            "Green-tinted lanterns flicker as you pass — one goes out entirely.",
            "A shop window displays a cursed object that seems to lean toward you as you pass.",
            "You hear muffled chanting from a basement grate beneath your feet.",
        ],
    },
    "loc_004": {  # Forbidden Forest
        "danger": 0.65,
        "enemies": ["Acromantula", "Werewolf", "Inferi"],
        "flavor": [
            "A thestral lifts its head from a kill in the underbrush and watches you pass with calm white eyes.",
            "The trees ahead form a path that wasn't there a moment ago — the forest is showing you something.",
            "Bioluminescent fungi pulse once in unison, then go dark.",
            "Far off, a single centaur watches from a ridge, neither hostile nor welcoming.",
        ],
    },
    "loc_005": {  # Ministry
        "danger": 0.30,
        "enemies": ["Dark Wizard"],
        "flavor": [
            "An enchanted paper aeroplane circles you twice before darting off down a corridor.",
            "Two Aurors in dress robes stop talking abruptly as you pass.",
            "The lift gives a tone it shouldn't — the Department of Mysteries floor button is briefly warm.",
            "A memo flutters down at your feet, addressed to no one, with a single Ministry seal.",
        ],
    },
    "loc_006": {  # Azkaban
        "danger": 0.80,
        "enemies": ["Dementor", "Inferi"],
        "flavor": [
            "A scream from a distant cell cuts off mid-breath, then resumes — but in a different voice.",
            "The temperature drops abruptly; your breath fogs in the air despite the indoor torches.",
            "Footsteps in the corridor behind you stop exactly when yours do.",
            "A prisoner you cannot see whispers your name as you pass their cell.",
        ],
    },
    "loc_007": {  # Godric's Hollow
        "danger": 0.25,
        "enemies": ["Boggart"],
        "flavor": [
            "Fresh lilies on the war memorial weren't there an hour ago.",
            "A figure stands at the edge of the cemetery, motionless — when you look again, they're gone.",
            "An owl perches on the ruined cottage roof and watches you with unusual attention.",
            "The wind carries a snatch of song from far down the lane — a lullaby in a language you almost recognize.",
        ],
    },
}

DEFAULT_PROFILE = {"danger": 0.15, "enemies": [], "flavor": ["A small magical sound echoes nearby."]}


@dataclass
class Encounter:
    type: str               # "flavor" | "combat" | "discovery"
    description: str
    enemy_archetype: Optional[str] = None
    item_id: Optional[str] = None
    severity: float = 0.0


class EncounterEngine:
    """
    Per-session encounter tracker. Rolls encounters based on:
      - Location danger level
      - Day/night cycle (night = 1.5× more dangerous)
      - Encounter cooldown (min 3 turns between)
      - Difficulty setting
    """

    def __init__(self):
        self._last_turn: Dict[str, int] = {}   # session_id → last encounter turn

    def roll(
        self,
        session_id: str,
        location_id: str,
        turn_number: int,
        is_night: bool = False,
        player_level: int = 1,
        difficulty: str = "medium",
    ) -> Optional[Encounter]:
        """Roll for an encounter. Returns None if nothing happens this turn."""
        profile = LOCATION_PROFILES.get(location_id, DEFAULT_PROFILE)
        base = profile["danger"]

        # Cooldown: at least 3 turns between encounters
        last = self._last_turn.get(session_id, -10)
        if turn_number - last < 3:
            return None

        # Time-of-day modifier
        if is_night:
            base *= 1.5

        # Difficulty modifier
        diff_mod = {"easy": 0.5, "medium": 1.0, "hard": 1.3, "very_hard": 1.6, "legendary": 2.0}
        base *= diff_mod.get(difficulty, 1.0)

        if random.random() > base:
            return None

        # Pick type based on danger (high danger = more combat)
        roll = random.random()
        if roll < base * 0.4 and profile["enemies"]:
            archetype = random.choice(profile["enemies"])
            enc = Encounter(
                type="combat",
                description=f"A {archetype} blocks your path!",
                enemy_archetype=archetype,
                severity=base,
            )
        elif roll < 0.15:
            enc = Encounter(
                type="discovery",
                description="You notice something half-hidden — worth investigating.",
                item_id=random.choice(["butterbeer", "antidote", "mana_draught", "healing_potion"]),
                severity=base,
            )
        else:
            enc = Encounter(
                type="flavor",
                description=random.choice(profile["flavor"]),
                severity=base,
            )

        self._last_turn[session_id] = turn_number
        return enc


encounter_engine = EncounterEngine()
