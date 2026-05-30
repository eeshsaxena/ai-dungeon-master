"""
Player progression: XP thresholds (levels 1-20), spell unlocks, titles.
Authoritative source — frontend mirrors these values, never recomputes them.
"""
from typing import List, Tuple, Dict

# Cumulative XP needed to REACH each level (index 0 = level 1 = 0 XP).
# Formula: round(100 * level^1.5 / 50) * 50, capped at 20 levels.
XP_THRESHOLDS: List[int] = [
    0,     # level 1
    300,   # level 2
    500,   # level 3
    800,   # level 4
    1100,  # level 5
    1450,  # level 6
    1850,  # level 7
    2250,  # level 8
    2700,  # level 9
    3150,  # level 10
    3650,  # level 11
    4150,  # level 12
    4700,  # level 13
    5250,  # level 14
    5800,  # level 15
    6400,  # level 16
    7000,  # level 17
    7650,  # level 18
    8300,  # level 19
    8950,  # level 20
]

# Spells unlocked at or before each level.
SPELL_UNLOCKS: Dict[int, List[str]] = {
    1:  ["Expelliarmus", "Stupefy", "Protego", "Lumos", "Accio"],
    3:  ["Aguamenti", "Incendio", "Wingardium Leviosa"],
    5:  ["Expecto Patronum", "Impedimenta"],
    7:  ["Sectumsempra", "Obliviate"],
    10: ["Finite Incantatem", "Reparo"],
    12: ["Homenum Revelio"],
    15: ["Fiendfyre"],
    18: ["Priori Incantatem"],
    20: ["Occlumency", "Legilimency"],
}

LEVEL_TITLES: Dict[int, str] = {
    1:  "First-Year",
    3:  "Apprentice Witch/Wizard",
    5:  "Adept",
    8:  "Journeyman Duelist",
    10: "Auror Trainee",
    13: "Auror",
    15: "Senior Auror",
    18: "Master Witch/Wizard",
    20: "Legendary Champion",
}


def level_from_xp(total_xp: int) -> int:
    """Determine level from total accumulated XP (clamped to 1–20)."""
    level = 1
    for lvl in range(2, 21):
        if total_xp >= XP_THRESHOLDS[lvl - 1]:
            level = lvl
        else:
            break
    return level


def xp_to_next(total_xp: int, current_level: int) -> int:
    """XP still needed to reach the next level."""
    if current_level >= 20:
        return 0
    return max(0, XP_THRESHOLDS[current_level] - total_xp)


def spells_for_level(level: int) -> List[str]:
    """All spells unlocked at or below this level."""
    spells: List[str] = []
    for unlock_level, spell_list in sorted(SPELL_UNLOCKS.items()):
        if unlock_level <= level:
            spells.extend(spell_list)
    return spells


def title_for_level(level: int) -> str:
    title = "First-Year"
    for lvl, t in sorted(LEVEL_TITLES.items()):
        if level >= lvl:
            title = t
    return title


def process_xp_gain(old_xp: int, added_xp: int) -> Tuple[int, int, List[str], bool]:
    """
    Apply an XP gain and return (new_xp, new_level, newly_unlocked_spells, leveled_up).
    """
    new_xp = old_xp + added_xp
    old_level = level_from_xp(old_xp)
    new_level = level_from_xp(new_xp)
    leveled_up = new_level > old_level

    newly_unlocked: List[str] = []
    if leveled_up:
        old_set = set(spells_for_level(old_level))
        new_set = set(spells_for_level(new_level))
        newly_unlocked = sorted(new_set - old_set)

    return new_xp, new_level, newly_unlocked, leveled_up
