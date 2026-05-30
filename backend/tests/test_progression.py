"""Tests for backend/progression.py — XP curve, level-up, spell unlocks."""
from progression import (
    process_xp_gain, xp_to_next, title_for_level, spells_for_level,
    level_from_xp, XP_THRESHOLDS, SPELL_UNLOCKS
)


def test_xp_thresholds_monotonic_and_capped():
    assert len(XP_THRESHOLDS) == 20
    assert XP_THRESHOLDS[0] == 0
    for i in range(1, 20):
        assert XP_THRESHOLDS[i] > XP_THRESHOLDS[i-1]


def test_level_from_xp_at_thresholds():
    assert level_from_xp(0) == 1
    assert level_from_xp(299) == 1
    assert level_from_xp(300) == 2
    assert level_from_xp(8950) == 20
    assert level_from_xp(999999) == 20


def test_xp_gain_no_level_up():
    new_xp, new_level, new_spells, leveled = process_xp_gain(0, 100)
    assert new_xp == 100
    assert new_level == 1
    assert new_spells == []
    assert leveled is False


def test_xp_gain_single_level_up():
    new_xp, new_level, new_spells, leveled = process_xp_gain(0, 350)
    assert new_xp == 350
    assert new_level == 2
    assert leveled is True
    # Level 2 doesn't unlock new spells (level 3 does)
    assert new_spells == []


def test_xp_gain_unlocks_level_3_spells():
    new_xp, new_level, new_spells, leveled = process_xp_gain(300, 250)
    assert new_level == 3
    assert leveled is True
    assert "Aguamenti" in new_spells
    assert "Incendio" in new_spells
    assert "Wingardium Leviosa" in new_spells


def test_xp_gain_multi_level_jump_collects_all_spells():
    # Jump from level 1 to level 5 in one gain
    new_xp, new_level, new_spells, leveled = process_xp_gain(0, 1100)
    assert new_level == 5
    assert "Expecto Patronum" in new_spells  # level 5 unlock
    assert "Aguamenti" in new_spells          # level 3 unlock too


def test_xp_to_next_at_max_level():
    assert xp_to_next(99999, 20) == 0


def test_title_for_level():
    assert title_for_level(1) == "First-Year"
    assert title_for_level(20) == "Legendary Champion"
    # Boundary correctness
    assert title_for_level(5) == "Adept"


def test_spells_for_level_cumulative():
    level_1 = spells_for_level(1)
    level_5 = spells_for_level(5)
    assert set(level_1).issubset(set(level_5))
    assert len(level_5) > len(level_1)
