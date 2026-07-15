"""Tests for achievements.py — unlock tracking, counter thresholds, persistence."""

import achievements as ach_mod
from achievements import AchievementTracker, ACHIEVEMENTS


def make_tracker(tmp_path):
    """Helper: AchievementTracker with disk persistence pointed at tmp."""
    ach_mod.ACHIEVEMENT_DIR = tmp_path
    return AchievementTracker(session_id="test_session")


def test_total_achievements():
    assert len(ACHIEVEMENTS) == 30


def test_achievements_have_required_fields():
    for aid, ach in ACHIEVEMENTS.items():
        assert "name" in ach
        assert "emoji" in ach
        assert "desc" in ach


def test_unlock_first_time_returns_true(tmp_path):
    t = make_tracker(tmp_path)
    assert t.unlock("first_steps") is True
    assert t.is_unlocked("first_steps")


def test_unlock_twice_returns_false(tmp_path):
    t = make_tracker(tmp_path)
    t.unlock("first_steps")
    assert t.unlock("first_steps") is False


def test_unlock_unknown_returns_false(tmp_path):
    t = make_tracker(tmp_path)
    assert t.unlock("totally_made_up") is False


def test_increment_threshold_unlock(tmp_path):
    t = make_tracker(tmp_path)
    for _ in range(4):
        new = t.increment("death_eater_kills")
        assert "death_eater_slayer" not in new
    new = t.increment("death_eater_kills")  # 5th time
    assert "death_eater_slayer" in new


def test_combat_won_first_blood(tmp_path):
    t = make_tracker(tmp_path)
    unlocked = t.on_event("combat_won", {"archetype": "Death Eater", "took_damage": True})
    assert "first_blood" in [u.get("name") if isinstance(u, dict) else u for u in unlocked] or "first_blood" in str(unlocked)


def test_level_up_triggers_threshold(tmp_path):
    t = make_tracker(tmp_path)
    unlocked = t.on_event("level_up", {"level": 5})
    names = [u.get("name") for u in unlocked if isinstance(u, dict)]
    assert "Adept Witch/Wizard" in names


def test_progress_hides_secrets_when_locked(tmp_path):
    t = make_tracker(tmp_path)
    progress = t.progress()
    secret_locked = [a for a in progress["achievements"]
                     if a.get("secret") and not a.get("unlocked")]
    for a in secret_locked:
        assert a["name"] == "???"


def test_progress_shows_secret_when_unlocked(tmp_path):
    t = make_tracker(tmp_path)
    t.unlock("hollow_mage_defeated")
    progress = t.progress()
    boss_ach = next(a for a in progress["achievements"] if a["id"] == "hollow_mage_defeated")
    assert boss_ach["unlocked"]
    assert boss_ach["name"] != "???"


def test_persistence_roundtrip(tmp_path):
    t = make_tracker(tmp_path)
    t.unlock("first_steps")
    t.increment("locations_visited", 3)

    # Create a fresh tracker — should load from disk
    t2 = AchievementTracker(session_id="test_session")
    assert t2.is_unlocked("first_steps")
    assert t2.counters.get("locations_visited") == 3


def test_progress_summary(tmp_path):
    t = make_tracker(tmp_path)
    p = t.progress()
    assert p["total"] == 30
    assert p["unlocked"] == 0
    t.unlock("first_steps")
    p = t.progress()
    assert p["unlocked"] == 1
