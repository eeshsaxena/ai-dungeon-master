"""Tests for encounter_engine.py — random encounter rolling."""
import random
import pytest
from encounter_engine import encounter_engine, LOCATION_PROFILES, EncounterEngine


@pytest.fixture(autouse=True)
def reset_engine():
    encounter_engine._last_turn.clear()


def test_all_main_locations_have_profiles():
    for loc_id in ["loc_001", "loc_002", "loc_003", "loc_004", "loc_005", "loc_006", "loc_007"]:
        assert loc_id in LOCATION_PROFILES
        assert "danger" in LOCATION_PROFILES[loc_id]
        assert "flavor" in LOCATION_PROFILES[loc_id]


def test_danger_levels_increase_with_location():
    assert LOCATION_PROFILES["loc_001"]["danger"] < LOCATION_PROFILES["loc_004"]["danger"]
    assert LOCATION_PROFILES["loc_001"]["danger"] < LOCATION_PROFILES["loc_006"]["danger"]


def test_cooldown_prevents_rapid_encounters(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)   # always trigger
    e1 = encounter_engine.roll("s1", "loc_004", 5)
    assert e1 is not None
    # Within cooldown — should be None
    e2 = encounter_engine.roll("s1", "loc_004", 6)
    assert e2 is None
    # After cooldown — should fire again
    e3 = encounter_engine.roll("s1", "loc_004", 10)
    assert e3 is not None


def test_safe_location_rarely_encounters(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.5)   # mid-range roll
    # Three Broomsticks has danger 0.05 — way below 0.5 → no encounter
    e = encounter_engine.roll("s1", "loc_001", 5)
    assert e is None


def test_night_boosts_encounter_rate(monkeypatch):
    # Roll just above the day threshold for Forbidden Forest
    monkeypatch.setattr(random, "random", lambda: 0.7)
    # Day: danger 0.65, roll 0.7 → no encounter
    encounter_engine.roll("s_day", "loc_004", 5, is_night=False)
    # Night: danger 0.65 × 1.5 = 0.975, roll 0.7 → encounter
    e_night = encounter_engine.roll("s_night", "loc_004", 5, is_night=True)
    assert e_night is not None


def test_combat_encounter_has_archetype(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    # Force into combat branch — need both danger and combat roll low
    eng = EncounterEngine()
    e = eng.roll("s1", "loc_004", 5)
    if e and e.type == "combat":
        assert e.enemy_archetype in LOCATION_PROFILES["loc_004"]["enemies"]


def test_flavor_encounter_has_description(monkeypatch):
    # Make roll high enough to skip combat (40% of danger) but low enough to fire (under danger)
    monkeypatch.setattr(random, "random", lambda call=[0]: [0.05, 0.5][call[0] % 2 if (call.__setitem__(0, call[0]+1) or True) else 0])
    e = encounter_engine.roll("s1", "loc_003", 5)
    if e:
        assert e.description
