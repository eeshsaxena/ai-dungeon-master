"""Tests for day_night.py — time-of-day cycle."""
from day_night import (
    turn_to_time, get_phase, time_prompt,
    night_encounter_multiplier, TURNS_PER_DAY, START_HOUR
)


def test_turn_1_is_start_hour():
    h, m = turn_to_time(1)
    assert h == START_HOUR
    assert m == 0


def test_turn_2_is_30_min_later():
    h, m = turn_to_time(2)
    assert h == START_HOUR
    assert m == 30


def test_full_day_wraps():
    # After TURNS_PER_DAY (48 turns), should return to start
    h, m = turn_to_time(1 + TURNS_PER_DAY)
    h_start, m_start = turn_to_time(1)
    assert h == h_start
    assert m == m_start


def test_phase_morning():
    p = get_phase(1)   # 8:00 AM
    assert p["phase"] == "morning"
    assert p["is_night"] is False


def test_phase_midday():
    # 11 AM = turn corresponding to (11 - 8) * 2 + 1 = 7
    p = get_phase(7)
    assert p["phase"] == "midday"


def test_phase_evening_is_night():
    # 8 PM = turn (20 - 8) * 2 + 1 = 25
    p = get_phase(25)
    assert p["is_night"] is True


def test_phase_has_display_format():
    p = get_phase(5)   # 10:00 AM
    assert p["display"] == "10:00"


def test_phase_has_required_keys():
    p = get_phase(1)
    for key in ("phase", "emoji", "hour", "minute", "display", "is_night", "is_dawn_dusk"):
        assert key in p


def test_night_multiplier_higher_at_night():
    day_mult = night_encounter_multiplier(1)   # morning
    night_mult = night_encounter_multiplier(30) # late evening
    assert night_mult > day_mult
    assert night_mult >= 1.5


def test_time_prompt_includes_time_string():
    prompt = time_prompt(1)
    assert "TIME OF DAY" in prompt
    assert "08:00" in prompt
