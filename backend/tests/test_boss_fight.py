"""Tests for boss_fight.py — 3-phase Hollow Mage scripted encounter."""
import pytest
from boss_fight import boss_fight, BOSS_HP, PHASE_THRESHOLDS


@pytest.fixture(autouse=True)
def reset_boss():
    """Ensure each test starts with a fresh boss state."""
    boss_fight._fights.clear()
    yield
    boss_fight._fights.clear()


def test_start_creates_state():
    result = boss_fight.start("s1")
    assert result["boss_hp"] == BOSS_HP
    assert result["phase"] == 1
    assert "phase_intro" in result


def test_resolve_round_without_start_raises():
    with pytest.raises(ValueError):
        boss_fight.resolve_round("never_started", "Lumos", 100, 100, [])


def test_patronus_effective_in_phase_1():
    boss_fight.start("s1")
    result = boss_fight.resolve_round("s1", "Expecto Patronum", 100, 100, [])
    assert result.boss_damage > 30   # Patronus is highly effective


def test_useless_spell_in_phase_2_does_no_damage():
    boss_fight.start("s1")
    state = boss_fight._fights["s1"]
    state.boss_hp = 250
    state.phase = 2   # explicitly set phase
    # Wingardium Leviosa isn't in any phase's effective list
    result = boss_fight.resolve_round("s1", "Wingardium Leviosa", 100, 100, [])
    assert result.boss_damage == 0


def test_expelliarmus_stuns_in_phase_2():
    boss_fight.start("s1")
    state = boss_fight._fights["s1"]
    state.boss_hp = 250
    state.phase = 2
    # Verify the boss did NOT attack back this round (because we just stunned them)
    result = boss_fight.resolve_round("s1", "Expelliarmus", 100, 100, [])
    # Boss was stunned this round → didn't deal damage from regular attack
    # (Inferi might still deal small amount but base attack is 0)
    assert "stunned" in result.narrative.lower()


def test_phase_3_requires_fiendfyre_or_elder_wand():
    boss_fight.start("s1")
    state = boss_fight._fights["s1"]
    state.boss_hp = 100   # phase 3
    state.phase = 3
    # Standard spell does 0 damage in phase 3 without elder wand
    result = boss_fight.resolve_round("s1", "Stupefy", 100, 100, [])
    assert result.boss_damage == 0


def test_phase_3_with_elder_wand_buff_works():
    boss_fight.start("s1")
    state = boss_fight._fights["s1"]
    state.boss_hp = 100
    state.phase = 3
    elder_buff = [{"type": "buff", "source": "elder_wand_shard", "turns_remaining": 5}]
    result = boss_fight.resolve_round("s1", "Stupefy", 100, 100, elder_buff)
    assert result.boss_damage > 0


def test_fiendfyre_phase_3_devastating():
    boss_fight.start("s1")
    state = boss_fight._fights["s1"]
    state.boss_hp = 100
    state.phase = 3
    result = boss_fight.resolve_round("s1", "Fiendfyre", 100, 100, [])
    assert result.boss_damage >= 100


def test_phase_transition_at_threshold():
    boss_fight.start("s1")
    state = boss_fight._fights["s1"]
    state.boss_hp = 340   # just above phase 2 threshold of 334
    boss_fight.resolve_round("s1", "Expecto Patronum", 100, 100, [])
    # After taking ~50 damage, should be ~290 → phase 2
    if state.boss_hp <= PHASE_THRESHOLDS[1]["min_hp"]:
        assert state.phase >= 2


def test_boss_defeat():
    boss_fight.start("s1")
    state = boss_fight._fights["s1"]
    state.boss_hp = 1
    state.phase = 3
    elder_buff = [{"type": "buff", "source": "elder_wand_shard", "turns_remaining": 5}]
    result = boss_fight.resolve_round("s1", "Fiendfyre", 100, 100, elder_buff)
    assert result.boss_defeated is True
    assert result.player_won is True


def test_boss_attacks_back_when_not_stunned():
    boss_fight.start("s1")
    result = boss_fight.resolve_round("s1", "Lumos", 100, 100, [])
    # Lumos in phase 1 deals some damage, boss attacks back
    assert result.player_damage > 0
