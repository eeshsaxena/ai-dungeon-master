"""Tests for item_system.py — DB, use effects, loot tables."""
import pytest
from item_system import (
    ITEM_DB, make_item_instance, use_item,
    roll_loot, tick_effects, has_buff, LOOT_TABLES
)


@pytest.fixture
def fresh_player():
    return {
        "hp": 50, "max_hp": 100,
        "mana": 30, "max_mana": 80,
        "xp": 0, "level": 1,
        "spells_known": ["Lumos"],
    }


def test_item_db_integrity():
    assert len(ITEM_DB) >= 20
    for item_id, item in ITEM_DB.items():
        assert "name" in item
        assert "type" in item
        assert "rarity" in item
        assert "description" in item
        assert "effects" in item


def test_make_item_instance_unknown_id_returns_none():
    assert make_item_instance("not_a_real_item") is None


def test_make_item_instance_includes_timestamp():
    inst = make_item_instance("butterbeer")
    assert inst is not None
    assert "acquired_at" in inst


def test_use_healing_potion(fresh_player):
    item = make_item_instance("healing_potion")
    result = use_item(item, fresh_player, [])
    assert result["success"] is True
    assert fresh_player["hp"] == 100  # was 50, +50, capped at max


def test_heal_does_not_exceed_max_hp(fresh_player):
    fresh_player["hp"] = 95
    item = make_item_instance("healing_potion")
    use_item(item, fresh_player, [])
    assert fresh_player["hp"] == 100


def test_use_butterbeer(fresh_player):
    item = make_item_instance("butterbeer")
    use_item(item, fresh_player, [])
    assert fresh_player["hp"] == 70  # 50 + 20


def test_use_mana_draught(fresh_player):
    item = make_item_instance("mana_draught")
    use_item(item, fresh_player, [])
    assert fresh_player["mana"] == 70  # 30 + 40


def test_use_buff_creates_timed_effect(fresh_player):
    item = make_item_instance("felix_felicis")
    active = []
    use_item(item, fresh_player, active)
    assert len(active) == 1
    assert active[0]["stat"] == "luck"
    assert active[0]["turns_remaining"] == 1


def test_teach_spell_adds_to_spellbook(fresh_player):
    item = make_item_instance("scroll_patronum")
    use_item(item, fresh_player, [])
    assert "Expecto Patronum" in fresh_player["spells_known"]


def test_teach_spell_no_dupes(fresh_player):
    fresh_player["spells_known"] = ["Expecto Patronum"]
    item = make_item_instance("scroll_patronum")
    use_item(item, fresh_player, [])
    assert fresh_player["spells_known"].count("Expecto Patronum") == 1


def test_tick_effects_decrements_and_removes_expired():
    active = [
        {"type": "buff", "stat": "luck", "turns_remaining": 1},
        {"type": "buff", "stat": "spell_power", "turns_remaining": 3},
    ]
    result = tick_effects(active)
    assert len(result) == 1   # luck expired
    assert result[0]["stat"] == "spell_power"
    assert result[0]["turns_remaining"] == 2


def test_has_buff_returns_amount():
    active = [{"type": "buff", "stat": "spell_power", "amount": 50}]
    assert has_buff(active, "spell_power") == 50
    assert has_buff(active, "luck") is None


def test_roll_loot_returns_list():
    drops = roll_loot("Death Eater", 5)
    assert isinstance(drops, list)


def test_roll_loot_higher_level_more_drops(monkeypatch):
    import random
    # Force every probability check to succeed
    monkeypatch.setattr(random, "random", lambda: 0.01)
    low_drops = roll_loot("Death Eater", 1)
    high_drops = roll_loot("Death Eater", 20)
    assert len(high_drops) >= len(low_drops)


def test_loot_tables_cover_main_archetypes():
    for archetype in ["Death Eater", "Dementor", "Hollow Mage", "Boggart"]:
        assert archetype in LOOT_TABLES
