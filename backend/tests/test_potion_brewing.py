"""Tests for potion_brewing.py — 12-recipe brewing system."""
import pytest
import random
from potion_brewing import (
    RECIPES, list_recipes, can_brew, brew, consume_ingredients, count_ingredient
)


@pytest.fixture
def basic_inventory():
    """Inventory with common ingredients for testing."""
    return [
        {"id": "crystallised_pineapple"},
        {"id": "crystallised_pineapple"},
        {"id": "crystallised_pineapple"},
        {"id": "antidote"},
        {"id": "antidote"},
        {"id": "boomslang_skin"},
    ]


def test_all_recipes_have_required_fields():
    for rid, recipe in RECIPES.items():
        assert "name" in recipe
        assert "result_item" in recipe
        assert "ingredients" in recipe
        assert "difficulty" in recipe
        assert "min_level" in recipe


def test_count_ingredient(basic_inventory):
    assert count_ingredient(basic_inventory, "crystallised_pineapple") == 3
    assert count_ingredient(basic_inventory, "antidote") == 2
    assert count_ingredient(basic_inventory, "nonexistent") == 0


def test_can_brew_with_ingredients(basic_inventory):
    can, missing = can_brew("wiggenweld", basic_inventory, player_level=5)
    assert can is True
    assert missing == []


def test_cannot_brew_missing_ingredients():
    inv = [{"id": "crystallised_pineapple"}]   # missing antidote
    can, missing = can_brew("wiggenweld", inv, player_level=5)
    assert can is False
    assert len(missing) > 0


def test_cannot_brew_below_min_level(basic_inventory):
    # Felix Felicis requires level 12
    can, missing = can_brew("felix_brew", basic_inventory, player_level=3)
    assert can is False


def test_consume_ingredients_removes_correct_count(basic_inventory):
    before_count = len(basic_inventory)
    consume_ingredients("wiggenweld", basic_inventory)
    after_count = len(basic_inventory)
    # Wiggenweld uses 1 pineapple + 1 antidote = removes 2
    assert before_count - after_count == 2


def test_brew_success_path(basic_inventory, monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.1)   # always succeeds
    result = brew("wiggenweld", basic_inventory, player_level=10)
    assert result["success"] is True
    assert result["item"]["id"] == "healing_potion"


def test_brew_failure_consumes_ingredients(basic_inventory, monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.99)   # always fails
    before_count = len(basic_inventory)
    result = brew("wiggenweld", basic_inventory, player_level=1)
    assert result["success"] is False
    # Ingredients still consumed
    assert len(basic_inventory) < before_count


def test_brew_unknown_recipe_returns_failure():
    result = brew("not_a_recipe", [], player_level=5)
    assert result["success"] is False
    assert "Unknown recipe" in result["message"]


def test_list_recipes_marks_availability():
    recipes = list_recipes(player_level=1)
    wiggenweld = next(r for r in recipes if r["id"] == "wiggenweld")
    felix = next(r for r in recipes if r["id"] == "felix_brew")
    assert wiggenweld["available"] is True
    assert felix["available"] is False


def test_brew_higher_level_higher_success(monkeypatch):
    # The success chance scales with level above min_level
    inv1 = [{"id": "crystallised_pineapple"}, {"id": "antidote"}]
    inv2 = [{"id": "crystallised_pineapple"}, {"id": "antidote"}]
    # At level 1 (min), base chance is 50%
    # At level 11 (10 above min), capped at 95%
    monkeypatch.setattr(random, "random", lambda: 0.85)  # would fail at low level, pass at high
    brew("wiggenweld", inv1, player_level=1)
    result_high = brew("wiggenweld", inv2, player_level=11)
    assert result_high["success"] is True
