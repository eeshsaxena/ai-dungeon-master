"""
Potion Brewing System (Phase 11).

Players combine ingredients from their inventory into potions via known recipes.
12 recipes — some classic HP brews, some custom for the game.

Usage:
    list_recipes() → all known recipes
    can_brew(recipe_id, inventory) → bool + missing ingredients
    brew(recipe_id, inventory, player_level) → result with new item or failure
"""
import random
from typing import Any, Dict, List, Optional, Tuple

RECIPES: Dict[str, Dict[str, Any]] = {
    "wiggenweld": {
        "id": "wiggenweld",
        "name": "Wiggenweld Potion",
        "emoji": "🧪",
        "result_item": "healing_potion",
        "ingredients": {"crystallised_pineapple": 1, "antidote": 1},
        "difficulty": 1,
        "min_level": 1,
        "description": "A basic healing potion. The first brew most students learn.",
    },
    "pepperup": {
        "id": "pepperup",
        "name": "Pepperup Potion",
        "emoji": "🌶️",
        "result_item": "butterbeer",  # repurposed: minor refresh
        "ingredients": {"crystallised_pineapple": 2},
        "difficulty": 1,
        "min_level": 1,
        "description": "Cures the common cold. Causes steam to come out of the drinker's ears.",
    },
    "draught_of_peace": {
        "id": "draught_of_peace",
        "name": "Draught of Peace",
        "emoji": "🌊",
        "result_item": "mana_draught",
        "ingredients": {"crystallised_pineapple": 1, "antidote": 2},
        "difficulty": 2,
        "min_level": 3,
        "description": "Calms anxiety and restores magical reserves.",
    },
    "skele_gro_brew": {
        "id": "skele_gro_brew",
        "name": "Skele-Gro",
        "emoji": "🦴",
        "result_item": "skele_gro",
        "ingredients": {"boomslang_skin": 1, "antidote": 2, "crystallised_pineapple": 1},
        "difficulty": 3,
        "min_level": 5,
        "description": "Madam Pomfrey's bone-regrowing brew. Tastes terrible.",
    },
    "polyjuice_base": {
        "id": "polyjuice_base",
        "name": "Polyjuice Base",
        "emoji": "🧫",
        "result_item": "invigoration_draught",
        "ingredients": {"boomslang_skin": 2, "antidote": 1, "crystallised_pineapple": 2},
        "difficulty": 4,
        "min_level": 7,
        "description": "Without the final ingredient (a hair), it stays as a useful invigoration draught.",
    },
    "felix_brew": {
        "id": "felix_brew",
        "name": "Felix Felicis",
        "emoji": "✨",
        "result_item": "felix_felicis",
        "ingredients": {"boomslang_skin": 3, "antidote": 3, "crystallised_pineapple": 3},
        "difficulty": 5,
        "min_level": 12,
        "description": "Liquid luck. Notoriously difficult to brew. Demands perfect timing.",
    },
    "antidote_advanced": {
        "id": "antidote_advanced",
        "name": "Advanced Antidote",
        "emoji": "🌿",
        "result_item": "antidote",
        "ingredients": {"crystallised_pineapple": 2},
        "difficulty": 1,
        "min_level": 1,
        "description": "A simple antidote brewed from common herbal ingredients.",
    },
    "gillyweed_extract": {
        "id": "gillyweed_extract",
        "name": "Gillyweed Extract",
        "emoji": "🌿",
        "result_item": "gillyweed",
        "ingredients": {"boomslang_skin": 1, "crystallised_pineapple": 2},
        "difficulty": 3,
        "min_level": 5,
        "description": "Concentrated gillyweed for underwater work.",
    },
    "amortentia_decoy": {
        "id": "amortentia_decoy",
        "name": "Amortentia Decoy",
        "emoji": "💗",
        "result_item": "felix_felicis",  # close enough for game purposes
        "ingredients": {"crystallised_pineapple": 5, "antidote": 1},
        "difficulty": 4,
        "min_level": 10,
        "description": "An imperfect Amortentia that grants a lucky charm effect instead.",
    },
    "veritaserum_lite": {
        "id": "veritaserum_lite",
        "name": "Veritaserum Lite",
        "emoji": "💧",
        "result_item": "pensieve_fragment",
        "ingredients": {"boomslang_skin": 2, "crystallised_pineapple": 1},
        "difficulty": 3,
        "min_level": 7,
        "description": "A diluted truth serum that crystallizes into a memory fragment.",
    },
    "regeneration_draught": {
        "id": "regeneration_draught",
        "name": "Regeneration Draught",
        "emoji": "💉",
        "result_item": "bezoar",
        "ingredients": {"boomslang_skin": 1, "antidote": 3, "crystallised_pineapple": 1},
        "difficulty": 3,
        "min_level": 6,
        "description": "A vital all-purpose restoration. Crystallizes into a bezoar.",
    },
    "patronus_distillate": {
        "id": "patronus_distillate",
        "name": "Patronus Distillate",
        "emoji": "✨",
        "result_item": "scroll_patronum",
        "ingredients": {"boomslang_skin": 2, "antidote": 2, "crystallised_pineapple": 3},
        "difficulty": 4,
        "min_level": 8,
        "description": "Crystallizes happy memories into a scroll that teaches Expecto Patronum.",
    },
}


def list_recipes(player_level: int = 1) -> List[Dict[str, Any]]:
    """Return all recipes available to the player (sorted by difficulty)."""
    out = []
    for rid, recipe in RECIPES.items():
        entry = dict(recipe)
        entry["available"] = player_level >= recipe["min_level"]
        out.append(entry)
    return sorted(out, key=lambda r: (r["difficulty"], r["min_level"]))


def count_ingredient(inventory: List[Dict], ingredient_id: str) -> int:
    return sum(1 for item in inventory if item.get("id") == ingredient_id)


def can_brew(recipe_id: str, inventory: List[Dict], player_level: int) -> Tuple[bool, List[str]]:
    """Check if the player can brew this recipe. Returns (can_brew, missing_ingredients)."""
    recipe = RECIPES.get(recipe_id)
    if not recipe:
        return False, ["Unknown recipe"]
    if player_level < recipe["min_level"]:
        return False, [f"Requires Level {recipe['min_level']}"]

    missing = []
    for ing_id, amount in recipe["ingredients"].items():
        have = count_ingredient(inventory, ing_id)
        if have < amount:
            missing.append(f"{ing_id} x{amount - have}")
    return len(missing) == 0, missing


def consume_ingredients(recipe_id: str, inventory: List[Dict]) -> None:
    """Mutate inventory in place — remove ingredients used."""
    recipe = RECIPES.get(recipe_id)
    if not recipe:
        return
    for ing_id, amount in recipe["ingredients"].items():
        remaining = amount
        i = 0
        while i < len(inventory) and remaining > 0:
            if inventory[i].get("id") == ing_id:
                inventory.pop(i)
                remaining -= 1
            else:
                i += 1


def brew(recipe_id: str, inventory: List[Dict], player_level: int) -> Dict[str, Any]:
    """
    Attempt to brew a potion. Success rate based on player level vs recipe difficulty.
    On success, ingredients are consumed and the result item is returned.
    On failure, ingredients are consumed but no item is created.
    """
    recipe = RECIPES.get(recipe_id)
    if not recipe:
        return {"success": False, "message": "Unknown recipe", "item": None}

    can, missing = can_brew(recipe_id, inventory, player_level)
    if not can:
        return {"success": False, "message": f"Cannot brew: {', '.join(missing)}", "item": None}

    # Success chance: base 50% + (level - min_level) * 10%, max 95%
    level_bonus = (player_level - recipe["min_level"]) * 0.10
    base_chance = 0.5 + level_bonus
    success_chance = max(0.2, min(0.95, base_chance))

    consume_ingredients(recipe_id, inventory)

    if random.random() < success_chance:
        # Import here to avoid circular dep
        from item_system import make_item_instance
        item = make_item_instance(recipe["result_item"])
        if item:
            inventory.append(item)
        return {
            "success": True,
            "message": f"Successfully brewed {recipe['name']}!",
            "item": item,
            "recipe": recipe,
        }
    else:
        return {
            "success": False,
            "message": f"The {recipe['name']} brew failed — the cauldron's contents fizzle into useless sludge.",
            "item": None,
            "recipe": recipe,
        }
