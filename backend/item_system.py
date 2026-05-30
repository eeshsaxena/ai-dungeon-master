"""
Item System for AI Dungeon Master (Phase 10).

Full HP-universe item database with typed effects, inventory management,
loot tables for combat, and quest/exploration rewards.

Effects applied immediately by use_item(); any temporal effects (buffs) are
stored in the player's active_effects list and checked on each combat round.
"""
import random
import time
from typing import Any, Dict, List, Optional, Tuple

# ── Item database ─────────────────────────────────────────────────────────────

ITEM_DB: Dict[str, Dict[str, Any]] = {
    # ── Consumables ─────────────────────────────────────────────────────────
    "butterbeer": {
        "id": "butterbeer", "name": "Butterbeer", "emoji": "🍺",
        "type": "consumable", "rarity": "common",
        "description": "A warm, butterscotch-sweet wizarding drink. Restores a little HP and lifts spirits.",
        "effects": [{"type": "heal_hp", "amount": 20}],
        "value": 3,
    },
    "healing_potion": {
        "id": "healing_potion", "name": "Healing Potion", "emoji": "🧪",
        "type": "consumable", "rarity": "uncommon",
        "description": "A vivid crimson potion brewed by Madam Pomfrey's recipe. Restores significant HP.",
        "effects": [{"type": "heal_hp", "amount": 50}],
        "value": 25,
    },
    "mana_draught": {
        "id": "mana_draught", "name": "Mana Draught", "emoji": "💙",
        "type": "consumable", "rarity": "uncommon",
        "description": "A shimmering blue liquid that replenishes magical reserves quickly.",
        "effects": [{"type": "restore_mana", "amount": 40}],
        "value": 20,
    },
    "felix_felicis": {
        "id": "felix_felicis", "name": "Felix Felicis", "emoji": "✨",
        "type": "consumable", "rarity": "legendary",
        "description": "Liquid luck. Extraordinarily difficult to make. Grants fortune for the next combat encounter.",
        "effects": [{"type": "buff", "stat": "luck", "amount": 100, "duration": 1}],
        "value": 500,
    },
    "skele_gro": {
        "id": "skele_gro", "name": "Skele-Gro", "emoji": "🦴",
        "type": "consumable", "rarity": "uncommon",
        "description": "Agonising but effective. Fully restores HP and removes all negative status effects.",
        "effects": [{"type": "heal_hp", "amount": 999}, {"type": "cleanse"}],
        "value": 80,
    },
    "invigoration_draught": {
        "id": "invigoration_draught", "name": "Invigoration Draught", "emoji": "⚗️",
        "type": "consumable", "rarity": "rare",
        "description": "Boosts magical power and mana for the next three combat rounds.",
        "effects": [
            {"type": "restore_mana", "amount": 30},
            {"type": "buff", "stat": "spell_power", "amount": 25, "duration": 3},
        ],
        "value": 60,
    },
    "antidote": {
        "id": "antidote", "name": "Antidote", "emoji": "🌿",
        "type": "consumable", "rarity": "common",
        "description": "A general-purpose antidote to mild poisons and hexes.",
        "effects": [{"type": "cleanse"}],
        "value": 15,
    },

    # ── Artifacts ────────────────────────────────────────────────────────────
    "remembrall": {
        "id": "remembrall", "name": "Remembrall", "emoji": "🔮",
        "type": "artifact", "rarity": "rare",
        "description": "A glass ball filled with smoke. It turns red when you've forgotten something important. Reveals one hidden lore secret.",
        "effects": [{"type": "reveal_secret"}],
        "value": 200,
    },
    "marauders_map": {
        "id": "marauders_map", "name": "Marauder's Map", "emoji": "🗺️",
        "type": "artifact", "rarity": "legendary",
        "description": "I solemnly swear I am up to no good. Shows all NPC locations in current area for 5 turns.",
        "effects": [{"type": "buff", "stat": "npc_tracking", "amount": 1, "duration": 5}],
        "value": 1000,
    },
    "time_turner": {
        "id": "time_turner", "name": "Time-Turner Fragment", "emoji": "⏳",
        "type": "artifact", "rarity": "legendary",
        "description": "A broken Time-Turner. Can be used once to undo the last combat loss.",
        "effects": [{"type": "revive"}],
        "value": 2000,
    },
    "invisibility_cloak": {
        "id": "invisibility_cloak", "name": "Invisibility Cloak", "emoji": "🌫️",
        "type": "artifact", "rarity": "legendary",
        "description": "A true Invisibility Cloak — rare and extraordinary. Skip the next random encounter.",
        "effects": [{"type": "buff", "stat": "invisibility", "amount": 1, "duration": 1}],
        "value": 5000,
    },
    "pensieve_fragment": {
        "id": "pensieve_fragment", "name": "Pensieve Fragment", "emoji": "💭",
        "type": "artifact", "rarity": "rare",
        "description": "A shard of silver memory. Using it surfaces a forgotten key story beat.",
        "effects": [{"type": "recall_memory"}],
        "value": 300,
    },
    "elder_wand_shard": {
        "id": "elder_wand_shard", "name": "Elder Wand Shard", "emoji": "🪄",
        "type": "artifact", "rarity": "legendary",
        "description": "A fragment of legend. Temporarily makes all spells 50% more powerful.",
        "effects": [{"type": "buff", "stat": "spell_power", "amount": 50, "duration": 5}],
        "value": 3000,
    },

    # ── Key items ────────────────────────────────────────────────────────────
    "azkaban_key": {
        "id": "azkaban_key", "name": "Azkaban Master Key", "emoji": "🗝️",
        "type": "key", "rarity": "unique",
        "description": "A heavy iron key etched with Ministry seals. Opens restricted areas of Azkaban.",
        "effects": [{"type": "unlock_location", "location": "loc_006_restricted"}],
        "value": 0,
    },
    "knockturn_dossier": {
        "id": "knockturn_dossier", "name": "Knockturn Alley Dossier", "emoji": "📋",
        "type": "key", "rarity": "unique",
        "description": "Ministry intelligence files on the Hollow Circle's Knockturn operations.",
        "effects": [{"type": "reveal_secret"}, {"type": "xp_bonus", "amount": 150}],
        "value": 0,
    },
    "gillyweed": {
        "id": "gillyweed", "name": "Gillyweed", "emoji": "🌿",
        "type": "consumable", "rarity": "rare",
        "description": "A grey-green plant found in Mediterranean waters. Grants temporary water-breathing and swimming speed.",
        "effects": [{"type": "buff", "stat": "underwater", "amount": 1, "duration": 3}],
        "value": 50,
    },

    # ── Ingredients ──────────────────────────────────────────────────────────
    "boomslang_skin": {
        "id": "boomslang_skin", "name": "Boomslang Skin", "emoji": "🐍",
        "type": "ingredient", "rarity": "rare",
        "description": "A key Polyjuice Potion ingredient. Valuable to brewers and potion-makers.",
        "effects": [],
        "value": 100,
    },
    "bezoar": {
        "id": "bezoar", "name": "Bezoar", "emoji": "💎",
        "type": "consumable", "rarity": "rare",
        "description": "A stone from a goat's stomach that can counteract most poisons. Emergency antidote.",
        "effects": [{"type": "cleanse"}, {"type": "heal_hp", "amount": 30}],
        "value": 150,
    },
    "crystallised_pineapple": {
        "id": "crystallised_pineapple", "name": "Crystallised Pineapple", "emoji": "🍍",
        "type": "ingredient", "rarity": "common",
        "description": "Used in Honeydukes sweets and certain confectionery potions.",
        "effects": [],
        "value": 5,
    },

    # ── Scrolls ──────────────────────────────────────────────────────────────
    "scroll_patronum": {
        "id": "scroll_patronum", "name": "Patronus Scroll", "emoji": "📜",
        "type": "scroll", "rarity": "rare",
        "description": "A scroll inscribed with the incantation for Expecto Patronum. Teaches the spell permanently.",
        "effects": [{"type": "teach_spell", "spell": "Expecto Patronum"}],
        "value": 200,
    },
    "scroll_reparo": {
        "id": "scroll_reparo", "name": "Reparo Scroll", "emoji": "📜",
        "type": "scroll", "rarity": "uncommon",
        "description": "A mending scroll. Teaches Reparo and restores 10 HP from minor wounds.",
        "effects": [{"type": "teach_spell", "spell": "Reparo"}, {"type": "heal_hp", "amount": 10}],
        "value": 60,
    },
}

# ── Loot tables per enemy archetype ───────────────────────────────────────────

LOOT_TABLES: Dict[str, List[Tuple[str, float]]] = {
    "Death Eater":  [("healing_potion", 0.4), ("mana_draught", 0.3), ("knockturn_dossier", 0.1), ("antidote", 0.5)],
    "Dementor":     [("healing_potion", 0.5), ("felix_felicis", 0.05)],
    "Acromantula":  [("boomslang_skin", 0.3), ("antidote", 0.6), ("healing_potion", 0.3)],
    "Troll":        [("healing_potion", 0.5), ("skele_gro", 0.1)],
    "Werewolf":     [("antidote", 0.5), ("healing_potion", 0.4), ("bezoar", 0.15)],
    "Dark Wizard":  [("mana_draught", 0.4), ("healing_potion", 0.3), ("scroll_reparo", 0.1)],
    "Hollow Mage":  [("elder_wand_shard", 0.25), ("time_turner", 0.1), ("felix_felicis", 0.2)],
    "Boggart":      [("butterbeer", 0.7), ("remembrall", 0.1)],
    "Inferi":       [("antidote", 0.6), ("healing_potion", 0.3)],
}

RARITY_GALLEONS = {"common": (2, 10), "uncommon": (10, 30), "rare": (25, 75), "legendary": (100, 300), "unique": (0, 0)}


# ── Core functions ────────────────────────────────────────────────────────────

def get_item(item_id: str) -> Optional[Dict[str, Any]]:
    return ITEM_DB.get(item_id)


def make_item_instance(item_id: str) -> Optional[Dict[str, Any]]:
    """Return a fresh item instance dict (copy of DB entry + instance metadata)."""
    template = ITEM_DB.get(item_id)
    if not template:
        return None
    inst = dict(template)
    inst["acquired_at"] = time.time()
    return inst


def roll_loot(enemy_archetype: str, player_level: int) -> List[Dict[str, Any]]:
    """
    Roll loot for a defeated enemy. Higher level = better chance of multiple drops.
    Returns list of item instance dicts.
    """
    table = LOOT_TABLES.get(enemy_archetype, [("healing_potion", 0.3)])
    drops: List[Dict] = []
    # Level bonus: +2% per level above 1
    level_bonus = (player_level - 1) * 0.02
    for item_id, base_chance in table:
        if random.random() < min(base_chance + level_bonus, 0.95):
            inst = make_item_instance(item_id)
            if inst:
                drops.append(inst)
    return drops


def use_item(
    item: Dict[str, Any],
    player: Dict[str, Any],
    active_effects: List[Dict[str, Any]],
    story_beats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Apply item effects to a player dict (mutated in place).
    Returns a result dict: {success, message, effects_applied}.
    """
    effects_applied = []
    messages = []

    for eff in item.get("effects", []):
        t = eff["type"]

        if t == "heal_hp":
            amt = min(eff["amount"], player["max_hp"] - player["hp"])
            player["hp"] = min(player["hp"] + eff["amount"], player["max_hp"])
            messages.append(f"Restored {amt} HP (now {player['hp']}/{player['max_hp']})")
            effects_applied.append(eff)

        elif t == "restore_mana":
            amt = min(eff["amount"], player["max_mana"] - player["mana"])
            player["mana"] = min(player["mana"] + eff["amount"], player["max_mana"])
            messages.append(f"Restored {amt} Mana (now {player['mana']}/{player['max_mana']})")
            effects_applied.append(eff)

        elif t == "buff":
            active_effects.append({**eff, "turns_remaining": eff.get("duration", 1), "source": item["id"]})
            messages.append(f"Gained {eff['stat']} +{eff['amount']} for {eff.get('duration', 1)} turns")
            effects_applied.append(eff)

        elif t == "cleanse":
            before = len(active_effects)
            active_effects[:] = [e for e in active_effects if e.get("type") != "debuff"]
            removed = before - len(active_effects)
            messages.append(f"Cleansed {removed} negative effect(s)")
            effects_applied.append(eff)

        elif t == "teach_spell":
            spell = eff["spell"]
            if spell not in player.get("spells_known", []):
                player.setdefault("spells_known", []).append(spell)
                messages.append(f"Learned {spell}!")
            else:
                messages.append(f"Already know {spell}")
            effects_applied.append(eff)

        elif t == "xp_bonus":
            player["xp"] = player.get("xp", 0) + eff["amount"]
            messages.append(f"Gained {eff['amount']} XP")
            effects_applied.append(eff)

        elif t in ("reveal_secret", "recall_memory", "revive", "unlock_location"):
            messages.append(f"Special effect '{t}' triggered")
            effects_applied.append(eff)

    return {
        "success": bool(effects_applied),
        "message": "; ".join(messages) if messages else "No effect",
        "effects_applied": effects_applied,
        "item_name": item["name"],
        "item_emoji": item.get("emoji", "📦"),
    }


def tick_effects(active_effects: List[Dict]) -> List[Dict]:
    """Decrement durations; remove expired effects. Returns still-active effects."""
    updated = []
    for eff in active_effects:
        if "turns_remaining" in eff:
            eff["turns_remaining"] -= 1
            if eff["turns_remaining"] > 0:
                updated.append(eff)
        else:
            updated.append(eff)
    return updated


def has_buff(active_effects: List[Dict], stat: str) -> Optional[int]:
    """Return total buff amount for a stat, or None if not buffed."""
    total = sum(e["amount"] for e in active_effects
                if e.get("type") == "buff" and e.get("stat") == stat)
    return total if total else None


def inventory_to_display(inventory: List[Dict]) -> List[Dict]:
    """Enrich raw inventory items with full DB metadata for display."""
    result = []
    for item in inventory:
        db_item = ITEM_DB.get(item.get("id", ""))
        if db_item:
            merged = {**db_item, **item}
        else:
            merged = item
        result.append(merged)
    return result


def item_summary(item: Dict) -> str:
    """One-line summary for UI tooltip."""
    eff_strs = []
    for e in item.get("effects", []):
        t = e["type"]
        if t == "heal_hp":        eff_strs.append(f"+{e['amount']} HP")
        elif t == "restore_mana": eff_strs.append(f"+{e['amount']} MP")
        elif t == "buff":         eff_strs.append(f"{e['stat']} +{e['amount']} ({e.get('duration',1)} turns)")
        elif t == "teach_spell":  eff_strs.append(f"Teaches {e['spell']}")
        elif t == "cleanse":      eff_strs.append("Cleanse")
    return " | ".join(eff_strs) if eff_strs else "No combat effect"
