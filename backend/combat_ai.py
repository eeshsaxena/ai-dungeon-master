"""
Enemy AI for AI Dungeon Master.

Phase 4: the decision core is a tabular Q-learning agent (see rl_agent.py) that
adapts to the player's fighting style. The rule-based personality logic is kept
as a warm-start prior for states the agent has not yet encountered, and as a
fallback when RL is disabled (ENEMY_AI=rule).
"""
import os
import random
from typing import Dict, Optional, Tuple
from models.schemas import CombatAction, EnemyArchetype, Enemy, CombatRequest, CombatResponse
from rl_agent import rl_agent, encode_state, ACTION_EFFECTS, ACTIONS

USE_RL = os.getenv("ENEMY_AI", "rl").lower() == "rl"

# Map the strategic action string used by the RL agent to a CombatAction enum.
ACTION_TO_ENUM = {
    "attack": CombatAction.ATTACK,
    "defend": CombatAction.DEFEND,
    "special": CombatAction.SPECIAL,
}


# ── Enemy Definitions ──────────────────────────────────────────────────────────

ENEMY_TEMPLATES = {
    EnemyArchetype.DEMENTOR: {
        "hp": 80, "max_hp": 80, "attack": 25, "defense": 5,
        "special_ability": "Soul Drain — reduces player's mana by 30",
        "weakness": "Expecto Patronum",
        "personality": "aggressive",
        "spells": ["Soul Drain", "Despair Wave", "Chill Aura"],
        "loot": [{"name": "Dark Essence", "type": "ingredient", "galleons": 15}],
        "xp": 200,
        "description": "A cloaked wraith that glides soundlessly. The temperature drops as it approaches."
    },
    EnemyArchetype.DEATH_EATER: {
        "hp": 120, "max_hp": 120, "attack": 35, "defense": 20,
        "special_ability": "Crucio — stuns player for 1 turn",
        "weakness": "Expelliarmus",
        "personality": "cunning",
        "spells": ["Avada Kedavra", "Crucio", "Sectumsempra", "Expelliarmus"],
        "loot": [{"name": "Death Eater Mask", "type": "key_item", "galleons": 50}],
        "xp": 450,
        "description": "A masked witch or wizard in dark robes. Their wand hand trembles with barely contained dark magic."
    },
    EnemyArchetype.ACROMANTULA: {
        "hp": 150, "max_hp": 150, "attack": 40, "defense": 15,
        "special_ability": "Web Snare — player loses their next action",
        "weakness": "Lumos Maxima",
        "personality": "aggressive",
        "spells": ["Venom Bite", "Web Snare", "Charge"],
        "loot": [{"name": "Acromantula Venom", "type": "ingredient", "galleons": 100}],
        "xp": 600,
        "description": "Eight terrible eyes glitter in the darkness. Mandibles click rhythmically. The spider is enormous."
    },
    EnemyArchetype.TROLL: {
        "hp": 200, "max_hp": 200, "attack": 50, "defense": 30,
        "special_ability": "Club Smash — massive damage but skips next turn",
        "weakness": "Wingardium Leviosa",
        "personality": "aggressive",
        "spells": ["Club Smash", "Roar", "Stomp"],
        "loot": [{"name": "Troll Club Fragment", "type": "item", "galleons": 20}],
        "xp": 400,
        "description": "A mountain troll. Enormous, green-grey, smells like a blocked drain. Not very bright."
    },
    EnemyArchetype.WEREWOLF: {
        "hp": 160, "max_hp": 160, "attack": 45, "defense": 20,
        "special_ability": "Feral Rage — doubles attack for 2 turns when below 50% HP",
        "weakness": "Silver-tipped spells",
        "personality": "erratic",
        "spells": ["Savage Bite", "Feral Charge", "Howl", "Feral Rage"],
        "loot": [{"name": "Wolfsbane Residue", "type": "ingredient", "galleons": 80}],
        "xp": 550,
        "description": "A cursed witch or wizard mid-transformation. Their eyes are wild with a pain that has turned to rage."
    },
    EnemyArchetype.DARK_WIZARD: {
        "hp": 140, "max_hp": 140, "attack": 40, "defense": 25,
        "special_ability": "Counter-Jinx — reflects 50% of player's spell damage",
        "weakness": "Unpredictable spell combinations",
        "personality": "cunning",
        "spells": ["Sectumsempra", "Confundus", "Legilimens", "Avada Kedavra"],
        "loot": [{"name": "Enchanted Grimoire", "type": "book", "galleons": 200}],
        "xp": 500,
        "description": "A pale figure with cold eyes and a wand that seems to move before thought."
    },
    EnemyArchetype.HOLLOW_MAGE: {
        "hp": 350, "max_hp": 350, "attack": 70, "defense": 40,
        "special_ability": "Soul Hollow — drains player's XP and adds to own HP",
        "weakness": "Love and self-sacrifice magic",
        "personality": "cunning",
        "spells": ["Soul Hollow", "Dark Binding", "Void Pulse", "Unravel", "Lich's Grasp"],
        "loot": [{"name": "The Hollow Orb", "type": "artifact", "galleons": 1000}],
        "xp": 5000,
        "description": "The Hollow Mage. His form seems to absorb light. Where his eyes should be: two voids."
    },
    EnemyArchetype.BOGGART: {
        "hp": 60, "max_hp": 60, "attack": 20, "defense": 5,
        "special_ability": "Shapeshift — takes the form of your greatest fear",
        "weakness": "Riddikulus",
        "personality": "erratic",
        "spells": ["Fear Manifestation", "Shapeshift", "Nightmare"],
        "loot": [],
        "xp": 150,
        "description": "A shape-shifting creature that takes the form of whatever you fear most."
    },
    EnemyArchetype.INFERI: {
        "hp": 90, "max_hp": 90, "attack": 28, "defense": 10,
        "special_ability": "Undying — respawns once unless fire damage applied",
        "weakness": "Fire spells",
        "personality": "aggressive",
        "spells": ["Grasp", "Overwhelm", "Undying Rise"],
        "loot": [{"name": "Rotten Cloth Fragment", "type": "trash", "galleons": 2}],
        "xp": 180,
        "description": "A corpse reanimated by dark magic. Its movements are jerky and wrong. It feels no pain."
    }
}

# ── Behavior Tracking (RL Scaffold) ───────────────────────────────────────────

class BehaviorTracker:
    """
    Tracks player behavior patterns for adaptive AI.
    Phase 4: This will feed into the RL agent's state space.
    Currently: stores data for future training.
    """
    def __init__(self):
        self.player_spell_counts: Dict[str, int] = {}
        self.player_action_counts: Dict[str, int] = {}
        self.rounds_survived: int = 0
        self.consecutive_attacks: int = 0
        self.defensive_turns: int = 0
        self.last_player_action: Optional[str] = None

    def record_player_action(self, action: str, spell: Optional[str] = None):
        self.player_action_counts[action] = self.player_action_counts.get(action, 0) + 1
        if spell:
            self.player_spell_counts[spell] = self.player_spell_counts.get(spell, 0) + 1
        if action == "attack":
            self.consecutive_attacks += 1
            self.defensive_turns = 0
        elif action == "defend":
            self.defensive_turns += 1
            self.consecutive_attacks = 0
        self.last_player_action = action
        self.rounds_survived += 1

    def get_player_tendency(self) -> str:
        total = sum(self.player_action_counts.values()) or 1
        attack_rate = self.player_action_counts.get("attack", 0) / total
        defend_rate = self.player_action_counts.get("defend", 0) / total
        if attack_rate > 0.6:
            return "aggressive"
        elif defend_rate > 0.4:
            return "defensive"
        else:
            return "balanced"


# ── Enemy AI Engine ────────────────────────────────────────────────────────────

class EnemyAI:
    """
    Rule-based enemy AI with personality-driven decision making.
    Tracks player patterns and adapts strategy accordingly.
    RL upgrade hook: replace select_action() with a trained Q-network in Phase 4.
    """

    def __init__(self):
        self.behavior_trackers: Dict[str, BehaviorTracker] = {}

    def get_tracker(self, session_id: str) -> BehaviorTracker:
        if session_id not in self.behavior_trackers:
            self.behavior_trackers[session_id] = BehaviorTracker()
        return self.behavior_trackers[session_id]

    def select_action(
        self,
        enemy: Enemy,
        tracker: BehaviorTracker,
        round_number: int,
        player_hp_ratio: float,
        enemy_hp_ratio: float
    ) -> Tuple[CombatAction, Optional[str], str, str, str]:
        """
        Decide the enemy's strategic action for this round.
        Returns: (combat_action, spell, narrative, state_key, action_str)

        The rule-based personality logic supplies a prior; the RL agent makes
        the final choice (and learns from the outcome in resolve_combat_round).
        """
        template = ENEMY_TEMPLATES.get(enemy.archetype, ENEMY_TEMPLATES[EnemyArchetype.DARK_WIZARD])
        spells = template["spells"]

        prior = self._rule_based_strategy(
            enemy, tracker, round_number, player_hp_ratio, enemy_hp_ratio
        )

        state = encode_state(
            enemy_hp_ratio, player_hp_ratio,
            tracker.get_player_tendency(), tracker.last_player_action,
        )

        action_str = rl_agent.select_action(enemy.archetype.value, state, prior) if USE_RL else prior

        combat_action = ACTION_TO_ENUM.get(action_str, CombatAction.ATTACK)
        spell = None if action_str == "defend" else random.choice(spells)
        narrative = self._narrate_action(enemy, action_str, state)
        return combat_action, spell, narrative, state, action_str

    def _rule_based_strategy(
        self, enemy: Enemy, tracker: BehaviorTracker,
        round_number: int, player_hp_ratio: float, enemy_hp_ratio: float
    ) -> str:
        """Personality-driven heuristic, collapsed to one of ACTIONS. Used as
        the RL warm-start prior and as the policy when ENEMY_AI=rule."""
        personality = enemy.personality
        tendency = tracker.get_player_tendency()

        if personality == "aggressive":
            return "special" if enemy_hp_ratio < 0.3 else "attack"
        if personality == "cunning":
            if tendency == "aggressive" and round_number > 2:
                return "defend"   # bait the predictable attacker
            if tracker.defensive_turns > 2:
                return "special"  # punish turtling
            return "attack"
        if personality == "defensive":
            if player_hp_ratio < 0.4:
                return "attack"   # press the advantage
            if enemy_hp_ratio < 0.5:
                return "defend"
            return "attack"
        if personality == "erratic":
            return random.choice(ACTIONS)
        return "attack"

    def _narrate_action(self, enemy: Enemy, action_str: str, state: str) -> str:
        """Flavor text for the chosen action; hints when the agent is acting on
        learned experience rather than the default prior."""
        name = enemy.name
        learned = USE_RL and not rl_agent._is_unseen(enemy.archetype.value, state)
        if action_str == "special":
            tail = " — it has seen this fight before" if learned else ""
            return f"The {name} channels its full power into a devastating strike{tail}!"
        if action_str == "defend":
            tail = ", reading your rhythm" if learned else ""
            return f"The {name} guards{tail}, deflecting the brunt of your assault."
        return f"The {name} presses the attack!"

    def resolve_combat_round(self, request: CombatRequest) -> CombatResponse:
        """Resolve a full combat round and let the RL agent learn from it."""
        tracker = self.get_tracker(request.session_id)
        tracker.record_player_action(request.player_action, request.player_spell)

        enemy = request.enemy
        player = request.player_stats

        player_hp_ratio = player.hp / max(player.max_hp, 1)
        enemy_hp_ratio = enemy.hp / max(enemy.max_hp, 1)

        # Enemy chooses its strategic action (RL policy or rule prior)
        action, spell, narrative, state, action_str = self.select_action(
            enemy, tracker, request.round_number, player_hp_ratio, enemy_hp_ratio
        )
        effects = ACTION_EFFECTS.get(action_str, ACTION_EFFECTS["attack"])

        # Enemy's outgoing damage (0 when defending)
        enemy_damage = int(enemy.attack * effects["out_mult"] * random.uniform(0.85, 1.15))

        # Player's outgoing damage, mitigated when the enemy guards
        player_damage = self._resolve_player_attack(
            request.player_action, request.player_spell, enemy, action
        )
        player_damage = int(player_damage * effects["incoming_mult"])

        # Weakness exploit
        template = ENEMY_TEMPLATES.get(enemy.archetype, {})
        weakness = template.get("weakness", "")
        if request.player_spell and weakness and weakness.lower() in request.player_spell.lower():
            player_damage = int(player_damage * 2.5)
            narrative += f"\n*{enemy.name} recoils — that's their weakness!*"

        # Resolve HP
        new_enemy_hp = max(0, enemy.hp - player_damage)
        new_player_hp = max(0, player.hp - enemy_damage)

        combat_over = new_enemy_hp <= 0 or new_player_hp <= 0
        player_won = (new_enemy_hp <= 0) if combat_over else None
        enemy_won = new_player_hp <= 0

        # ── RL: reward + Q-update (from the enemy's perspective) ───────────
        if USE_RL:
            reward = (enemy_damage / max(player.max_hp, 1)) - (player_damage / max(enemy.max_hp, 1))
            if combat_over:
                reward += 1.0 if enemy_won else -1.0
            next_state = encode_state(
                new_enemy_hp / max(enemy.max_hp, 1),
                new_player_hp / max(player.max_hp, 1),
                tracker.get_player_tendency(),
                request.player_action,
            )
            rl_agent.learn(
                enemy.archetype.value, state, action_str, reward, next_state, combat_over
            )
            if combat_over:
                rl_agent.end_episode(enemy_won)

        loot = []
        xp_gained = 0
        if player_won:
            loot = template.get("loot", [])
            xp_gained = template.get("xp", 100)

        return CombatResponse(
            enemy_action=action,
            enemy_spell=spell,
            narrative=narrative,
            player_damage=player_damage,
            enemy_damage=enemy_damage,
            enemy_hp_remaining=new_enemy_hp,
            player_hp_remaining=new_player_hp,
            combat_over=combat_over,
            player_won=player_won,
            loot=loot,
            xp_gained=xp_gained
        )

    def _resolve_player_attack(
        self, action: str, spell: Optional[str], enemy: Enemy, enemy_action: CombatAction
    ) -> int:
        """Calculate player's damage output."""
        SPELL_DAMAGE = {
            "Expelliarmus": 25,
            "Stupefy": 35,
            "Protego": 0,
            "Expecto Patronum": 60,
            "Sectumsempra": 70,
            "Fiendfyre": 120,
            "Accio": 10,
            "Lumos": 15,
            "Lumos Maxima": 45,
            "Wingardium Leviosa": 30,
            "Riddikulus": 50,
        }

        if action == "defend":
            return 0
        elif action == "flee":
            return 0

        base = SPELL_DAMAGE.get(spell or "", 30) if spell else 20
        variance = random.uniform(0.85, 1.15)
        return int(base * variance)

    def spawn_enemy(self, archetype: EnemyArchetype, level_scaling: float = 1.0) -> Enemy:
        """Create an enemy instance from an archetype template."""
        template = ENEMY_TEMPLATES.get(archetype, ENEMY_TEMPLATES[EnemyArchetype.DARK_WIZARD])
        hp = int(template["hp"] * level_scaling)

        enemy_names = {
            EnemyArchetype.DEATH_EATER: random.choice(["Lucius Malfoy's Follower", "Bellatrix's Disciple", "Avery", "Nott", "Rookwood"]),
            EnemyArchetype.DEMENTOR: "Dementor",
            EnemyArchetype.ACROMANTULA: random.choice(["Aragog's Offspring", "Giant Acromantula", "Shadow Spinner"]),
            EnemyArchetype.TROLL: random.choice(["Mountain Troll", "Forest Troll"]),
            EnemyArchetype.WEREWOLF: random.choice(["Vyla's Pack Member", "Feral Werewolf"]),
            EnemyArchetype.DARK_WIZARD: random.choice(["Hollow Circle Agent", "Rogue Auror", "Shadow Cultist"]),
            EnemyArchetype.HOLLOW_MAGE: "The Hollow Mage",
            EnemyArchetype.BOGGART: "Boggart",
            EnemyArchetype.INFERI: "Inferius"
        }

        return Enemy(
            id=f"{archetype.value.lower().replace(' ', '_')}_{random.randint(100, 999)}",
            name=enemy_names.get(archetype, archetype.value),
            archetype=archetype,
            hp=hp,
            max_hp=hp,
            attack=int(template["attack"] * level_scaling),
            defense=int(template["defense"] * level_scaling),
            special_ability=template["special_ability"],
            weakness=template.get("weakness"),
            personality=template["personality"]
        )


# Global AI instance
enemy_ai = EnemyAI()
