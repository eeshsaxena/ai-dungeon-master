"""
Reinforcement-learning enemy agent for the AI Dungeon Master.

Phase 4: replaces the rule-based combat scaffold's decision core with a
tabular Q-learning policy that adapts to how a given player fights.

Why tabular Q-learning (and not a deep net): the combat state factorises
cleanly into a handful of discrete buckets, so the table is tiny (81 states ×
3 actions) and learns from the few dozen rounds a real session produces — no
GPU, no training loop, no external deps beyond numpy. It genuinely improves
online and the learned policy is fully inspectable for the /api/rl-stats view.

State  : (enemy_hp_bucket, player_hp_bucket, player_tendency, last_player_action)
Action : attack | defend | special
Reward : damage dealt to player − damage taken, plus a terminal win/loss bonus
         (from the *enemy's* perspective).

Unseen states fall back to the rule-based prior so the agent fights sensibly
from the very first turn, then improves as it gathers experience. Q-tables are
persisted per enemy archetype so learning survives restarts.
"""
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

RL_DATA_DIR = Path(__file__).parent / "rl_data"
Q_TABLE_PATH = RL_DATA_DIR / "q_tables.json"

ACTIONS: List[str] = ["attack", "defend", "special"]

# Hyperparameters
ALPHA = 0.2          # learning rate
GAMMA = 0.9          # discount factor
EPSILON_START = 0.30  # initial exploration
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.995  # per learning step

# Mechanical effects of each enemy action (used by combat resolution).
# Each action is a genuine tradeoff so the optimal choice is state-dependent:
#   attack  — balanced
#   special — hits 1.8x but the enemy over-commits and takes 1.3x in return
#   defend  — deals nothing but only takes 0.4x
ACTION_EFFECTS = {
    "attack": {"out_mult": 1.0, "incoming_mult": 1.0},
    "special": {"out_mult": 1.8, "incoming_mult": 1.3},
    "defend": {"out_mult": 0.0, "incoming_mult": 0.4},
}


def hp_bucket(ratio: float) -> str:
    if ratio > 0.66:
        return "high"
    if ratio > 0.33:
        return "mid"
    return "low"


def action_bucket(player_action: Optional[str]) -> str:
    a = (player_action or "").lower()
    if "defend" in a or "protego" in a or "shield" in a:
        return "defend"
    if "special" in a or "flee" in a or "taunt" in a:
        return "other"
    return "attack"  # default: casting an offensive spell / attacking


def encode_state(
    enemy_hp_ratio: float,
    player_hp_ratio: float,
    player_tendency: str,
    last_player_action: Optional[str],
) -> str:
    """Discretise the combat situation into a hashable state key."""
    return "|".join([
        hp_bucket(enemy_hp_ratio),
        hp_bucket(player_hp_ratio),
        player_tendency,
        action_bucket(last_player_action),
    ])


class RLEnemyAgent:
    """
    One Q-learning policy per enemy archetype.

    The player's fighting tendency (aggressive/defensive/balanced) is part of
    the state, so a single per-archetype table naturally specialises its
    response to whoever it is currently facing.
    """

    def __init__(self, persist: bool = True):
        self.persist = persist
        # archetype -> { state_key -> [q_attack, q_defend, q_special] }
        self.q_tables: Dict[str, Dict[str, List[float]]] = {}
        self.epsilon: float = EPSILON_START
        self.episodes: int = 0       # combats finished
        self.updates: int = 0        # Q-updates performed
        self.wins: int = 0           # combats the enemy won
        self.losses: int = 0
        if self.persist:
            self._load()

    # -- table access ---------------------------------------------------------

    def _table(self, archetype: str) -> Dict[str, List[float]]:
        return self.q_tables.setdefault(archetype, {})

    def _q(self, archetype: str, state: str) -> List[float]:
        return self._table(archetype).setdefault(state, [0.0, 0.0, 0.0])

    def _is_unseen(self, archetype: str, state: str) -> bool:
        q = self._table(archetype).get(state)
        return q is None or all(v == 0.0 for v in q)

    # -- policy ---------------------------------------------------------------

    def select_action(
        self,
        archetype: str,
        state: str,
        rule_based_action: Optional[str] = None,
        explore: bool = True,
    ) -> str:
        """
        Choose an enemy action for the given state.

        Unseen state -> use the rule-based prior (warm start).
        Otherwise -> epsilon-greedy over the learned Q-values.
        """
        if self._is_unseen(archetype, state) and rule_based_action in ACTIONS:
            return rule_based_action

        if explore and random.random() < self.epsilon:
            return random.choice(ACTIONS)

        q = self._q(archetype, state)
        best = max(range(len(ACTIONS)), key=lambda i: q[i])
        return ACTIONS[best]

    # -- learning -------------------------------------------------------------

    def learn(
        self,
        archetype: str,
        state: str,
        action: str,
        reward: float,
        next_state: Optional[str],
        terminal: bool,
    ):
        """Single-step Q-learning update."""
        if action not in ACTIONS:
            return
        ai = ACTIONS.index(action)
        q = self._q(archetype, state)
        future = 0.0
        if not terminal and next_state is not None:
            future = max(self._q(archetype, next_state))
        td_target = reward + GAMMA * future
        q[ai] += ALPHA * (td_target - q[ai])
        self.updates += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def end_episode(self, enemy_won: bool):
        self.episodes += 1
        if enemy_won:
            self.wins += 1
        else:
            self.losses += 1
        if self.persist:
            self._save()

    # -- persistence ----------------------------------------------------------

    def _load(self):
        if not Q_TABLE_PATH.exists():
            return
        try:
            data = json.loads(Q_TABLE_PATH.read_text(encoding="utf-8"))
            self.q_tables = data.get("q_tables", {})
            self.epsilon = data.get("epsilon", EPSILON_START)
            self.episodes = data.get("episodes", 0)
            self.updates = data.get("updates", 0)
            self.wins = data.get("wins", 0)
            self.losses = data.get("losses", 0)
        except Exception as e:
            print(f"[RL] Could not load Q-tables ({e}); starting fresh.")

    def _save(self):
        RL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        Q_TABLE_PATH.write_text(json.dumps({
            "q_tables": self.q_tables,
            "epsilon": round(self.epsilon, 5),
            "episodes": self.episodes,
            "updates": self.updates,
            "wins": self.wins,
            "losses": self.losses,
        }, indent=2), encoding="utf-8")

    # -- introspection --------------------------------------------------------

    def policy_for(self, archetype: str) -> Dict[str, Dict]:
        """Human-readable best action + Q-values per learned state."""
        out = {}
        for state, q in self._table(archetype).items():
            best = ACTIONS[max(range(len(ACTIONS)), key=lambda i: q[i])]
            out[state] = {
                "best_action": best,
                "q": {a: round(v, 3) for a, v in zip(ACTIONS, q)},
            }
        return out

    def stats(self) -> Dict:
        total = self.wins + self.losses
        return {
            "episodes": self.episodes,
            "updates": self.updates,
            "epsilon": round(self.epsilon, 4),
            "enemy_win_rate": round(self.wins / total, 3) if total else None,
            "archetypes_learned": {
                a: len(t) for a, t in self.q_tables.items()
            },
            "total_states": sum(len(t) for t in self.q_tables.values()),
        }


# Global agent instance
rl_agent = RLEnemyAgent()
