"""
Self-play harness for the RL enemy agent.

Drives the same combat dynamics the live game uses (see ACTION_EFFECTS) against
a scripted player, so we can show — quantitatively — that the agent learns:
its win rate after training should beat the hand-coded rule baseline.

Used by the /api/rl-simulate endpoint and runnable directly:
    python rl_train.py
"""
import random
from typing import Dict, Optional

from rl_agent import RLEnemyAgent, rl_agent, encode_state, ACTION_EFFECTS
from combat_ai import ENEMY_TEMPLATES
from models.schemas import EnemyArchetype

# Scripted player archetypes: how they act, how hard they hit, how the agent
# perceives their tendency (the value that goes into the state).
# Damage values sit in the "hard but winnable" zone where a naive rule policy
# loses the damage race but a learned policy can flip it — that's what makes the
# learning measurable. Each style differs in the tendency the agent keys on.
PLAYER_STYLES: Dict[str, Dict] = {
    "aggressive": {"tendency": "aggressive", "action": "attack", "damage": 72},
    "balanced":   {"tendency": "balanced",   "action": "attack", "damage": 66},
    "defensive":  {"tendency": "defensive",  "action": "defend", "damage": 60},
}


def _rule_prior(enemy_hp_ratio: float) -> str:
    """Mirror of combat_ai._rule_based_strategy for a generic enemy."""
    return "special" if enemy_hp_ratio < 0.3 else "attack"


def _play_combat(agent: RLEnemyAgent, archetype: str, tmpl: dict,
                 style: dict, *, learn: bool, explore: bool) -> bool:
    """Run one combat to its end. Returns True if the enemy won."""
    enemy_hp, enemy_max, enemy_atk = tmpl["max_hp"], tmpl["max_hp"], tmpl["attack"]
    player_hp = player_max = 100
    last_action: Optional[str] = None
    rounds = 0

    while enemy_hp > 0 and player_hp > 0 and rounds < 60:
        rounds += 1
        ehr, phr = enemy_hp / enemy_max, player_hp / player_max
        state = encode_state(ehr, phr, style["tendency"], last_action)
        action = agent.select_action(archetype, state, _rule_prior(ehr), explore=explore)
        eff = ACTION_EFFECTS[action]

        enemy_dmg = int(enemy_atk * eff["out_mult"] * random.uniform(0.85, 1.15))
        player_dmg = int(style["damage"] * eff["incoming_mult"] * random.uniform(0.85, 1.15))

        new_enemy_hp = max(0, enemy_hp - player_dmg)
        new_player_hp = max(0, player_hp - enemy_dmg)
        over = new_enemy_hp <= 0 or new_player_hp <= 0
        enemy_won = new_player_hp <= 0

        if learn:
            reward = (enemy_dmg / player_max) - (player_dmg / enemy_max)
            if over:
                reward += 1.0 if enemy_won else -1.0
            next_state = encode_state(
                new_enemy_hp / enemy_max, new_player_hp / player_max,
                style["tendency"], style["action"],
            )
            agent.learn(archetype, state, action, reward, next_state, over)

        enemy_hp, player_hp = new_enemy_hp, new_player_hp
        last_action = style["action"]

    return player_hp <= 0


def simulate(archetype: str = "Death Eater", train_combats: int = 400,
             eval_combats: int = 300, player_style: str = "aggressive",
             use_global: bool = False) -> Dict:
    """
    Measure the agent's enemy-win-rate before vs after learning.

    use_global=False (default): trains a throwaway agent so the demo doesn't
    touch the persisted table. Set True to train the live agent on this style.
    """
    try:
        arch_enum = EnemyArchetype(archetype)
    except ValueError:
        arch_enum = EnemyArchetype.DARK_WIZARD
    archetype = arch_enum.value
    tmpl = ENEMY_TEMPLATES[arch_enum]
    style = PLAYER_STYLES.get(player_style, PLAYER_STYLES["aggressive"])

    agent = rl_agent if use_global else RLEnemyAgent(persist=False)

    # 1) Baseline: greedy, no learning. Unseen states use the rule prior.
    baseline_wins = sum(
        _play_combat(agent, archetype, tmpl, style, learn=False, explore=False)
        for _ in range(eval_combats)
    )

    # 2) Train with exploration + learning, capturing a coarse learning curve.
    curve, window, won = [], max(1, train_combats // 10), 0
    for i in range(1, train_combats + 1):
        if _play_combat(agent, archetype, tmpl, style, learn=True, explore=True):
            won += 1
        if i % window == 0:
            curve.append(round(won / window, 3))
            won = 0

    # 3) Evaluate the learned greedy policy.
    learned_wins = sum(
        _play_combat(agent, archetype, tmpl, style, learn=False, explore=False)
        for _ in range(eval_combats)
    )

    if use_global:
        agent._save()

    return {
        "archetype": archetype,
        "player_style": player_style,
        "train_combats": train_combats,
        "eval_combats": eval_combats,
        "baseline_win_rate": round(baseline_wins / eval_combats, 3),
        "learned_win_rate": round(learned_wins / eval_combats, 3),
        "improvement": round((learned_wins - baseline_wins) / eval_combats, 3),
        "learning_curve": curve,
        "learned_policy": agent.policy_for(archetype),
    }


if __name__ == "__main__":
    for style in ("aggressive", "balanced", "defensive"):
        r = simulate(player_style=style)
        print(f"\n[{style}] baseline={r['baseline_win_rate']} "
              f"learned={r['learned_win_rate']} (+{r['improvement']})")
        print(f"  curve: {r['learning_curve']}")
