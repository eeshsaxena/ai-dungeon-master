"""
Hollow Mage Boss Fight (Phase 11).

A scripted 3-phase boss encounter, separate from the regular combat system.
Each phase has unique mechanics and narrative beats.

Phase 1: The Voice from the Dark (HP 100%-66%)
  - Mage attacks with shadow tendrils, deals heavy damage
  - Player must use Patronus or Lumos Maxima to break the darkness
  - Mage taunts about Hollow Circle's plan

Phase 2: The Hollow Form (HP 66%-33%)
  - Mage manifests physically; counters all standard spells
  - Player must exploit specific spells: Sectumsempra (damage), Expelliarmus
    (stun for 2 turns), Reducto (break shield)
  - Adds spawn: 2 Inferi every 3 turns

Phase 3: The Soul Anchor (HP 33%-0%)
  - Mage exposes the Horcrux fragment they're channeling
  - Player must use Fiendfyre OR have Elder Wand Shard buff active to deal real damage
  - Final stand — all-or-nothing
"""
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

BOSS_HP = 500
BOSS_NAME = "The Hollow Mage"

PHASE_THRESHOLDS = [
    {"phase": 1, "min_hp": 334, "name": "The Voice from the Dark"},
    {"phase": 2, "min_hp": 167, "name": "The Hollow Form"},
    {"phase": 3, "min_hp": 0,   "name": "The Soul Anchor"},
]

# Spells that work effectively in each phase
EFFECTIVE_SPELLS = {
    1: {"Expecto Patronum": 60, "Lumos Maxima": 50, "Lumos": 25},
    2: {"Sectumsempra": 70, "Expelliarmus": 30, "Reducto": 45, "Stupefy": 25},
    3: {"Fiendfyre": 120, "Sectumsempra": 50, "Expelliarmus": 40},
}

# Mage attacks per phase
MAGE_ATTACKS = {
    1: [
        ("Shadow Tendril", 18, "Black smoke lashes from the darkness, leaving a burning gash."),
        ("Whispered Curse", 14, "Your name is whispered in your own voice — and pain follows."),
        ("Void Pull", 12, "An emptiness opens beneath your ribs."),
    ],
    2: [
        ("Crucio Variant", 28, "A familiar curse, twisted into something colder."),
        ("Hollow Strike", 22, "The Mage moves faster than you can track."),
        ("Soul Drain", 20, "A grey light leaches from your wand toward the Mage."),
    ],
    3: [
        ("Anchor Backlash", 35, "The Horcrux fragment lashes out as you wound it."),
        ("Final Curse", 40, "A spell older than language, weaponized."),
        ("Death's Whisper", 30, "The Mage speaks the killing curse — but the syllables are wrong, deliberately."),
    ],
}

PHASE_INTRO = {
    1: "The air thickens until you can barely breathe. From the darkness ahead, a voice that sounds almost — *almost* — like someone you love.\n\n*\"You came. I'd hoped you would. It saves me looking for you.\"*\n\nThe Hollow Mage steps into the edge of your wandlight. Their face is a void. Your wand is the only thing keeping you from running.",
    2: "*\"You're better than I expected,\"* the Mage says, voice flat. *\"Most witches break here.\"*\n\nThe air around them solidifies. They have a form now — gaunt, robed, ancient. The Hollow Mage draws a wand that shouldn't exist, and you realize the previous attacks were them holding back.\n\nFrom the shadows behind them, two Inferi rise from the floor.",
    3: "*\"This is what you came for,\"* the Mage rasps, and they tear open their own chest with their wand hand.\n\nInside — pulsing, anchoring them to this body — is a Horcrux fragment. A piece of something far older and worse than Voldemort ever was.\n\n*\"You can't kill what's already been dead for five thousand years. But you can try. They all try.\"*",
}

PHASE_HINT = {
    1: "Light spells are most effective here — Expecto Patronum, Lumos Maxima.",
    2: "Standard spells are weakened. Try Sectumsempra, Reducto, or Expelliarmus.",
    3: "Only Fiendfyre or an Elder Wand-empowered spell can damage the Horcrux fragment.",
}


@dataclass
class BossState:
    session_id: str
    boss_hp: int = BOSS_HP
    boss_max_hp: int = BOSS_HP
    phase: int = 1
    rounds: int = 0
    player_stunned_by: int = 0      # turns of stun applied to mage
    inferi_alive: int = 0            # adds in phase 2
    fragment_exposed: bool = False  # phase 3 only
    started_at: float = 0.0
    log: List[str] = field(default_factory=list)
    active: bool = True


@dataclass
class BossRoundResult:
    boss_hp_remaining: int
    boss_max_hp: int
    player_hp_remaining: int
    player_damage: int
    boss_damage: int
    boss_action: str
    narrative: str
    phase: int
    phase_name: str
    phase_changed: bool
    phase_hint: str
    boss_defeated: bool
    player_won: bool
    inferi_alive: int = 0


class BossFight:
    """Stateful 3-phase scripted boss encounter."""

    def __init__(self):
        self._fights: Dict[str, BossState] = {}

    def start(self, session_id: str) -> Dict[str, Any]:
        """Begin the boss fight. Returns intro narrative + initial state."""
        state = BossState(session_id=session_id)
        self._fights[session_id] = state
        return {
            "boss_name": BOSS_NAME,
            "boss_hp": state.boss_hp,
            "boss_max_hp": state.boss_max_hp,
            "phase": 1,
            "phase_name": PHASE_THRESHOLDS[0]["name"],
            "phase_intro": PHASE_INTRO[1],
            "phase_hint": PHASE_HINT[1],
            "inferi_alive": 0,
        }

    def resolve_round(
        self,
        session_id: str,
        spell: str,
        player_hp: int,
        player_max_hp: int,
        active_effects: Optional[List[Dict]] = None,
    ) -> BossRoundResult:
        """Resolve one round of boss combat."""
        state = self._fights.get(session_id)
        if not state or not state.active:
            raise ValueError("No active boss fight for this session")

        active_effects = active_effects or []
        elder_wand_active = any(
            e.get("type") == "buff" and e.get("source") == "elder_wand_shard" and e.get("turns_remaining", 0) > 0
            for e in active_effects
        )

        state.rounds += 1

        # ── Player attacks the boss ──
        spell_damage = self._compute_damage(spell, state.phase, elder_wand_active)
        narrative_lines = [f"You cast {spell}."]
        if spell_damage <= 0:
            narrative_lines.append("It has no real effect — the Mage absorbs it without flinching.")
        else:
            state.boss_hp = max(0, state.boss_hp - spell_damage)
            narrative_lines.append(f"The spell strikes home for {spell_damage} damage.")

        # Apply stuns
        if spell == "Expelliarmus" and state.phase == 2:
            state.player_stunned_by = 2
            narrative_lines.append("The Mage is briefly disarmed and stunned!")

        # Phase 3 fragment exposure
        if state.phase == 3 and not state.fragment_exposed:
            state.fragment_exposed = True
            narrative_lines.append("The Horcrux fragment is now exposed and pulsing.")

        # Boss defeat
        if state.boss_hp <= 0:
            state.active = False
            return BossRoundResult(
                boss_hp_remaining=0, boss_max_hp=state.boss_max_hp,
                player_hp_remaining=player_hp, player_damage=0,
                boss_damage=spell_damage, boss_action="defeated",
                narrative="\n".join(narrative_lines) + "\n\n*The Hollow Mage collapses. The Horcrux fragment shatters into dust, releasing centuries of trapped malice into the air — and then nothing. Silence. The threat is ended.*",
                phase=state.phase, phase_name=PHASE_THRESHOLDS[state.phase-1]["name"],
                phase_changed=False, phase_hint="",
                boss_defeated=True, player_won=True,
                inferi_alive=state.inferi_alive,
            )

        # ── Phase transition check ──
        new_phase = 1
        for thresh in PHASE_THRESHOLDS:
            if state.boss_hp >= thresh["min_hp"]:
                new_phase = thresh["phase"]
                break
        phase_changed = False
        if new_phase != state.phase:
            state.phase = new_phase
            phase_changed = True
            narrative_lines.append(f"\n— PHASE {new_phase}: {PHASE_THRESHOLDS[new_phase-1]['name']} —")
            narrative_lines.append(PHASE_INTRO[new_phase])
            if new_phase == 2:
                state.inferi_alive = 2

        # ── Boss attacks back (unless stunned) ──
        boss_damage = 0
        boss_action = "stunned"
        if state.player_stunned_by > 0:
            state.player_stunned_by -= 1
            narrative_lines.append("The Mage is still stunned — they cannot retaliate this round.")
        else:
            attack_name, base_dmg, flavor = random.choice(MAGE_ATTACKS[state.phase])
            boss_damage = base_dmg + random.randint(-3, 5)
            # Reduce if player has shield buff
            shield_buff = any(e.get("type") == "buff" and e.get("stat") == "defense" for e in active_effects)
            if shield_buff:
                boss_damage = int(boss_damage * 0.7)
                narrative_lines.append("Your defenses absorb part of the impact.")
            boss_action = attack_name
            narrative_lines.append(f"\n{flavor} ({attack_name} — {boss_damage} damage.)")

        # ── Inferi adds (phase 2) ──
        if state.phase == 2:
            if state.rounds % 3 == 0:
                state.inferi_alive += 1
                narrative_lines.append("Another Inferius rises from the floor.")
            if state.inferi_alive > 0:
                add_damage = state.inferi_alive * 5
                boss_damage += add_damage
                narrative_lines.append(f"The {state.inferi_alive} Inferi claw at you for {add_damage} extra damage.")

        new_player_hp = max(0, player_hp - boss_damage)
        player_won = False
        boss_defeated = False

        # Player death check
        if new_player_hp <= 0:
            state.active = False
            narrative_lines.append("\n*The world goes white. You collapse. The Hollow Mage stands over you, victorious — for now.*")

        return BossRoundResult(
            boss_hp_remaining=state.boss_hp,
            boss_max_hp=state.boss_max_hp,
            player_hp_remaining=new_player_hp,
            player_damage=boss_damage,
            boss_damage=spell_damage,
            boss_action=boss_action,
            narrative="\n".join(narrative_lines),
            phase=state.phase,
            phase_name=PHASE_THRESHOLDS[state.phase-1]["name"],
            phase_changed=phase_changed,
            phase_hint=PHASE_HINT[state.phase] if phase_changed else "",
            boss_defeated=boss_defeated,
            player_won=player_won,
            inferi_alive=state.inferi_alive,
        )

    def _compute_damage(self, spell: str, phase: int, elder_wand: bool) -> int:
        """Spell effectiveness per phase. Phase 3 requires Fiendfyre or Elder Wand."""
        effective = EFFECTIVE_SPELLS.get(phase, {})
        if spell in effective:
            dmg = effective[spell] + random.randint(-8, 12)
            if elder_wand:
                dmg = int(dmg * 1.5)
            return max(1, dmg)
        # Phase 3 with Elder Wand: any spell works (reduced)
        if phase == 3 and elder_wand:
            return random.randint(15, 30)
        # Default: minor damage in phase 1, none in 2/3
        if phase == 1:
            return random.randint(3, 8)
        return 0

    def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        state = self._fights.get(session_id)
        if not state:
            return None
        return {
            "active": state.active,
            "boss_hp": state.boss_hp,
            "boss_max_hp": state.boss_max_hp,
            "phase": state.phase,
            "phase_name": PHASE_THRESHOLDS[state.phase-1]["name"],
            "rounds": state.rounds,
            "inferi_alive": state.inferi_alive,
        }

    def end(self, session_id: str) -> None:
        self._fights.pop(session_id, None)


boss_fight = BossFight()
