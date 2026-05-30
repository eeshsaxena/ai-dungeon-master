"""Pydantic schemas for AI Dungeon Master API."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class PlayerHouse(str, Enum):
    GRYFFINDOR = "Gryffindor"
    HUFFLEPUFF = "Hufflepuff"
    RAVENCLAW = "Ravenclaw"
    SLYTHERIN = "Slytherin"
    UNSELECTED = "Unselected"


class EmotionState(str, Enum):
    EXCITED = "excited"
    CURIOUS = "curious"
    FRUSTRATED = "frustrated"
    BORED = "bored"
    CONFUSED = "confused"
    FEARFUL = "fearful"
    NEUTRAL = "neutral"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"
    LEGENDARY = "legendary"


class PlayerStats(BaseModel):
    name: str = "Unnamed Witch/Wizard"
    house: PlayerHouse = PlayerHouse.UNSELECTED
    level: int = 1
    xp: int = 0
    xp_to_next: int = 300  # XP needed to reach next level
    title: str = "First-Year"  # Wizard title for current level
    hp: int = 100
    max_hp: int = 100
    mana: int = 80
    max_mana: int = 80
    galleons: int = 50
    spells_known: List[str] = Field(default_factory=lambda: ["Expelliarmus", "Stupefy", "Protego", "Lumos", "Accio"])
    inventory: List[Dict[str, Any]] = Field(default_factory=list)
    active_quests: List[str] = Field(default_factory=lambda: ["quest_001"])
    completed_quests: List[str] = Field(default_factory=list)
    reputation: Dict[str, int] = Field(default_factory=lambda: {
        "Ministry of Magic": 0,
        "Order of the Phoenix": 10,
        "Hollow Circle": -100,
        "Centaur Herd": 0
    })
    current_location: str = "loc_001"
    emotion_state: EmotionState = EmotionState.NEUTRAL
    turns_played: int = 0
    active_effects: List[Dict[str, Any]] = Field(default_factory=list)  # timed buffs/debuffs


class NarrateRequest(BaseModel):
    session_id: str
    player_input: str
    player_stats: PlayerStats
    current_location: str
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM


class NarrateResponse(BaseModel):
    narrative: str
    atmosphere: str
    suggested_actions: List[str]
    world_state_updates: Dict[str, Any] = Field(default_factory=dict)
    scene_prompt: str  # Prompt for image generation
    detected_emotion: EmotionState = EmotionState.NEUTRAL
    difficulty_adjustment: Optional[str] = None
    new_quest: Optional[Dict[str, Any]] = None  # dynamically generated quest, if any
    encounter: Optional[Dict[str, Any]] = None  # Phase 11: random encounter, if rolled
    time_of_day: Optional[Dict[str, Any]] = None  # Phase 11: in-game time
    achievements_unlocked: List[Dict[str, Any]] = Field(default_factory=list)


class CombatAction(str, Enum):
    ATTACK = "attack"
    DEFEND = "defend"
    FLEE = "flee"
    SPECIAL = "special"
    TAUNT = "taunt"


class EnemyArchetype(str, Enum):
    DEMENTOR = "Dementor"
    DEATH_EATER = "Death Eater"
    ACROMANTULA = "Acromantula"
    TROLL = "Troll"
    WEREWOLF = "Werewolf"
    DARK_WIZARD = "Dark Wizard"
    HOLLOW_MAGE = "Hollow Mage"
    BOGGART = "Boggart"
    INFERI = "Inferi"


class Enemy(BaseModel):
    id: str
    name: str
    archetype: EnemyArchetype
    hp: int
    max_hp: int
    attack: int
    defense: int
    special_ability: str
    weakness: Optional[str] = None
    personality: str = "aggressive"  # aggressive, defensive, cunning, erratic


class CombatRequest(BaseModel):
    session_id: str
    enemy: Enemy
    player_stats: PlayerStats
    player_action: str
    player_spell: Optional[str] = None
    round_number: int = 1
    combat_history: List[Dict[str, str]] = Field(default_factory=list)


class CombatResponse(BaseModel):
    enemy_action: CombatAction
    enemy_spell: Optional[str]
    narrative: str
    player_damage: int
    enemy_damage: int
    enemy_hp_remaining: int
    player_hp_remaining: int
    combat_over: bool
    player_won: Optional[bool] = None
    loot: List[Dict[str, Any]] = Field(default_factory=list)
    xp_gained: int = 0
    # Level-up info — populated when the player levels up this combat
    level_up: bool = False
    new_level: Optional[int] = None
    new_title: Optional[str] = None
    new_spells: List[str] = Field(default_factory=list)
    updated_player: Optional[Dict] = None  # full updated player stats if level_up


class EmotionRequest(BaseModel):
    session_id: str
    recent_inputs: List[str]
    turns_without_progress: int = 0
    time_between_inputs_seconds: float = 30.0


class EmotionResponse(BaseModel):
    emotion: EmotionState
    confidence: float
    difficulty_recommendation: DifficultyLevel
    hint_triggered: bool
    hint_text: Optional[str] = None


class WorldStateNode(BaseModel):
    id: str
    label: str
    type: str  # location, npc, quest, item, faction
    properties: Dict[str, Any] = Field(default_factory=dict)


class WorldStateEdge(BaseModel):
    source: str
    target: str
    relationship: str


class WorldStateResponse(BaseModel):
    nodes: List[WorldStateNode]
    edges: List[WorldStateEdge]
    player_location: str
    active_quests: List[str]


class SessionState(BaseModel):
    session_id: str
    player: PlayerStats
    story_history: List[Dict[str, str]] = Field(default_factory=list)
    world_events: List[str] = Field(default_factory=list)
    current_difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    in_combat: bool = False
    current_enemy: Optional[Enemy] = None
