"""
Multi-turn NPC Dialogue Engine (Phase 10).

Enables structured, stateful conversations with any NPC.
The LLM generates the NPC's response and 3-4 topic options per turn.
Conversations can: reveal lore, trigger quests, change faction reputation,
gift items, and unlock secrets.

Usage in main.py:
    POST /api/dialogue/start  → DialogueStartResponse
    POST /api/dialogue/reply  → DialogueTurnResponse
    GET  /api/dialogue/{session_id}/{npc_id} → history
    POST /api/dialogue/end    → close conversation
"""
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

LLM_PROVIDER    = __import__("os").getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = __import__("os").getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = __import__("os").getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_API_KEY  = __import__("os").getenv("OPENAI_API_KEY", "")
OPENAI_MODEL    = __import__("os").getenv("OPENAI_MODEL", "gpt-4o")

# NPC personality prompts — keep the DM persona intact
_NPC_SYSTEM = """You are roleplaying a specific Harry Potter character in a text RPG.
Stay completely in character. Never break the fourth wall. Never mention being an AI.
Write the NPC's dialogue in their authentic voice as established in the HP universe.

Return ONLY a valid JSON object with this exact shape:
{{
  "npc_says": "What the NPC says (1-4 sentences, in-character, atmospheric)",
  "mood": "friendly|neutral|suspicious|worried|excited|guarded|amused",
  "topics": ["Topic option 1", "Topic option 2", "Topic option 3"],
  "quest_hint": null or "quest_id if this conversation reveals/advances a quest",
  "reputation_change": {{}},
  "lore_revealed": null or "one sentence of lore the NPC reveals",
  "item_offered": null or "item_id from the HP world if NPC offers an item"
}}

Topics should be natural conversation choices (3-4 options), always including one to end the conversation."""

_NPC_PERSONALITIES: Dict[str, str] = {
    "npc_001": "Madam Rosmerta: proprietor of the Three Broomsticks, shrewd, warm but guarded, knows everything that happens in Hogsmeade, protective of her regulars, uses understated British humour.",
    "npc_002": "Professor Neville Longbottom: humble, brave, genuinely fond of the player, tends to understate danger, references Herbology and his own past adventures, becomes quietly intense when serious.",
    "npc_003": "Headmistress Minerva McGonagall: precise, formal, deeply caring beneath stern exterior, references Hogwarts history and tradition, does not suffer fools but respects competence.",
    "npc_004": "Professor Filius Flitwick: enthusiastic, jovial, prone to excited digressions about charms theory, deeply kind, excellent at reading moods.",
    "npc_005": "Borgin: oily, carefully neutral, speaks in transaction terms, offers information only when it serves him, deeply knowledgeable about dark artifacts.",
    "npc_006": "A Knockturn Alley informant: shifty, speaks in fragments and implications, asks as many questions as they answer, wants something in return.",
    "npc_007": "Firenze the centaur: speaks in careful, prophetic cadences, references stars and planets, cryptic but genuinely helpful, does not see time linearly.",
    "npc_008": "Minister Kingsley Shacklebolt: calm, measured, carries authority without displaying it, speaks very directly, gives information in the exact amount needed.",
    "npc_009": "Unspeakable Vane: professionally evasive, every statement has a qualifier, speaks of knowledge as something to be managed not shared, unusually observant.",
    "npc_010": "The Hollow Mage: speaks rarely, each word precisely chosen, radiates cold certainty, references ancient magic and inevitable outcomes, never explains motives.",
}

_DEFAULT_PERSONALITY = "A wizarding world character: knowledgeable, guarded, authentic to the Harry Potter universe."


# ── Context & State ───────────────────────────────────────────────────────────

@dataclass
class DialogueContext:
    session_id: str
    npc_id: str
    npc_name: str
    player_name: str
    player_level: int
    player_house: str
    player_reputation: Dict[str, int]
    location_name: str
    npc_memories: List[str]
    active_quest_titles: List[str]
    key_beats: List[str]


@dataclass
class Turn:
    speaker: str          # "player" | "npc"
    content: str
    mood: str = "neutral"
    topics: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Conversation:
    session_id: str
    npc_id: str
    npc_name: str
    turns: List[Turn] = field(default_factory=list)
    reputation_changes: Dict[str, int] = field(default_factory=dict)
    lore_revealed: List[str] = field(default_factory=list)
    items_offered: List[str] = field(default_factory=list)
    quests_triggered: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    active: bool = True


# ── Engine ────────────────────────────────────────────────────────────────────

class DialogueEngine:
    """Stateful multi-turn NPC conversation manager."""

    def __init__(self):
        # session_id → {npc_id → Conversation}
        self._conversations: Dict[str, Dict[str, Conversation]] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    async def start(self, ctx: DialogueContext) -> Dict[str, Any]:
        """Open a new conversation with an NPC. Returns opening + topics."""
        conv = Conversation(
            session_id=ctx.session_id,
            npc_id=ctx.npc_id,
            npc_name=ctx.npc_name,
        )
        self._conversations.setdefault(ctx.session_id, {})[ctx.npc_id] = conv

        result = await self._generate_response(ctx, conv, player_says="[OPENING — NPC greets the player]")
        if result:
            self._apply_side_effects(conv, result)
            conv.turns.append(Turn(
                speaker="npc",
                content=result["npc_says"],
                mood=result.get("mood", "neutral"),
                topics=result.get("topics", self._default_topics()),
            ))
        else:
            # LLM unavailable — template greeting
            result = self._template_opening(ctx)
            conv.turns.append(Turn(
                speaker="npc",
                content=result["npc_says"],
                mood="neutral",
                topics=result["topics"],
            ))

        return {
            "npc_id": ctx.npc_id,
            "npc_name": ctx.npc_name,
            "npc_says": conv.turns[-1].content,
            "mood": conv.turns[-1].mood,
            "topics": conv.turns[-1].topics,
            "lore_revealed": result.get("lore_revealed"),
            "item_offered": result.get("item_offered"),
            "quest_hint": result.get("quest_hint"),
            "reputation_change": result.get("reputation_change", {}),
        }

    async def reply(
        self,
        session_id: str,
        npc_id: str,
        player_says: str,
        ctx: DialogueContext,
    ) -> Dict[str, Any]:
        """Player says something; NPC responds. Returns updated dialogue state."""
        conv = self._get_conversation(session_id, npc_id)
        if not conv or not conv.active:
            return await self.start(ctx)

        # Check for goodbye
        if any(kw in player_says.lower() for kw in ("goodbye", "farewell", "leave", "end conversation", "bye")):
            return self._end_conversation(conv)

        conv.turns.append(Turn(speaker="player", content=player_says))

        result = await self._generate_response(ctx, conv, player_says)
        if result:
            self._apply_side_effects(conv, result)
        else:
            result = self._template_response(ctx, player_says)

        conv.turns.append(Turn(
            speaker="npc",
            content=result["npc_says"],
            mood=result.get("mood", "neutral"),
            topics=result.get("topics", self._default_topics()),
        ))

        return {
            "npc_id": npc_id,
            "npc_name": ctx.npc_name,
            "npc_says": result["npc_says"],
            "mood": result.get("mood", "neutral"),
            "topics": result.get("topics", self._default_topics()),
            "lore_revealed": result.get("lore_revealed"),
            "item_offered": result.get("item_offered"),
            "quest_hint": result.get("quest_hint"),
            "reputation_change": result.get("reputation_change", {}),
            "conversation_active": conv.active,
        }

    def get_history(self, session_id: str, npc_id: str) -> List[Dict]:
        conv = self._get_conversation(session_id, npc_id)
        if not conv:
            return []
        return [
            {"speaker": t.speaker, "content": t.content,
             "mood": t.mood, "topics": t.topics, "timestamp": t.timestamp}
            for t in conv.turns
        ]

    def end_conversation(self, session_id: str, npc_id: str) -> None:
        conv = self._get_conversation(session_id, npc_id)
        if conv:
            conv.active = False

    def summary(self, session_id: str) -> Dict[str, Any]:
        convs = self._conversations.get(session_id, {})
        return {
            "active_conversations": [nid for nid, c in convs.items() if c.active],
            "total_npcs_spoken_to": len(convs),
            "total_turns": sum(len(c.turns) for c in convs.values()),
        }

    # ── LLM call ───────────────────────────────────────────────────────────

    async def _generate_response(
        self, ctx: DialogueContext, conv: Conversation, player_says: str
    ) -> Optional[Dict]:
        personality = _NPC_PERSONALITIES.get(ctx.npc_id, _DEFAULT_PERSONALITY)
        memories = "\n".join(f"- {m}" for m in ctx.npc_memories[:5]) or "No prior encounters."
        history_text = "\n".join(
            f"{t.speaker.upper()}: {t.content}"
            for t in conv.turns[-6:]
        )

        user_prompt = f"""CHARACTER: {ctx.npc_name}
PERSONALITY: {personality}

CONTEXT:
- Location: {ctx.location_name}
- Player: {ctx.player_name}, Level {ctx.player_level}, House {ctx.player_house}
- This NPC remembers: {memories}
- Active quests: {', '.join(ctx.active_quest_titles[:4]) or 'none'}
- Recent story: {'; '.join(ctx.key_beats[-3:]) or 'adventure just beginning'}

CONVERSATION SO FAR:
{history_text}

PLAYER JUST SAID: "{player_says}"

Now respond as {ctx.npc_name}. Return only the JSON object."""

        messages = [
            {"role": "system", "content": _NPC_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ]
        try:
            raw = await self._call_llm(messages)
            if raw:
                return self._parse_json(raw)
        except Exception as e:
            print(f"[Dialogue] LLM error: {e}")
        return None

    async def _call_llm(self, messages: List[Dict]) -> Optional[str]:
        if LLM_PROVIDER == "ollama":
            async with httpx.AsyncClient(timeout=45.0) as c:
                resp = await c.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={"model": OLLAMA_MODEL, "messages": messages, "stream": False,
                          "options": {"temperature": 0.85, "top_p": 0.9}},
                )
                return resp.json().get("message", {}).get("content", "")
        if LLM_PROVIDER == "openai":
            async with httpx.AsyncClient(timeout=45.0) as c:
                resp = await c.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={"model": OPENAI_MODEL, "messages": messages,
                          "temperature": 0.85, "max_tokens": 400},
                )
                return resp.json()["choices"][0]["message"]["content"]
        return None

    def _parse_json(self, raw: str) -> Optional[Dict]:
        raw = re.sub(r"```(?:json)?", "", raw).strip("` \n")
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None

    # ── Side-effects ───────────────────────────────────────────────────────

    def _apply_side_effects(self, conv: Conversation, result: Dict) -> None:
        if result.get("lore_revealed"):
            conv.lore_revealed.append(result["lore_revealed"])
        if result.get("item_offered"):
            conv.items_offered.append(result["item_offered"])
        if result.get("quest_hint"):
            conv.quests_triggered.append(result["quest_hint"])
        if result.get("reputation_change"):
            for faction, delta in result["reputation_change"].items():
                conv.reputation_changes[faction] = (
                    conv.reputation_changes.get(faction, 0) + int(delta)
                )

    # ── Fallback templates ─────────────────────────────────────────────────

    def _template_opening(self, ctx: DialogueContext) -> Dict:
        GREETINGS = {
            "npc_001": f"*Rosmerta looks up from the glass she's polishing.* 'Well then. Haven't seen you here in a while, {ctx.player_name}. What'll it be — a drink, or information? Both, usually, with people who look the way you do right now.'",
            "npc_002": "*Neville straightens up from his notes and extends a hand.* 'Good timing, actually. I've been meaning to talk to someone about this who isn't going to immediately try to write a report.'",
            "npc_003": "*McGonagall looks up over her glasses with the expression of someone who has been expecting you.* 'Sit down. I'll come straight to the point, as I'm sure you'd prefer.'",
            "npc_007": "*Firenze turns slowly, starlight in his eyes.* 'You come at the right time. The stars have been speaking of you.'",
        }
        greeting = GREETINGS.get(ctx.npc_id, f"*{ctx.npc_name} regards you with measured attention.* 'What brings you here?'")
        return {
            "npc_says": greeting,
            "mood": "neutral",
            "topics": self._default_topics(ctx.npc_id),
            "lore_revealed": None,
            "item_offered": None,
            "quest_hint": None,
            "reputation_change": {},
        }

    def _template_response(self, ctx: DialogueContext, player_says: str) -> Dict:
        return {
            "npc_says": f"*{ctx.npc_name} considers your words carefully.* 'That's worth thinking about. I'll tell you what I can, but some things aren't mine to share.'",
            "mood": "neutral",
            "topics": self._default_topics(ctx.npc_id),
            "lore_revealed": None,
            "item_offered": None,
            "quest_hint": None,
            "reputation_change": {},
        }

    def _default_topics(self, npc_id: str = "") -> List[str]:
        NPC_TOPICS = {
            "npc_001": ["Ask about the mysterious figure", "Ask about recent strange events", "Ask for information about the area", "Farewell"],
            "npc_002": ["Ask about the greenhouse disturbances", "Ask about Hogwarts defenses", "Offer to help investigate", "Farewell"],
            "npc_003": ["Ask about recent threats", "Discuss the Hollow Mage", "Ask for guidance", "Farewell"],
            "npc_007": ["Ask what the stars say", "Ask about the dark presence", "Ask about the future", "Farewell"],
        }
        return NPC_TOPICS.get(npc_id, ["Ask about recent events", "Ask for advice", "Share what you've discovered", "Farewell"])

    def _end_conversation(self, conv: Conversation) -> Dict:
        conv.active = False
        return {
            "npc_id": conv.npc_id,
            "npc_name": conv.npc_name,
            "npc_says": "The conversation draws to a close.",
            "mood": "neutral",
            "topics": [],
            "conversation_active": False,
            "reputation_change": dict(conv.reputation_changes),
            "lore_revealed": None,
            "item_offered": None,
            "quest_hint": None,
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_conversation(self, session_id: str, npc_id: str) -> Optional[Conversation]:
        return self._conversations.get(session_id, {}).get(npc_id)


# Global instance
dialogue_engine = DialogueEngine()
