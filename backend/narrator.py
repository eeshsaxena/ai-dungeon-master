"""
LLM Narrator Agent for AI Dungeon Master.
Connects to Ollama (local) or OpenAI.
Maintains the Harry Potter Dungeon Master persona.
"""
import os
import json
import httpx
import random
from typing import List, Dict, Optional, AsyncGenerator
from models.schemas import (
    NarrateRequest, NarrateResponse, EmotionState, DifficultyLevel, PlayerHouse
)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Dungeon Master for an immersive Harry Potter RPG adventure game set in the Wizarding World. You narrate the story with vivid, atmospheric descriptions in the style of J.K. Rowling's writing.

WORLD CONTEXT:
- Setting: Post-Voldemort era, a new dark threat called the Hollow Mage is rising
- The player is a witch or wizard investigating dark disturbances
- Key locations: Three Broomsticks (Hogsmeade), Hogwarts, Knockturn Alley, Forbidden Forest, Ministry of Magic, Azkaban

YOUR ROLE:
- Narrate in second person ("You see...", "You feel...")
- Be descriptive: smells, sounds, textures, magical atmosphere
- Reference canon HP lore naturally (Butterbeer, Quidditch, house points, spells)
- Present meaningful choices but don't list them mechanically
- Track player choices and remember them
- Vary tone: wonder for Hogwarts, dread for Knockturn Alley, melancholy for Godric's Hollow
- Keep responses between 100-200 words unless asked for more
- End each response with 1-2 subtle hooks or choices (not a numbered list)
- Occasionally have NPCs reference past player actions
- Use [SPELL_CAST: SpellName] markup when the player casts a spell
- Use [LOCATION_CHANGE: loc_id] markup when the player moves
- Use [COMBAT_START: EnemyType] markup when combat begins

TONE GUIDELINES BY LOCATION:
- Three Broomsticks: warm, gossipy, safe but full of secrets
- Hogwarts: majestic, alive, full of history and magic
- Knockturn Alley: sinister, suspicious, dangerous
- Forbidden Forest: primordial, vast, deeply magical
- Ministry of Magic: bureaucratic, tense, political
- Azkaban: cold, despairing, oppressive

Never break character. Never mention being an AI. If asked about rules or mechanics, answer as the Dungeon Master ("The ancient magic here responds to...").
"""

DIFFICULTY_MODIFIERS = {
    DifficultyLevel.EASY: "\n[DIFFICULTY: Easy — be forgiving, offer more hints, enemies are weaker]",
    DifficultyLevel.MEDIUM: "",
    DifficultyLevel.HARD: "\n[DIFFICULTY: Hard — be less forgiving, more consequence to actions]",
    DifficultyLevel.VERY_HARD: "\n[DIFFICULTY: Very Hard — high stakes, every choice matters greatly]",
    DifficultyLevel.LEGENDARY: "\n[DIFFICULTY: Legendary — maximum peril, this is the final confrontation]"
}

EMOTION_MODIFIERS = {
    EmotionState.FRUSTRATED: "\n[PLAYER STATE: Frustrated — ease difficulty slightly, offer clearer options]",
    EmotionState.BORED: "\n[PLAYER STATE: Bored — inject unexpected drama, a surprise encounter or discovery]",
    EmotionState.CONFUSED: "\n[PLAYER STATE: Confused — clarify situation, offer clearer guidance from an NPC]",
    EmotionState.EXCITED: "\n[PLAYER STATE: Excited — match their energy, raise the stakes]",
    EmotionState.FEARFUL: "\n[PLAYER STATE: Fearful — heighten tension but give a path forward]",
}

# ── Mock responses (when no LLM available) ────────────────────────────────────

MOCK_RESPONSES = [
    {
        "narrative": "The Three Broomsticks hums with warmth and chatter as you push through the heavy oak door. Floating candles cast amber pools of light over worn wooden tables, and the rich smell of Butterbeer mingles with pipe smoke. Madam Rosmerta catches your eye from behind the bar — she has something in her expression you've seen before: worry masked as welcome.\n\n*'I've been expecting someone like you,'* she says quietly, sliding a Butterbeer across the counter without you asking. *'Word travels fast in Hogsmeade. Something's not right in that forest.'*\n\nProfessor Longbottom sits alone in the corner booth, his hands wrapped around a mug that's gone cold. He looks up when you approach — relief crossing his face.",
        "atmosphere": "warm",
        "suggested_actions": ["Sit with Professor Longbottom", "Ask Rosmerta what she's heard", "Look around the pub for other familiar faces", "Order a Butterbeer and listen to conversations"],
        "scene_prompt": "warm cozy wizarding pub interior, Three Broomsticks, floating candles, Butterbeer, dark wood, magical atmosphere"
    },
    {
        "narrative": "Professor Longbottom leans forward, keeping his voice low enough that only you can hear over the pub's noise.\n\n*'The greenhouses,'* he says, jaw tight. *'Three nights in a row, every Mandrake has wilted overnight. Not natural wilting — they've been... drained. Of magic. Something is feeding on magical flora, and it's getting bolder. Last night the wards around Greenhouse Seven flickered.'*\n\nHe presses a folded parchment into your hand. A hand-drawn map of the Hogwarts grounds, with three locations circled in red.\n\n*'McGonagall doesn't want to alarm the students. But she agreed we need someone to look into this quietly.'* His eyes hold yours steadily. *'I thought of you.'*",
        "atmosphere": "mysterious",
        "suggested_actions": ["Accept the mission and examine the map", "Ask Neville what he thinks is causing this", "Suggest informing the Ministry", "Ask if there have been any other strange incidents"],
        "scene_prompt": "two wizards in a dark pub booth, secret conversation, candlelight, parchment map, concerned expressions"
    },
    {
        "narrative": "The Hogwarts greenhouses loom ahead in the pre-dawn grey, glass panels fogged from within. The grounds are utterly silent — no owls, no wind. Even the ghosts seem to be absent.\n\nYou push open the door to Greenhouse Seven. The smell hits you first: rot, and beneath it something else — a cold metallic scent, like lightning and old bones. Every plant on the left row has turned completely black. The soil around them is crystalline, as if frozen, though the air isn't cold.\n\nThen you see it. On the far wall, scratched into the stone in a script you almost recognize — almost — are three symbols. They pulse with a dim, sickly light that fades even as you watch.",
        "atmosphere": "dangerous",
        "suggested_actions": ["Cast Lumos and examine the symbols closely", "Take a clipping of the blackened plants for study", "Search the rest of the greenhouse", "Try to replicate the symbols on parchment"],
        "scene_prompt": "dark greenhouse at dawn, wilted magical plants turned black, glowing mysterious symbols on stone wall, eerie atmosphere, Harry Potter world"
    }
]


class NarratorAgent:
    """
    The LLM-powered Dungeon Master.
    Connects to Ollama (local) or OpenAI API.
    Falls back to mock responses when neither is available.
    """

    def __init__(self):
        self.provider = LLM_PROVIDER
        self._mock_index = 0

    async def narrate(
        self,
        request: NarrateRequest,
        story_history: List[Dict[str, str]],
        world_context: str,
        emotion: EmotionState = EmotionState.NEUTRAL
    ) -> NarrateResponse:
        """Generate narrative response for a player action."""

        # Build system prompt with context
        system = SYSTEM_PROMPT
        system += DIFFICULTY_MODIFIERS.get(request.difficulty, "")
        system += EMOTION_MODIFIERS.get(emotion, "")
        if world_context:
            system += f"\n\n[WORLD STATE]\n{world_context}"

        # Add player stats context
        stats = request.player_stats
        system += (
            f"\n\n[PLAYER STATUS]\n"
            f"Name: {stats.name} | House: {stats.house} | HP: {stats.hp}/{stats.max_hp} | "
            f"Mana: {stats.mana}/{stats.max_mana} | Galleons: {stats.galleons} | "
            f"Level: {stats.level} | Location: {request.current_location}"
        )

        messages = [{"role": "system", "content": system}]
        messages.extend(story_history[-20:])  # Last 20 turns
        messages.append({"role": "user", "content": request.player_input})

        try:
            if self.provider == "ollama":
                raw = await self._call_ollama(messages)
            elif self.provider == "openai":
                raw = await self._call_openai(messages)
            else:
                raw = None

            if raw:
                return self._parse_response(raw, request.current_location)

        except Exception as e:
            print(f"[Narrator] LLM call failed: {e}")

        # Fallback: contextual mock response
        return self._get_mock_response(request)

    async def _call_ollama(self, messages: List[Dict]) -> Optional[str]:
        """Call local Ollama API."""
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.85,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload
            )
            data = response.json()
            return data.get("message", {}).get("content", "")

    async def _call_openai(self, messages: List[Dict]) -> Optional[str]:
        """Call OpenAI API."""
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "temperature": 0.85,
            "max_tokens": 400
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            data = response.json()
            return data["choices"][0]["message"]["content"]

    def _parse_response(self, raw: str, location_id: str) -> NarrateResponse:
        """Parse raw LLM response into structured NarrateResponse."""
        from image_gen import build_scene_prompt

        # Extract markup tags from response
        world_updates = {}
        import re

        # Check for location change
        loc_match = re.search(r'\[LOCATION_CHANGE: (\w+)\]', raw)
        if loc_match:
            world_updates["new_location"] = loc_match.group(1)
            raw = raw.replace(loc_match.group(0), "").strip()

        # Check for combat start
        combat_match = re.search(r'\[COMBAT_START: (\w+)\]', raw)
        if combat_match:
            world_updates["combat_start"] = combat_match.group(1)
            raw = raw.replace(combat_match.group(0), "").strip()

        # Check for spell cast
        spell_match = re.search(r'\[SPELL_CAST: ([^\]]+)\]', raw)
        if spell_match:
            world_updates["spell_cast"] = spell_match.group(1)
            raw = raw.replace(spell_match.group(0), "").strip()

        # Extract suggested actions from last paragraph if present
        lines = raw.strip().split('\n')
        suggested = []
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                suggested.append(line[2:])
            elif line:
                clean_lines.append(line)

        narrative = '\n\n'.join(clean_lines)
        if not suggested:
            suggested = ["Continue exploring", "Investigate further", "Talk to someone nearby", "Check your surroundings"]

        scene_prompt = build_scene_prompt(location_id, narrative[:100])

        return NarrateResponse(
            narrative=narrative,
            atmosphere=self._detect_atmosphere(narrative),
            suggested_actions=suggested[:4],
            world_state_updates=world_updates,
            scene_prompt=scene_prompt,
            detected_emotion=EmotionState.NEUTRAL
        )

    def _detect_atmosphere(self, text: str) -> str:
        """Simple atmosphere detection from narrative text."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["dark", "shadow", "danger", "threat", "evil"]):
            return "dangerous"
        elif any(w in text_lower for w in ["warm", "cozy", "friendly", "safe"]):
            return "warm"
        elif any(w in text_lower for w in ["mysterious", "strange", "unusual", "puzzle"]):
            return "mysterious"
        elif any(w in text_lower for w in ["ancient", "old", "forgotten", "ruins"]):
            return "ancient"
        return "magical"

    def _get_mock_response(self, request: NarrateRequest) -> NarrateResponse:
        """Return a contextual mock response (no LLM needed)."""
        from image_gen import build_scene_prompt

        mock = MOCK_RESPONSES[self._mock_index % len(MOCK_RESPONSES)]
        self._mock_index += 1

        return NarrateResponse(
            narrative=mock["narrative"],
            atmosphere=mock["atmosphere"],
            suggested_actions=mock["suggested_actions"],
            world_state_updates={},
            scene_prompt=build_scene_prompt(request.current_location, mock["narrative"][:100]),
            detected_emotion=EmotionState.NEUTRAL
        )


# Global narrator instance
narrator = NarratorAgent()
