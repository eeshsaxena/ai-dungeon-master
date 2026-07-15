"""
Dynamic Quest Generator for AI Dungeon Master (Phase 7).

Generates contextual side quests on-the-fly using the LLM, grounded in the
player's current location, active NPCs, story memory, and progression level.
Falls back to a curated template library when the LLM is unavailable so the
feature always works — even in offline/mock mode.

Trigger logic (evaluated in main.py):
  - First time the player visits a location this session.
  - Every QUEST_COOLDOWN turns while in the same location.
  - Only when fewer than MAX_ACTIVE_DYNAMIC quests are currently active.
  - At most QUEST_COOLDOWN turns between any two generations globally.
"""
import json
import re
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import httpx

LLM_PROVIDER    = __import__("os").getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = __import__("os").getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = __import__("os").getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_API_KEY  = __import__("os").getenv("OPENAI_API_KEY", "")
OPENAI_MODEL    = __import__("os").getenv("OPENAI_MODEL", "gpt-4o")

QUEST_COOLDOWN       = 5    # minimum turns between generations
MAX_ACTIVE_DYNAMIC   = 3    # max simultaneous generated quests per session


# ── Context ───────────────────────────────────────────────────────────────────

@dataclass
class QuestContext:
    session_id: str
    player_name: str
    player_level: int
    player_house: str
    player_reputation: Dict[str, int]
    location_id: str
    location_name: str
    npcs_here: List[str]          # NPC names present
    key_beats: List[str]          # recent story beats from memory
    existing_quest_titles: List[str]
    turn_number: int


# ── Templates (fallback when LLM unavailable) ─────────────────────────────────

_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "loc_001": [  # Three Broomsticks
        {
            "title": "The Hissing Package",
            "description": "A small, cloth-wrapped package sits abandoned on a stool at the far end of the bar. It has been there since last night — and it's hissing. Madam Rosmerta keeps glancing at it but won't touch it.",
            "objectives": [{"text": "Examine the hissing package safely", "completed": False},
                           {"text": "Identify what is inside", "completed": False},
                           {"text": "Decide what to do with it", "completed": False}],
            "giver": "Madam Rosmerta",
            "difficulty": "easy",
            "rewards": {"xp": 150, "galleons": 30},
            "hook": "A low, rhythmic hissing comes from the end of the bar — something wrapped in cloth that shouldn't be moving is.",
        },
        {
            "title": "Ink on the Mirror",
            "description": "Someone has written a message on the mirror above the bar in what looks like invisible ink — only you can see it. The message reads: 'Meet me where the Butterbeer goes cold. Come alone.' No signature.",
            "objectives": [{"text": "Decode when and where the meeting is", "completed": False},
                           {"text": "Attend the clandestine meeting", "completed": False},
                           {"text": "Decide whether to trust the contact", "completed": False}],
            "giver": "Unknown contact",
            "difficulty": "medium",
            "rewards": {"xp": 200, "galleons": 0},
            "hook": "You catch a shimmer in the bar mirror — words forming just for your eyes.",
        },
        {
            "title": "Overheard at Last Orders",
            "description": "Two cloaked figures at a corner table are arguing in low voices about a map — something hidden beneath one of the Hogwarts staircases. They don't notice you listening.",
            "objectives": [{"text": "Learn more without being detected", "completed": False},
                           {"text": "Investigate the staircase they mentioned", "completed": False},
                           {"text": "Recover or secure whatever is hidden there", "completed": False}],
            "giver": "the environment",
            "difficulty": "medium",
            "rewards": {"xp": 250, "galleons": 50},
            "hook": "Two figures argue in hushed, urgent tones — and you catch the words 'beneath the seventh stair.'",
        },
    ],
    "loc_002": [  # Hogwarts
        {
            "title": "The Rearranged Stairs",
            "description": "Three staircases have locked into a new configuration overnight, forming a path that leads to a corridor not shown on any map. The castle, it seems, is trying to show you something.",
            "objectives": [{"text": "Follow the staircase path before it changes again", "completed": False},
                           {"text": "Find what lies at the end of the hidden corridor", "completed": False},
                           {"text": "Document the discovery for Hogwarts records", "completed": False}],
            "giver": "the castle itself",
            "difficulty": "medium",
            "rewards": {"xp": 280, "galleons": 20},
            "hook": "The grand staircase clicks and groans, locking into a shape you have never seen — pointing somewhere new.",
        },
        {
            "title": "A House Elf's Plea",
            "description": "A terrified house elf named Trix materializes beside you, wringing its hands. Something has been taken from the kitchens — not food, but a small vial that 'smells of very old magic.' It must be found before the next meal service.",
            "objectives": [{"text": "Hear Trix's full account of what was stolen", "completed": False},
                           {"text": "Search the most likely hiding spots", "completed": False},
                           {"text": "Return the vial to the kitchens", "completed": False}],
            "giver": "Trix the house elf",
            "difficulty": "easy",
            "rewards": {"xp": 120, "galleons": 0},
            "hook": "A small hand tugs your robe — a house elf's huge eyes are full of tears.",
        },
        {
            "title": "Empty Classroom, Full Shelves",
            "description": "Classroom 3C has been locked for a month — but the light under the door is wrong. Inside, all the desks are pushed aside and the shelves are full of books that don't belong to this classroom. Someone has been using this room, and recently.",
            "objectives": [{"text": "Enter the locked classroom without triggering alarms", "completed": False},
                           {"text": "Catalogue the unusual books and trace their origin", "completed": False},
                           {"text": "Determine who has been using the room and why", "completed": False}],
            "giver": "the environment",
            "difficulty": "medium",
            "rewards": {"xp": 220, "galleons": 10},
            "hook": "A warm yellow light seeps under the door of a classroom that should be empty.",
        },
    ],
    "loc_003": [  # Knockturn Alley
        {
            "title": "Borgin's New Acquisition",
            "description": "Something new sits in Borgin and Burkes' window — a small hand mirror that reflects not what stands before it, but what stood there a century ago. Borgin claims he doesn't know where it came from.",
            "objectives": [{"text": "Examine the mirror and its properties", "completed": False},
                           {"text": "Trace how it arrived at the shop", "completed": False},
                           {"text": "Decide whether to acquire, secure, or destroy it", "completed": False}],
            "giver": "the environment",
            "difficulty": "medium",
            "rewards": {"xp": 300, "galleons": 0},
            "hook": "A mirror in the shop window shows not your reflection — but someone else's.",
        },
        {
            "title": "The Marked Letter",
            "description": "An unsealed letter bearing a faint Dark Mark lies open on the cobblestones. It's addressed to no one, and what's written inside is partly in code — but the last line is chillingly clear.",
            "objectives": [{"text": "Read and memorize the letter's contents", "completed": False},
                           {"text": "Decode the encrypted portion", "completed": False},
                           {"text": "Report to the appropriate authority — or use the information yourself", "completed": False}],
            "giver": "the environment",
            "difficulty": "easy",
            "rewards": {"xp": 180, "galleons": 0},
            "hook": "Something pale on the wet cobblestones catches your eye — a letter, face-up, unsealed.",
        },
        {
            "title": "Eyes in the Dark",
            "description": "You've been followed since you entered the alley. Whoever it is keeps to the shadows, but three times you've caught a glimpse: short, fast, and wearing Ministry-issue boots. An unsanctioned tail.",
            "objectives": [{"text": "Confirm you are being followed", "completed": False},
                           {"text": "Lose or confront your shadow", "completed": False},
                           {"text": "Discover who sent them and why", "completed": False}],
            "giver": "the environment",
            "difficulty": "medium",
            "rewards": {"xp": 240, "galleons": 30},
            "hook": "You hear footsteps stop exactly when yours do. Someone very professional is behind you.",
        },
    ],
    "loc_004": [  # Forbidden Forest
        {
            "title": "The Wounded Thestral",
            "description": "A thestral is lying just off the path, unable to rise. Its wing is caught in an enchanted snare — not a mundane trap, but something woven with deliberate dark magic. The rest of the herd watches from the treeline.",
            "objectives": [{"text": "Approach the thestral without startling the herd", "completed": False},
                           {"text": "Identify and safely dismantle the enchanted snare", "completed": False},
                           {"text": "Ensure the thestral can rejoin the herd", "completed": False}],
            "giver": "the environment",
            "difficulty": "easy",
            "rewards": {"xp": 160, "galleons": 0},
            "hook": "A sound that shouldn't come from a horse — a low, pained sound — reaches you from just off the path.",
        },
        {
            "title": "Where the Path Was",
            "description": "A path you've taken before has vanished entirely — the trees have closed over it completely. This doesn't happen on its own. Something in the forest has deliberately sealed this route, and the only way forward is to find out why.",
            "objectives": [{"text": "Determine when and how the path disappeared", "completed": False},
                           {"text": "Find an alternative route or force the path open", "completed": False},
                           {"text": "Discover what is being hidden or protected", "completed": False}],
            "giver": "the environment",
            "difficulty": "medium",
            "rewards": {"xp": 200, "galleons": 0},
            "hook": "You stop short. The forest has sealed itself — where the path was, there is now only old, dense wood.",
        },
        {
            "title": "The Spider's Message",
            "description": "A large acromantula — not aggressive, which is remarkable — blocks the path and drops a scroll of silk-webbing at your feet. The centaurs say the spiders have no written language. Someone taught this one to carry messages.",
            "objectives": [{"text": "Decipher the webbed message", "completed": False},
                           {"text": "Follow the instruction or refuse it", "completed": False},
                           {"text": "Determine who taught the acromantula to communicate this way", "completed": False}],
            "giver": "the acromantula colony",
            "difficulty": "hard",
            "rewards": {"xp": 400, "galleons": 0},
            "hook": "An acromantula — far too large to outrun — steps into the path, drops something, and waits.",
        },
    ],
    "loc_005": [  # Ministry of Magic
        {
            "title": "Department Twenty-Seven",
            "description": "Every floor directory lists twenty-six departments. But one lift button — unmarked, slightly warm to the touch — leads to a floor that isn't on any map. The lift opens. Something is down there.",
            "objectives": [{"text": "Investigate Department Twenty-Seven discreetly", "completed": False},
                           {"text": "Document what you find without being caught", "completed": False},
                           {"text": "Determine who knows about this department and why it's hidden", "completed": False}],
            "giver": "the environment",
            "difficulty": "medium",
            "rewards": {"xp": 320, "galleons": 50},
            "hook": "The lift has one more button than the directory says it should.",
        },
        {
            "title": "The Circling Memo",
            "description": "An enchanted paper aeroplane memo has been circling the Atrium for three days. It's addressed to no one, and every owl sent to intercept it has returned with a burnt tail. It carries a name — a name no one at the Ministry will say aloud.",
            "objectives": [{"text": "Intercept the enchanted memo", "completed": False},
                           {"text": "Read its contents safely", "completed": False},
                           {"text": "Decide what to do with the name on the memo", "completed": False}],
            "giver": "the environment",
            "difficulty": "easy",
            "rewards": {"xp": 140, "galleons": 0},
            "hook": "A paper aeroplane keeps circling near the golden statue, refusing to land.",
        },
        {
            "title": "The Classified File",
            "description": "A locked cabinet in a records corridor has sprung open — not from magic, but from the inside. Inside: a single file, stamped CLASSIFIED in three colors, referencing an operation that officially ended in 1998. It didn't.",
            "objectives": [{"text": "Secure the file before someone else finds it", "completed": False},
                           {"text": "Decode the operational details within", "completed": False},
                           {"text": "Find out who opened the cabinet — and why now", "completed": False}],
            "giver": "the environment",
            "difficulty": "hard",
            "rewards": {"xp": 380, "galleons": 0},
            "hook": "A locked cabinet in the records corridor is standing open. The lock is intact — it opened from the inside.",
        },
    ],
    "loc_006": [  # Azkaban
        {
            "title": "The Seventh Cell",
            "description": "The official manifest says Cell 7, Block C is empty. The guard's log says it has been empty for a year. But the food slot has been used from the inside — recently.",
            "objectives": [{"text": "Access Block C without triggering the alarm wards", "completed": False},
                           {"text": "Determine who or what is in Cell 7", "completed": False},
                           {"text": "Decide whether to report it, free them, or leave them", "completed": False}],
            "giver": "the environment",
            "difficulty": "hard",
            "rewards": {"xp": 500, "galleons": 0},
            "hook": "The manifest says empty. The food slot says otherwise.",
        },
        {
            "title": "Scratched Into Stone",
            "description": "Someone has scratched symbols into the wall of a cell — thousands of them, covering every surface. They're not random. Arranged correctly, they form a sequence of coordinates. But the prisoner who scratched them has been dead for six years.",
            "objectives": [{"text": "Map and photograph the full symbol array", "completed": False},
                           {"text": "Decode the coordinate sequence", "completed": False},
                           {"text": "Investigate what the coordinates point to", "completed": False}],
            "giver": "the environment",
            "difficulty": "medium",
            "rewards": {"xp": 350, "galleons": 0},
            "hook": "Every surface of one cell is covered in tiny, methodical scratches — a message from the dead.",
        },
    ],
    "loc_007": [  # Godric's Hollow
        {
            "title": "Fresh Flowers",
            "description": "Lilies on the Potters' grave — fresh, still dewy. No one in the village remembers placing them. The local magical signature detection spell is silent: whoever left them knew how to leave no trace.",
            "objectives": [{"text": "Determine when the flowers were placed", "completed": False},
                           {"text": "Find who visits this grave without leaving a trace", "completed": False},
                           {"text": "Follow up on what you discover about the visitor", "completed": False}],
            "giver": "the environment",
            "difficulty": "easy",
            "rewards": {"xp": 140, "galleons": 0},
            "hook": "Fresh lilies on a grave that no one in the village claims to have visited.",
        },
        {
            "title": "The Wall's Message",
            "description": "On the outer wall of the ruined Potter cottage — beneath decades of tourists' signatures — someone has scratched a message in a language that predates modern English. It was not there last month.",
            "objectives": [{"text": "Translate the ancient message", "completed": False},
                           {"text": "Understand what it refers to", "completed": False},
                           {"text": "Determine who could have left it and why here", "completed": False}],
            "giver": "the environment",
            "difficulty": "medium",
            "rewards": {"xp": 220, "galleons": 0},
            "hook": "Beneath all the ink and signatures, something older has been scratched into the stone — recently.",
        },
    ],
}

# XP scaling by difficulty
_DIFF_XP = {"easy": (100, 200), "medium": (200, 400), "hard": (350, 600)}

# ── LLM prompt ────────────────────────────────────────────────────────────────

_QUEST_PROMPT = """You are a quest writer for an immersive Harry Potter RPG set in the post-Voldemort wizarding world.

PLAYER: {player_name}, Level {level}, House {house}
CURRENT LOCATION: {location_name}
NPCS PRESENT: {npcs}
RECENT STORY EVENTS: {story_beats}
EXISTING QUEST TITLES (do NOT duplicate): {existing_titles}

Generate ONE new side quest for this specific location and context. The quest must:
- Feel authentically Harry Potter — specific spells, objects, creatures, places
- Be completable in 3–5 player actions
- Have 2–4 concrete objectives (not vague)
- Match the {location_name} atmosphere
- Be appropriate for Level {level} (difficulty: {difficulty})
- NOT duplicate any existing quest title

Respond with ONLY a single valid JSON object — no markdown, no commentary:
{{
  "title": "3-6 word evocative title",
  "description": "2-3 sentence scene-setting description in past-tense narrative style",
  "objectives": [
    {{"text": "specific action objective", "completed": false}},
    {{"text": "specific action objective", "completed": false}},
    {{"text": "specific action objective", "completed": false}}
  ],
  "giver": "NPC name or 'the environment'",
  "difficulty": "{difficulty}",
  "rewards": {{"xp": {xp}, "galleons": {galleons}}},
  "hook": "one vivid atmospheric sentence that makes the player notice this quest"
}}"""


# ── Generator ─────────────────────────────────────────────────────────────────

class DynamicQuestGenerator:
    """Generates side quests via LLM (+ template fallback) based on story context."""

    def __init__(self):
        # session_id -> set of location_ids where we already generated a quest
        self._visited: Dict[str, set] = {}
        # session_id -> last turn number a quest was generated
        self._last_turn: Dict[str, int] = {}
        # session_id -> list of generated quest dicts
        self._quests: Dict[str, List[Dict]] = {}

    # ── public ────────────────────────────────────────────────────────────────

    def should_generate(self, ctx: QuestContext) -> bool:
        """Return True when conditions are right for a new quest."""
        sid = ctx.session_id
        active = self._count_active(sid)
        if active >= MAX_ACTIVE_DYNAMIC:
            return False
        last = self._last_turn.get(sid, -QUEST_COOLDOWN)
        if ctx.turn_number - last < QUEST_COOLDOWN:
            return False
        return True

    async def generate(self, ctx: QuestContext) -> Optional[Dict]:
        """Generate a quest. Returns the quest dict or None on failure."""
        difficulty = self._pick_difficulty(ctx.player_level)
        xp, galleons = self._pick_rewards(difficulty)

        quest_dict = await self._try_llm(ctx, difficulty, xp, galleons)
        if quest_dict is None:
            quest_dict = self._template_fallback(ctx, difficulty, xp, galleons)
        if quest_dict is None:
            return None

        # Stamp with runtime metadata
        ts = int(time.time())
        quest_dict.update({
            "id": f"dq_{ts}_{ctx.session_id[:4]}",
            "type": "dynamic",
            "status": "available",
            "generated": True,
            "session_id": ctx.session_id,
            "location_id": ctx.location_id,
            "turn_generated": ctx.turn_number,
            "connected_quests": [],
            "house_points": {},
        })

        # Record
        self._quests.setdefault(ctx.session_id, []).append(quest_dict)
        self._visited.setdefault(ctx.session_id, set()).add(ctx.location_id)
        self._last_turn[ctx.session_id] = ctx.turn_number
        return quest_dict

    def get_quests(self, session_id: str) -> List[Dict]:
        return list(self._quests.get(session_id, []))

    def complete_quest(self, session_id: str, quest_id: str) -> bool:
        for q in self._quests.get(session_id, []):
            if q["id"] == quest_id:
                q["status"] = "completed"
                return True
        return False

    # ── LLM path ──────────────────────────────────────────────────────────────

    async def _try_llm(
        self, ctx: QuestContext, difficulty: str, xp: int, galleons: int
    ) -> Optional[Dict]:
        story_beats = "; ".join(ctx.key_beats[-5:]) or "Adventure just beginning."
        npcs = ", ".join(ctx.npcs_here) or "none visible"
        existing = ", ".join(f'"{t}"' for t in ctx.existing_quest_titles[-10:]) or "none"

        prompt = _QUEST_PROMPT.format(
            player_name=ctx.player_name,
            level=ctx.player_level,
            house=ctx.player_house,
            location_name=ctx.location_name,
            npcs=npcs,
            story_beats=story_beats,
            existing_titles=existing,
            difficulty=difficulty,
            xp=xp,
            galleons=galleons,
        )
        messages = [
            {"role": "system", "content": "You are a creative Harry Potter RPG quest designer. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = await self._call_llm(messages)
            if raw:
                return self._parse_json(raw, ctx.existing_quest_titles)
        except Exception as e:
            print(f"[QuestGen] LLM call failed ({e})")
        return None

    async def _call_llm(self, messages: List[Dict]) -> Optional[str]:
        if LLM_PROVIDER == "ollama":
            async with httpx.AsyncClient(timeout=60.0) as c:
                resp = await c.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={"model": OLLAMA_MODEL, "messages": messages, "stream": False,
                          "options": {"temperature": 0.9, "top_p": 0.95}},
                )
                return resp.json().get("message", {}).get("content", "")
        if LLM_PROVIDER == "openai":
            async with httpx.AsyncClient(timeout=60.0) as c:
                resp = await c.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={"model": OPENAI_MODEL, "messages": messages,
                          "temperature": 0.9, "max_tokens": 500},
                )
                return resp.json()["choices"][0]["message"]["content"]
        return None

    def _parse_json(self, raw: str, existing_titles: List[str]) -> Optional[Dict]:
        """Extract and validate the JSON block from the LLM response."""
        # Strip markdown code fences
        raw = re.sub(r"```(?:json)?", "", raw).strip("` \n")
        # Find the first complete JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        # Validate required fields
        required = {"title", "description", "objectives", "giver", "difficulty",
                    "rewards", "hook"}
        if not required.issubset(data.keys()):
            return None
        if not isinstance(data.get("objectives"), list) or not data["objectives"]:
            return None
        if data["title"] in existing_titles:
            return None  # duplicate — let fallback handle it

        # Normalise objectives
        objs = []
        for o in data["objectives"][:4]:
            if isinstance(o, str):
                objs.append({"text": o, "completed": False})
            elif isinstance(o, dict) and "text" in o:
                objs.append({"text": o["text"], "completed": False})
        data["objectives"] = objs or [{"text": "Complete the quest", "completed": False}]

        # Normalise rewards
        rw = data.get("rewards", {})
        if not isinstance(rw, dict):
            data["rewards"] = {"xp": 150, "galleons": 20}
        else:
            data["rewards"] = {"xp": int(rw.get("xp", 150)), "galleons": int(rw.get("galleons", 0))}

        return data

    # ── Template fallback ─────────────────────────────────────────────────────

    def _template_fallback(
        self, ctx: QuestContext, difficulty: str, xp: int, galleons: int
    ) -> Optional[Dict]:
        pool = _TEMPLATES.get(ctx.location_id, [])
        if not pool:
            pool = [t for loc_templates in _TEMPLATES.values() for t in loc_templates]
        # Exclude already-generated titles
        used = {q["title"] for q in self._quests.get(ctx.session_id, [])}
        used |= set(ctx.existing_quest_titles)
        available = [t for t in pool if t["title"] not in used]
        if not available:
            available = pool  # all used — allow repeats
        template = random.choice(available)
        quest = dict(template)
        quest["objectives"] = [dict(o) for o in template["objectives"]]
        quest["difficulty"] = difficulty
        quest["rewards"] = {"xp": xp, "galleons": galleons}
        return quest

    # ── helpers ───────────────────────────────────────────────────────────────

    def _count_active(self, session_id: str) -> int:
        return sum(
            1 for q in self._quests.get(session_id, [])
            if q.get("status") == "available"
        )

    @staticmethod
    def _pick_difficulty(level: int) -> str:
        if level <= 3:
            return "easy"
        if level <= 8:
            return random.choice(["easy", "medium"])
        if level <= 14:
            return random.choice(["medium", "medium", "hard"])
        return random.choice(["medium", "hard", "hard"])

    @staticmethod
    def _pick_rewards(difficulty: str) -> tuple:
        lo, hi = _DIFF_XP.get(difficulty, (150, 300))
        xp = random.randrange(lo, hi + 1, 25)
        galleons = {"easy": random.randint(0, 40),
                    "medium": random.randint(20, 80),
                    "hard": random.randint(50, 150)}.get(difficulty, 30)
        return xp, galleons


# Global instance
quest_generator = DynamicQuestGenerator()
