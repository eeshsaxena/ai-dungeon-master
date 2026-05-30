"""
Self-distillation data generator (Phase 8).

Uses the running Ollama instance to generate additional HP DM training examples,
grounded in the world lore. This augments the handcrafted examples in
dataset_builder.py with LLM-generated ones that teach the same style.

Usage:
    python generate_data.py [--num 100] [--out extra_examples.jsonl]

The generated JSONL can then be fed back into dataset_builder.build_dataset()
for a larger fine-tuning corpus.
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import httpx

OLLAMA_BASE_URL = __import__("os").getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = __import__("os").getenv("OLLAMA_MODEL", "llama3.2")
OUT_DIR         = Path(__file__).parent / "training_data"

# ── Prompt inputs ──────────────────────────────────────────────────────────────

LOCATIONS = [
    ("loc_001", "The Three Broomsticks", "warm pub"),
    ("loc_002", "Hogwarts Castle",       "majestic school"),
    ("loc_003", "Knockturn Alley",       "dark district"),
    ("loc_004", "The Forbidden Forest",  "ancient wilderness"),
    ("loc_005", "Ministry of Magic",     "magical government"),
    ("loc_006", "Azkaban",              "grim prison"),
    ("loc_007", "Godric's Hollow",      "haunted village"),
]

PLAYER_ACTIONS = [
    "I look around carefully to assess the situation.",
    "I examine the strange markings on the wall.",
    "I try to speak with the mysterious figure.",
    "I cast Lumos to see what's in the dark corner.",
    "I search the room for anything unusual.",
    "I follow the sound coming from further in.",
    "I read the notice board to gather information.",
    "I find a hidden passageway and decide whether to enter.",
    "I notice someone watching me from across the room.",
    "I cast Accio to retrieve something I spotted.",
    "I try to decipher the ancient text on the stone.",
    "I approach the unusual creature cautiously.",
    "I pick up the strange artifact that was left behind.",
    "I cast Revelio to check for hidden things.",
    "I speak to the portrait on the wall.",
    "I check my watch and realize I've lost track of time.",
    "I study the footprints leading away from the scene.",
    "I attempt to heal the injured figure I found.",
    "I send an owl with an urgent message.",
    "I hide quickly as I hear footsteps approaching.",
]

_GENERATION_SYSTEM = """You are a Harry Potter RPG Dungeon Master. When given a player action, write an atmospheric narrative response.

RULES:
- Second person narration ("You see...", "You feel...")
- 100-180 words
- HP-specific detail: spells, creatures, magical items, locations, characters
- End with a subtle story hook, not a list of options
- Use [SPELL_CAST: SpellName] when spells are cast
- Never break character or mention being an AI
- Dark and atmospheric in dangerous locations; warm and alive in safe ones
- Make the world feel lived-in and specific"""


async def generate_one(location_id: str, location_name: str, atmosphere: str, action: str) -> dict | None:
    prompt = f"[LOCATION: {location_name} — {atmosphere}]\n\nPlayer: {action}\n\nDungeon Master:"
    messages = [
        {"role": "system", "content": _GENERATION_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": False,
                      "options": {"temperature": 0.92, "top_p": 0.95}},
            )
            content = resp.json().get("message", {}).get("content", "").strip()
            if len(content.split()) < 60:  # too short
                return None
            return {"user": action, "assistant": content, "location": location_id}
    except Exception as e:
        print(f"  [!] Generation error: {e}", file=sys.stderr)
        return None


async def run(num: int, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    combos = [(loc, action) for loc in LOCATIONS for action in PLAYER_ACTIONS]
    random.shuffle(combos)
    targets = combos[:num]

    print(f"[GenerateData] Generating {len(targets)} examples via {OLLAMA_MODEL}…")
    for i, ((lid, lname, latm), action) in enumerate(targets, 1):
        ex = await generate_one(lid, lname, latm, action)
        if ex:
            results.append(ex)
            print(f"  [{i}/{len(targets)}] {lname}: {action[:50]}…")
        else:
            print(f"  [{i}/{len(targets)}] SKIPPED (quality filter)")

    with open(out, "w", encoding="utf-8") as f:
        for ex in results:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[GenerateData] Saved {len(results)} examples → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num",  type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--out",  type=str, default=str(OUT_DIR / "extra_examples.jsonl"))
    args = parser.parse_args()
    asyncio.run(run(args.num, Path(args.out)))
