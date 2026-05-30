"""
FastAPI main application for AI Dungeon Master.
Endpoints: narrate, combat, world-state, emotion, image generation, session management.
"""
import os
import sys
import uuid
import time
import json
from typing import Dict, Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend dir to path
sys.path.insert(0, str(Path(__file__).parent))

from models.schemas import (
    NarrateRequest, NarrateResponse,
    CombatRequest, CombatResponse,
    EmotionRequest, EmotionResponse,
    WorldStateResponse, SessionState,
    PlayerStats, Enemy, EnemyArchetype,
    DifficultyLevel, EmotionState
)
from narrator import narrator
from knowledge_graph import world_graph
from combat_ai import enemy_ai
from emotion import emotion_classifier
from memory import memory_manager
from image_gen import generate_scene_image, build_scene_prompt, warmup_diffusers
import image_gen as _image_gen
from rag import lore_retriever
from rl_agent import rl_agent
from rl_train import simulate as rl_simulate
from npc_memory import npc_memory_store
from save_manager import save_manager
from progression import process_xp_gain, xp_to_next, title_for_level, spells_for_level
from quest_generator import quest_generator, QuestContext
from world_state_store import get_world_state, drop_world_state
from tts_engine import synthesize as tts_synthesize, tts_status, warmup_coqui
from item_system import (
    ITEM_DB, get_item, make_item_instance, use_item, roll_loot,
    inventory_to_display, item_summary, tick_effects
)
from dialogue_engine import dialogue_engine, DialogueContext

# ── App Setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Dungeon Master API",
    description="Harry Potter RPG powered by LLM + Knowledge Graph + RL",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ── Session Storage ────────────────────────────────────────────────────────────

sessions: Dict[str, SessionState] = {}


def get_or_create_session(session_id: str) -> SessionState:
    if session_id not in sessions:
        sessions[session_id] = SessionState(
            session_id=session_id,
            player=PlayerStats(),
        )
    return sessions[session_id]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the game frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "AI Dungeon Master API", "status": "running", "version": "1.0.0"}


@app.get("/api/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
        "image_provider": os.getenv("IMAGE_PROVIDER", "placeholder"),
        "active_sessions": len(sessions),
        "timestamp": time.time()
    }


@app.post("/api/session/create")
async def create_session(player_name: str = "Unnamed Witch/Wizard", house: str = "Unselected"):
    """Create a new game session."""
    session_id = str(uuid.uuid4())
    player = PlayerStats(name=player_name, house=house)
    sessions[session_id] = SessionState(session_id=session_id, player=player)
    memory = memory_manager.get_or_create(session_id)

    # Log session start
    intro_event = f"New adventure begins. Player: {player_name}, House: {house}"
    memory.add_key_beat(intro_event)

    return {
        "session_id": session_id,
        "player": player.model_dump(),
        "message": f"⚡ Welcome to Hogwarts, {player_name}!"
    }


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get current session state."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.model_dump()


@app.post("/api/narrate")
async def narrate(request: NarrateRequest) -> NarrateResponse:
    """
    Main narrative endpoint.
    Takes player input → returns LLM narrative response.
    """
    session = get_or_create_session(request.session_id)
    memory = memory_manager.get_or_create(request.session_id)

    # Add player input to memory
    memory.add_turn("user", request.player_input)

    # Get world context
    world_context = world_graph.get_location_context(request.current_location)
    world_summary = world_graph.get_world_summary()
    full_context = f"{world_context}\n\n{world_summary}"

    # Detect player emotion from recent inputs
    recent_inputs = [
        t["content"] for t in memory.full_history[-5:]
        if t["role"] == "user"
    ]
    emotion, confidence, hint = emotion_classifier.classify(recent_inputs)

    # RAG: retrieve lore passages relevant to the player's action + recent context
    retrieval_query = " ".join(recent_inputs[-3:]) or request.player_input
    lore_context = lore_retriever.retrieve_context(retrieval_query, top_k=4)
    if lore_context:
        full_context = f"{full_context}\n\n{lore_context}"

    # NPC memory: surface what NPCs present here remember about the player
    npcs_here = world_graph.get_npcs_at_location(request.current_location)
    npc_recall = [
        block for npc in npcs_here
        if (block := npc_memory_store.recall_text(
            npc["id"], npc.get("label", npc["id"]), request.player_input, top_k=2))
    ]
    if npc_recall:
        full_context += "\n\n[NPC MEMORY — these characters recall the player]\n" + "\n".join(npc_recall)

    # Get narrative from LLM
    response = await narrator.narrate(
        request=request,
        story_history=memory.get_messages_for_llm(),
        world_context=full_context,
        emotion=emotion
    )

    # Add response to memory
    memory.add_turn("assistant", response.narrative)

    # NPC long-term memory: NPCs present here remember what the player just did
    for npc in npcs_here:
        npc_memory_store.record(
            npc["id"], f"The player {request.player_input}", request.session_id
        )

    # Apply world state updates + persist location visits
    ws = get_world_state(request.session_id)
    ws.visit_location(request.current_location)
    updates = response.world_state_updates
    if "new_location" in updates:
        new_loc = updates["new_location"]
        world_graph.update_player_location(new_loc)
        session.player.current_location = new_loc
        memory.add_world_change(f"Player moved to {new_loc}")
        first_visit = ws.visit_location(new_loc)
        if first_visit:
            ws.add_event(f"First visited {new_loc}")

    # Update session
    session.player.turns_played += 1
    session.player.emotion_state = emotion
    session.story_history.append({"role": "user", "content": request.player_input})
    session.story_history.append({"role": "assistant", "content": response.narrative})

    # Auto-save every 5 turns
    if session.player.turns_played % 5 == 0:
        _auto_save(request.session_id, session)

    # Dynamic quest generation — check conditions and generate if warranted
    all_quest_titles = [q.get("title", "") for q in world_graph.get_active_quests() + world_graph.get_available_quests()]
    all_quest_titles += [q["title"] for q in quest_generator.get_quests(request.session_id)]
    ctx = QuestContext(
        session_id=request.session_id,
        player_name=session.player.name,
        player_level=session.player.level,
        player_house=str(session.player.house),
        player_reputation=dict(session.player.reputation),
        location_id=request.current_location,
        location_name=(world_graph.get_location_info(request.current_location) or {}).get("label", request.current_location),
        npcs_here=[n.get("label", n["id"]) for n in npcs_here],
        key_beats=memory.key_beats[-8:],
        existing_quest_titles=all_quest_titles,
        turn_number=session.player.turns_played,
    )
    if quest_generator.should_generate(ctx):
        new_quest = await quest_generator.generate(ctx)
        if new_quest:
            response.new_quest = new_quest

    # Add hint if triggered
    if hint:
        response.narrative += f"\n\n{hint}"

    response.detected_emotion = emotion
    response.difficulty_adjustment = emotion_classifier.get_difficulty_recommendation(emotion).value

    return response


def _auto_save(session_id: str, session) -> None:
    """Background auto-save (player + memory + world state) — never raises."""
    try:
        mem = memory_manager.get(session_id)
        memory_data = mem.to_dict() if mem else {}
        ws  = get_world_state(session_id)
        player_dict = session.player.model_dump()
        player_dict["_world_state"] = ws.to_dict()   # bundle into the save
        save_manager.save(session_id, player_dict, memory_data)
    except Exception:
        pass


@app.post("/api/combat")
async def resolve_combat(request: CombatRequest) -> CombatResponse:
    """Resolve a combat round using the enemy AI."""
    session = get_or_create_session(request.session_id)

    result = enemy_ai.resolve_combat_round(request)
    session.player.hp = result.player_hp_remaining

    if result.combat_over:
        memory = memory_manager.get_or_create(request.session_id)
        if result.player_won:
            # Server-authoritative XP + level-up
            old_xp = request.player_stats.xp
            old_level = request.player_stats.level
            new_xp, new_level, new_spells, leveled_up = process_xp_gain(old_xp, result.xp_gained)

            session.player.xp = new_xp
            session.player.level = new_level
            session.player.xp_to_next = xp_to_next(new_xp, new_level)
            session.player.title = title_for_level(new_level)

            if leveled_up:
                levels_gained = new_level - old_level
                session.player.max_hp = request.player_stats.max_hp + levels_gained * 10
                session.player.hp = session.player.max_hp
                session.player.max_mana = request.player_stats.max_mana + levels_gained * 5
                session.player.mana = session.player.max_mana
                for spell in new_spells:
                    if spell not in session.player.spells_known:
                        session.player.spells_known.append(spell)
                result.level_up = True
                result.new_level = new_level
                result.new_title = session.player.title
                result.new_spells = new_spells
                result.updated_player = session.player.model_dump()
                memory.add_key_beat(
                    f"Leveled up to {new_level} ({session.player.title})!"
                    + (f" Unlocked: {', '.join(new_spells)}" if new_spells else "")
                )

            # Roll loot from item system
            loot_items = roll_loot(request.enemy.archetype, session.player.level)
            if loot_items:
                session.player.inventory.extend(loot_items)
                loot_names = [i["name"] for i in loot_items]
                result.loot = loot_items
                memory.add_key_beat(f"Found: {', '.join(loot_names)}")

            memory.add_key_beat(f"Defeated {request.enemy.name}! Gained {result.xp_gained} XP")
        else:
            session.player.hp = 10
            memory.add_key_beat(f"Defeated by {request.enemy.name}. Barely escaped.")

    return result


# ── Inventory / Item System ────────────────────────────────────────────────────

@app.get("/api/items")
async def list_all_items():
    """Return the full item database for the frontend."""
    return {"items": ITEM_DB}


@app.get("/api/inventory/{session_id}")
async def get_inventory(session_id: str):
    """Get the current player inventory with enriched item data."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"inventory": inventory_to_display(session.player.inventory)}


@app.post("/api/inventory/{session_id}/use/{item_id}")
async def use_item_endpoint(session_id: str, item_id: str):
    """Use an item from the player's inventory. Applies effects immediately."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Find item in inventory
    inv = session.player.inventory
    idx = next((i for i, it in enumerate(inv) if it.get("id") == item_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not in inventory")

    item = inv[idx]
    player_dict = session.player.model_dump()
    active_effects = player_dict.get("active_effects", [])
    mem = memory_manager.get_or_create(session_id)
    result = use_item(item, player_dict, active_effects, mem.key_beats)

    # Consume the item (remove from inventory) unless it's an artifact/key
    if item.get("type") not in ("artifact", "key") or not item.get("effects"):
        inv.pop(idx)
    elif item.get("type") == "artifact":
        # Artifacts are consumed after single use
        inv.pop(idx)

    # Apply updated stats back to session
    from models.schemas import PlayerStats
    updated = PlayerStats(**player_dict)
    session.player = updated

    mem.add_key_beat(f"Used {item['name']}: {result['message']}")
    return result


@app.post("/api/inventory/{session_id}/drop/{item_id}")
async def drop_item_endpoint(session_id: str, item_id: str):
    """Remove an item from inventory without using it."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    inv = session.player.inventory
    idx = next((i for i, it in enumerate(inv) if it.get("id") == item_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Item not found")
    dropped = inv.pop(idx)
    return {"dropped": True, "item": dropped}


@app.post("/api/inventory/{session_id}/give")
async def give_item_endpoint(session_id: str, item_id: str):
    """Give the player an item (for testing / quest rewards)."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    inst = make_item_instance(item_id)
    if not inst:
        raise HTTPException(status_code=404, detail=f"Unknown item: {item_id}")
    session.player.inventory.append(inst)
    return {"given": True, "item": inst}


# ── NPC Dialogue ───────────────────────────────────────────────────────────────

def _build_dialogue_ctx(session_id: str, npc_id: str, npc_name: str) -> DialogueContext:
    session = get_or_create_session(session_id)
    mem     = memory_manager.get_or_create(session_id)
    npcs_h  = world_graph.get_npcs_at_location(session.player.current_location)
    memories = npc_memory_store.recall(npc_id, "", top_k=5)
    mem_texts = [m.get("text", "") for m in memories if m.get("text")]
    all_titles = [q.get("title", "") for q in world_graph.get_active_quests() + world_graph.get_available_quests()]

    return DialogueContext(
        session_id=session_id,
        npc_id=npc_id,
        npc_name=npc_name,
        player_name=session.player.name,
        player_level=session.player.level,
        player_house=str(session.player.house),
        player_reputation=dict(session.player.reputation),
        location_name=(world_graph.get_location_info(session.player.current_location) or {}).get("label", session.player.current_location),
        npc_memories=mem_texts,
        active_quest_titles=all_titles,
        key_beats=mem.key_beats[-6:],
    )


@app.post("/api/dialogue/start")
async def start_dialogue(session_id: str, npc_id: str):
    """Begin a conversation with an NPC."""
    node = world_graph.graph.nodes.get(npc_id, {})
    npc_name = node.get("label", npc_id)
    ctx = _build_dialogue_ctx(session_id, npc_id, npc_name)
    result = await dialogue_engine.start(ctx)

    # Record in NPC memory
    npc_memory_store.record(npc_id, f"The player started a conversation", session_id)
    return result


@app.post("/api/dialogue/reply")
async def dialogue_reply(session_id: str, npc_id: str, player_says: str):
    """Player says something in an ongoing conversation."""
    node = world_graph.graph.nodes.get(npc_id, {})
    npc_name = node.get("label", npc_id)
    ctx = _build_dialogue_ctx(session_id, npc_id, npc_name)
    result = await dialogue_engine.reply(session_id, npc_id, player_says, ctx)

    # Apply reputation changes from this dialogue turn
    if result.get("reputation_change"):
        session = get_or_create_session(session_id)
        ws = get_world_state(session_id)
        for faction, delta in result["reputation_change"].items():
            ws.adjust_reputation(faction, delta)
            if faction in session.player.reputation:
                session.player.reputation[faction] += delta

    # If NPC offered an item, give it to the player
    if result.get("item_offered"):
        session = get_or_create_session(session_id)
        inst = make_item_instance(result["item_offered"])
        if inst and len(session.player.inventory) < 20:
            session.player.inventory.append(inst)
            result["item_given"] = inst

    # Record what the player said
    npc_memory_store.record(npc_id, f"The player said: {player_says[:100]}", session_id)
    return result


@app.get("/api/dialogue/{session_id}/{npc_id}")
async def get_dialogue_history(session_id: str, npc_id: str):
    """Get conversation history with an NPC."""
    return {"history": dialogue_engine.get_history(session_id, npc_id)}


@app.post("/api/dialogue/end")
async def end_dialogue(session_id: str, npc_id: str):
    """End the current conversation."""
    dialogue_engine.end_conversation(session_id, npc_id)
    return {"ended": True, "summary": dialogue_engine.summary(session_id)}


# ── Dynamic Quests ─────────────────────────────────────────────────────────────

@app.post("/api/generate-quest")
async def generate_quest_endpoint(
    session_id: str,
    location_id: str = "loc_001",
    force: bool = False,
):
    """Manually trigger a dynamic quest generation for the current location."""
    session = get_or_create_session(session_id)
    memory  = memory_manager.get_or_create(session_id)
    npcs    = world_graph.get_npcs_at_location(location_id)
    all_titles = [q.get("title", "") for q in world_graph.get_active_quests() + world_graph.get_available_quests()]
    all_titles += [q["title"] for q in quest_generator.get_quests(session_id)]

    ctx = QuestContext(
        session_id=session_id,
        player_name=session.player.name,
        player_level=session.player.level,
        player_house=str(session.player.house),
        player_reputation=dict(session.player.reputation),
        location_id=location_id,
        location_name=(world_graph.get_location_info(location_id) or {}).get("label", location_id),
        npcs_here=[n.get("label", n["id"]) for n in npcs],
        key_beats=memory.key_beats[-8:],
        existing_quest_titles=all_titles,
        turn_number=session.player.turns_played,
    )
    if not force and not quest_generator.should_generate(ctx):
        return {"generated": False, "reason": "cooldown or max active quests reached"}

    quest = await quest_generator.generate(ctx)
    return {"generated": quest is not None, "quest": quest}


@app.get("/api/dynamic-quests/{session_id}")
async def list_dynamic_quests(session_id: str):
    """Return all generated quests for a session."""
    return {"quests": quest_generator.get_quests(session_id)}


# ── Save / Load ────────────────────────────────────────────────────────────────

@app.post("/api/save/{session_id}")
async def save_game(session_id: str):
    """Manually save the current session to disk."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    mem = memory_manager.get(session_id)
    memory_data = mem.to_dict() if mem else {}
    filename = save_manager.save(session_id, session.player.model_dump(), memory_data)
    return {"saved": True, "filename": filename, "player": session.player.model_dump()}


@app.get("/api/saves")
async def list_saves():
    """List all save files, newest first."""
    return {"saves": save_manager.list_saves(), **save_manager.stats()}


@app.post("/api/load")
async def load_game(filename: str):
    """
    Load a save file and reconstitute a live session.
    Returns a new session_id and the restored player stats.
    """
    data = save_manager.load(filename)
    if not data:
        raise HTTPException(status_code=404, detail=f"Save file '{filename}' not found")

    from models.schemas import PlayerStats as PS
    player = PS(**data["player"])
    # Recompute progression fields in case the save predates Phase 5
    player.xp_to_next = xp_to_next(player.xp, player.level)
    player.title = title_for_level(player.level)
    # Make sure all spells for current level are known
    for spell in spells_for_level(player.level):
        if spell not in player.spells_known:
            player.spells_known.append(spell)

    new_sid = str(uuid.uuid4())
    sessions[new_sid] = SessionState(session_id=new_sid, player=player)

    # Restore story memory key beats so the narrator has context
    mem = memory_manager.get_or_create(new_sid)
    saved_mem = data.get("memory", {})
    if saved_mem.get("compressed_summary"):
        mem.compressed_summary = saved_mem["compressed_summary"]
    for beat in saved_mem.get("key_beats", []):
        mem.add_key_beat(beat)
    for change in saved_mem.get("world_changes", []):
        mem.add_world_change(change)

    # Restore world state (quest statuses, visited locations, flags)
    saved_ws = data["player"].get("_world_state")
    if saved_ws:
        ws = get_world_state(new_sid)
        ws.from_dict(saved_ws)
        ws.session_id = new_sid   # restamp with new session id

    return {
        "session_id": new_sid,
        "player": player.model_dump(),
        "loaded_from": filename,
        "world_state": get_world_state(new_sid).summary(),
    }


@app.delete("/api/save/{filename}")
async def delete_save(filename: str):
    """Delete a save file."""
    deleted = save_manager.delete(filename)
    if not deleted:
        raise HTTPException(status_code=404, detail="Save file not found")
    return {"deleted": True, "filename": filename}


@app.get("/api/rl-stats")
async def rl_stats(archetype: Optional[str] = None):
    """Inspect what the RL enemy agent has learned."""
    payload = {"stats": rl_agent.stats()}
    if archetype:
        payload["policy"] = rl_agent.policy_for(archetype)
    return payload


@app.post("/api/rl-simulate")
async def rl_simulate_endpoint(
    archetype: str = "Death Eater",
    player_style: str = "aggressive",
    train_combats: int = 400,
    eval_combats: int = 300,
    train_live: bool = False,
):
    """
    Run self-play to demonstrate the enemy agent learning a player's style.
    Returns baseline (rule) vs learned win rates and the learning curve.
    Set train_live=true to train the persistent live agent on this style.
    """
    train_combats = max(20, min(train_combats, 5000))
    eval_combats = max(20, min(eval_combats, 2000))
    return rl_simulate(
        archetype=archetype,
        train_combats=train_combats,
        eval_combats=eval_combats,
        player_style=player_style,
        use_global=train_live,
    )


@app.post("/api/spawn-enemy")
async def spawn_enemy(archetype: str = "Death Eater", level_scaling: float = 1.0):
    """Spawn an enemy for combat."""
    try:
        arch = EnemyArchetype(archetype)
    except ValueError:
        arch = EnemyArchetype.DARK_WIZARD

    enemy = enemy_ai.spawn_enemy(arch, level_scaling)
    return enemy.model_dump()


@app.post("/api/classify-emotion")
async def classify_emotion(request: EmotionRequest) -> EmotionResponse:
    """Classify player emotional state."""
    emotion, confidence, hint = emotion_classifier.classify(
        request.recent_inputs,
        request.turns_without_progress,
        request.time_between_inputs_seconds
    )

    difficulty = emotion_classifier.get_difficulty_recommendation(emotion)

    return EmotionResponse(
        emotion=emotion,
        confidence=confidence,
        difficulty_recommendation=difficulty,
        hint_triggered=hint is not None,
        hint_text=hint
    )


@app.get("/api/world-state")
async def get_world_state(session_id: Optional[str] = None) -> dict:
    """Export world graph for D3.js visualization."""
    session = sessions.get(session_id) if session_id else None
    player_location = session.player.current_location if session else "loc_001"

    d3_data = world_graph.to_d3_json(player_location)
    active_quests = world_graph.get_active_quests()
    available_quests = world_graph.get_available_quests()

    return {
        **d3_data,
        "player_location": player_location,
        "active_quests": active_quests,
        "available_quests": available_quests
    }


@app.get("/api/location/{location_id}")
async def get_location(location_id: str):
    """Get information about a specific location."""
    info = world_graph.get_location_info(location_id)
    if not info:
        raise HTTPException(status_code=404, detail="Location not found")
    npcs = world_graph.get_npcs_at_location(location_id)
    return {"location": info, "npcs": npcs}


@app.get("/api/sd-status")
async def sd_status():
    """Report whether the Stable Diffusion pipeline is loaded and ready."""
    provider = os.getenv("IMAGE_PROVIDER", "procedural")
    if provider != "diffusers":
        return {"provider": provider, "ready": provider == "procedural", "model": None}
    ready = _image_gen._diffusers_pipeline is not None
    error = _image_gen._diffusers_error
    return {
        "provider": "diffusers",
        "ready": ready,
        "loading": not ready and not error,
        "error": error,
        "model": _image_gen.SD_MODEL_ID,
    }


@app.post("/api/tts")
async def text_to_speech(text: str, session_id: str = ""):
    """Convert DM narrative text to speech. Returns base64 audio data URI or null."""
    audio = await tts_synthesize(text)
    return {"audio": audio, "provider": os.getenv("TTS_PROVIDER", "disabled")}


@app.get("/api/tts-status")
async def get_tts_status():
    """Report TTS provider readiness."""
    return tts_status()


@app.get("/api/world-state-store/{session_id}")
async def get_world_state_store(session_id: str):
    """Return the persistent world state delta for a session."""
    ws = get_world_state(session_id)
    return ws.summary()


@app.post("/api/world-state-store/{session_id}/quest")
async def update_quest_status(session_id: str, quest_id: str, status: str):
    """Mark a quest active/completed/failed."""
    ws = get_world_state(session_id)
    ws.set_quest_status(quest_id, status)
    if status == "completed":
        memory = memory_manager.get_or_create(session_id)
        memory.add_key_beat(f"Completed quest: {quest_id}")
    return {"updated": True, "quest_id": quest_id, "status": status}


@app.post("/api/generate-scene")
async def generate_scene(
    location_id: str = "loc_001",
    situation: str = "",
    mood: str = "mysterious"
):
    """Generate a scene image for the current location."""
    prompt = build_scene_prompt(location_id, situation, mood)
    result = await generate_scene_image(prompt, location_id, mood=mood)
    return result


@app.get("/api/world-lore")
async def get_world_lore():
    """Get world lore entries."""
    lore_path = Path(__file__).parent / "data" / "world_lore.json"
    with open(lore_path) as f:
        return json.load(f)


@app.get("/api/lore-search")
async def lore_search(q: str, top_k: int = 4):
    """Semantic search over the world lore corpus (RAG retriever)."""
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query 'q' is required")
    top_k = max(1, min(top_k, 10))
    results = lore_retriever.search(q, top_k=top_k)
    return {
        "query": q,
        "results": results,
        "retriever": lore_retriever.stats(),
    }


@app.get("/api/npc-memory")
async def npc_memory(npc_id: Optional[str] = None, q: str = "", top_k: int = 3):
    """Inspect long-term NPC memory. With npc_id, recall that NPC's memories
    (semantically if a query is given); without it, return store stats."""
    if not npc_id:
        return npc_memory_store.stats()
    top_k = max(1, min(top_k, 10))
    return {
        "npc_id": npc_id,
        "query": q,
        "memories": npc_memory_store.recall(npc_id, q, top_k=top_k),
    }


@app.get("/api/quests")
async def get_quests():
    """Get all quests."""
    quests_path = Path(__file__).parent / "data" / "quests.json"
    with open(quests_path) as f:
        return json.load(f)


@app.put("/api/session/{session_id}/player")
async def update_player(session_id: str, player: PlayerStats):
    """Update player stats in a session."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.player = player
    return {"updated": True, "player": player.model_dump()}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session (quit game)."""
    if session_id in sessions:
        del sessions[session_id]
    memory_manager.clear(session_id)
    return {"deleted": True}


# ── WebSocket for real-time streaming ─────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, ws: WebSocket):
        await ws.accept()
        self.active[session_id] = ws

    def disconnect(self, session_id: str):
        self.active.pop(session_id, None)

    async def send(self, session_id: str, data: dict):
        ws = self.active.get(session_id)
        if ws:
            await ws.send_json(data)


ws_manager = ConnectionManager()


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket for real-time narrative streaming."""
    await ws_manager.connect(session_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            event_type = data.get("type", "")

            if event_type == "ping":
                await ws_manager.send(session_id, {"type": "pong"})

            elif event_type == "narrate":
                # Build request from WebSocket data
                try:
                    session = get_or_create_session(session_id)
                    req = NarrateRequest(
                        session_id=session_id,
                        player_input=data.get("input", ""),
                        player_stats=session.player,
                        current_location=session.player.current_location,
                        difficulty=DifficultyLevel(data.get("difficulty", "medium"))
                    )
                    # Send typing indicator
                    await ws_manager.send(session_id, {"type": "typing", "data": True})
                    # Get narrative
                    response = await narrate(req)
                    await ws_manager.send(session_id, {
                        "type": "narrative",
                        "data": response.model_dump()
                    })
                except Exception as e:
                    await ws_manager.send(session_id, {"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        ws_manager.disconnect(session_id)


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    print("AI Dungeon Master starting...")
    print(f"   LLM Provider: {os.getenv('LLM_PROVIDER', 'ollama')}")
    print(f"   Ollama Model: {os.getenv('OLLAMA_MODEL', 'llama3.2')}")
    print(f"   Image Provider: {os.getenv('IMAGE_PROVIDER', 'placeholder')}")
    print(f"   World: The Wizarding World (Harry Potter)")
    print(f"   Frontend: {FRONTEND_DIR}")
    # Warm the RAG index so the first narration isn't slow
    try:
        stats = lore_retriever.initialize().stats()
        print(f"   RAG: {stats['passages']} passages via {stats['backend']} ({stats['model']})")
    except Exception as e:
        print(f"   RAG: failed to initialize ({e})")
    rls = rl_agent.stats()
    print(f"   RL Enemy AI: {'on' if os.getenv('ENEMY_AI', 'rl').lower() == 'rl' else 'off'} "
          f"| {rls['episodes']} episodes, {rls['total_states']} states learned")
    ss = save_manager.stats()
    print(f"   Save System: {ss['save_count']} saves on disk ({ss['save_dir']})")
    # Pre-warm SD pipeline in background (no-op if IMAGE_PROVIDER != "diffusers")
    warmup_diffusers()
    # Pre-warm LoRA model in background (no-op if LLM_PROVIDER != "lora")
    if os.getenv("LLM_PROVIDER") == "lora":
        try:
            from lora_train.infer import warmup_lora
            warmup_lora()
            print("   LoRA: loading adapter in background…")
        except ImportError:
            print("   LoRA: peft/transformers not installed — run: pip install peft trl bitsandbytes")
    # Pre-warm Coqui TTS in background (no-op if TTS_PROVIDER != "coqui"/"auto")
    warmup_coqui()
    ts = tts_status()
    print(f"   TTS: {ts['provider']}" + (" (loading…)" if ts.get("loading") else " (ready)" if ts.get("ready") else " (disabled)"))
    print("   Ready!")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=os.getenv("DEBUG", "true").lower() == "true"
    )
