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

    # Apply world state updates
    updates = response.world_state_updates
    if "new_location" in updates:
        world_graph.update_player_location(updates["new_location"])
        session.player.current_location = updates["new_location"]
        memory.add_world_change(f"Player moved to {updates['new_location']}")

    # Update session
    session.player.turns_played += 1
    session.player.emotion_state = emotion
    session.story_history.append({"role": "user", "content": request.player_input})
    session.story_history.append({"role": "assistant", "content": response.narrative})

    # Auto-save every 5 turns
    if session.player.turns_played % 5 == 0:
        _auto_save(request.session_id, session)

    # Add hint if triggered
    if hint:
        response.narrative += f"\n\n{hint}"

    response.detected_emotion = emotion
    response.difficulty_adjustment = emotion_classifier.get_difficulty_recommendation(emotion).value

    return response


def _auto_save(session_id: str, session) -> None:
    """Background auto-save — never raises."""
    try:
        mem = memory_manager.get(session_id)
        memory_data = mem.to_dict() if mem else {}
        save_manager.save(session_id, session.player.model_dump(), memory_data)
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

            memory.add_key_beat(f"Defeated {request.enemy.name}! Gained {result.xp_gained} XP")
        else:
            session.player.hp = 10
            memory.add_key_beat(f"Defeated by {request.enemy.name}. Barely escaped.")

    return result


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

    return {
        "session_id": new_sid,
        "player": player.model_dump(),
        "loaded_from": filename,
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
    print("   Ready!")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=os.getenv("DEBUG", "true").lower() == "true"
    )
