# AI Dungeon Master — Harry Potter Universe RPG

> *"Happiness can be found even in the darkest of times, if one only remembers to turn on the light."*

A fully AI-powered text + image RPG set in the Wizarding World. Powered by **Ollama** (local LLM), a **Knowledge Graph** world engine, adaptive **Emotion Detection**, and a **Rule-based Enemy AI** scaffold ready for RL upgrade.

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- A modern web browser

### 1. Install Ollama Model
```bash
ollama pull llama3.2
```

### 2. Install Python Dependencies
```bash
cd ai-dungeon-master
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
copy .env.example .env
# Edit .env if needed — defaults work with Ollama + llama3.2
```

### 4. Start the Backend
```bash
cd backend
python main.py
```
Backend runs at: `http://localhost:8000`

### 5. Open the Game
Open `frontend/index.html` in your browser, OR navigate to `http://localhost:8000`

---

## 🗺️ Architecture

```
Player Input (Text)
       │
       ▼
┌─────────────────┐     ┌──────────────────┐
│  Intent Parser  │────▶│ Knowledge Graph  │
│  (JS Frontend)  │     │ (NetworkX/Python) │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│      Narrator Agent (Ollama LLM)         │
│    llama3.2 with HP DM system prompt     │
│    Story memory with compression         │
└──────────┬──────────────────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐   ┌──────────────────┐
│ Scene  │   │   Enemy RL Agent  │
│ Canvas │   │ (Rule-based→RL)  │
│ (HTML5)│   └──────────────────┘
└────────┘
```

---

## 🧠 AI Components

| Component | Technology | Phase |
|---|---|---|
| Narrator LLM | Ollama (llama3.2) | ✅ Phase 1 |
| Story Memory | In-memory compression | ✅ Phase 1 |
| Knowledge Graph | NetworkX + D3.js | ✅ Phase 1 |
| Emotion Classifier | Keyword + behavioral | ✅ Phase 1 |
| Enemy AI | Rule-based (RL scaffold) | ✅ Phase 1 |
| Scene Images | Procedural Pillow renderer (real PNGs) | ✅ Phase 3 |
| RAG Pipeline | sentence-transformers + cosine | ✅ Phase 2 |
| Persistent Vectors | Disk-backed NPC embedding store | ✅ Phase 2.5 |
| RL Enemy Agent | Tabular Q-learning (adaptive) | ✅ Phase 4 |
| Stable Diffusion | Photoreal scene images (diffusers or A1111) | ✅ Phase 3+ |
| Save / Load System | Disk-backed session persistence (JSON) | ✅ Phase 5 |
| Level Progression | XP → Level 1-20, spell unlocks, titles | ✅ Phase 5 |
| Vector DB | ChromaDB / numpy pluggable vector store | ✅ Phase 6 |
| Dynamic Quests | LLM-generated side quests from story state | ✅ Phase 7 |
| LoRA Fine-tune | QLoRA training pipeline + HP DM dataset | ✅ Phase 8 |

---

## 📁 Project Structure

```
ai-dungeon-master/
├── frontend/               # Game UI (HTML/CSS/JS)
│   ├── index.html          # Main game layout
│   ├── style.css           # Dark fantasy design system
│   ├── game.js             # Core game engine
│   ├── narrator.js         # LLM API interface
│   ├── knowledge_graph.js  # D3.js world visualization
│   ├── combat.js           # Combat system UI
│   └── emotion.js          # Emotion tracking
├── backend/                # FastAPI Python backend
│   ├── main.py             # API server entry point
│   ├── narrator.py         # Ollama/OpenAI narrator agent
│   ├── knowledge_graph.py  # NetworkX world graph
│   ├── combat_ai.py        # Combat resolution (drives the RL agent)
│   ├── rl_agent.py         # Q-learning enemy agent (adaptive)
│   ├── rl_train.py         # Self-play harness / learning demo
│   ├── emotion.py          # Player emotion classifier
│   ├── memory.py           # Story memory management
│   ├── rag.py              # RAG retriever — semantic lore search
│   ├── npc_memory.py       # Long-term per-NPC memory (Phase 2.5)
│   ├── image_gen.py        # Scene image generation hook
│   ├── models/schemas.py   # Pydantic data models
│   └── data/
│       ├── world_lore.json # Harry Potter world data
│       └── quests.json     # Quest definitions
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🎮 Features

### ✅ Phase 1 (Current)
- **Dark fantasy premium UI** — glassmorphism panels, animated scenes, HP house themes
- **Ollama narrator** — llama3.2 as your personal Dungeon Master
- **Mock mode** — fully playable without any AI setup
- **Knowledge graph** — D3.js force-directed world map (locations, NPCs, quests)
- **Combat system** — turn-based dueling with 9 enemy types
- **Emotion detection** — adapts difficulty based on player mood
- **Quest log** — tracks active, available, and completed quests
- **Story memory** — remembers plot history with compression
- **Inventory & spellbook** — track items and known spells
- **WebSocket streaming** — real-time narrative delivery
- **Spell animations** — visual effects on casting

### ✅ Phase 2 (Current)
- **RAG pipeline** — every lore passage (locations, NPCs, factions, items, spells, lore entries) embedded with `sentence-transformers` (all-MiniLM-L6-v2)
- **Semantic retrieval** — each turn pulls the most relevant canon and injects it into the narrator's context, so the DM stays lore-accurate
- **Embedding cache** — vectors computed once and cached to disk; rebuilt automatically when the corpus changes
- **Graceful fallback** — degrades to keyword retrieval if the embedding model is unavailable (offline)
- **`/api/lore-search`** — query the corpus semantically over HTTP

### ✅ Phase 4 (Current)
- **Adaptive enemy AI** — tabular Q-learning agent that learns how each player fights and counters it
- **Real tradeoffs** — `attack` / `special` (more damage out, more taken) / `defend` (no damage out, much less taken); the optimal choice is state-dependent
- **Warm start** — unseen states fall back to the rule-based personality prior, so enemies fight sensibly from turn one, then improve
- **Persistent learning** — per-archetype Q-tables saved to disk; enemies keep getting smarter across sessions
- **Measurable** — self-play shows learned win rate jumping from ~0% (rule baseline) to ~100% vs a hard aggressive player
- **`/api/rl-stats`** + **`/api/rl-simulate`** — inspect the learned policy and run a before/after demonstration

### ✅ Phase 2.5 (Current)
- **Long-term NPC memory** — each NPC remembers what the player did in their presence, persisted across sessions
- **Semantic recall** — NPCs surface their most relevant memories of you (shared embeddings), with a recency fallback offline
- **Persistent vector store** — memory embeddings are cached to disk (`npc_vectors.npz`) so recall no longer recomputes every vector on startup; a per-NPC content hash rebuilds only changed NPCs, and a model-name guard invalidates the cache if the embedding model changes
- **`/api/npc-memory`** — inspect or query any NPC's memory of the player

### ✅ Phase 3 (Current)
- **Procedural scene art** — server-side Pillow renderer produces real atmospheric scene PNGs (per-location silhouettes, mood-driven particles, gradient skies) with no GPU, model, or network required; `IMAGE_PROVIDER=procedural` is the default
- **Layered display** — the frontend fetches and caches the generated image per location and fades it in over the canvas fallback
- **`/api/generate-scene`** — returns a base64 PNG data URL

### ✅ Phase 5 (Current)
- **Full game persistence** — every session auto-saves every 5 turns to `backend/saves/` as JSON; manual save via the 💾 button (server-backed when online, localStorage when offline)
- **Load saved game** — "Load Saved Game" on the character-creation screen fetches all saves from the server, shows name / level / title / turn count, and resumes a full live session
- **Server-side progression** — authoritative XP/level system (levels 1–20); XP thresholds follow a `100·lvl^1.5` curve; spell unlocks at levels 1, 3, 5, 7, 10, 12, 15, 18, 20; wizard title advances with each tier
- **Level-up toast** — animated full-screen overlay fires on level-up, showing the new level, title, and any newly unlocked spells; frontend is immediately updated with server-authoritative `updated_player` data
- **Endpoints**: `POST /api/save/{session_id}`, `GET /api/saves`, `POST /api/load?filename=`, `DELETE /api/save/{filename}`
- **`backend/progression.py`** — authoritative XP table, `process_xp_gain()`, `spells_for_level()`, `title_for_level()`
- **`backend/save_manager.py`** — `SaveManager` class with save/load/list/delete and per-player pruning (keeps 30 most recent saves)

### ✅ Phase 3+ (Current)
- **Stable Diffusion integration** — two backends in one:
  - `IMAGE_PROVIDER=diffusers`: HuggingFace `diffusers` pipeline runs directly in the backend process — no separate server, just `pip install diffusers transformers accelerate safetensors` + a CUDA-enabled PyTorch. Pipeline loads on startup in a background thread; requests block until ready then stream from the cache.
  - `IMAGE_PROVIDER=stable_diffusion`: calls an AUTOMATIC1111 `/sdapi/v1/txt2img` API (point `SD_API_URL` at your running instance).
- **Configurable model** — `SD_MODEL_ID=runwayml/stable-diffusion-v1-5` by default; swap in any HuggingFace model ID
- **Deterministic per-location seed** — same location always produces the same art; mood changes the prompt
- **SD generating overlay** — spinner appears over the scene panel while the pipeline runs; auto-clears when the image arrives
- **Pipeline warmup polling** — frontend polls `/api/sd-status` at startup; once the model finishes loading it automatically fetches the first scene
- **Auto-fallback** — if diffusers errors (VRAM OOM, model missing), falls back to the procedural renderer silently

### ✅ Phase 6 (Current)
- **Pluggable vector store** — new `backend/vector_store.py` provides a unified `VectorStore` interface backed by ChromaDB (persistent, production-ready) or numpy (zero-install fallback). Auto-detects which is available at startup.
- **ChromaDB backend** — `pip install chromadb` to activate; data persists to `backend/chroma_db/` (gitignored); uses `PersistentClient` with HNSWLib cosine similarity and `SentenceTransformerEmbeddingFunction`; metadata filtering for NPC memory (filter by `npc_id`)
- **numpy fallback** — all features continue to work without ChromaDB; includes keyword overlap fallback when embeddings are unavailable
- **RAG upgrade** — `rag.py` now uses `VectorStore`; lore index upserted once and reused (Chroma is persistent across restarts — no more `.npz` cache files to manage)
- **NPC memory upgrade** — `npc_memory.py` uses `VectorStore`; memories are seeded from persisted JSON on startup; new memories upserted individually on `record()`; recall uses Chroma's `where` filter for efficient per-NPC queries
- **Config**: `VECTOR_BACKEND=auto|chroma|numpy`, `EMBED_MODEL=all-MiniLM-L6-v2`

### ✅ Phase 7 (Current)
- **Dynamic quest generation** — `backend/quest_generator.py` builds a `QuestContext` from the player's current location, NPCs present, recent story beats, level, and reputation, then asks the LLM (Ollama/OpenAI) to output a structured JSON quest. Falls back to a hand-authored library of 19 location-specific templates (3 per location) when the LLM is offline or unavailable.
- **Smart trigger logic** — generation fires automatically every 5 turns and/or on location change, capped at 3 active generated quests per session; a `POST /api/generate-quest?force=true` endpoint allows manual triggering
- **Quest discovery UX** — the hook sentence is woven into the narrative as a DM entry 0.8 s after generation; a purple slide-in toast (top-right) shows the title, difficulty, and XP reward for 4 s
- **✨ badge in quest log** — generated quests render with a purple glow border and ✨ prefix to distinguish them from static quests; clicking shows the same detail view
- **Session restore** — dynamic quests are fetched via `GET /api/dynamic-quests/{session_id}` when loading a saved game, so the quest log is fully restored
- **New API**: `POST /api/generate-quest`, `GET /api/dynamic-quests/{session_id}`

### ✅ Phase 8 (Current)
- **HP DM dataset** — `backend/lora_train/dataset_builder.py` builds a ShareGPT-format JSONL: 37 hand-authored DM examples (atmospheric narration per location, NPC conversations, combat, spellcasting) + auto-generated examples from world lore (location arrivals, NPC encounters, spell effects) = **67 examples** from static corpus alone
- **Self-distillation generator** — `lora_train/generate_data.py` uses the running Ollama instance to generate 100+ additional examples (20 player actions × 7 locations), teaching the same HP DM voice at scale; output merged into the training corpus via `build_dataset(extra_path=...)`
- **QLoRA training** — `lora_train/train.py`: 4-bit quantized LoRA on `microsoft/phi-3-mini-4k-instruct` (3.8 B params, ~8 GB VRAM for training). PEFT + TRL `SFTTrainer`. LoRA rank 16 / alpha 32, cosine LR schedule, 3 epochs, paged AdamW 8-bit. Adapter saved to `lora_train/adapter/` (~50–150 MB). Model family auto-detects correct LoRA target modules (Phi-3, Llama, Mistral, Qwen).
- **LoRA inference** — `lora_train/infer.py` lazy-loads model + adapter once at startup in a background thread (same pattern as SD diffusers), runs generation in `run_in_executor` so the event loop stays free
- **`LLM_PROVIDER=lora`** — `narrator.py` routes to `lora_train/infer.py`; `main.py` calls `warmup_lora()` at startup and falls back to mock mode if the adapter isn't found

**To train and activate:**
```bash
# 1. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install peft trl bitsandbytes datasets

# 2. (Optional) generate extra training data using your running Ollama
python backend/lora_train/generate_data.py --num 100

# 3. Train (~30-60 min on a mid-range GPU)
python backend/lora_train/train.py

# 4. Activate in .env
LLM_PROVIDER=lora
LORA_BASE_MODEL=microsoft/phi-3-mini-4k-instruct
LORA_ADAPTER_PATH=backend/lora_train/adapter
```

### 📅 Next
- World state persistence (completed quests / visited locations survive restarts)
- Voice narration via TTS (ElevenLabs / Coqui)

---

## 🌍 The Harry Potter World

**Setting:** Post-Voldemort era. A new dark threat — the **Hollow Mage** — rises from within Azkaban.

**Key Locations:**
- 🍺 The Three Broomsticks (starting location)
- 🏰 Hogwarts Castle
- 🕯️ Knockturn Alley
- 🌲 The Forbidden Forest
- 🏛️ Ministry of Magic
- ⛓️ Azkaban

**Key NPCs:**
- Professor Neville Longbottom (ally)
- Headmistress McGonagall (quest giver)
- Madam Rosmerta (information broker)
- The Hollow Mage (final boss)

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/session/create` | Create new game session |
| POST | `/api/narrate` | Get narrative from LLM |
| POST | `/api/combat` | Resolve combat round (RL enemy agent learns) |
| GET | `/api/world-state` | Get D3.js graph data |
| GET | `/api/lore-search` | Semantic search over the lore corpus (RAG) |
| GET | `/api/npc-memory` | Inspect/query long-term NPC memory of the player |
| GET | `/api/rl-stats` | Inspect the enemy agent's learned policy |
| POST | `/api/rl-simulate` | Self-play demo: rule baseline vs learned win rate |
| POST | `/api/classify-emotion` | Detect player emotion |
| POST | `/api/generate-scene` | Generate scene image |
| WS | `/ws/{session_id}` | Real-time streaming |

---

## 🛠️ Switching to OpenAI

Edit `.env`:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

---

## 📜 License

MIT — build your own adventure!
