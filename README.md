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
| Stable Diffusion | Photoreal scene images (optional upgrade) | 📅 Phase 3+ |
| Fine-tuned LLM | LoRA on HP corpus | 📅 Phase 6 |

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

### 📅 Next
- **Stable Diffusion upgrade** (`IMAGE_PROVIDER=stable_diffusion`) — photoreal scene art via a running AUTOMATIC1111 server; the hook already exists in `image_gen.py`
- Larger-scale persistent vector DB (Chroma/FAISS) for cross-session recall beyond the current disk cache

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
