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
| Scene Images | HTML5 Canvas + SD hook | ✅ Phase 1 |
| Fine-tuned LLM | LoRA on HP corpus | 📅 Phase 6 |
| RL Enemy Agent | PPO/DQN | 📅 Phase 4 |
| Stable Diffusion | Scene images | 📅 Phase 3 |
| RAG Pipeline | Vector DB + embeddings | 📅 Phase 2 |

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
│   ├── combat_ai.py        # Enemy AI (RL-ready scaffold)
│   ├── emotion.py          # Player emotion classifier
│   ├── memory.py           # Story memory management
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

### 📅 Phase 2 (Next)
- Real RAG pipeline with vector DB
- Semantic search over HP corpus
- Long-term NPC memory

### 📅 Phase 3
- Stable Diffusion scene images
- Style: HP illustrated book aesthetic

### 📅 Phase 4
- PPO/DQN RL enemy agent
- Learns player patterns over time

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
| POST | `/api/combat` | Resolve combat round |
| GET | `/api/world-state` | Get D3.js graph data |
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
