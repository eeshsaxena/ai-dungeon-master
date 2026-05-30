/**
 * narrator.js — LLM Narrator Interface
 * Handles communication with the FastAPI backend for narrative generation.
 * Supports HTTP REST and WebSocket streaming.
 */

const Narrator = (() => {
  const API_BASE = 'http://localhost:8000';
  let ws = null;
  let wsConnected = false;
  let sessionId = null;

  // ── WebSocket Connection ────────────────────────────────────────────────
  function connectWebSocket(sid, onMessage, onConnect, onError) {
    sessionId = sid;
    const wsUrl = `ws://localhost:8000/ws/${sid}`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      wsConnected = true;
      console.log('[Narrator] WebSocket connected');
      // Ping to confirm
      ws.send(JSON.stringify({ type: 'ping' }));
      onConnect && onConnect();
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage && onMessage(data);
      } catch (e) {
        console.error('[Narrator] WS parse error', e);
      }
    };

    ws.onerror = (e) => {
      wsConnected = false;
      console.warn('[Narrator] WebSocket error, falling back to HTTP');
      onError && onError(e);
    };

    ws.onclose = () => {
      wsConnected = false;
      console.log('[Narrator] WebSocket closed');
    };
  }

  function disconnectWebSocket() {
    if (ws) { ws.close(); ws = null; wsConnected = false; }
  }

  // ── HTTP API Calls ──────────────────────────────────────────────────────
  async function checkHealth() {
    try {
      const r = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(3000) });
      return await r.json();
    } catch {
      return null;
    }
  }

  async function createSession(playerName, house) {
    const params = new URLSearchParams({ player_name: playerName, house });
    const r = await fetch(`${API_BASE}/api/session/create?${params}`, { method: 'POST' });
    if (!r.ok) throw new Error('Failed to create session');
    return await r.json();
  }

  async function narrate(sessionId, playerInput, playerStats, currentLocation, difficulty = 'medium') {
    const body = {
      session_id: sessionId,
      player_input: playerInput,
      player_stats: playerStats,
      current_location: currentLocation,
      difficulty
    };

    // Try WebSocket first
    if (wsConnected && ws) {
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('WS timeout')), 30000);
        const originalOnMessage = ws.onmessage;
        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.type === 'narrative') {
            clearTimeout(timeout);
            ws.onmessage = originalOnMessage;
            resolve(data.data);
          } else if (data.type === 'error') {
            clearTimeout(timeout);
            ws.onmessage = originalOnMessage;
            reject(new Error(data.message));
          }
          // Pass typing indicators through
          if (data.type === 'typing') {
            originalOnMessage({ data: JSON.stringify(data) });
          }
        };
        ws.send(JSON.stringify({ type: 'narrate', input: playerInput, difficulty }));
      }).catch(() => narrateHTTP(body));
    }

    return narrateHTTP(body);
  }

  async function narrateHTTP(body) {
    const r = await fetch(`${API_BASE}/api/narrate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(60000)
    });
    if (!r.ok) throw new Error(`Narrator error: ${r.status}`);
    return await r.json();
  }

  async function spawnEnemy(archetype = 'Death Eater', levelScaling = 1.0) {
    const params = new URLSearchParams({ archetype, level_scaling: levelScaling });
    const r = await fetch(`${API_BASE}/api/spawn-enemy?${params}`, { method: 'POST' });
    return await r.json();
  }

  async function resolveCombat(combatRequest) {
    const r = await fetch(`${API_BASE}/api/combat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(combatRequest)
    });
    return await r.json();
  }

  async function getWorldState(sessionId) {
    const params = sessionId ? `?session_id=${sessionId}` : '';
    const r = await fetch(`${API_BASE}/api/world-state${params}`);
    return await r.json();
  }

  async function getQuests() {
    const r = await fetch(`${API_BASE}/api/quests`);
    return await r.json();
  }

  async function generateScene(locationId, situation = '', mood = 'mysterious') {
    const params = new URLSearchParams({ location_id: locationId, situation, mood });
    const r = await fetch(`${API_BASE}/api/generate-scene?${params}`, { method: 'POST' });
    return await r.json();
  }

  // ── Save / Load ─────────────────────────────────────────────────────────

  // ── Dynamic Quests ───────────────────────────────────────────────────────

  async function generateQuest(sessionId, locationId, force = false) {
    const params = new URLSearchParams({ session_id: sessionId, location_id: locationId, force });
    const r = await fetch(`${API_BASE}/api/generate-quest?${params}`, { method: 'POST' });
    if (!r.ok) throw new Error(`Quest gen failed: ${r.status}`);
    return await r.json();
  }

  async function getDynamicQuests(sessionId) {
    const r = await fetch(`${API_BASE}/api/dynamic-quests/${sessionId}`);
    return await r.json();
  }

  async function saveGame(sessionId) {
    const r = await fetch(`${API_BASE}/api/save/${sessionId}`, { method: 'POST' });
    if (!r.ok) throw new Error(`Save failed: ${r.status}`);
    return await r.json();
  }

  async function listSaves() {
    const r = await fetch(`${API_BASE}/api/saves`);
    if (!r.ok) throw new Error(`List saves failed: ${r.status}`);
    return await r.json();
  }

  async function loadGame(filename) {
    const params = new URLSearchParams({ filename });
    const r = await fetch(`${API_BASE}/api/load?${params}`, { method: 'POST' });
    if (!r.ok) throw new Error(`Load failed: ${r.status}`);
    return await r.json();
  }

  async function deleteSave(filename) {
    const r = await fetch(`${API_BASE}/api/save/${encodeURIComponent(filename)}`, { method: 'DELETE' });
    if (!r.ok) throw new Error(`Delete failed: ${r.status}`);
    return await r.json();
  }

  async function classifyEmotion(sessionId, recentInputs, turnsWithoutProgress = 0) {
    const body = {
      session_id: sessionId,
      recent_inputs: recentInputs,
      turns_without_progress: turnsWithoutProgress,
      time_between_inputs_seconds: 30
    };
    const r = await fetch(`${API_BASE}/api/classify-emotion`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    return await r.json();
  }

  // ── Mock Narrator (offline mode) ────────────────────────────────────────
  const MOCK_OPENING = {
    narrative: `The heavy oak door of the Three Broomsticks swings open before you, releasing a warm fog of Butterbeer steam and pipe smoke into the cold Hogsmeade night. Floating candles drift overhead, their amber light catching the brass fixtures and the steam rising from dozens of clay mugs.

Every table is occupied by the usual mix: Hogwarts professors nursing their drinks, Hogsmeade shopkeepers unwinding after market day, and a few figures in traveling cloaks whose faces you can't quite see.

*"You there,"* says a voice to your left. Professor Longbottom sits alone in a corner booth, both hands wrapped around a mug that's stopped steaming. His expression is carefully neutral — the expression of someone who doesn't want to alarm the people around them. *"Sit down. I've been hoping someone trustworthy would walk through that door."*

Behind the bar, Madam Rosmerta polishes a glass and watches your exchange with eyes that miss absolutely nothing.`,
    atmosphere: 'warm',
    suggested_actions: [
      'Sit with Professor Longbottom',
      'Order a Butterbeer first',
      'Ask Rosmerta what she knows',
      'Scan the room for suspicious faces'
    ],
    scene_prompt: 'Three Broomsticks interior, warm candlelight, Butterbeer, Harry Potter',
    world_state_updates: {},
    detected_emotion: 'neutral',
    difficulty_adjustment: 'medium'
  };

  function getMockResponse(input) {
    const responses = [
      {
        narrative: `You lean across the table, keeping your voice low. Neville's jaw tightens.\n\n*"Three greenhouses,"* he says quietly. *"Greenhouse Seven was first — every Mandrake wilted overnight. Not natural wilting. They were... drained. The soil around them turned crystalline, like frozen glass, though the air was perfectly warm."*\n\nHe slides a folded piece of parchment across the table. A hand-drawn map of the Hogwarts grounds, three locations circled in red ink.\n\n*"It started moving closer to the castle each night. McGonagall knows, but she doesn't want to frighten the students. We need someone to investigate who isn't on the Hogwarts staff — someone the Board of Governors can't question."*\n\nHis eyes hold yours. *"Whatever is doing this, it's intelligent. And it's hungry."*`,
        atmosphere: 'mysterious',
        suggested_actions: ['Examine the map carefully', 'Ask about the crystalline soil', 'Agree to investigate', 'Mention it to the Ministry']
      },
      {
        narrative: `The night air bites as you step out onto the cobblestones of Hogsmeade, parchment tucked inside your robes. The village is quiet — a few lit windows, the distant sound of the Hogwarts Express horn echoing off the mountains.\n\nThe path up to the castle is well-lit by enchanted lanterns, but the Greenhouse wing sits at the far eastern edge of the grounds, where the lanterns thin out and the Forbidden Forest begins to press close.\n\nAs you approach Greenhouse Seven, you notice something immediately: the glass panels are fogged from within, though it's not warm enough tonight for that. The door is ajar — and there's a smell you can't quite name. Cold metal and old lightning.\n\nInside, something moves.`,
        atmosphere: 'dangerous',
        suggested_actions: ['Enter the greenhouse carefully', 'Cast Lumos and look through the glass', 'Circle the building first', 'Send a Patronus message to Neville']
      },
      {
        narrative: `Inside the greenhouse, the damage is worse than the map suggested. Every plant on the left row has turned completely black — not from rot, but from something more absolute. When you reach out to touch a blackened leaf, it crumbles to a fine, silver-grey dust that rises slowly rather than falling.\n\nOn the far wall, three symbols pulse with a dying light. They're not runes you recognize from standard Ancient Runes curriculum — they're older, angular, and they seem to pull at something behind your eyes when you look at them too long.\n\nA sound. Behind you, near the door — not breathing, exactly. More like the absence of sound in the shape of breathing.`,
        atmosphere: 'dangerous',
        suggested_actions: ['Cast Stupefy toward the sound', 'Spin around with wand ready', 'Press your back against the wall', 'Cast Lumos Maxima to flood the room with light']
      }
    ];
    const idx = Math.floor(Math.random() * responses.length);
    return { ...responses[idx], world_state_updates: {}, scene_prompt: '', detected_emotion: 'curious', difficulty_adjustment: 'medium' };
  }

  return {
    connectWebSocket,
    disconnectWebSocket,
    checkHealth,
    createSession,
    narrate,
    spawnEnemy,
    resolveCombat,
    getWorldState,
    getQuests,
    generateScene,
    classifyEmotion,
    generateQuest,
    getDynamicQuests,
    saveGame,
    listSaves,
    loadGame,
    deleteSave,
    getMockResponse,
    MOCK_OPENING,
    get isConnected() { return wsConnected; }
  };
})();
