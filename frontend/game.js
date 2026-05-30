/**
 * game.js — Core Game Engine
 * Orchestrates the entire AI Dungeon Master game loop.
 * Handles: loading, character creation, narrative flow, scene rendering, UI state.
 */

// ═══════════════════════════════════════════════════════════════════════════
// GAME STATE
// ═══════════════════════════════════════════════════════════════════════════
const GameState = {
  sessionId: null,
  backendConnected: false,
  playerStats: {
    name: 'Unnamed Witch/Wizard',
    house: 'Unselected',
    level: 1,
    xp: 0,
    xp_to_next: 300,
    title: 'First-Year',
    hp: 100,
    max_hp: 100,
    mana: 80,
    max_mana: 80,
    galleons: 50,
    spells_known: ['Expelliarmus', 'Stupefy', 'Protego', 'Lumos', 'Accio', 'Expecto Patronum'],
    inventory: [],
    active_quests: ['quest_001'],
    completed_quests: [],
    reputation: {
      'Ministry of Magic': 0,
      'Order of the Phoenix': 10,
      'Hollow Circle': -100,
      'Centaur Herd': 0
    },
    current_location: 'loc_001',
    emotion_state: 'neutral',
    turns_played: 0
  },
  currentLocation: 'loc_001',
  difficulty: 'medium',
  turnCount: 1,
  inCombat: false,
  isLoading: false,
  recentInputs: [],
  selectedHouse: null,

  // Location data cache
  locations: {
    loc_001: { name: 'The Three Broomsticks', atmosphere: 'warm' },
    loc_002: { name: 'Hogwarts Castle', atmosphere: 'magical' },
    loc_003: { name: 'Knockturn Alley', atmosphere: 'dangerous' },
    loc_004: { name: 'The Forbidden Forest', atmosphere: 'ancient' },
    loc_005: { name: 'Ministry of Magic', atmosphere: 'mysterious' },
    loc_006: { name: 'Azkaban', atmosphere: 'cold' },
    loc_007: { name: "Godric's Hollow", atmosphere: 'melancholic' }
  },

  // Quests cache
  quests: [],
};

// XP required to REACH each level (matches backend progression.py, 20 levels)
const XP_TABLE = [
  0,    // 1
  300,  // 2
  500,  // 3
  800,  // 4
  1100, // 5
  1450, // 6
  1850, // 7
  2250, // 8
  2700, // 9
  3150, // 10
  3650, // 11
  4150, // 12
  4700, // 13
  5250, // 14
  5800, // 15
  6400, // 16
  7000, // 17
  7650, // 18
  8300, // 19
  8950, // 20
];

// ═══════════════════════════════════════════════════════════════════════════
// LOADING SEQUENCE
// ═══════════════════════════════════════════════════════════════════════════
async function runLoadingSequence() {
  const bar = document.getElementById('loading-bar');
  const text = document.getElementById('loading-text');

  const steps = [
    [10, 'Summoning the magic...'],
    [25, 'Consulting the Sorting Hat...'],
    [40, 'Opening the Marauder\'s Map...'],
    [55, 'Connecting to Ollama...'],
    [70, 'Loading Wizarding World lore...'],
    [85, 'Preparing your adventure...'],
    [100, 'Ready! ⚡']
  ];

  for (const [pct, msg] of steps) {
    bar.style.width = pct + '%';
    text.textContent = msg;
    await sleep(300 + Math.random() * 200);
  }

  // Try to connect to backend
  await checkBackendConnection();

  await sleep(400);

  // Hide loading screen
  const loading = document.getElementById('loading-screen');
  loading.style.opacity = '0';
  await sleep(700);
  loading.style.display = 'none';

  // Show character creation
  document.getElementById('char-creation-modal').classList.remove('hidden');
}

async function checkBackendConnection() {
  try {
    const health = await Narrator.checkHealth();
    if (health && health.status === 'healthy') {
      GameState.backendConnected = true;
      updateModelBadge(true, health.ollama_model || 'llama3.2');
      return true;
    }
  } catch {}
  GameState.backendConnected = false;
  updateModelBadge(false, 'Offline Mode (Mock Narrator)');
  return false;
}

function updateModelBadge(connected, modelName) {
  const dot = document.querySelector('.model-dot');
  const name = document.getElementById('model-name');
  if (dot) {
    dot.className = 'model-dot ' + (connected ? 'connected' : 'error');
  }
  if (name) name.textContent = connected ? `Ollama: ${modelName}` : modelName;
}

// ═══════════════════════════════════════════════════════════════════════════
// CHARACTER CREATION
// ═══════════════════════════════════════════════════════════════════════════
function initCharacterCreation() {
  const nameInput = document.getElementById('player-name-input');
  const beginBtn = document.getElementById('begin-adventure-btn');
  const houseBtns = document.querySelectorAll('.house-btn');

  nameInput.addEventListener('input', () => { validateCharCreation(); });

  houseBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      houseBtns.forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
      GameState.selectedHouse = btn.dataset.house;
      validateCharCreation();
    });
  });

  beginBtn.addEventListener('click', async () => {
    const name = nameInput.value.trim();
    if (!name || !GameState.selectedHouse) return;
    await startGame(name, GameState.selectedHouse);
  });

  nameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const name = nameInput.value.trim();
      if (name && GameState.selectedHouse) startGame(name, GameState.selectedHouse);
    }
  });

  // Load Game button — opens save list modal
  document.getElementById('load-game-btn').addEventListener('click', async () => {
    await openSaveListModal();
  });

  // Close save list modal
  document.getElementById('close-save-list-btn').addEventListener('click', () => {
    document.getElementById('save-list-modal').classList.add('hidden');
  });
}

async function openSaveListModal() {
  const modal = document.getElementById('save-list-modal');
  const listEl = document.getElementById('save-list-items');
  modal.classList.remove('hidden');
  listEl.innerHTML = '<p class="saves-empty">Loading saves...</p>';

  try {
    const data = await Narrator.listSaves();
    const saves = data.saves || [];
    if (saves.length === 0) {
      listEl.innerHTML = '<p class="saves-empty">No saved games found.</p>';
      return;
    }
    listEl.innerHTML = '';
    saves.forEach(save => {
      const item = document.createElement('div');
      item.className = 'save-item';
      const ts = save.saved_at ? new Date(save.saved_at * 1000).toLocaleString() : 'Unknown time';
      const HOUSE_ICONS = { Gryffindor: '🦁', Hufflepuff: '🦡', Ravenclaw: '🦅', Slytherin: '🐍', Unselected: '⚡' };
      item.innerHTML = `
        <div class="save-item-info">
          <div class="save-item-name">${HOUSE_ICONS[save.house] || '⚡'} ${save.player_name}</div>
          <div class="save-item-meta">${save.title} · Turn ${save.turns} · ${ts}</div>
        </div>
        <div class="save-item-level">Lv ${save.level}<br>${save.xp} XP</div>
        <button class="save-delete-btn" data-filename="${save.filename}" title="Delete save">🗑</button>
      `;
      // Load on click (not on delete button)
      item.querySelector('.save-item-info, .save-item-level').addEventListener
        ? item.addEventListener('click', async (e) => {
            if (e.target.classList.contains('save-delete-btn')) return;
            await loadSaveFile(save.filename);
          })
        : null;
      item.addEventListener('click', async (e) => {
        if (e.target.classList.contains('save-delete-btn')) {
          e.stopPropagation();
          if (!confirm(`Delete save for ${save.player_name}?`)) return;
          try { await Narrator.deleteSave(save.filename); } catch {}
          await openSaveListModal();
          return;
        }
        await loadSaveFile(save.filename);
      });
      listEl.appendChild(item);
    });
  } catch {
    listEl.innerHTML = '<p class="saves-empty">Could not load saves (backend offline?)</p>';
  }
}

async function loadSaveFile(filename) {
  const modal = document.getElementById('save-list-modal');
  modal.classList.add('hidden');
  document.getElementById('char-creation-modal').classList.add('hidden');

  try {
    const data = await Narrator.loadGame(filename);
    GameState.sessionId = data.session_id;
    GameState.backendConnected = true;
    Object.assign(GameState.playerStats, data.player);
    GameState.currentLocation = data.player.current_location || 'loc_001';
    GameState.turnCount = data.player.turns_played || 1;

    document.getElementById('game-container').classList.remove('hidden');
    updatePlayerUI();
    renderSpellbook();
    renderReputation();
    initKnowledgeGraph();
    try {
      const questData = await Narrator.getQuests();
      GameState.quests = questData.quests || [];
    } catch {}
    renderQuestLog();
    renderScene(GameState.currentLocation, 'mysterious');

    addNarrativeEntry('dm', `✨ *Welcome back, ${data.player.name}!* Your adventure continues where you left off...`, 'warm');
    checkSdStatus();
    document.getElementById('player-input').focus();
  } catch (e) {
    document.getElementById('char-creation-modal').classList.remove('hidden');
    alert('Failed to load save: ' + e.message);
  }
}

function validateCharCreation() {
  const name = document.getElementById('player-name-input').value.trim();
  const beginBtn = document.getElementById('begin-adventure-btn');
  beginBtn.disabled = !(name.length >= 2 && GameState.selectedHouse);
}

async function startGame(playerName, house) {
  // Update game state
  GameState.playerStats.name = playerName;
  GameState.playerStats.house = house;

  // Close modal
  document.getElementById('char-creation-modal').classList.add('hidden');

  // Try create server session
  if (GameState.backendConnected) {
    try {
      const session = await Narrator.createSession(playerName, house);
      GameState.sessionId = session.session_id;
    } catch {
      GameState.sessionId = 'local-' + Date.now();
    }
  } else {
    GameState.sessionId = 'local-' + Date.now();
  }

  // Show game
  document.getElementById('game-container').classList.remove('hidden');

  // Initialize UI
  updatePlayerUI();
  renderSpellbook();
  renderReputation();
  initKnowledgeGraph();

  // Load quests
  try {
    const questData = await Narrator.getQuests();
    GameState.quests = questData.quests || [];
  } catch {}
  renderQuestLog();

  // Draw opening scene
  renderScene('loc_001', 'warm');

  // Show opening narrative
  await showOpeningNarrative(playerName, house);

  // Check if SD pipeline is warming up
  checkSdStatus();

  // Focus input
  document.getElementById('player-input').focus();
}

// ═══════════════════════════════════════════════════════════════════════════
// OPENING NARRATIVE
// ═══════════════════════════════════════════════════════════════════════════
async function showOpeningNarrative(playerName, house) {
  const houseEmojis = {
    Gryffindor: '🦁', Hufflepuff: '🦡', Ravenclaw: '🦅', Slytherin: '🐍', Unselected: '⚡'
  };

  const houseColors = {
    Gryffindor: '#AE0001', Hufflepuff: '#FFD800', Ravenclaw: '#4A90D9', Slytherin: '#1A472A', Unselected: '#D4AF37'
  };

  addNarrativeEntry('dm', `⚡ *A new wizard enters the story...*\n\n**${playerName}** — ${houseEmojis[house] || '⚡'} ${house}`);

  await sleep(600);

  // Try backend for opening narrative
  if (GameState.backendConnected) {
    setTyping(true);
    try {
      const response = await Narrator.narrate(
        GameState.sessionId,
        `I am ${playerName} of ${house} house. I've just arrived at the Three Broomsticks in Hogsmeade.`,
        GameState.playerStats,
        'loc_001',
        GameState.difficulty
      );
      setTyping(false);
      addNarrativeEntry('dm', response.narrative, response.atmosphere);
      updateSuggestions(response.suggested_actions || []);
      handleWorldUpdates(response.world_state_updates || {});
      updateEmotionUI(response.detected_emotion, response.difficulty_adjustment);
    } catch {
      setTyping(false);
      showMockOpening();
    }
  } else {
    await sleep(800);
    setTyping(true);
    await sleep(1200);
    setTyping(false);
    showMockOpening();
  }
}

function showMockOpening() {
  const mock = Narrator.MOCK_OPENING;
  addNarrativeEntry('dm', mock.narrative, mock.atmosphere);
  updateSuggestions(mock.suggested_actions);
}

// ═══════════════════════════════════════════════════════════════════════════
// GAME LOOP — PLAYER INPUT
// ═══════════════════════════════════════════════════════════════════════════
function initInputHandlers() {
  const input = document.getElementById('player-input');
  const sendBtn = document.getElementById('send-btn');
  const charCount = document.getElementById('char-count');

  input.addEventListener('input', () => {
    charCount.textContent = `${input.value.length}/500`;
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitPlayerInput();
    }
  });

  sendBtn.addEventListener('click', () => submitPlayerInput());
}

async function submitPlayerInput() {
  const input = document.getElementById('player-input');
  const text = input.value.trim();
  if (!text || GameState.isLoading) return;

  input.value = '';
  document.getElementById('char-count').textContent = '0/500';

  // Record for emotion tracking
  EmotionTracker.recordInput(text);
  GameState.recentInputs.push(text);
  if (GameState.recentInputs.length > 10) GameState.recentInputs = GameState.recentInputs.slice(-10);

  // Show player's action
  addNarrativeEntry('player', text);

  // Increment turn
  GameState.turnCount++;
  GameState.playerStats.turns_played++;
  document.getElementById('turn-count').textContent = GameState.turnCount;

  // Check for combat triggers
  const combatTrigger = checkCombatTrigger(text);
  if (combatTrigger) {
    triggerCombat(combatTrigger);
    return;
  }

  // Get narrative
  await processPlayerAction(text);
}

async function processPlayerAction(text) {
  GameState.isLoading = true;
  setTyping(true);

  try {
    let response;
    if (GameState.backendConnected) {
      try {
        response = await Narrator.narrate(
          GameState.sessionId,
          text,
          GameState.playerStats,
          GameState.currentLocation,
          GameState.difficulty
        );
      } catch {
        GameState.backendConnected = false;
        updateModelBadge(false, 'Connection Lost (Mock Mode)');
        response = Narrator.getMockResponse(text);
      }
    } else {
      await sleep(800 + Math.random() * 600);
      response = Narrator.getMockResponse(text);
    }

    setTyping(false);
    addNarrativeEntry('dm', response.narrative, response.atmosphere);
    updateSuggestions(response.suggested_actions || []);
    handleWorldUpdates(response.world_state_updates || {});
    updateEmotionUI(response.detected_emotion, response.difficulty_adjustment);

    // Update scene
    if (response.atmosphere) {
      updateSceneAtmosphere(response.atmosphere);
    }

  } catch (e) {
    setTyping(false);
    addNarrativeEntry('dm', '*The magic wavers momentarily... [Error connecting to narrator]*', 'mysterious');
    console.error('[Game] Narrative error:', e);
  }

  GameState.isLoading = false;
}

// ═══════════════════════════════════════════════════════════════════════════
// COMBAT TRIGGERS
// ═══════════════════════════════════════════════════════════════════════════
const COMBAT_KEYWORDS = {
  'death eater': 'Death Eater',
  'dementor': 'Dementor',
  'fight': 'Dark Wizard',
  'attack': 'Dark Wizard',
  'combat': 'Dark Wizard',
  'troll': 'Troll',
  'acromantula': 'Acromantula',
  'spider': 'Acromantula',
  'werewolf': 'Werewolf',
  'boggart': 'Boggart',
  'fight them': 'Death Eater',
  'i attack': 'Death Eater'
};

function checkCombatTrigger(input) {
  const lower = input.toLowerCase();
  for (const [keyword, archetype] of Object.entries(COMBAT_KEYWORDS)) {
    if (lower.includes(keyword)) return archetype;
  }
  return null;
}

async function triggerCombat(archetype) {
  let enemy;
  try {
    enemy = await Narrator.spawnEnemy(archetype, 1.0);
  } catch {
    // Mock enemy
    enemy = {
      id: 'enemy_001',
      name: archetype,
      archetype,
      hp: 120,
      max_hp: 120,
      attack: 35,
      defense: 20,
      special_ability: 'Dark Magic',
      weakness: 'Expelliarmus',
      personality: 'aggressive'
    };
  }

  addNarrativeEntry('dm', `*A ${enemy.name} emerges from the shadows! You raise your wand...*\n\n⚔️ **Combat begins!** Choose your spell wisely.`, 'dangerous');

  CombatSystem.startCombat(
    enemy,
    GameState.sessionId,
    GameState.playerStats,
    GameState.playerStats.spells_known,
    (result) => {
      GameState.inCombat = false;
      if (result.player_won) {
        GameState.playerStats.xp += result.xp_gained || 0;
        if (result.loot?.length > 0) {
          result.loot.forEach(item => GameState.playerStats.inventory.push(item));
          updateInventory();
        }
        addNarrativeEntry('dm', `*You stand victorious! The ${enemy.name} is defeated.* ✨`, 'warm');
        // Server-authoritative level-up (or client fallback in offline mode)
        if (result.level_up) {
          applyServerLevelUp(result);
        } else {
          checkLevelUp();
        }
      } else {
        GameState.playerStats.hp = 10;
        addNarrativeEntry('dm', `*You barely escape with your life... You need to recover.* 💀`, 'dangerous');
      }
      updatePlayerUI();
    }
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// WORLD STATE UPDATES
// ═══════════════════════════════════════════════════════════════════════════
function handleWorldUpdates(updates) {
  if (!updates) return;

  if (updates.new_location) {
    const locId = updates.new_location;
    GameState.currentLocation = locId;
    GameState.playerStats.current_location = locId;
    const locData = GameState.locations[locId] || { name: locId, atmosphere: 'mysterious' };
    updateLocationBadge(locData.name);
    renderScene(locId, locData.atmosphere);
    KnowledgeGraph.updatePlayerLocation(locId);
  }

  if (updates.spell_cast) {
    CombatSystem.showSpellCast(updates.spell_cast);
  }

  if (updates.combat_start) {
    triggerCombat(updates.combat_start);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// SCENE RENDERING
// ═══════════════════════════════════════════════════════════════════════════
const SCENE_PALETTES = {
  loc_001: { bg: '#2C1810', mid: '#5C3A2A', accent: '#D4A017', particles: ['✨', '🕯️'] },
  loc_002: { bg: '#0A1628', mid: '#1A2E50', accent: '#4A90D9', particles: ['⭐', '🌙', '✨'] },
  loc_003: { bg: '#1A0A2E', mid: '#2D1550', accent: '#8B5CF6', particles: ['💀', '🔮'] },
  loc_004: { bg: '#0D1F12', mid: '#1A3A20', accent: '#22C55E', particles: ['🌿', '🍄', '🌑'] },
  loc_005: { bg: '#1C1C2E', mid: '#2C2C45', accent: '#60A5FA', particles: ['📝', '⚡'] },
  loc_006: { bg: '#0F0F23', mid: '#1A1A35', accent: '#6366F1', particles: ['💀', '❄️'] },
  loc_007: { bg: '#1F1B18', mid: '#2E2A25', accent: '#78716C', particles: ['🪦', '🕊️'] }
};

function renderScene(locationId, atmosphere) {
  const canvas = document.getElementById('scene-canvas');
  const locationTag = document.getElementById('scene-location-tag');
  const atmosphereTag = document.getElementById('scene-atmosphere-tag');

  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || 600;
  const H = canvas.offsetHeight || 180;
  canvas.width = W;
  canvas.height = H;

  const palette = SCENE_PALETTES[locationId] || SCENE_PALETTES.loc_001;

  // Draw gradient background
  const grad = ctx.createLinearGradient(0, 0, W, H);
  grad.addColorStop(0, palette.bg);
  grad.addColorStop(0.5, palette.mid);
  grad.addColorStop(1, palette.bg);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  // Draw atmospheric elements
  drawSceneAtmosphere(ctx, W, H, palette, locationId);

  // Update overlays
  const locData = GameState.locations[locationId];
  if (locationTag) locationTag.textContent = locData ? locData.name : locationId;

  const atmosphereIcons = {
    warm: '✨ Warm', dangerous: '⚠️ Dangerous', mysterious: '🔮 Mysterious',
    ancient: '🏛️ Ancient', magical: '✨ Magical', cold: '❄️ Cold', melancholic: '🌧️ Melancholic'
  };
  if (atmosphereTag) atmosphereTag.textContent = atmosphereIcons[atmosphere] || '✨ Magical';

  // Layer the richer backend-generated scene art over the canvas fallback
  loadSceneImage(locationId, atmosphere);
}

// Backend scene art is fetched once per location and cached, so the periodic
// canvas animation loop never re-hits the API.
const sceneImageCache = {};
let sceneImageToken = 0;

function loadSceneImage(locationId, mood) {
  const imgEl = document.getElementById('scene-image');
  if (!imgEl || !GameState.backendConnected || typeof Narrator === 'undefined') return;

  const token = ++sceneImageToken;
  const apply = (dataUrl) => {
    if (token !== sceneImageToken) return;
    imgEl.src = dataUrl;
    imgEl.classList.add('loaded');
    setSdGenerating(false);
  };

  if (sceneImageCache[locationId]) {
    apply(sceneImageCache[locationId]);
    return;
  }

  imgEl.classList.remove('loaded');
  setSdGenerating(true, 'Generating scene art...');
  Narrator.generateScene(locationId, '', mood)
    .then((res) => {
      if (res && res.image_base64) {
        sceneImageCache[locationId] = res.image_base64;
        apply(res.image_base64);
      } else {
        setSdGenerating(false);
      }
    })
    .catch(() => { setSdGenerating(false); });
}

function setSdGenerating(active, text = 'Generating scene art...') {
  const overlay = document.getElementById('sd-generating-overlay');
  const textEl  = document.getElementById('sd-generating-text');
  if (!overlay) return;
  if (active) {
    if (textEl) textEl.textContent = text;
    overlay.classList.remove('hidden');
  } else {
    overlay.classList.add('hidden');
  }
}

// Poll /api/sd-status once on startup to show pipeline warmup state in model badge
async function checkSdStatus() {
  if (!GameState.backendConnected) return;
  try {
    const r = await fetch('http://localhost:8000/api/sd-status');
    const data = await r.json();
    if (data.provider === 'diffusers' && data.loading) {
      setSdGenerating(true, `⚙️ Loading SD model (${(data.model || '').split('/').pop()})...`);
      // Keep polling until ready
      const poll = setInterval(async () => {
        try {
          const r2 = await fetch('http://localhost:8000/api/sd-status');
          const d2 = await r2.json();
          if (!d2.loading) {
            clearInterval(poll);
            setSdGenerating(false);
            if (d2.ready) {
              // Reload current scene with SD
              delete sceneImageCache[GameState.currentLocation];
              loadSceneImage(GameState.currentLocation, 'mysterious');
            }
          }
        } catch { clearInterval(poll); setSdGenerating(false); }
      }, 3000);
    }
  } catch {}
}

function drawSceneAtmosphere(ctx, W, H, palette, locationId) {
  // Stars / particles
  const numParticles = 40;
  for (let i = 0; i < numParticles; i++) {
    const x = Math.random() * W;
    const y = Math.random() * H;
    const size = Math.random() * 2 + 0.5;
    const alpha = Math.random() * 0.6 + 0.1;

    ctx.fillStyle = palette.accent + Math.floor(alpha * 255).toString(16).padStart(2, '0');
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fill();
  }

  // Atmospheric glow
  const radGrad = ctx.createRadialGradient(W / 2, H * 0.3, 0, W / 2, H * 0.3, W * 0.4);
  radGrad.addColorStop(0, palette.accent + '25');
  radGrad.addColorStop(1, 'transparent');
  ctx.fillStyle = radGrad;
  ctx.fillRect(0, 0, W, H);

  // Ground fog / vignette
  const fogGrad = ctx.createLinearGradient(0, H * 0.6, 0, H);
  fogGrad.addColorStop(0, 'transparent');
  fogGrad.addColorStop(1, palette.bg + 'CC');
  ctx.fillStyle = fogGrad;
  ctx.fillRect(0, 0, W, H);

  // Side vignettes
  const leftVig = ctx.createLinearGradient(0, 0, W * 0.3, 0);
  leftVig.addColorStop(0, palette.bg + 'AA');
  leftVig.addColorStop(1, 'transparent');
  ctx.fillStyle = leftVig;
  ctx.fillRect(0, 0, W, H);

  const rightVig = ctx.createLinearGradient(W, 0, W * 0.7, 0);
  rightVig.addColorStop(0, palette.bg + 'AA');
  rightVig.addColorStop(1, 'transparent');
  ctx.fillStyle = rightVig;
  ctx.fillRect(0, 0, W, H);

  // Location-specific elements
  drawLocationDetails(ctx, W, H, palette, locationId);
}

function drawLocationDetails(ctx, W, H, palette, locationId) {
  ctx.save();

  if (locationId === 'loc_001') {
    // Candles
    for (let i = 0; i < 5; i++) {
      const x = (W / 6) * (i + 1);
      const y = H * 0.4 + Math.sin(Date.now() / 1000 + i) * 3;
      ctx.fillStyle = '#FFD700AA';
      ctx.font = '16px serif';
      ctx.fillText('🕯️', x - 8, y);
    }
  } else if (locationId === 'loc_002') {
    // Castle silhouette
    ctx.fillStyle = '#0A1628DD';
    ctx.fillRect(W * 0.3, H * 0.1, W * 0.4, H * 0.7);
    // Tower tops
    for (let i = 0; i < 4; i++) {
      ctx.fillRect(W * (0.3 + i * 0.12), H * 0.05, W * 0.05, H * 0.2);
    }
    // Moon
    ctx.fillStyle = '#FFFDE7AA';
    ctx.beginPath();
    ctx.arc(W * 0.8, H * 0.2, 20, 0, Math.PI * 2);
    ctx.fill();
  } else if (locationId === 'loc_004') {
    // Tree silhouettes
    for (let i = 0; i < 6; i++) {
      const x = (W / 7) * (i + 0.5);
      ctx.fillStyle = '#0D1F12';
      ctx.fillRect(x - 3, H * 0.3, 6, H * 0.7);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x - 20, H * 0.35);
      ctx.lineTo(x + 20, H * 0.35);
      ctx.fill();
    }
  }

  ctx.restore();
}

function updateSceneAtmosphere(atmosphere) {
  const tag = document.getElementById('scene-atmosphere-tag');
  const atmosphereIcons = {
    warm: '✨ Warm', dangerous: '⚠️ Dangerous', mysterious: '🔮 Mysterious',
    ancient: '🏛️ Ancient', magical: '✨ Magical', cold: '❄️ Cold', melancholic: '🌧️ Melancholic'
  };
  if (tag) tag.textContent = atmosphereIcons[atmosphere] || '✨ Magical';
}

// ═══════════════════════════════════════════════════════════════════════════
// UI UPDATE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════
function updatePlayerUI() {
  const p = GameState.playerStats;

  // Name, title & house
  document.getElementById('player-name-display').textContent = p.name;
  document.getElementById('player-level').textContent = p.level;
  // Title badge — create once, update thereafter
  let titleEl = document.getElementById('player-title-badge');
  if (!titleEl) {
    titleEl = document.createElement('div');
    titleEl.id = 'player-title-badge';
    titleEl.className = 'player-title-badge';
    const nameEl = document.getElementById('player-name-display');
    nameEl.parentNode.insertBefore(titleEl, nameEl.nextSibling);
  }
  titleEl.textContent = p.title || 'First-Year';

  // House badge
  const HOUSE_CONFIG = {
    Gryffindor: { icon: '🦁', color: '#AE0001' },
    Hufflepuff: { icon: '🦡', color: '#FFD800' },
    Ravenclaw:  { icon: '🦅', color: '#4A90D9' },
    Slytherin:  { icon: '🐍', color: '#1A472A' },
    Unselected: { icon: '⚡', color: '#D4AF37' }
  };

  const houseConf = HOUSE_CONFIG[p.house] || HOUSE_CONFIG.Unselected;
  document.getElementById('house-badge-icon').textContent = houseConf.icon;
  document.getElementById('house-badge-name').textContent = p.house;
  document.getElementById('house-badge').style.borderColor = houseConf.color + '60';

  // HP
  const hpPct = (p.hp / p.max_hp) * 100;
  document.getElementById('hp-bar').style.width = hpPct + '%';
  document.getElementById('hp-numbers').textContent = `${p.hp}/${p.max_hp}`;
  const hpBar = document.getElementById('hp-bar');
  if (hpPct < 25) { hpBar.style.background = 'linear-gradient(90deg, #8B0000, #C0392B)'; }
  else if (hpPct < 50) { hpBar.style.background = 'linear-gradient(90deg, #922B21, #E74C3C)'; }
  else { hpBar.style.background = 'linear-gradient(90deg, #C0392B, #E74C3C)'; }

  // Mana
  const manaPct = (p.mana / p.max_mana) * 100;
  document.getElementById('mana-bar').style.width = manaPct + '%';
  document.getElementById('mana-numbers').textContent = `${p.mana}/${p.max_mana}`;

  // XP — use server-provided xp_to_next when available
  const currentLevelXp = XP_TABLE[p.level - 1] || 0;
  const nextLevelXp = XP_TABLE[p.level] || 99999;
  const xpInLevel = p.xp - currentLevelXp;
  const xpNeeded = nextLevelXp - currentLevelXp;
  const xpPct = xpNeeded > 0 ? (xpInLevel / xpNeeded) * 100 : 100;
  document.getElementById('xp-bar').style.width = Math.min(100, xpPct) + '%';
  if (p.level >= 20) {
    document.getElementById('xp-text').textContent = `${p.xp} XP — MAX LEVEL`;
  } else {
    document.getElementById('xp-text').textContent = `${p.xp} / ${nextLevelXp} XP`;
  }

  // Galleons
  document.getElementById('galleon-count').textContent = p.galleons;

  // Turn counter (also update quest ticker)
  document.getElementById('turn-count').textContent = GameState.turnCount;
}

function checkLevelUp() {
  // Client-side level-up for offline mode only.
  // When connected to the backend, level-up is driven by combat response data.
  if (GameState.backendConnected) return;
  const p = GameState.playerStats;
  const nextLevelXp = XP_TABLE[p.level] || 99999;
  if (p.xp >= nextLevelXp && p.level < 20) {
    p.level++;
    p.max_hp += 10;
    p.hp = p.max_hp;
    p.max_mana += 5;
    p.mana = p.max_mana;
    showLevelUpToast(p.level, p.title || '', []);
    updatePlayerUI();
  }
}

function applyServerLevelUp(combatResult) {
  if (!combatResult.level_up) return;
  const p = GameState.playerStats;
  if (combatResult.updated_player) {
    // Use the full authoritative player from the server
    Object.assign(p, combatResult.updated_player);
  } else {
    p.level = combatResult.new_level;
    p.title = combatResult.new_title || p.title;
    for (const spell of (combatResult.new_spells || [])) {
      if (!p.spells_known.includes(spell)) p.spells_known.push(spell);
    }
  }
  showLevelUpToast(p.level, p.title, combatResult.new_spells || []);
  renderSpellbook();
  updatePlayerUI();
}

function showLevelUpToast(level, title, newSpells) {
  const toast = document.getElementById('levelup-toast');
  if (!toast) return;
  document.getElementById('levelup-title').textContent = `Level Up! → Level ${level}`;
  document.getElementById('levelup-subtitle').textContent = title || '';
  const spellsEl = document.getElementById('levelup-spells');
  if (newSpells && newSpells.length > 0) {
    spellsEl.textContent = `✨ New spells: ${newSpells.join(', ')}`;
    spellsEl.style.display = '';
  } else {
    spellsEl.style.display = 'none';
  }
  toast.classList.remove('hidden');
  toast.classList.add('show');
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.classList.add('hidden'), 450);
  }, 4000);
}

function renderSpellbook() {
  const list = document.getElementById('spell-list');
  if (!list) return;
  list.innerHTML = '';

  const SPELL_TYPES = {
    'Expelliarmus': 'offensive', 'Stupefy': 'offensive', 'Protego': 'defensive',
    'Expecto Patronum': 'defensive', 'Accio': 'utility', 'Lumos': 'utility',
    'Alohomora': 'utility', 'Sectumsempra': 'offensive', 'Episkey': 'healing',
    'Lumos Maxima': 'utility', 'Wingardium Leviosa': 'utility',
    'Riddikulus': 'utility', 'Fiendfyre': 'offensive', 'Legilimens': 'mental',
    'Occlumens': 'mental'
  };

  const SPELL_MANA = {
    'Expelliarmus': 10, 'Stupefy': 15, 'Protego': 12, 'Expecto Patronum': 40,
    'Accio': 8, 'Lumos': 3, 'Alohomora': 5, 'Sectumsempra': 35,
    'Episkey': 20, 'Lumos Maxima': 15, 'Wingardium Leviosa': 10,
    'Riddikulus': 12, 'Fiendfyre': 60
  };

  GameState.playerStats.spells_known.forEach(spell => {
    const type = SPELL_TYPES[spell] || 'utility';
    const mana = SPELL_MANA[spell] || 10;

    const item = document.createElement('div');
    item.className = `spell-item spell-${type}`;
    item.innerHTML = `
      <div class="spell-dot"></div>
      <span class="spell-name-text">${spell}</span>
      <span class="spell-mana-cost">💙${mana}</span>
    `;
    item.title = `${spell} — ${type} — ${mana} mana`;
    list.appendChild(item);
  });
}

function renderReputation() {
  const list = document.getElementById('reputation-list');
  if (!list) return;
  list.innerHTML = '';

  const rep = GameState.playerStats.reputation;
  Object.entries(rep).forEach(([faction, value]) => {
    const clampedVal = Math.max(-100, Math.min(100, value));
    const isPositive = clampedVal >= 0;
    const pct = Math.abs(clampedVal / 2); // 0–50% width from center

    const repLabel = value > 50 ? '✅ Allied' : value > 0 ? '😊 Friendly' :
                     value > -25 ? '😐 Neutral' : value > -75 ? '😤 Hostile' : '💀 Enemy';

    const item = document.createElement('div');
    item.className = 'rep-item';
    item.innerHTML = `
      <div class="rep-label">
        <span>${faction}</span>
        <span>${repLabel}</span>
      </div>
      <div class="rep-bar-bg">
        <div class="rep-center"></div>
        <div class="rep-bar-fill ${isPositive ? 'rep-positive' : 'rep-negative'}" style="width:${pct}%"></div>
      </div>
    `;
    list.appendChild(item);
  });
}

function renderQuestLog() {
  const list = document.getElementById('quest-list');
  if (!list) return;
  list.innerHTML = '';

  const DIFF_EMOJI = { easy: '🟢', medium: '🟡', hard: '🔴', very_hard: '💀', legendary: '⭐' };

  if (GameState.quests.length === 0) {
    list.innerHTML = '<div style="color:var(--text-muted);font-size:0.75rem;padding:8px;text-align:center;font-style:italic">No quests loaded...</div>';
    return;
  }

  GameState.quests.filter(q => q.status !== 'locked').forEach(quest => {
    const item = document.createElement('div');
    item.className = `quest-item ${quest.status}`;
    item.innerHTML = `
      <div class="quest-title-text">${quest.title}</div>
      <div class="quest-status">
        <span class="quest-difficulty">${DIFF_EMOJI[quest.difficulty] || '⬜'} ${quest.difficulty}</span>
        <span>${quest.status === 'completed' ? '✅ Done' : quest.status === 'active' ? '▶️ Active' : '📋 Available'}</span>
      </div>
    `;
    item.addEventListener('click', () => showQuestDetails(quest));
    list.appendChild(item);

    // Update quest ticker for first active quest
    if (quest.status === 'active') {
      document.getElementById('quest-ticker-text').textContent = `Active Quest: ${quest.title}`;
    }
  });
}

function updateInventory() {
  const grid = document.getElementById('inventory-grid');
  const count = document.getElementById('inventory-count');
  if (!grid) return;

  const items = GameState.playerStats.inventory;
  count.textContent = `${items.length}/12`;

  if (items.length === 0) {
    grid.innerHTML = '<div class="inventory-empty">Your bag is empty...</div>';
    return;
  }

  grid.innerHTML = '';
  items.forEach(item => {
    const slot = document.createElement('div');
    slot.className = 'inventory-slot';
    slot.textContent = item.emoji || '📦';
    slot.title = `${item.name}\n${item.galleons ? `Worth: ${item.galleons}G` : ''}`;
    grid.appendChild(slot);
  });
}

function updateSuggestions(actions) {
  const bar = document.getElementById('suggestions-bar');
  if (!bar) return;
  bar.innerHTML = '';

  (actions || []).forEach(action => {
    const btn = document.createElement('button');
    btn.className = 'suggestion-btn';
    btn.textContent = action;
    btn.addEventListener('click', () => {
      document.getElementById('player-input').value = action;
      document.getElementById('char-count').textContent = `${action.length}/500`;
      document.getElementById('player-input').focus();
    });
    bar.appendChild(btn);
  });
}

function updateLocationBadge(locationName) {
  document.getElementById('location-name').textContent = locationName;
}

function setTyping(isTyping) {
  const indicator = document.getElementById('typing-indicator');
  if (indicator) {
    if (isTyping) {
      indicator.classList.remove('hidden');
    } else {
      indicator.classList.add('hidden');
    }
  }
}

function addNarrativeEntry(role, text, atmosphere) {
  const scroll = document.getElementById('narrative-scroll');
  if (!scroll) return;

  const entry = document.createElement('div');

  if (role === 'dm') {
    entry.className = 'narrative-entry';
    entry.innerHTML = `
      <div class="narrative-role-dm">
        <div class="narrative-avatar-dm">🧙</div>
        <div>
          <div class="narrative-label">Dungeon Master</div>
        </div>
      </div>
      <div class="narrative-text">${formatNarrativeText(text)}</div>
    `;
  } else {
    entry.className = 'narrative-entry-player';
    entry.innerHTML = `
      <div class="player-input-display">
        <span class="player-input-arrow">▶</span>
        <span>${escapeHtml(text)}</span>
      </div>
    `;
  }

  scroll.appendChild(entry);

  // Auto scroll
  requestAnimationFrame(() => {
    scroll.scrollTop = scroll.scrollHeight;
  });
}

function formatNarrativeText(text) {
  // Convert markdown-like syntax to HTML
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>');
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(text));
  return div.innerHTML;
}

function updateEmotionUI(emotion, difficulty) {
  if (emotion && emotion !== 'neutral') {
    EmotionTracker.updateUI(emotion, difficulty || 'medium', 0.7);
  }
  // Also update difficulty in game state
  if (difficulty) {
    GameState.difficulty = difficulty;
  }
}

function showQuestDetails(quest) {
  addNarrativeEntry('dm', `📜 **${quest.title}**\n\n${quest.description || 'A mysterious quest awaits...'}\n\nObjectives:\n${(quest.objectives || []).map(o => `• ${typeof o === 'string' ? o : o.text}`).join('\n')}`, 'mysterious');
}

// ═══════════════════════════════════════════════════════════════════════════
// KNOWLEDGE GRAPH INIT
// ═══════════════════════════════════════════════════════════════════════════
async function initKnowledgeGraph() {
  KnowledgeGraph.init('world-graph-svg');

  try {
    const worldData = await Narrator.getWorldState(GameState.sessionId);
    KnowledgeGraph.render(worldData);
  } catch {
    // Use mock graph
    const mockData = KnowledgeGraph.buildMockGraph();
    KnowledgeGraph.render(mockData);
  }

  // Listen for node selection
  document.addEventListener('graph:nodeSelected', (e) => {
    const node = e.detail;
    addNarrativeEntry('dm', `📍 *${node.label}* — ${node.type}\n${node.properties?.description || ''}`, 'mysterious');
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// BOTTOM BAR HANDLERS
// ═══════════════════════════════════════════════════════════════════════════
function initBottomBarHandlers() {
  document.getElementById('save-btn').addEventListener('click', async () => {
    const btn = document.getElementById('save-btn');
    const orig = btn.textContent;
    btn.textContent = '💾 Saving...';
    btn.disabled = true;
    try {
      if (GameState.backendConnected && GameState.sessionId && !GameState.sessionId.startsWith('local-')) {
        await Narrator.saveGame(GameState.sessionId);
        btn.textContent = '✅ Saved!';
      } else {
        // Offline fallback — localStorage
        localStorage.setItem('ai-dungeon-save', JSON.stringify({
          playerStats: GameState.playerStats,
          currentLocation: GameState.currentLocation,
          difficulty: GameState.difficulty,
          turnCount: GameState.turnCount,
          savedAt: new Date().toISOString()
        }));
        btn.textContent = '✅ Saved (local)!';
      }
    } catch {
      btn.textContent = '❌ Save failed';
    } finally {
      btn.disabled = false;
      setTimeout(() => { btn.textContent = orig; }, 2000);
    }
  });

  document.getElementById('history-btn').addEventListener('click', () => {
    addNarrativeEntry('dm', `📚 *Turn ${GameState.turnCount} — You've adventured through ${GameState.turnCount - 1} actions so far. Current location: ${GameState.locations[GameState.currentLocation]?.name || 'Unknown'}.*`, 'mysterious');
  });

  document.getElementById('quit-btn').addEventListener('click', () => {
    if (confirm('Are you sure you want to end your adventure?')) {
      location.reload();
    }
  });

  document.getElementById('settings-btn').addEventListener('click', () => {
    addNarrativeEntry('dm', `⚙️ *Settings: Difficulty is currently set to **${GameState.difficulty}**. The emotion tracker adjusts this dynamically based on your gameplay.*\n\nTo change: type "change difficulty to easy/medium/hard"`, 'mysterious');
  });

  document.getElementById('help-btn').addEventListener('click', () => {
    addNarrativeEntry('dm', `❓ **How to Play:**\n\n• Type any action in the input box below\n• Click ⚡ or press Enter to submit\n• Use quick actions from the suggestion bar\n• Combat triggers when you mention fighting enemies\n• Click nodes in the World Map to explore them\n• The Dungeon Master adapts to your choices!\n\n*Your emotion is tracked to adjust difficulty automatically.*`, 'warm');
  });

  document.getElementById('quest-toggle').addEventListener('click', (e) => {
    const list = document.getElementById('quest-list');
    const btn = e.target;
    if (list.style.display === 'none') { list.style.display = ''; btn.textContent = '−'; }
    else { list.style.display = 'none'; btn.textContent = '+'; }
  });

  document.getElementById('graph-toggle').addEventListener('click', (e) => {
    const container = document.getElementById('graph-container');
    const btn = e.target;
    if (container.style.display === 'none') { container.style.display = ''; btn.textContent = '−'; }
    else { container.style.display = 'none'; btn.textContent = '+'; }
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ═══════════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initCharacterCreation();
  initInputHandlers();
  initBottomBarHandlers();

  // Periodic emotion check
  setInterval(async () => {
    if (GameState.sessionId && GameState.backendConnected && GameState.recentInputs.length > 0) {
      try {
        const result = await Narrator.classifyEmotion(
          GameState.sessionId,
          EmotionTracker.getRecentInputs(),
          EmotionTracker.getTurnsWithoutProgress()
        );
        EmotionTracker.updateUI(result.emotion, result.difficulty_recommendation, result.confidence);
      } catch {}
    }
  }, 30000);

  // Animate scene periodically
  let animFrame = null;
  function animateScene() {
    if (!GameState.inCombat) {
      renderScene(GameState.currentLocation, 'magical');
    }
    animFrame = setTimeout(animateScene, 3000);
  }
  setTimeout(animateScene, 5000);

  // Start loading sequence
  runLoadingSequence();
});
