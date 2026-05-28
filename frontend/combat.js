/**
 * combat.js — Combat System UI
 * Manages the combat overlay, spell selection, HP bars, and combat log.
 */

const CombatSystem = (() => {
  let inCombat = false;
  let currentEnemy = null;
  let combatRound = 1;
  let playerSpells = [];
  let selectedSpell = null;
  let sessionId = null;
  let playerStats = null;
  let onCombatEnd = null;

  const SPELL_COLORS = {
    offensive: '#E74C3C',
    defensive: '#27AE60',
    utility:   '#F5A623',
    dark:      '#6B21A8',
    healing:   '#16A085',
    shield:    '#4A90D9',
    mental:    '#8C5CF6'
  };

  const ENEMY_EMOJIS = {
    'Dementor':   '👻',
    'Death Eater': '🧙‍♂️',
    'Acromantula': '🕷️',
    'Troll':      '👹',
    'Werewolf':   '🐺',
    'Dark Wizard': '🧙',
    'Hollow Mage': '💀',
    'Boggart':    '😱',
    'Inferi':     '🧟'
  };

  const SPELL_TYPES = {
    'Expelliarmus':  { type: 'offensive', mana: 10, desc: 'Disarms opponent' },
    'Stupefy':       { type: 'offensive', mana: 15, desc: 'Stuns target' },
    'Protego':       { type: 'defensive', mana: 12, desc: 'Creates shield' },
    'Expecto Patronum': { type: 'defensive', mana: 40, desc: 'Repels dark entities' },
    'Accio':         { type: 'utility', mana: 8, desc: 'Summons objects' },
    'Lumos':         { type: 'utility', mana: 3, desc: 'Creates light' },
    'Alohomora':     { type: 'utility', mana: 5, desc: 'Unlocks doors' },
    'Sectumsempra':  { type: 'offensive', mana: 35, desc: 'Slashing curse' },
    'Episkey':       { type: 'healing', mana: 20, desc: 'Heals minor wounds' },
    'Lumos Maxima':  { type: 'utility', mana: 15, desc: 'Intense flash of light' },
    'Wingardium Leviosa': { type: 'utility', mana: 10, desc: 'Levitation charm' },
    'Riddikulus':    { type: 'utility', mana: 12, desc: 'Defeats Boggarts' },
    'Fiendfyre':     { type: 'offensive', mana: 60, desc: 'Cursed fire — DANGEROUS' }
  };

  function startCombat(enemy, sid, stats, spells, onEnd) {
    inCombat = true;
    currentEnemy = enemy;
    combatRound = 1;
    sessionId = sid;
    playerStats = { ...stats };
    playerSpells = spells || ['Expelliarmus', 'Stupefy', 'Protego'];
    onCombatEnd = onEnd;
    selectedSpell = null;

    // Update combat overlay
    updateEnemyDisplay();
    updatePlayerDisplay();
    renderSpellButtons();
    clearCombatLog();
    addCombatLog(`⚔️ A ${enemy.name} appears! Combat begins!`);
    addCombatLog(`💡 Weakness: ${enemy.weakness || 'Unknown'}`);

    // Show overlay
    document.getElementById('combat-overlay').classList.remove('hidden');
    document.getElementById('combat-panel').classList.remove('hidden');
    document.getElementById('combat-round').textContent = 'Round 1';

    // Setup buttons
    document.getElementById('defend-btn').onclick = () => resolveAction('defend', null);
    document.getElementById('flee-btn').onclick = () => resolveAction('flee', null);
  }

  function updateEnemyDisplay() {
    if (!currentEnemy) return;
    const archetype = currentEnemy.archetype || currentEnemy.name;
    document.getElementById('enemy-art').textContent = ENEMY_EMOJIS[archetype] || '🧙‍♂️';
    document.getElementById('enemy-name').textContent = currentEnemy.name;
    updateEnemyHPBar(currentEnemy.hp, currentEnemy.max_hp);
  }

  function updatePlayerDisplay() {
    if (!playerStats) return;
    document.getElementById('combat-player-name').textContent = playerStats.name || 'You';
    updatePlayerHPBar(playerStats.hp, playerStats.max_hp);
  }

  function updateEnemyHPBar(hp, maxHp) {
    const pct = Math.max(0, Math.min(100, (hp / maxHp) * 100));
    const bar = document.getElementById('enemy-hp-bar');
    const txt = document.getElementById('enemy-hp-text');
    if (bar) {
      bar.style.width = pct + '%';
      bar.style.background = pct > 50 ? 'linear-gradient(90deg,#8B0000,#E74C3C)' : 'linear-gradient(90deg,#4A0000,#8B0000)';
    }
    if (txt) txt.textContent = `${Math.max(0,hp)}/${maxHp}`;
  }

  function updatePlayerHPBar(hp, maxHp) {
    const pct = Math.max(0, Math.min(100, (hp / maxHp) * 100));
    const bar = document.getElementById('combat-player-hp-bar');
    const txt = document.getElementById('combat-player-hp-text');
    if (bar) {
      bar.style.width = pct + '%';
      bar.style.background = pct > 50
        ? 'linear-gradient(90deg,#16A085,#27AE60)'
        : pct > 25
          ? 'linear-gradient(90deg,#D35400,#E67E22)'
          : 'linear-gradient(90deg,#8B0000,#C0392B)';
    }
    if (txt) txt.textContent = `${Math.max(0,hp)}/${maxHp}`;
  }

  function renderSpellButtons() {
    const container = document.getElementById('spell-buttons');
    if (!container) return;
    container.innerHTML = '';

    playerSpells.forEach(spell => {
      const info = SPELL_TYPES[spell] || { type: 'offensive', mana: 10 };
      const btn = document.createElement('button');
      btn.className = 'spell-cast-btn';
      btn.textContent = spell;
      btn.title = info.desc || spell;
      btn.style.borderColor = SPELL_COLORS[info.type] || '#6C63FF';

      btn.addEventListener('click', () => {
        selectedSpell = spell;
        container.querySelectorAll('.spell-cast-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        // Auto-resolve after spell selection
        setTimeout(() => resolveAction('attack', spell), 300);
      });

      container.appendChild(btn);
    });
  }

  async function resolveAction(action, spell) {
    if (!inCombat || !currentEnemy) return;

    // Disable buttons during resolution
    setButtonsEnabled(false);

    // Show spell cast animation
    if (spell && action === 'attack') {
      showSpellCast(spell);
    }

    const combatRequest = {
      session_id: sessionId,
      enemy: currentEnemy,
      player_stats: playerStats,
      player_action: action,
      player_spell: spell,
      round_number: combatRound,
      combat_history: []
    };

    try {
      let result;
      try {
        result = await Narrator.resolveCombat(combatRequest);
      } catch {
        // Mock combat resolution
        result = mockCombatResolution(action, spell);
      }

      await processCombatResult(result);
    } catch (e) {
      addCombatLog('⚠️ Combat error. Try again.');
      setButtonsEnabled(true);
    }
  }

  function mockCombatResolution(action, spell) {
    const enemyActions = ['attacks', 'casts a spell at', 'charges'];
    const spellDamage = { 'Expelliarmus': 25, 'Stupefy': 35, 'Protego': 0,
      'Expecto Patronum': 60, 'Sectumsempra': 70, 'Episkey': -30 };

    const playerDmg = action === 'defend' ? 0 : (spellDamage[spell] || 30) + Math.floor(Math.random() * 10);
    const enemyDmg = action === 'defend' ? 0 : Math.floor(currentEnemy.attack * 0.8 + Math.random() * 20);

    const newEnemyHp = Math.max(0, (currentEnemy.hp || 120) - playerDmg);
    const newPlayerHp = Math.max(0, (playerStats.hp || 100) - enemyDmg);

    return {
      enemy_action: 'attack',
      enemy_spell: 'Avada Kedavra',
      narrative: `The ${currentEnemy.name} ${enemyActions[Math.floor(Math.random() * 3)]} you!`,
      player_damage: playerDmg,
      enemy_damage: enemyDmg,
      enemy_hp_remaining: newEnemyHp,
      player_hp_remaining: newPlayerHp,
      combat_over: newEnemyHp <= 0 || newPlayerHp <= 0,
      player_won: newEnemyHp <= 0 || undefined,
      loot: newEnemyHp <= 0 ? [{ name: 'Dark Essence', galleons: 15 }] : [],
      xp_gained: newEnemyHp <= 0 ? 200 : 0
    };
  }

  async function processCombatResult(result) {
    combatRound++;
    document.getElementById('combat-round-indicator').textContent = `Round ${combatRound}`;
    document.getElementById('combat-round').textContent = `Round ${combatRound}`;

    // Update HP
    if (currentEnemy) currentEnemy.hp = result.enemy_hp_remaining;
    if (playerStats) playerStats.hp = result.player_hp_remaining;

    updateEnemyHPBar(result.enemy_hp_remaining, currentEnemy?.max_hp || 120);
    updatePlayerHPBar(result.player_hp_remaining, playerStats?.max_hp || 100);

    // Enemy action
    const enemyBubble = document.getElementById('enemy-action-bubble');
    if (enemyBubble && result.enemy_spell) {
      enemyBubble.textContent = `"${result.enemy_spell}!"`;
      setTimeout(() => { enemyBubble.textContent = ''; }, 2000);
    }

    // Log
    if (result.player_damage > 0) {
      addCombatLog(`✨ Your spell dealt ${result.player_damage} damage!`);
    } else if (result.player_damage === 0) {
      addCombatLog(`🛡️ You defend! No damage dealt.`);
    }

    if (result.enemy_damage > 0) {
      addCombatLog(`💥 ${currentEnemy?.name} deals ${result.enemy_damage} damage to you!`);
      shakePlayerSide();
    } else {
      addCombatLog(`🛡️ Your defense held! No damage taken.`);
    }

    if (result.narrative) {
      addCombatLog(`📖 ${result.narrative}`);
    }

    // Wait a moment
    await sleep(500);

    if (result.combat_over) {
      await endCombat(result);
    } else {
      setButtonsEnabled(true);
      selectedSpell = null;
      document.querySelectorAll('.spell-cast-btn').forEach(b => b.classList.remove('selected'));
    }
  }

  async function endCombat(result) {
    if (result.player_won) {
      addCombatLog(`🏆 VICTORY! The ${currentEnemy?.name} has been defeated!`);
      if (result.xp_gained > 0) addCombatLog(`⭐ +${result.xp_gained} XP gained!`);
      if (result.loot?.length > 0) {
        result.loot.forEach(item => addCombatLog(`💰 Looted: ${item.name}!`));
      }
      await sleep(1500);
    } else {
      addCombatLog(`💀 You have been defeated... but live to fight another day.`);
      await sleep(1500);
    }

    inCombat = false;
    document.getElementById('combat-overlay').classList.add('hidden');
    document.getElementById('combat-panel').classList.add('hidden');
    onCombatEnd && onCombatEnd(result);
  }

  function addCombatLog(text) {
    const log = document.getElementById('combat-log');
    if (!log) return;
    const entry = document.createElement('p');
    entry.className = 'combat-log-entry';
    entry.textContent = text;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
  }

  function clearCombatLog() {
    const log = document.getElementById('combat-log');
    if (log) log.innerHTML = '';
  }

  function setButtonsEnabled(enabled) {
    document.querySelectorAll('.spell-cast-btn, .combat-btn').forEach(btn => {
      btn.disabled = !enabled;
      btn.style.opacity = enabled ? '1' : '0.5';
    });
  }

  function showSpellCast(spellName) {
    const overlay = document.getElementById('spell-overlay');
    const nameEl = document.getElementById('spell-overlay-name');
    if (!overlay || !nameEl) return;

    nameEl.textContent = spellName + '!';
    overlay.classList.remove('hidden');

    // Create particles
    const particles = document.getElementById('spell-particles');
    if (particles) {
      particles.innerHTML = '';
      for (let i = 0; i < 12; i++) {
        const p = document.createElement('div');
        const angle = (i / 12) * 360;
        p.style.cssText = `
          position: absolute;
          width: 6px; height: 6px;
          border-radius: 50%;
          background: ${['#FFD700','#8C5CF6','#4A90D9'][i % 3]};
          top: 50%; left: 50%;
          --tx: ${Math.cos(angle * Math.PI / 180) * 150}px;
          animation: particle-float 0.8s ease-out ${i * 50}ms forwards;
        `;
        particles.appendChild(p);
      }
    }

    setTimeout(() => { overlay.classList.add('hidden'); }, 900);
  }

  function shakePlayerSide() {
    const playerSide = document.querySelector('.player-side');
    if (!playerSide) return;
    playerSide.style.animation = 'none';
    playerSide.style.transform = 'translateX(-8px)';
    setTimeout(() => { playerSide.style.transform = 'translateX(8px)'; }, 100);
    setTimeout(() => { playerSide.style.transform = ''; }, 200);
  }

  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  function isInCombat() { return inCombat; }

  return { startCombat, isInCombat, addCombatLog, showSpellCast };
})();
