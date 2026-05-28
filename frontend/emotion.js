/**
 * emotion.js — Player Emotion Tracker (Frontend)
 * Tracks player input patterns, detects emotional state,
 * and communicates with the backend emotion classifier.
 */

const EmotionTracker = (() => {
  const EMOTIONS = {
    neutral:    { icon: '😐', color: '#B8A890', label: 'Neutral' },
    excited:    { icon: '🤩', color: '#FFD700', label: 'Excited' },
    curious:    { icon: '🤔', color: '#4A90D9', label: 'Curious' },
    frustrated: { icon: '😤', color: '#E74C3C', label: 'Frustrated' },
    bored:      { icon: '😴', color: '#7A6A5A', label: 'Bored' },
    confused:   { icon: '😵', color: '#F5A623', label: 'Confused' },
    fearful:    { icon: '😨', color: '#8C5CF6', label: 'Fearful' }
  };

  const DIFFICULTY_COLORS = {
    easy:      '#27AE60',
    medium:    '#F5A623',
    hard:      '#E74C3C',
    very_hard: '#8B0000',
    legendary: '#FFD700'
  };

  let currentEmotion = 'neutral';
  let currentDifficulty = 'medium';
  let recentInputs = [];
  let turnsWithoutProgress = 0;
  let lastInputTime = Date.now();

  function recordInput(inputText) {
    recentInputs.push(inputText);
    if (recentInputs.length > 10) recentInputs = recentInputs.slice(-10);
    lastInputTime = Date.now();
  }

  function recordNoProgress() {
    turnsWithoutProgress++;
  }

  function resetProgress() {
    turnsWithoutProgress = 0;
  }

  function updateUI(emotion, difficulty, confidence = 0.5) {
    currentEmotion = emotion;
    currentDifficulty = difficulty;

    const data = EMOTIONS[emotion] || EMOTIONS.neutral;

    const icon = document.getElementById('emotion-icon');
    const name = document.getElementById('emotion-name');
    const bar = document.getElementById('emotion-bar');
    const adj = document.getElementById('emotion-adjustment');

    if (icon) {
      icon.textContent = data.icon;
      icon.style.filter = `drop-shadow(0 0 8px ${data.color}40)`;
    }

    if (name) {
      name.textContent = data.label;
      name.style.color = data.color;
    }

    if (bar) {
      bar.style.width = `${Math.round(confidence * 100)}%`;
      bar.style.background = `linear-gradient(90deg, ${data.color}80, ${data.color})`;
    }

    if (adj) {
      const diffColor = DIFFICULTY_COLORS[difficulty] || '#F5A623';
      const diffLabel = difficulty.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
      adj.innerHTML = `Difficulty: <span style="color:${diffColor}">${diffLabel}</span>`;
    }

    // Flash the emotion panel
    const panel = document.querySelector('.emotion-panel');
    if (panel) {
      panel.style.borderColor = data.color + '60';
      setTimeout(() => { panel.style.borderColor = ''; }, 2000);
    }
  }

  function getCurrentEmotion() { return currentEmotion; }
  function getCurrentDifficulty() { return currentDifficulty; }
  function getRecentInputs() { return [...recentInputs]; }
  function getTimeSinceLastInput() { return (Date.now() - lastInputTime) / 1000; }
  function getTurnsWithoutProgress() { return turnsWithoutProgress; }

  return {
    recordInput,
    recordNoProgress,
    resetProgress,
    updateUI,
    getCurrentEmotion,
    getCurrentDifficulty,
    getRecentInputs,
    getTimeSinceLastInput,
    getTurnsWithoutProgress,
    EMOTIONS
  };
})();
