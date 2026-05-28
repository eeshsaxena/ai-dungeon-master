"""
Player Emotion Classifier for AI Dungeon Master.
Detects player frustration, boredom, excitement etc. from text + behavior patterns.
Adjusts difficulty and provides hints accordingly.
"""
import re
import time
from typing import List, Dict, Optional, Tuple
from models.schemas import EmotionState, DifficultyLevel


# ── Keyword dictionaries ───────────────────────────────────────────────────────

FRUSTRATED_KEYWORDS = [
    "hate", "stupid", "this sucks", "wtf", "what the hell", "broken", "impossible",
    "unfair", "cheating", "why", "ugh", "argh", "seriously", "again", "killed again",
    "lost again", "i give up", "this is too hard", "can't", "cant", "doesn't work",
    "not working", "bug", "glitch", "dumb", "annoying", "frustrating"
]

BORED_KEYWORDS = [
    "boring", "bored", "slow", "nothing", "same thing", "repeat", "again", "meh",
    "ok", "okay", "whatever", "skip", "fast forward", "hurry", "move on", "next",
    "let's go", "already", "when does it get good", "is this all", "nothing happens"
]

EXCITED_KEYWORDS = [
    "awesome", "amazing", "wow", "great", "cool", "epic", "love", "yes!", "finally",
    "yes yes yes", "omg", "incredible", "insane", "legendary", "brilliant", "perfect",
    "woah", "whoa", "lets go", "let's go!", "hype", "this is amazing", "best"
]

CURIOUS_KEYWORDS = [
    "what", "how", "why", "where", "who", "tell me", "explain", "what if", "i wonder",
    "explore", "investigate", "look", "search", "examine", "inspect", "curious",
    "interesting", "i want to know", "find out", "discover"
]

CONFUSED_KEYWORDS = [
    "confused", "don't understand", "what do i do", "how do i", "help", "stuck",
    "lost", "what now", "i don't get it", "doesn't make sense", "unclear", "??",
    "huh", "what?", "which way", "no idea", "not sure"
]

FEARFUL_KEYWORDS = [
    "scared", "afraid", "terrified", "run", "flee", "escape", "dangerous",
    "i'm scared", "too scary", "help me", "protect", "hide", "danger", "careful",
    "is it safe", "will i die"
]

# ── Hint templates per emotion ─────────────────────────────────────────────────

HINTS = {
    EmotionState.FRUSTRATED: [
        "💡 *Hint: Try using a defensive spell like Protego before attacking.*",
        "💡 *Hint: Talk to NPCs — they often have useful information you might have missed.*",
        "💡 *Hint: Every enemy has a weakness. Check the combat log for patterns.*",
        "💡 *Hint: You can type 'help' or 'what can I do?' to see available actions.*"
    ],
    EmotionState.BORED: [
        "✨ *Something stirs in the shadows nearby... perhaps worth investigating.*",
        "✨ *A mysterious owl arrives with an unmarked letter. It's addressed to you.*",
        "✨ *You notice something glinting under the floorboards — something magical.*"
    ],
    EmotionState.CONFUSED: [
        "📜 *Hint: Type 'look around' to assess your surroundings.*",
        "📜 *Hint: Check your quest log (press Q) for your current objectives.*",
        "📜 *Hint: NPCs with a ⚡ icon are willing to talk and may give guidance.*"
    ],
    EmotionState.EXCITED: [
        "⚡ *Your excitement fuels your magic! Spell power temporarily increased.*",
        "⚡ *Word of your deeds reaches Hogwarts. A new rumor spreads...*"
    ]
}


class EmotionClassifier:
    """
    Classifies player emotional state from recent text inputs and behavioral signals.
    Uses keyword matching + behavioral heuristics (time between inputs, death count, etc.)
    """

    def __init__(self):
        self.emotion_history: List[EmotionState] = []
        self.death_count: int = 0
        self.hint_cooldown: Dict[str, float] = {}  # emotion -> last hint timestamp
        self.hint_cooldown_seconds: int = 120

    def classify(
        self,
        recent_inputs: List[str],
        turns_without_progress: int = 0,
        time_between_inputs: float = 30.0,
        deaths_recently: int = 0
    ) -> Tuple[EmotionState, float, Optional[str]]:
        """
        Classify emotion from recent player inputs and behavior.
        Returns: (emotion, confidence, optional_hint)
        """
        if not recent_inputs:
            return EmotionState.NEUTRAL, 0.5, None

        # Combine recent inputs into one text blob
        text = " ".join(recent_inputs).lower()
        text = re.sub(r'[^\w\s!?]', ' ', text)

        # Score each emotion category
        scores = {
            EmotionState.FRUSTRATED: self._score(text, FRUSTRATED_KEYWORDS),
            EmotionState.BORED: self._score(text, BORED_KEYWORDS),
            EmotionState.EXCITED: self._score(text, EXCITED_KEYWORDS),
            EmotionState.CURIOUS: self._score(text, CURIOUS_KEYWORDS),
            EmotionState.CONFUSED: self._score(text, CONFUSED_KEYWORDS),
            EmotionState.FEARFUL: self._score(text, FEARFUL_KEYWORDS),
        }

        # Behavioral boosts
        if deaths_recently >= 2:
            scores[EmotionState.FRUSTRATED] += 0.3
        if turns_without_progress >= 5:
            scores[EmotionState.BORED] += 0.25
            scores[EmotionState.FRUSTRATED] += 0.15
        if time_between_inputs > 120:  # 2 min between inputs
            scores[EmotionState.BORED] += 0.2
        if time_between_inputs < 5:  # rapid fire inputs
            scores[EmotionState.EXCITED] += 0.2

        # Find dominant emotion
        best_emotion = max(scores, key=scores.get)
        best_score = scores[best_emotion]

        if best_score < 0.1:
            emotion = EmotionState.NEUTRAL
            confidence = 0.5
        else:
            emotion = best_emotion
            confidence = min(0.95, 0.4 + best_score)

        # Update history
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > 20:
            self.emotion_history = self.emotion_history[-20:]

        # Decide if hint should trigger
        hint = self._maybe_get_hint(emotion)

        return emotion, confidence, hint

    def _score(self, text: str, keywords: List[str]) -> float:
        """Score text against a keyword list."""
        hits = sum(1 for kw in keywords if kw in text)
        return min(1.0, hits * 0.15)

    def _maybe_get_hint(self, emotion: EmotionState) -> Optional[str]:
        """Return a hint if cooldown has expired."""
        emotion_key = emotion.value
        now = time.time()
        last = self.hint_cooldown.get(emotion_key, 0)

        if emotion in HINTS and (now - last) > self.hint_cooldown_seconds:
            hints = HINTS[emotion]
            idx = len(self.emotion_history) % len(hints)
            self.hint_cooldown[emotion_key] = now
            return hints[idx]
        return None

    def get_difficulty_recommendation(self, emotion: EmotionState) -> DifficultyLevel:
        """
        Recommend difficulty adjustment based on detected emotion.
        Frustrated → ease up | Bored → ramp up | Confused → give hints
        """
        if emotion == EmotionState.FRUSTRATED:
            return DifficultyLevel.EASY
        elif emotion == EmotionState.BORED:
            return DifficultyLevel.HARD
        elif emotion == EmotionState.EXCITED:
            return DifficultyLevel.MEDIUM
        elif emotion == EmotionState.CONFUSED:
            return DifficultyLevel.EASY
        else:
            return DifficultyLevel.MEDIUM

    def get_persistent_emotion(self) -> EmotionState:
        """Get the most common recent emotion (last 10 turns)."""
        recent = self.emotion_history[-10:] if self.emotion_history else [EmotionState.NEUTRAL]
        from collections import Counter
        return Counter(recent).most_common(1)[0][0]


# Global classifier instance
emotion_classifier = EmotionClassifier()
