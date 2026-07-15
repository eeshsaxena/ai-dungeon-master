"""
Story memory management for the AI Dungeon Master.
Maintains per-session narrative history, compresses older turns, and
provides context windows for the LLM narrator.
"""
import time
from typing import List, Dict, Optional, Any


class StoryMemory:
    """
    Manages narrative history for a single game session.
    - Keeps last MAX_RECENT turns in full detail
    - Compresses older turns into summaries
    - Tracks key story beats, NPC interactions, and world changes
    """

    def __init__(self, session_id: str, max_recent: int = 20):
        self.session_id = session_id
        self.max_recent = max_recent
        self.full_history: List[Dict[str, Any]] = []
        self.compressed_summary: str = ""
        self.key_beats: List[str] = []  # Important story moments
        self.npc_interactions: Dict[str, List[str]] = {}  # npc_id -> interactions
        self.world_changes: List[str] = []  # Things that changed in the world
        self.created_at = time.time()
        self.last_updated = time.time()

    def add_turn(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a conversation turn to memory."""
        turn = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.full_history.append(turn)
        self.last_updated = time.time()

        # Auto-compress if too long
        if len(self.full_history) > self.max_recent + 10:
            self._compress_old_turns()

    def _compress_old_turns(self):
        """Compress turns older than max_recent into a summary."""
        overflow = len(self.full_history) - self.max_recent
        if overflow <= 0:
            return

        old_turns = self.full_history[:overflow]
        self.full_history = self.full_history[overflow:]

        # Build summary from old turns
        summaries = []
        for turn in old_turns:
            if turn["role"] == "user":
                summaries.append(f"Player: {turn['content'][:100]}")
            elif turn["role"] == "assistant":
                # Extract first sentence as summary
                first_sentence = turn["content"].split(".")[0][:150]
                summaries.append(f"DM: {first_sentence}...")

        new_summary = " | ".join(summaries)
        if self.compressed_summary:
            self.compressed_summary += "\n[Earlier] " + new_summary
        else:
            self.compressed_summary = "[Earlier in the adventure] " + new_summary

    def add_key_beat(self, beat: str):
        """Record an important story moment."""
        self.key_beats.append(beat)
        if len(self.key_beats) > 50:
            self.key_beats = self.key_beats[-50:]

    def add_npc_interaction(self, npc_id: str, interaction: str):
        """Track interaction with an NPC."""
        if npc_id not in self.npc_interactions:
            self.npc_interactions[npc_id] = []
        self.npc_interactions[npc_id].append(interaction)

    def add_world_change(self, change: str):
        """Record a change to the world state."""
        self.world_changes.append(change)

    def get_context_window(self, max_tokens: int = 3000) -> str:
        """
        Build a context string for the LLM.
        Includes: compressed summary + recent turns + key beats.
        """
        parts = []

        if self.compressed_summary:
            parts.append(self.compressed_summary)

        if self.key_beats:
            parts.append("\n[KEY STORY BEATS]\n" + "\n".join(f"• {b}" for b in self.key_beats[-10:]))

        if self.world_changes:
            parts.append("\n[WORLD CHANGES]\n" + "\n".join(f"• {c}" for c in self.world_changes[-5:]))

        if self.full_history:
            parts.append("\n[RECENT CONVERSATION]")
            for turn in self.full_history[-self.max_recent:]:
                role_label = "PLAYER" if turn["role"] == "user" else "DUNGEON MASTER"
                parts.append(f"{role_label}: {turn['content']}")

        return "\n".join(parts)

    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get message list formatted for LLM API (OpenAI / Ollama)."""
        messages = []
        for turn in self.full_history[-self.max_recent:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        return messages

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "history_length": len(self.full_history),
            "key_beats": self.key_beats,
            "world_changes": self.world_changes,
            "compressed_summary": self.compressed_summary,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }


class MemoryManager:
    """Manages memory across all active sessions."""

    def __init__(self, max_recent: int = 20):
        self.sessions: Dict[str, StoryMemory] = {}
        self.max_recent = max_recent

    def get_or_create(self, session_id: str) -> StoryMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = StoryMemory(session_id, self.max_recent)
        return self.sessions[session_id]

    def get(self, session_id: str) -> Optional[StoryMemory]:
        return self.sessions.get(session_id)

    def clear(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def cleanup_old_sessions(self, timeout_seconds: int = 3600):
        """Remove sessions inactive for longer than timeout."""
        now = time.time()
        to_remove = [
            sid for sid, mem in self.sessions.items()
            if now - mem.last_updated > timeout_seconds
        ]
        for sid in to_remove:
            del self.sessions[sid]
        return len(to_remove)

    def list_sessions(self) -> List[str]:
        return list(self.sessions.keys())


# Global memory manager instance
memory_manager = MemoryManager()
