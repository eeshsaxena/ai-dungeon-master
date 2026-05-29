"""
Long-term NPC memory for the AI Dungeon Master (Phase 2.5).

Phase 1 memory (memory.py) is per-session story history. This adds a layer that
persists *per NPC across sessions*: each character remembers what the player did
in their presence, so when you return to the Three Broomsticks weeks later,
Madam Rosmerta can reference that time you smashed up Knockturn Alley.

Recall is semantic — when the player interacts near an NPC, we surface that
NPC's most relevant past memories (via shared embeddings from rag.py), falling
back to plain recency when the embedding model is unavailable. Memories persist
to disk as human-readable JSON; vectors are recomputed on load.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from rag import lore_retriever

STORE_DIR = Path(__file__).parent / "npc_memory_store"
STORE_PATH = STORE_DIR / "npc_memories.json"

MAX_MEMORIES_PER_NPC = 50


class NPCMemoryStore:
    """Per-NPC long-term memory with semantic recall."""

    def __init__(self, persist: bool = True):
        self.persist = persist
        # npc_id -> list of memory dicts {text, ts, session_id}
        self.memories: Dict[str, List[Dict[str, Any]]] = {}
        # npc_id -> np.ndarray of embeddings aligned with self.memories[npc_id]
        self._vectors: Dict[str, Optional[np.ndarray]] = {}
        if self.persist:
            self._load()

    # -- writing --------------------------------------------------------------

    def record(self, npc_id: str, text: str, session_id: str = "") -> None:
        """Store a memory for an NPC (something the player did in their presence)."""
        if not npc_id or not text or not text.strip():
            return
        bucket = self.memories.setdefault(npc_id, [])
        bucket.append({"text": text.strip(), "ts": time.time(), "session_id": session_id})

        # Cap per-NPC history (keep most recent)
        if len(bucket) > MAX_MEMORIES_PER_NPC:
            self.memories[npc_id] = bucket[-MAX_MEMORIES_PER_NPC:]

        # Invalidate cached vectors for this NPC; rebuilt lazily on recall
        self._vectors.pop(npc_id, None)
        if self.persist:
            self._save()

    # -- reading --------------------------------------------------------------

    def recall(self, npc_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return this NPC's most relevant past memories for the query."""
        bucket = self.memories.get(npc_id, [])
        if not bucket:
            return []

        vecs = self._ensure_vectors(npc_id)
        if vecs is not None and query.strip():
            qv = lore_retriever.embed([query])
            if qv is not None:
                scores = vecs @ qv[0]
                order = np.argsort(scores)[::-1][:top_k]
                return [
                    {**bucket[i], "score": round(float(scores[i]), 4)}
                    for i in order
                ]

        # Fallback: most recent memories
        return [{**m, "score": None} for m in bucket[-top_k:][::-1]]

    def recall_text(self, npc_id: str, npc_name: str, query: str, top_k: int = 3) -> str:
        """Format an NPC's recalled memories as a context block, or '' if none."""
        hits = self.recall(npc_id, query, top_k)
        if not hits:
            return ""
        lines = [f"{npc_name} remembers:"]
        for h in hits:
            lines.append(f"  - {h['text']}")
        return "\n".join(lines)

    def _ensure_vectors(self, npc_id: str) -> Optional[np.ndarray]:
        if npc_id in self._vectors:
            return self._vectors[npc_id]
        texts = [m["text"] for m in self.memories.get(npc_id, [])]
        vecs = lore_retriever.embed(texts) if texts else None
        self._vectors[npc_id] = vecs
        return vecs

    # -- persistence ----------------------------------------------------------

    def _load(self) -> None:
        if not STORE_PATH.exists():
            return
        try:
            self.memories = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[NPCMemory] Could not load store ({e}); starting fresh.")
            self.memories = {}

    def _save(self) -> None:
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        STORE_PATH.write_text(json.dumps(self.memories, indent=2), encoding="utf-8")

    # -- introspection --------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "npcs_tracked": len(self.memories),
            "total_memories": sum(len(v) for v in self.memories.values()),
            "per_npc": {k: len(v) for k, v in self.memories.items()},
        }


# Global store instance
npc_memory_store = NPCMemoryStore()
