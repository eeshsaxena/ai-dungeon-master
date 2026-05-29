"""
Long-term NPC memory for the AI Dungeon Master (Phase 2.5).

Phase 1 memory (memory.py) is per-session story history. This adds a layer that
persists *per NPC across sessions*: each character remembers what the player did
in their presence, so when you return to the Three Broomsticks weeks later,
Madam Rosmerta can reference that time you smashed up Knockturn Alley.

Recall is semantic — when the player interacts near an NPC, we surface that
NPC's most relevant past memories (via shared embeddings from rag.py), falling
back to plain recency when the embedding model is unavailable.

Persistence has two layers, both on disk:
  - npc_memories.json   — human-readable memory text per NPC
  - npc_vectors.npz     — the embedding for each memory, so recall does NOT
                          recompute every vector on startup. A per-NPC content
                          hash (in npc_vectors.meta.json) rebuilds only the NPCs
                          whose memories actually changed, and a model-name guard
                          invalidates the whole cache if the embedding model
                          changes. This is the project's lightweight, dependency
                          -free persistent vector store (Chroma/FAISS optional later).
"""
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from rag import lore_retriever

STORE_DIR = Path(__file__).parent / "npc_memory_store"
STORE_PATH = STORE_DIR / "npc_memories.json"
VEC_PATH = STORE_DIR / "npc_vectors.npz"
VEC_META_PATH = STORE_DIR / "npc_vectors.meta.json"

MAX_MEMORIES_PER_NPC = 50


def _texts_hash(texts: List[str]) -> str:
    """Stable hash of an NPC's memory texts — changes iff the memories change."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class NPCMemoryStore:
    """Per-NPC long-term memory with semantic recall."""

    def __init__(self, persist: bool = True):
        self.persist = persist
        # npc_id -> list of memory dicts {text, ts, session_id}
        self.memories: Dict[str, List[Dict[str, Any]]] = {}
        # npc_id -> np.ndarray of embeddings aligned with self.memories[npc_id]
        self._vectors: Dict[str, Optional[np.ndarray]] = {}
        # npc_id -> hash of the texts the cached vectors were built from
        self._vec_hashes: Dict[str, str] = {}
        if self.persist:
            self._load()
            self._load_vectors()

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

        # Invalidate cached vectors for this NPC; rebuilt and re-persisted lazily on recall
        self._vectors.pop(npc_id, None)
        self._vec_hashes.pop(npc_id, None)
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
        texts = [m["text"] for m in self.memories.get(npc_id, [])]
        if not texts:
            self._vectors.pop(npc_id, None)
            self._vec_hashes.pop(npc_id, None)
            return None

        h = _texts_hash(texts)
        # Reuse persisted/cached vectors when the memories are unchanged
        if self._vec_hashes.get(npc_id) == h and self._vectors.get(npc_id) is not None:
            return self._vectors[npc_id]

        vecs = lore_retriever.embed(texts)
        if vecs is not None:
            self._vectors[npc_id] = vecs
            self._vec_hashes[npc_id] = h
            if self.persist:
                self._save_vectors()
        else:  # model unavailable — drop stale vectors, recall falls back to recency
            self._vectors.pop(npc_id, None)
            self._vec_hashes.pop(npc_id, None)
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

    def _save_vectors(self) -> None:
        """Persist all cached NPC embeddings + their content hashes to disk."""
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        arrays = {npc: v for npc, v in self._vectors.items() if v is not None}
        if not arrays:
            return
        np.savez_compressed(VEC_PATH, **arrays)
        VEC_META_PATH.write_text(json.dumps({
            "model": getattr(lore_retriever, "model_name", ""),
            "hashes": self._vec_hashes,
        }), encoding="utf-8")

    def _load_vectors(self) -> None:
        """Load persisted embeddings, keeping only those still matching the
        current memories and the current embedding model. Mismatches are left
        out and rebuilt lazily on first recall."""
        if not (VEC_PATH.exists() and VEC_META_PATH.exists()):
            return
        try:
            meta = json.loads(VEC_META_PATH.read_text(encoding="utf-8"))
            if meta.get("model") != getattr(lore_retriever, "model_name", ""):
                return  # model changed → all cached vectors are stale
            stored_hashes = meta.get("hashes", {})
            with np.load(VEC_PATH) as data:
                for npc_id in data.files:
                    texts = [m["text"] for m in self.memories.get(npc_id, [])]
                    if texts and stored_hashes.get(npc_id) == _texts_hash(texts):
                        self._vectors[npc_id] = data[npc_id].astype(np.float32)
                        self._vec_hashes[npc_id] = stored_hashes[npc_id]
        except Exception as e:
            print(f"[NPCMemory] Could not load vector cache ({e}); will rebuild lazily.")

    # -- introspection --------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "npcs_tracked": len(self.memories),
            "total_memories": sum(len(v) for v in self.memories.values()),
            "per_npc": {k: len(v) for k, v in self.memories.items()},
            "vectors_cached": sum(
                int(v.shape[0]) for v in self._vectors.values() if v is not None
            ),
            "vector_store_persisted": VEC_PATH.exists(),
        }


# Global store instance
npc_memory_store = NPCMemoryStore()
