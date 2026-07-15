"""
Long-term NPC memory for the AI Dungeon Master (Phase 2.5 / Phase 6 upgrade).

Each NPC accumulates memories of what the player did in their presence, persisted
across sessions. Recall is semantic — the most relevant past memories are surfaced
when the player interacts near an NPC.

Storage:
  - npc_memories.json — human-readable memory text per NPC (always written)
  - VectorStore("npc_memory") — embeddings backed by Chroma (persistent, no
    manual .npz management) or numpy (auto-fallback when chromadb is not installed)

The Phase 2.5 .npz vector cache is superseded by VectorStore. Existing data in
npc_memory_store/npc_memories.json is loaded as before; the vector index is
rebuilt from it on first startup then kept hot by VectorStore.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from vector_store import VectorStore, EMBED_MODEL

STORE_DIR  = Path(__file__).parent / "npc_memory_store"
STORE_PATH = STORE_DIR / "npc_memories.json"

MAX_MEMORIES_PER_NPC = 50


class NPCMemoryStore:
    """Per-NPC long-term memory with semantic recall, backed by VectorStore."""

    def __init__(self, persist: bool = True):
        self.persist = persist
        # npc_id -> list of {text, ts, session_id}
        self.memories: Dict[str, List[Dict[str, Any]]] = {}
        self._store: Optional[VectorStore] = None
        if persist:
            self._load()
        self._store = VectorStore("npc_memory", model_name=EMBED_MODEL)
        # Seed the vector store with persisted memories
        self._rebuild_index()

    # ── writing ───────────────────────────────────────────────────────────────

    def record(self, npc_id: str, text: str, session_id: str = "") -> None:
        """Store a new memory for an NPC and update the vector index."""
        if not npc_id or not text or not text.strip():
            return
        bucket = self.memories.setdefault(npc_id, [])
        bucket.append({"text": text.strip(), "ts": time.time(), "session_id": session_id})

        # Cap per-NPC history
        if len(bucket) > MAX_MEMORIES_PER_NPC:
            # Drop oldest — remove them from the vector store too
            if self._store:
                try:
                    self._store.delete(where={"npc_id": npc_id})
                    # Re-index the kept memories (simpler than tracking individual IDs)
                    self.memories[npc_id] = bucket[-MAX_MEMORIES_PER_NPC:]
                    self._index_npc(npc_id)
                except Exception:
                    self.memories[npc_id] = bucket[-MAX_MEMORIES_PER_NPC:]
            else:
                self.memories[npc_id] = bucket[-MAX_MEMORIES_PER_NPC:]
        else:
            # Just append the new memory to the index
            mem_idx = len(bucket) - 1
            mid = _mem_id(npc_id, mem_idx)
            if self._store:
                self._store.upsert(
                    ids=[mid],
                    texts=[text.strip()],
                    metadatas=[{"npc_id": npc_id, "ts": time.time(),
                                "session_id": session_id}],
                )

        if self.persist:
            self._save()

    # ── reading ───────────────────────────────────────────────────────────────

    def recall(self, npc_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return this NPC's most relevant past memories for the query."""
        bucket = self.memories.get(npc_id, [])
        if not bucket:
            return []

        if self._store and query.strip():
            n = min(top_k, len(bucket))
            hits = self._store.query(query, n=n, where={"npc_id": npc_id})
            if hits:
                return [
                    {"text": h.text, "ts": h.metadata.get("ts", 0),
                     "session_id": h.metadata.get("session_id", ""),
                     "score": h.score}
                    for h in hits
                ]

        # Recency fallback
        return [{**m, "score": None} for m in bucket[-top_k:][::-1]]

    def recall_text(self, npc_id: str, npc_name: str, query: str, top_k: int = 3) -> str:
        """Format recalled memories as a context block, or '' if none."""
        hits = self.recall(npc_id, query, top_k)
        if not hits:
            return ""
        lines = [f"{npc_name} remembers:"]
        for h in hits:
            lines.append(f"  - {h['text']}")
        return "\n".join(lines)

    # ── index helpers ─────────────────────────────────────────────────────────

    def _mem_id(self, npc_id: str, idx: int) -> str:
        return _mem_id(npc_id, idx)

    def _index_npc(self, npc_id: str) -> None:
        """(Re)index all memories for a single NPC."""
        bucket = self.memories.get(npc_id, [])
        if not bucket or not self._store:
            return
        ids   = [_mem_id(npc_id, i) for i in range(len(bucket))]
        texts = [m["text"] for m in bucket]
        metas = [{"npc_id": npc_id, "ts": m["ts"],
                  "session_id": m.get("session_id", "")} for m in bucket]
        self._store.upsert(ids, texts, metas)

    def _rebuild_index(self) -> None:
        """Index all persisted memories into VectorStore (idempotent via upsert)."""
        if not self._store:
            return
        for npc_id in self.memories:
            self._index_npc(npc_id)
        total = sum(len(v) for v in self.memories.values())
        if total:
            print(f"[NPCMemory] Indexed {total} memories across "
                  f"{len(self.memories)} NPCs via {self._store.backend}.")

    # ── persistence ───────────────────────────────────────────────────────────

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

    # ── introspection ─────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "npcs_tracked": len(self.memories),
            "total_memories": sum(len(v) for v in self.memories.values()),
            "per_npc": {k: len(v) for k, v in self.memories.items()},
            "vector_backend": self._store.backend if self._store else "none",
            "vector_count": self._store.count() if self._store else 0,
        }


def _mem_id(npc_id: str, idx: int) -> str:
    """Stable unique ID for a memory entry."""
    return f"{npc_id}__{idx}"


# Global store instance
npc_memory_store = NPCMemoryStore()
