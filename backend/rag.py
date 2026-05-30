"""
RAG (Retrieval-Augmented Generation) pipeline for the AI Dungeon Master.

Phase 2: semantic search over the Harry Potter world corpus.
Phase 6 upgrade: backed by VectorStore (Chroma when installed, numpy otherwise).

The narrator's static knowledge-graph context is good for "where am I / who is
here", but it can't answer "what does this player input *mean* in the wider
lore". This retriever embeds every lore passage (locations, NPCs, factions,
items, spells, lore entries) once and, on each turn, pulls the handful of
passages most relevant to the player's action so the LLM narrates with the
right canon at hand.
"""
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from vector_store import VectorStore, embed as vs_embed, EMBED_MODEL

DATA_DIR   = Path(__file__).parent / "data"
LORE_PATH  = DATA_DIR / "world_lore.json"


# ── Document building ────────────────────────────────────────────────────────

def _build_documents(lore: Dict[str, Any]) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []

    world = lore.get("world", {})
    if world:
        rules = " ".join(world.get("rules", []))
        docs.append({
            "id": "world", "type": "world",
            "title": world.get("name", "The Wizarding World"),
            "text": (f"{world.get('name', '')}. {world.get('description', '')} "
                     f"Era: {world.get('age', '')}. Tone: {world.get('tone', '')}. "
                     f"Rules of magic: {rules}"),
        })

    for loc in lore.get("locations", []):
        secrets = " ".join(loc.get("secrets", []))
        docs.append({
            "id": loc["id"], "type": "location",
            "title": loc.get("name", loc["id"]),
            "text": (f"Location: {loc.get('name', '')} ({loc.get('type', '')}). "
                     f"{loc.get('description', '')} "
                     f"Atmosphere: {loc.get('atmosphere', '')}. Secret: {secrets}"),
        })

    for npc in lore.get("npcs", []):
        knows = ", ".join(npc.get("knows", []))
        docs.append({
            "id": npc["id"], "type": "npc",
            "title": npc.get("name", npc["id"]),
            "text": (f"Character: {npc.get('name', '')}, {npc.get('role', '')} "
                     f"(attitude: {npc.get('attitude', '')}). "
                     f"{npc.get('description', '')} Knows about: {knows}."),
        })

    for fac in lore.get("factions", []):
        docs.append({
            "id": fac["id"], "type": "faction",
            "title": fac.get("name", fac["id"]),
            "text": (f"Faction: {fac.get('name', '')} ({fac.get('alignment', '')}). "
                     f"{fac.get('description', '')}"),
        })

    for item in lore.get("items", []):
        docs.append({
            "id": item["id"], "type": "item",
            "title": item.get("name", item["id"]),
            "text": (f"Item: {item.get('name', '')} ({item.get('type', '')}). "
                     f"{item.get('description', '')}"),
        })

    for spell in lore.get("spells", []):
        name = spell.get("name", "")
        docs.append({
            "id": f"spell_{name.lower().replace(' ', '_')}",
            "type": "spell", "title": name,
            "text": (f"Spell: {name} ({spell.get('type', '')}, "
                     f"difficulty {spell.get('difficulty', '')}). "
                     f"Effect: {spell.get('effect', '')}."),
        })

    for entry in lore.get("lore_entries", []):
        docs.append({
            "id": entry["id"], "type": "lore",
            "title": entry.get("title", entry["id"]),
            "text": f"{entry.get('title', '')}: {entry.get('content', '')}",
        })

    return docs


# ── Retriever ────────────────────────────────────────────────────────────────

class LoreRetriever:
    """Semantic retriever over the world lore corpus, backed by VectorStore."""

    def __init__(self, lore_path: Path = LORE_PATH, model_name: str = EMBED_MODEL):
        self.lore_path  = Path(lore_path)
        self.model_name = model_name
        self.docs: List[Dict[str, str]] = []
        self._store: Optional[VectorStore] = None
        self._ready = False

    # -- lifecycle ------------------------------------------------------------

    def _corpus_hash(self) -> str:
        raw = self.lore_path.read_bytes()
        return hashlib.sha256(raw + self.model_name.encode()).hexdigest()[:16]

    def initialize(self) -> "LoreRetriever":
        if self._ready:
            return self

        lore = json.loads(self.lore_path.read_text(encoding="utf-8"))
        self.docs = _build_documents(lore)

        self._store = VectorStore("lore", model_name=self.model_name)

        # Check if the store already has the right number of docs (Chroma is persistent)
        if self._store.backend == "chroma" and self._store.count() == len(self.docs):
            print(f"[RAG] Chroma store has {len(self.docs)} passages — reusing.")
        else:
            # (Re)build the index
            ids    = [d["id"] for d in self.docs]
            texts  = [d["text"] for d in self.docs]
            metas  = [{"type": d["type"], "title": d["title"]} for d in self.docs]
            self._store.upsert(ids, texts, metas)
            print(f"[RAG] Indexed {len(self.docs)} passages "
                  f"via {self._store.backend} ({self.model_name}).")

        self._ready = True
        return self

    # -- search ---------------------------------------------------------------

    def search(self, query: str, top_k: int = 4, min_score: float = 0.15) -> List[Dict[str, Any]]:
        if not self._ready:
            self.initialize()
        if not query.strip() or not self.docs:
            return []
        hits = self._store.query(query, n=top_k, min_score=min_score)
        return [
            {"id": h.id, "type": h.metadata.get("type", ""),
             "title": h.metadata.get("title", h.id),
             "text": h.text, "score": h.score}
            for h in hits
        ]

    def retrieve_context(self, query: str, top_k: int = 4) -> str:
        hits = self.search(query, top_k=top_k)
        if not hits:
            return ""
        lines = ["[RELEVANT LORE — weave in naturally, do not quote verbatim]"]
        for h in hits:
            lines.append(f"• ({h['type']}) {h['title']}: {h['text']}")
        return "\n".join(lines)

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """Shared embedding helper (normalized vectors) for other modules."""
        return vs_embed(texts, self.model_name)

    def stats(self) -> Dict[str, Any]:
        if not self._ready:
            self.initialize()
        return {
            "backend": self._store.backend if self._store else "uninitialized",
            "model":   self.model_name,
            "passages": len(self.docs),
            "ready":   self._ready,
            **(self._store.stats() if self._store else {}),
        }


# Global retriever instance (index built lazily on first use)
lore_retriever = LoreRetriever()
