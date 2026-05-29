"""
RAG (Retrieval-Augmented Generation) pipeline for the AI Dungeon Master.

Phase 2: semantic search over the Harry Potter world corpus.

The narrator's static knowledge-graph context is good for "where am I / who is
here", but it can't answer "what does this player input *mean* in the wider
lore". This retriever embeds every lore passage (locations, NPCs, factions,
items, spells, lore entries) once and, on each turn, pulls the handful of
passages most relevant to the player's action so the LLM narrates with the
right canon at hand.

Primary backend: sentence-transformers (all-MiniLM-L6-v2) + cosine similarity.
If the model cannot be loaded (offline, not installed), it degrades to a
keyword-overlap retriever so the feature still returns useful results.
"""
import os
import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
LORE_PATH = DATA_DIR / "world_lore.json"
CACHE_DIR = Path(__file__).parent / "embeddings_cache"
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


# ── Document building ────────────────────────────────────────────────────────

def _build_documents(lore: Dict[str, Any]) -> List[Dict[str, str]]:
    """Flatten the world lore JSON into self-contained retrievable passages."""
    docs: List[Dict[str, str]] = []

    world = lore.get("world", {})
    if world:
        rules = " ".join(world.get("rules", []))
        docs.append({
            "id": "world",
            "type": "world",
            "title": world.get("name", "The Wizarding World"),
            "text": f"{world.get('name', '')}. {world.get('description', '')} "
                    f"Era: {world.get('age', '')}. Tone: {world.get('tone', '')}. "
                    f"Rules of magic: {rules}",
        })

    for loc in lore.get("locations", []):
        secrets = " ".join(loc.get("secrets", []))
        docs.append({
            "id": loc["id"],
            "type": "location",
            "title": loc.get("name", loc["id"]),
            "text": f"Location: {loc.get('name', '')} ({loc.get('type', '')}). "
                    f"{loc.get('description', '')} "
                    f"Atmosphere: {loc.get('atmosphere', '')}. "
                    f"Secret: {secrets}",
        })

    for npc in lore.get("npcs", []):
        knows = ", ".join(npc.get("knows", []))
        docs.append({
            "id": npc["id"],
            "type": "npc",
            "title": npc.get("name", npc["id"]),
            "text": f"Character: {npc.get('name', '')}, {npc.get('role', '')} "
                    f"(attitude: {npc.get('attitude', '')}). "
                    f"{npc.get('description', '')} Knows about: {knows}.",
        })

    for fac in lore.get("factions", []):
        docs.append({
            "id": fac["id"],
            "type": "faction",
            "title": fac.get("name", fac["id"]),
            "text": f"Faction: {fac.get('name', '')} ({fac.get('alignment', '')}). "
                    f"{fac.get('description', '')}",
        })

    for item in lore.get("items", []):
        docs.append({
            "id": item["id"],
            "type": "item",
            "title": item.get("name", item["id"]),
            "text": f"Item: {item.get('name', '')} ({item.get('type', '')}). "
                    f"{item.get('description', '')}",
        })

    for spell in lore.get("spells", []):
        name = spell.get("name", "")
        docs.append({
            "id": f"spell_{name.lower().replace(' ', '_')}",
            "type": "spell",
            "title": name,
            "text": f"Spell: {name} ({spell.get('type', '')}, "
                    f"difficulty {spell.get('difficulty', '')}). "
                    f"Effect: {spell.get('effect', '')}.",
        })

    for entry in lore.get("lore_entries", []):
        docs.append({
            "id": entry["id"],
            "type": "lore",
            "title": entry.get("title", entry["id"]),
            "text": f"{entry.get('title', '')}: {entry.get('content', '')}",
        })

    return docs


# ── Retriever ────────────────────────────────────────────────────────────────

class LoreRetriever:
    """Semantic retriever over the world lore corpus."""

    def __init__(self, lore_path: Path = LORE_PATH, model_name: str = EMBED_MODEL):
        self.lore_path = Path(lore_path)
        self.model_name = model_name
        self.docs: List[Dict[str, str]] = []
        self.vectors: Optional[np.ndarray] = None
        self.backend = "uninitialized"  # -> "embeddings" | "keyword"
        self._model = None
        self._ready = False

    # -- lifecycle ------------------------------------------------------------

    def _corpus_hash(self) -> str:
        raw = self.lore_path.read_bytes()
        return hashlib.sha256(raw + self.model_name.encode()).hexdigest()[:16]

    def _load_model(self):
        """Lazy-load the embedding model; return True on success."""
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            return True
        except Exception as e:  # offline, missing weights, etc.
            print(f"[RAG] Embedding model unavailable ({e}); using keyword fallback.")
            return False

    def initialize(self) -> "LoreRetriever":
        """Build (or load cached) the index. Safe to call repeatedly."""
        if self._ready:
            return self

        lore = json.loads(self.lore_path.read_text(encoding="utf-8"))
        self.docs = _build_documents(lore)

        if self._try_load_cache():
            self.backend = "embeddings"
            self._ready = True
            print(f"[RAG] Loaded {len(self.docs)} passages from embedding cache.")
            return self

        if self._load_model():
            texts = [d["text"] for d in self.docs]
            vecs = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            self.vectors = np.asarray(vecs, dtype=np.float32)
            self.backend = "embeddings"
            self._save_cache()
            print(f"[RAG] Embedded {len(self.docs)} passages with {self.model_name}.")
        else:
            self.backend = "keyword"
            print(f"[RAG] Keyword index ready over {len(self.docs)} passages.")

        self._ready = True
        return self

    # -- cache ----------------------------------------------------------------

    def _cache_paths(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR / "lore_vectors.npz", CACHE_DIR / "lore_index.meta.json"

    def _save_cache(self):
        vec_path, meta_path = self._cache_paths()
        np.savez_compressed(vec_path, vectors=self.vectors)
        meta_path.write_text(json.dumps({
            "hash": self._corpus_hash(),
            "model": self.model_name,
            "doc_ids": [d["id"] for d in self.docs],
        }), encoding="utf-8")

    def _try_load_cache(self) -> bool:
        vec_path, meta_path = self._cache_paths()
        if not (vec_path.exists() and meta_path.exists()):
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("hash") != self._corpus_hash():
                return False
            if meta.get("doc_ids") != [d["id"] for d in self.docs]:
                return False
            with np.load(vec_path) as data:
                self.vectors = data["vectors"].astype(np.float32)
            return self.vectors.shape[0] == len(self.docs)
        except Exception:
            return False

    # -- search ---------------------------------------------------------------

    def search(self, query: str, top_k: int = 4, min_score: float = 0.15) -> List[Dict[str, Any]]:
        """Return the top_k most relevant passages for the query."""
        if not self._ready:
            self.initialize()
        if not query or not query.strip() or not self.docs:
            return []

        if self.backend == "embeddings" and self.vectors is not None and self._load_model():
            scores = self._embedding_scores(query)
        else:
            scores = self._keyword_scores(query)

        ranked = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in ranked:
            score = float(scores[idx])
            if score < min_score:
                continue
            doc = self.docs[idx]
            results.append({
                "id": doc["id"],
                "type": doc["type"],
                "title": doc["title"],
                "text": doc["text"],
                "score": round(score, 4),
            })
        return results

    def _embedding_scores(self, query: str) -> np.ndarray:
        q = self._model.encode([query], normalize_embeddings=True)[0]
        return self.vectors @ np.asarray(q, dtype=np.float32)

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """Shared embedding helper (normalized vectors) for other modules such
        as NPC memory. Returns None if the model is unavailable."""
        if not texts or not self._load_model():
            return None
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def _keyword_scores(self, query: str) -> np.ndarray:
        terms = set(re.findall(r"[a-z']+", query.lower()))
        terms = {t for t in terms if len(t) > 2}
        scores = np.zeros(len(self.docs), dtype=np.float32)
        if not terms:
            return scores
        for i, doc in enumerate(self.docs):
            words = set(re.findall(r"[a-z']+", doc["text"].lower()))
            overlap = len(terms & words)
            scores[i] = overlap / len(terms)
        return scores

    def retrieve_context(self, query: str, top_k: int = 4) -> str:
        """Format the top passages as a context block for the narrator prompt."""
        hits = self.search(query, top_k=top_k)
        if not hits:
            return ""
        lines = ["[RELEVANT LORE — weave in naturally, do not quote verbatim]"]
        for h in hits:
            lines.append(f"• ({h['type']}) {h['title']}: {h['text']}")
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        if not self._ready:
            self.initialize()
        return {
            "backend": self.backend,
            "model": self.model_name,
            "passages": len(self.docs),
            "ready": self._ready,
        }


# Global retriever instance (index built lazily on first use)
lore_retriever = LoreRetriever()
