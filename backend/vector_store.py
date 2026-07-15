"""
Pluggable vector store for AI Dungeon Master (Phase 6 upgrade).

Auto-detects ChromaDB when installed; falls back to the existing
numpy/cosine-similarity approach so the game always works.

Usage
-----
    from vector_store import VectorStore
    vs = VectorStore("my_collection", model_name="all-MiniLM-L6-v2")
    vs.upsert(ids, texts, metadatas)   # add / update documents
    hits = vs.query(query_text, n=4)   # returns List[Hit]
    hits = vs.query(query_text, n=4, where={"npc_id": "rosmerta"})

The caller never needs to know whether Chroma or numpy is running underneath.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── backend detection ─────────────────────────────────────────────────────────

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "auto")   # "auto" | "chroma" | "numpy"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

try:
    import chromadb  # noqa: F401
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


def _use_chroma() -> bool:
    if VECTOR_BACKEND == "numpy":
        return False
    if VECTOR_BACKEND == "chroma":
        return _CHROMA_AVAILABLE
    return _CHROMA_AVAILABLE   # "auto"


# ── shared embedding helper ───────────────────────────────────────────────────

_embed_model_cache: Dict[str, Any] = {}


def _get_embed_model(model_name: str):
    if model_name not in _embed_model_cache:
        from sentence_transformers import SentenceTransformer
        _embed_model_cache[model_name] = SentenceTransformer(model_name)
    return _embed_model_cache[model_name]


def embed(texts: List[str], model_name: str = EMBED_MODEL) -> Optional[np.ndarray]:
    """Embed texts with sentence-transformers. Returns (N, D) float32 or None."""
    if not texts:
        return None
    try:
        model = _get_embed_model(model_name)
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)
    except Exception as e:
        print(f"[VectorStore] Embedding failed ({e})")
        return None


# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class Hit:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── VectorStore ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    Thin wrapper around ChromaDB (preferred) or a numpy flat index (fallback).
    One instance per logical collection (e.g. "lore", "npc_memory").
    """

    def __init__(self, name: str, model_name: str = EMBED_MODEL):
        self.name = name
        self.model_name = model_name
        self._backend: str = "uninitialized"
        # numpy fallback state
        self._ids:   List[str] = []
        self._texts: List[str] = []
        self._metas: List[Dict] = []
        self._vecs:  Optional[np.ndarray] = None
        # chroma state
        self._chroma_client: Any = None
        self._chroma_coll:   Any = None

        if _use_chroma():
            ok = self._init_chroma()
            if ok:
                self._backend = "chroma"
                return
        self._backend = "numpy"

    # ── Chroma backend ────────────────────────────────────────────────────────

    def _init_chroma(self) -> bool:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            ef = SentenceTransformerEmbeddingFunction(model_name=self.model_name)
            coll = client.get_or_create_collection(
                name=self.name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            self._chroma_client = client
            self._chroma_coll = coll
            return True
        except Exception as exc:
            print(f"[VectorStore:{self.name}] Chroma init failed ({exc}); using numpy.")
            return False

    # ── public API ────────────────────────────────────────────────────────────

    def upsert(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        """Add or replace documents. Silently drops empty inputs."""
        if not ids or not texts:
            return
        metas = metadatas or [{} for _ in ids]

        if self._backend == "chroma":
            try:
                self._chroma_coll.upsert(ids=ids, documents=texts, metadatas=metas)
                return
            except Exception as exc:
                print(f"[VectorStore:{self.name}] Chroma upsert failed ({exc}); falling to numpy.")
                self._backend = "numpy"

        # numpy path — replace matching ids, append new ones
        id_set = {doc_id: i for i, doc_id in enumerate(self._ids)}
        vecs = embed(texts, self.model_name)
        for j, doc_id in enumerate(ids):
            if doc_id in id_set:
                i = id_set[doc_id]
                self._texts[i] = texts[j]
                self._metas[i] = metas[j]
                if vecs is not None and self._vecs is not None:
                    self._vecs[i] = vecs[j]
            else:
                self._ids.append(doc_id)
                self._texts.append(texts[j])
                self._metas.append(metas[j])
                if vecs is not None:
                    v = vecs[j:j+1]
                    self._vecs = v if self._vecs is None else np.concatenate([self._vecs, v])

    def query(
        self,
        query_text: str,
        n: int = 4,
        where: Optional[Dict] = None,
        min_score: float = 0.0,
    ) -> List[Hit]:
        """Semantic nearest-neighbour search. Returns up to n hits."""
        if not query_text.strip():
            return []

        if self._backend == "chroma":
            return self._query_chroma(query_text, n, where, min_score)
        return self._query_numpy(query_text, n, where, min_score)

    def count(self, where: Optional[Dict] = None) -> int:
        if self._backend == "chroma":
            try:
                if where:
                    # Chroma doesn't expose count+where; approximate via get
                    res = self._chroma_coll.get(where=where, include=[])
                    return len(res["ids"])
                return self._chroma_coll.count()
            except Exception:
                pass
        if where:
            return sum(1 for m in self._metas if all(m.get(k) == v for k, v in where.items()))
        return len(self._ids)

    def delete(self, where: Dict) -> None:
        """Delete all documents matching metadata filter."""
        if self._backend == "chroma":
            try:
                self._chroma_coll.delete(where=where)
                return
            except Exception:
                pass
        # numpy: filter out
        keep = [i for i, m in enumerate(self._metas)
                if not all(m.get(k) == v for k, v in where.items())]
        self._ids   = [self._ids[i] for i in keep]
        self._texts = [self._texts[i] for i in keep]
        self._metas = [self._metas[i] for i in keep]
        if self._vecs is not None and len(keep) < len(self._vecs):
            self._vecs = self._vecs[keep] if keep else None

    @property
    def backend(self) -> str:
        return self._backend

    # ── internal query helpers ────────────────────────────────────────────────

    def _query_chroma(
        self, query_text: str, n: int, where: Optional[Dict], min_score: float
    ) -> List[Hit]:
        try:
            # Cap n to collection size to avoid Chroma error
            total = self.count(where=where)
            if total == 0:
                return []
            n_safe = max(1, min(n, total))
            kwargs: Dict[str, Any] = {
                "query_texts": [query_text],
                "n_results": n_safe,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where
            res = self._chroma_coll.query(**kwargs)
            hits = []
            for i in range(len(res["ids"][0])):
                # Chroma returns cosine distance (0=identical, 2=opposite);
                # convert to similarity in [0, 1]
                dist = float(res["distances"][0][i])
                score = round(1.0 - dist / 2.0, 4)
                if score < min_score:
                    continue
                hits.append(Hit(
                    id=res["ids"][0][i],
                    text=res["documents"][0][i],
                    score=score,
                    metadata=res["metadatas"][0][i] or {},
                ))
            return hits
        except Exception as exc:
            print(f"[VectorStore:{self.name}] Chroma query failed ({exc}); falling to numpy.")
            self._backend = "numpy"
            return self._query_numpy(query_text, n, where, min_score)

    def _query_numpy(
        self, query_text: str, n: int, where: Optional[Dict], min_score: float
    ) -> List[Hit]:
        if not self._ids:
            return []
        # Filter indices by where
        indices = [
            i for i, m in enumerate(self._metas)
            if not where or all(m.get(k) == v for k, v in where.items())
        ]
        if not indices:
            return []

        qvec = embed([query_text], self.model_name)
        if qvec is None or self._vecs is None:
            # keyword fallback
            import re
            terms = {t for t in re.findall(r"[a-z']+", query_text.lower()) if len(t) > 2}
            scores_raw = np.zeros(len(indices), dtype=np.float32)
            for j, i in enumerate(indices):
                words = set(re.findall(r"[a-z']+", self._texts[i].lower()))
                scores_raw[j] = len(terms & words) / max(len(terms), 1)
        else:
            sub_vecs = self._vecs[indices]
            scores_raw = sub_vecs @ qvec[0].astype(np.float32)

        order = np.argsort(scores_raw)[::-1][:n]
        hits = []
        for j in order:
            score = float(scores_raw[j])
            if score < min_score:
                continue
            i = indices[j]
            hits.append(Hit(
                id=self._ids[i],
                text=self._texts[i],
                score=round(score, 4),
                metadata=self._metas[i],
            ))
        return hits

    def stats(self) -> Dict[str, Any]:
        return {
            "backend": self._backend,
            "collection": self.name,
            "model": self.model_name,
            "count": self.count(),
        }
