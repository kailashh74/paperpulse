import faiss
import json
import numpy as np
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

BASE = Path(os.path.dirname(os.path.abspath(__file__)))


class Retriever:
    def __init__(self, index_path=None, meta_path=None):
        self.index_path = str(index_path or BASE / "faiss.index")
        self.meta_path = str(meta_path or BASE / "metadata.json")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = None
        self.metadata = []
        self.bm25 = None
        self._load()

    def _load(self):
        print(f"[Retriever] index={self.index_path} exists={Path(self.index_path).exists()}")
        if not Path(self.index_path).exists() or not Path(self.meta_path).exists():
            print("[Retriever] Index files missing. Run live_ingest.py first.")
            return
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"[Retriever] Loaded {len(self.metadata)} papers.")
        self._build_bm25()

    def _build_bm25(self):
        if not self.metadata:
            self.bm25 = None
            return
        print("[Retriever] Building BM25...")
        tokenized = [doc["text"].lower().split() for doc in self.metadata]
        self.bm25 = BM25Okapi(tokenized)
        print(f"[Retriever] BM25 ready — {len(self.metadata)} docs.")

    def reload(self):
        self.index = None
        self.metadata = []
        self.bm25 = None
        self._load()

    def _rrf(self, sem_indices, bm25_indices, k=60):
        scores = {}
        for rank, idx in enumerate(sem_indices):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        for rank, idx in enumerate(bm25_indices):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _normalize(self, rrf_pairs):
        if not rrf_pairs:
            return {}
        scores = [s for _, s in rrf_pairs]
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return {idx: 1.0 for idx, _ in rrf_pairs}
        return {idx: (s - lo) / (hi - lo) for idx, s in rrf_pairs}

    def retrieve(self, query: str, top_k: int = 5) -> list:
        if self.index is None or not self.metadata:
            print("[Retriever] Index not loaded — returning empty.")
            return []

        fetch_k = top_k * 6
        vec = self.model.encode([query], normalize_embeddings=True)
        sem_scores, sem_indices = self.index.search(np.array(vec, dtype="float32"), fetch_k)
        sem_idx_list = [int(i) for i in sem_indices[0] if i != -1]
        sem_score_map = {int(i): float(s) for s, i in zip(sem_scores[0], sem_indices[0]) if i != -1}

        if self.bm25:
            bm25_scores = self.bm25.get_scores(query.lower().split())
            bm25_idx_list = np.argsort(bm25_scores)[::-1][:fetch_k].tolist()
            fused = self._rrf(sem_idx_list, bm25_idx_list)
        else:
            fused = [(idx, sem_score_map[idx]) for idx in sem_idx_list]

        normalized = self._normalize(fused)
        results = []
        seen_urls = set()

        for idx, rrf_score in fused:
            if idx >= len(self.metadata):
                continue
            chunk = self.metadata[idx].copy()
            url = chunk.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            chunk["score"] = normalized.get(idx, 0.0)
            chunk["rrf_score"] = float(rrf_score)
            chunk["sem_score"] = sem_score_map.get(idx, 0.0)
            results.append(chunk)
            if len(results) == top_k:
                break

        top = results[0]['score'] if results else 0
        print(f"[Retriever] '{query[:50]}' → {len(results)} results, top_score={top:.3f}")
        
        return results