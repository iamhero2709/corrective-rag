"""
Semantic cache for RAG queries.
Stores query-result pairs and retrieves them based on embedding similarity.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class CacheEntry:
    query: str
    query_embedding: list
    answer: str
    confidence: str
    timestamp: float
    ttl: float = 86400.0  # 24 hours default

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.92, ttl: float = 86400.0,
                 max_entries: int = 1000, cache_path: str = "data/query_cache.json"):
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.max_entries = max_entries
        self.cache_path = Path(cache_path)
        self.entries: list[CacheEntry] = []
        self._load()

    def _load(self):
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text())
                self.entries = [CacheEntry(**e) for e in data]
                self.entries = [e for e in self.entries if not e.is_expired()]
            except Exception:
                self.entries = []

    def _save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps([asdict(e) for e in self.entries]))

    def get(self, query_embedding: np.ndarray) -> Optional[dict]:
        if not self.entries:
            return None

        best_score = -1
        best_entry = None

        for entry in self.entries:
            if entry.is_expired():
                continue
            cached_emb = np.array(entry.query_embedding)
            sim = float(np.dot(query_embedding, cached_emb) /
                       (np.linalg.norm(query_embedding) * np.linalg.norm(cached_emb) + 1e-8))
            if sim > best_score:
                best_score = sim
                best_entry = entry

        if best_score >= self.similarity_threshold and best_entry:
            return {
                "answer": best_entry.answer,
                "confidence": best_entry.confidence,
                "cache_hit": True,
                "similarity": best_score,
            }
        return None

    def put(self, query: str, query_embedding: np.ndarray, answer: str, confidence: str):
        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding.tolist(),
            answer=answer,
            confidence=confidence,
            timestamp=time.time(),
            ttl=self.ttl,
        )
        self.entries.append(entry)

        if len(self.entries) > self.max_entries:
            self.entries.sort(key=lambda e: e.timestamp, reverse=True)
            self.entries = self.entries[:self.max_entries]

        self._save()

    def clear(self):
        self.entries.clear()
        self._save()

    def stats(self) -> dict:
        valid = [e for e in self.entries if not e.is_expired()]
        return {
            "total_entries": len(self.entries),
            "valid_entries": len(valid),
            "expired_entries": len(self.entries) - len(valid),
        }
