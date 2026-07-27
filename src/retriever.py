import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.exceptions import IndexNotBuiltError

log = logging.getLogger("rag.retriever")


@dataclass
class Chunk:
    doc_id: str
    text: str
    chunk_id: int
    meta: dict = field(default_factory=dict)

    def __hash__(self):
        return hash((self.doc_id, self.chunk_id))


class DenseRetriever:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu",
                 use_hybrid: bool = False):
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name, device=device)
        self.index = None
        self.chunks: list[Chunk] = []
        self.use_hybrid = use_hybrid
        self._bm25 = None

    def _ensure_bm25(self, texts: list[str]):
        if not self.use_hybrid:
            return
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [t.lower().split() for t in texts]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            log.warning("rank_bm25 not installed — install with: pip install rank-bm25")
            self.use_hybrid = False

    def _embed(self, texts: list[str]) -> np.ndarray:
        emb = self.encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
            batch_size=64,
        )
        return np.asarray(emb, dtype="float32")

    def add_documents(self, docs: dict[str, str], chunk_size: int = 220, overlap: int = 40):
        import faiss

        new_chunks: list[Chunk] = []
        for doc_id, text in docs.items():
            words = text.split()
            step = max(chunk_size - overlap, 1)
            for i, start in enumerate(range(0, max(len(words), 1), step)):
                piece = " ".join(words[start : start + chunk_size]).strip()
                if piece:
                    new_chunks.append(Chunk(doc_id=doc_id, text=piece, chunk_id=i))
        if not new_chunks:
            log.warning("add_documents called with empty corpus")
            return

        texts = [c.text for c in new_chunks]
        embs = self._embed(texts)
        if self.index is None:
            self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        self.chunks.extend(new_chunks)

        if self.use_hybrid:
            self._ensure_bm25([c.text for c in self.chunks])

        log.info("indexed %d chunks (total %d)", len(new_chunks), len(self.chunks))

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            raise IndexNotBuiltError("no documents indexed — call add_documents() or load()")

        q = self._embed([query])
        dense_scores, ids = self.index.search(q, min(top_k * 2, len(self.chunks)))
        dense_results = {self.chunks[i]: float(s) for i, s in zip(ids[0], dense_scores[0]) if i != -1}

        if self.use_hybrid and self._bm25 is not None:
            tokenized_query = query.lower().split()
            bm25_scores = self._bm25.get_scores(tokenized_query)
            for chunk, score in zip(self.chunks, bm25_scores):
                if chunk in dense_results:
                    dense_results[chunk] = self._rrf(dense_results[chunk], score, k=60)
                else:
                    dense_results[chunk] = score

        ranked = sorted(dense_results.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def _rrf(self, dense_score: float, bm25_score: float, k: int = 60) -> float:
        dense_rank = max(1, int((1 - dense_score) * 100))
        bm25_rank = max(1, int((1 - min(bm25_score / 10, 1)) * 100))
        return 1 / (k + dense_rank) + 1 / (k + bm25_rank)

    def save(self, dir_path: str):
        import faiss

        if self.index is None:
            raise IndexNotBuiltError("nothing to save")
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        (p / "chunks.jsonl").write_text(
            "\n".join(json.dumps(asdict(c)) for c in self.chunks), encoding="utf-8"
        )
        (p / "meta.json").write_text(json.dumps({"model": self.model_name}))
        log.info("index saved to %s (%d chunks)", dir_path, len(self.chunks))

    def load(self, dir_path: str) -> bool:
        import faiss

        p = Path(dir_path)
        if not (p / "index.faiss").exists():
            return False
        meta = json.loads((p / "meta.json").read_text())
        if meta.get("model") != self.model_name:
            log.warning("index built with %s, current model %s — rebuilding recommended",
                        meta.get("model"), self.model_name)
        self.index = faiss.read_index(str(p / "index.faiss"))
        self.chunks = [
            Chunk(**json.loads(line))
            for line in (p / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if self.use_hybrid:
            self._ensure_bm25([c.text for c in self.chunks])
        log.info("index loaded from %s (%d chunks)", dir_path, len(self.chunks))
        return True
