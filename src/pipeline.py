"""
Corrective RAG pipeline:

  query
    └─> retrieve top-k
          └─> VERIFY each chunk (embedding + HRR structural + NLI)
                ├─ enough CORRECT chunks  -> generate
                ├─ AMBIGUOUS-heavy        -> rewrite query, retrieve again, merge
                └─ all INCORRECT          -> rewrite query, retry (max_retries)
    └─> generate from verified context
          └─> self_check
                ├─ SUPPORTED    -> done
                ├─ PARTIAL      -> regenerate once with tightened context
                └─ UNSUPPORTED  -> return INSUFFICIENT + audit trail

Every decision is logged to `trace` — this is your mechanistic-analysis
data. The paper's Figure 2 comes straight out of these traces.
"""

import time
from dataclasses import dataclass, field

from src.generator import SmallModelGenerator
from src.retriever import DenseRetriever, Chunk
from src.verifier import RetrievalVerifier, Verdict


REWRITE_PROMPT = """Rewrite this search query to find better documents. Make it more specific and keyword-rich. Reply with ONLY the rewritten query.

Original query: {query}
Rewritten query:"""


@dataclass
class PipelineTrace:
    steps: list[dict] = field(default_factory=list)

    def log(self, step: str, **kw):
        self.steps.append({"step": step, **kw})


@dataclass
class Citation:
    doc_id: str
    chunk_id: int
    text: str
    score: float = 0.0


@dataclass
class RAGAnswer:
    answer: str
    confidence: str  # HIGH | MEDIUM | LOW | ABSTAIN
    used_chunks: list[str]
    citations: list[Citation] = field(default_factory=list)
    trace: PipelineTrace = field(default_factory=PipelineTrace)
    latency_s: float = 0.0


class CorrectiveRAG:
    def __init__(
        self,
        retriever: DenseRetriever,
        verifier: RetrievalVerifier,
        generator: SmallModelGenerator,
        top_k: int = 5,
        min_correct: int = 2,
        max_retries: int = 2,
    ):
        self.retriever = retriever
        self.verifier = verifier
        self.generator = generator
        self.top_k = top_k
        self.min_correct = min_correct
        self.max_retries = max_retries

    def _verify_batch(self, query: str, results, trace: PipelineTrace):
        buckets = {Verdict.CORRECT: [], Verdict.AMBIGUOUS: [], Verdict.INCORRECT: []}
        vrs = self.verifier.verify_many(query, [(c.text, s) for c, s in results])
        for (chunk, emb_sim), vr in zip(results, vrs):
            buckets[vr.verdict].append((chunk, vr))
            trace.log(
                "verify",
                doc=chunk.doc_id,
                verdict=vr.verdict.value,
                fused=round(vr.fused_score, 3),
                emb=round(vr.embedding_sim, 3),
                struct=round(vr.structural_score, 3),
                entail=round(vr.entailment_score, 3),
            )
        return buckets

    def _rewrite(self, query: str, trace: PipelineTrace) -> str:
        new_q = self.generator._chat(
            REWRITE_PROMPT.format(query=query), max_new_tokens=48
        ).strip().strip('"')
        trace.log("rewrite", original=query, rewritten=new_q)
        return new_q or query

    def run(self, query: str) -> RAGAnswer:
        t0 = time.time()
        trace = PipelineTrace()
        current_q = query
        verified_chunks: list = []

        for attempt in range(self.max_retries + 1):
            trace.log("retrieve", attempt=attempt, query=current_q)
            results = self.retriever.search(current_q, top_k=self.top_k)
            buckets = self._verify_batch(query, results, trace)  # verify vs ORIGINAL query

            verified_chunks = buckets[Verdict.CORRECT] + buckets[Verdict.AMBIGUOUS]
            usable = len(buckets[Verdict.CORRECT]) + len(buckets[Verdict.AMBIGUOUS])
            if usable >= self.min_correct:
                break
            if attempt < self.max_retries:
                current_q = self._rewrite(current_q, trace)

        if not verified_chunks:
            trace.log("abstain", reason="no verified chunks after retries")
            return RAGAnswer("I don't have reliable information to answer this.",
                             "ABSTAIN", [], latency_s=time.time() - t0, trace=trace)

        # Sort by fused score, take best chunks as context
        verified_chunks.sort(key=lambda x: -x[1].fused_score)
        selected = verified_chunks[: self.top_k]
        context = [c.text for c, _ in selected]

        out = self.generator.answer(query, context)
        trace.log("generate", self_check=out.self_check, insufficient=out.insufficient)

        if out.insufficient:
            return RAGAnswer("I don't have reliable information to answer this.",
                             "ABSTAIN", context, latency_s=time.time() - t0, trace=trace)

        if out.self_check == "UNSUPPORTED":
            # One tightened retry: only CORRECT-verdict chunks
            strict_list = [(c, v) for c, v in verified_chunks if v.verdict == Verdict.CORRECT]
            if strict_list:
                selected = strict_list[: self.top_k]
                strict = [c.text for c, _ in selected]
                out = self.generator.answer(query, strict)
                trace.log("regenerate_strict", self_check=out.self_check)
                context = strict

        confidence = {
            "SUPPORTED": "HIGH",
            "PARTIAL": "MEDIUM",
            "UNSUPPORTED": "LOW",
            "SKIPPED": "LOW",
        }[out.self_check]

        citations = [
            Citation(doc_id=c.doc_id, chunk_id=c.chunk_id, text=c.text[:150], score=v.fused_score)
            for c, v in selected
        ]
        return RAGAnswer(out.answer, confidence, context, citations=citations,
                         latency_s=time.time() - t0, trace=trace)
