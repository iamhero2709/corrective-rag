"""
Retrieval Verifier — the core contribution.

Three signals per (query, chunk) pair, fused into a verdict:

  1. embedding_sim  — standard cosine (baseline signal, what everyone uses)
  2. structural     — HRR role-filler check: does the chunk plausibly contain
                      a filler for the role the query asks about?
  3. entailment     — lightweight NLI: does the chunk entail a hypothesis
                      formed from the query? (small cross-encoder, CPU-friendly)

Verdicts (CRAG-style): CORRECT / AMBIGUOUS / INCORRECT
  CORRECT   -> pass chunk to generator
  AMBIGUOUS -> keep, but trigger query rewrite + supplementary retrieval
  INCORRECT -> drop chunk, force re-retrieval

Ablation flags let you turn each signal off — that's your paper's Table 3.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.hrr import RoleFillerEncoder, cosine


class Verdict(Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class VerificationResult:
    verdict: Verdict
    embedding_sim: float
    structural_score: float
    entailment_score: float
    fused_score: float
    details: dict


def extract_role_fillers(text: str, spacy_nlp) -> list[tuple[str, str]]:
    """
    Cheap structural parse: (dep_role, lemma) pairs from a dependency parse.
    Roles = {subject, object, root_verb, entity}. Deliberately coarse —
    the point is structural *consistency*, not full semantic parsing.
    """
    doc = spacy_nlp(text[:2000])  # cap for speed
    pairs = []
    for tok in doc:
        if tok.dep_ in ("nsubj", "nsubjpass"):
            pairs.append(("subject", tok.lemma_.lower()))
        elif tok.dep_ in ("dobj", "pobj", "attr"):
            pairs.append(("object", tok.lemma_.lower()))
        elif tok.dep_ == "ROOT" and tok.pos_ == "VERB":
            pairs.append(("action", tok.lemma_.lower()))
    for ent in doc.ents:
        pairs.append((f"entity_{ent.label_.lower()}", ent.text.lower()))
    return pairs


class RetrievalVerifier:
    def __init__(
        self,
        hrr_dim: int = 2048,
        use_structural: bool = True,
        use_entailment: bool = True,
        use_embedding: bool = True,
        nli_model: str = "cross-encoder/nli-deberta-v3-xsmall",
        device: str = "cpu",
        thresholds: tuple[float, float] = (0.62, 0.40),  # (correct, incorrect)
        weights: tuple[float, float, float] = (0.30, 0.30, 0.40),  # (emb, struct, entail)
    ):
        self.use_structural = use_structural
        self.use_entailment = use_entailment
        self.use_embedding = use_embedding
        self.t_correct, self.t_incorrect = thresholds
        self.w_emb, self.w_struct, self.w_ent = weights

        self.encoder = RoleFillerEncoder(dim=hrr_dim)

        self.nlp = None
        if use_structural:
            import spacy

            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError as e:
                from src.exceptions import ModelLoadError
                raise ModelLoadError(
                    "spacy model missing — run: python -m spacy download en_core_web_sm"
                ) from e

        self.nli = None
        if use_entailment:
            from sentence_transformers import CrossEncoder

            self.nli = CrossEncoder(nli_model, device=device)

    # ---------- individual signals ----------

    def structural_score(self, query: str, chunk_text: str) -> tuple[float, dict]:
        """HRR check: encode both structures, probe the chunk trace with the
        query's roles, measure how strongly query-roles resolve to concrete
        fillers in the chunk. High = chunk is structurally answer-bearing."""
        q_pairs = extract_role_fillers(query, self.nlp)
        c_pairs = extract_role_fillers(chunk_text, self.nlp)
        if not q_pairs or not c_pairs:
            return 0.5, {"note": "no structure extracted, neutral"}

        chunk_trace = self.encoder.encode_structure(c_pairs)

        # Direct trace similarity (shared structure)
        query_trace = self.encoder.encode_structure(q_pairs)
        trace_sim = cosine(query_trace, chunk_trace)

        # Probe: do the query's roles resolve to clean fillers in the chunk?
        probe_scores = []
        for role, _ in q_pairs:
            candidates = self.encoder.probe(chunk_trace, role, top_k=1)
            probe_scores.append(candidates[0][1] if candidates else 0.0)
        probe_mean = float(np.mean(probe_scores)) if probe_scores else 0.0

        # Map to [0,1]-ish: trace_sim in [-1,1], probe cleanup sims are small
        # positive when a genuine filler exists (~0.15-0.45 at dim=2048)
        score = 0.5 * (trace_sim + 1) * 0.4 + min(probe_mean / 0.35, 1.0) * 0.6
        return float(np.clip(score, 0, 1)), {
            "trace_sim": trace_sim,
            "probe_mean": probe_mean,
            "q_pairs": q_pairs[:6],
        }

    def entailment_score(self, query: str, chunk_text: str) -> float:
        """P(entailment) that the chunk supports an answer to the query."""
        return self.entailment_scores(query, [chunk_text])[0]

    def entailment_scores(self, query: str, chunk_texts: list[str]) -> list[float]:
        """Batched NLI — one forward pass for all chunks of a query.
        nli-deberta label order: [contradiction, entailment, neutral].
        Converts questions to declarative statements for better NLI scoring."""
        if not chunk_texts:
            return []
        hypothesis = self._query_to_statement(query)
        logits = np.asarray(self.nli.predict([(c[:1500], hypothesis) for c in chunk_texts]))
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return [float(p[1] + 0.5 * p[2]) for p in probs]  # entail + half-credit neutral

    @staticmethod
    def _query_to_statement(query: str) -> str:
        """Convert a question to a declarative statement for NLI.
        'Who directed Inception?' -> 'The text contains information about who directed Inception'"""
        q = query.strip().rstrip("?").strip()
        return f"The text contains information about {q.lower()}"

    def verify_many(
        self, query: str, chunks_with_sims: list[tuple[str, float]]
    ) -> list["VerificationResult"]:
        """Batch verification: structural per-chunk (CPU-cheap), NLI batched
        (the expensive call). Use this in serving paths instead of verify()."""
        texts = [t for t, _ in chunks_with_sims]

        any_active = self.use_embedding or self.use_entailment or self.use_structural
        if not any_active:
            # Pure vanilla: no verification, all chunks pass as CORRECT
            return [
                VerificationResult(
                    verdict=Verdict.CORRECT,
                    embedding_sim=emb_sim,
                    structural_score=0.0,
                    entailment_score=0.0,
                    fused_score=1.0,
                    details={"note": "no verification active"},
                )
                for _, emb_sim in chunks_with_sims
            ]

        ent = self.entailment_scores(query, texts) if self.use_entailment else [0.0] * len(texts)
        out = []
        for (text, emb_sim), s_entail in zip(chunks_with_sims, ent):
            s_struct, details = (
                self.structural_score(query, text) if self.use_structural else (0.0, {})
            )
            emb = emb_sim if self.use_embedding else 0.0
            out.append(self._fuse(emb, s_struct, s_entail, details))
        return out

    # ---------- fusion ----------

    def verify(self, query: str, chunk_text: str, embedding_sim: float) -> VerificationResult:
        any_active = self.use_embedding or self.use_entailment or self.use_structural
        if not any_active:
            return VerificationResult(
                verdict=Verdict.CORRECT,
                embedding_sim=embedding_sim,
                structural_score=0.0,
                entailment_score=0.0,
                fused_score=1.0,
                details={"note": "no verification active"},
            )

        s_struct = 0.0
        details = {}
        if self.use_structural:
            s_struct, details = self.structural_score(query, chunk_text)

        s_entail = 0.0
        if self.use_entailment:
            s_entail = self.entailment_score(query, chunk_text)

        emb = embedding_sim if self.use_embedding else 0.0
        return self._fuse(emb, s_struct, s_entail, details)

    def _fuse(self, emb: float, s_struct: float, s_entail: float, details: dict) -> VerificationResult:
        # Normalize weights: redistribute weight from disabled signals to enabled ones
        active_weights = []
        active_scores = []
        if self.use_embedding:
            active_weights.append(self.w_emb)
            active_scores.append(emb)
        if self.use_structural:
            active_weights.append(self.w_struct)
            active_scores.append(s_struct)
        if self.use_entailment:
            active_weights.append(self.w_ent)
            active_scores.append(s_entail)

        if not active_weights:
            fused = 1.0
        else:
            total_w = sum(active_weights)
            fused = sum(w * s for w, s in zip(active_weights, active_scores)) / total_w

        if fused >= self.t_correct:
            verdict = Verdict.CORRECT
        elif fused <= self.t_incorrect:
            verdict = Verdict.INCORRECT
        else:
            verdict = Verdict.AMBIGUOUS

        return VerificationResult(
            verdict=verdict,
            embedding_sim=emb,
            structural_score=s_struct,
            entailment_score=s_entail,
            fused_score=float(fused),
            details=details,
        )
