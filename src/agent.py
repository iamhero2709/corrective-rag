"""
Agentic controller — plan/act loop over dense retrieval + graph store.

Small models can't reliably emit JSON tool calls, so the action format is
one line: ACTION_NAME | argument. Parsing is forgiving.

Action space:
  DECOMPOSE | <question>      -> split multi-hop question into sub-questions
  DENSE     | <query>         -> dense retrieval (chunks go through the Verifier)
  GRAPH     | <entity>        -> neighbor triples for an entity
  HOP       | <ent1> -> <ent2>-> multi-hop paths between two entities
  ANSWER    | <final answer>  -> terminate

Every retrieved chunk still passes through the 3-signal Verifier before it
enters working memory — the agent cannot act on unverified context. That
invariant ("verify-before-memory") is the paper's framing: agentic Graph RAG
where the verification layer gates *every* channel, not just dense retrieval.

Budgeted: max_steps hard cap, with a forced ANSWER on budget exhaustion.
All steps logged to PipelineTrace for mechanistic analysis.
"""

from dataclasses import dataclass, field

from src.generator import SmallModelGenerator
from src.graph_store import HRRGraphStore
from src.pipeline import PipelineTrace
from src.retriever import DenseRetriever
from src.verifier import RetrievalVerifier, Verdict


PLANNER_PROMPT = """You are a research agent answering a question step by step. You have these actions:

DECOMPOSE | <complex question>   (split into sub-questions, use once at most)
DENSE | <search query>           (search documents)
GRAPH | <entity name>            (look up facts about an entity)
HOP | <entity1> -> <entity2>     (find connection between two entities)
ANSWER | <final answer>          (finish, only when confident)

Working memory (verified facts so far):
{memory}

Question: {question}
Steps used: {steps_used}/{max_steps}

Reply with EXACTLY ONE action line and nothing else:"""


@dataclass
class AgentState:
    question: str
    memory: list[str] = field(default_factory=list)
    steps_used: int = 0

    def memory_text(self) -> str:
        if not self.memory:
            return "(empty)"
        return "\n".join(f"- {m}" for m in self.memory[-12:])


class GraphAgenticRAG:
    def __init__(
        self,
        retriever: DenseRetriever,
        graph: HRRGraphStore,
        verifier: RetrievalVerifier,
        generator: SmallModelGenerator,
        max_steps: int = 6,
        dense_top_k: int = 4,
    ):
        self.retriever = retriever
        self.graph = graph
        self.verifier = verifier
        self.generator = generator
        self.max_steps = max_steps
        self.dense_top_k = dense_top_k

    # ---------- action handlers ----------

    def _parse_action(self, raw: str) -> tuple[str, str]:
        line = raw.strip().splitlines()[0]
        if "|" not in line:
            return "ANSWER", line  # model skipped format -> treat as answer
        name, _, arg = line.partition("|")
        return name.strip().upper(), arg.strip()

    def _do_dense(self, query: str, state: AgentState, trace: PipelineTrace):
        results = self.retriever.search(query, top_k=self.dense_top_k)
        kept = 0
        for chunk, emb_sim in results:
            vr = self.verifier.verify(state.question, chunk.text, emb_sim)
            trace.log("agent_verify", verdict=vr.verdict.value,
                      fused=round(vr.fused_score, 3), doc=chunk.doc_id)
            if vr.verdict != Verdict.INCORRECT:
                state.memory.append(f"[doc:{chunk.doc_id}] {chunk.text[:300]}")
                kept += 1
        if kept == 0:
            state.memory.append(f"(no verified results for query: {query})")

    def _do_graph(self, entity: str, state: AgentState, trace: PipelineTrace):
        triples = self.graph.neighbors(entity.lower())
        trace.log("agent_graph", entity=entity, n_edges=len(triples))
        if not triples:
            # fall back to HRR fuzzy lookup across common relations
            state.memory.append(f"(entity not found in graph: {entity})")
            return
        for t in triples:
            state.memory.append(f"[graph] {t.subj} --{t.rel}--> {t.obj}")

    def _do_hop(self, arg: str, state: AgentState, trace: PipelineTrace):
        if "->" not in arg:
            state.memory.append(f"(malformed HOP argument: {arg})")
            return
        a, b = [s.strip().lower() for s in arg.split("->", 1)]
        paths = self.graph.multi_hop(a, b)
        trace.log("agent_hop", start=a, end=b, n_paths=len(paths))
        if not paths:
            state.memory.append(f"(no path found: {a} -> {b})")
            return
        for chain in paths[:3]:
            path_str = " ; ".join(f"{t.subj} --{t.rel}--> {t.obj}" for t in chain)
            state.memory.append(f"[path] {path_str}")

    def _do_decompose(self, question: str, state: AgentState, trace: PipelineTrace):
        subqs = self.generator._chat(
            "Split this question into 2-3 simpler sub-questions, one per line, "
            f"no numbering:\n{question}",
            max_new_tokens=96,
        )
        for sq in [s.strip() for s in subqs.splitlines() if s.strip()][:3]:
            state.memory.append(f"[subq] {sq}")
        trace.log("agent_decompose", question=question)

    # ---------- main loop ----------

    def run(self, question: str):
        from src.pipeline import RAGAnswer

        state = AgentState(question=question)
        trace = PipelineTrace()

        while state.steps_used < self.max_steps:
            raw = self.generator._chat(
                PLANNER_PROMPT.format(
                    memory=state.memory_text(),
                    question=question,
                    steps_used=state.steps_used,
                    max_steps=self.max_steps,
                ),
                max_new_tokens=64,
            )
            action, arg = self._parse_action(raw)
            trace.log("agent_action", step=state.steps_used, action=action, arg=arg[:80])
            state.steps_used += 1

            if action == "ANSWER":
                return self._finalize(question, arg, state, trace)
            elif action == "DENSE":
                self._do_dense(arg, state, trace)
            elif action == "GRAPH":
                self._do_graph(arg, state, trace)
            elif action == "HOP":
                self._do_hop(arg, state, trace)
            elif action == "DECOMPOSE":
                self._do_decompose(arg, state, trace)
            else:
                state.memory.append(f"(unknown action ignored: {action})")

        # budget exhausted -> force a final answer from verified memory
        trace.log("agent_budget_exhausted")
        return self._finalize(question, None, state, trace)

    def _finalize(self, question: str, proposed: str | None, state: AgentState, trace: PipelineTrace):
        from src.pipeline import RAGAnswer

        context = [m for m in state.memory if not m.startswith("(")]
        if not context:
            return RAGAnswer("I don't have reliable information to answer this.",
                             "ABSTAIN", [], trace=trace)

        out = self.generator.answer(question, context)
        trace.log("agent_generate", self_check=out.self_check,
                  proposed_by_planner=bool(proposed))

        confidence = {"SUPPORTED": "HIGH", "PARTIAL": "MEDIUM",
                      "UNSUPPORTED": "LOW", "SKIPPED": "LOW"}[out.self_check]
        if out.insufficient:
            return RAGAnswer("I don't have reliable information to answer this.",
                             "ABSTAIN", context, trace=trace)
        return RAGAnswer(out.answer, confidence, context, trace=trace)
