"""
HRR-encoded knowledge graph — the Graph RAG layer.

Two parallel representations of the same triples:

  1. networkx MultiDiGraph  -> exact symbolic traversal (multi-hop paths)
  2. HRR trace per entity   -> approximate vector queries over the graph

The HRR side is the differentiator vs. Microsoft GraphRAG: each entity gets
a "memory trace" that superposes all its outgoing edges as
bind(REL_r, ENTITY_o). Querying "what is X's relation r?" becomes a single
unbind + cleanup — no traversal, O(1) in path length. This connects directly
to the arXiv 2606.24948 line of work on HRR capacity limits in KG reasoning:
the graph layer here is also an *instrument* for measuring when bundled
traces saturate (how many edges per entity before cleanup fails).

Triple extraction is deliberately simple (spacy SVO + entity co-occurrence).
Swap in an LLM extractor later behind the same add_triples() interface.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from src.hrr import RoleFillerEncoder, bind, bundle, unbind


@dataclass(frozen=True)
class Triple:
    subj: str
    rel: str
    obj: str
    source_doc: str = ""


def extract_triples(text: str, doc_id: str, spacy_nlp) -> list[Triple]:
    """Cheap SVO extraction from dependency parse. Coarse but serviceable
    for bootstrapping the graph; precision improves with an LLM extractor."""
    doc = spacy_nlp(text)
    triples = []
    for sent in doc.sents:
        root = sent.root
        if root.pos_ != "VERB":
            continue
        subjs = [t for t in root.children if t.dep_ in ("nsubj", "nsubjpass")]
        objs = [t for t in root.children if t.dep_ in ("dobj", "attr", "dative")]
        # prepositional objects: "directed by X", "released in 2010"
        for prep in [t for t in root.children if t.dep_ == "prep"]:
            for pobj in [t for t in prep.children if t.dep_ == "pobj"]:
                objs.append(pobj)
        for s in subjs:
            for o in objs:
                triples.append(
                    Triple(
                        subj=" ".join(w.text for w in s.subtree).lower(),
                        rel=root.lemma_.lower(),
                        obj=" ".join(w.text for w in o.subtree).lower(),
                        source_doc=doc_id,
                    )
                )
    return triples


class HRRGraphStore:
    def __init__(self, hrr_dim: int = 4096, seed: int = 7):
        import networkx as nx

        self.nx = nx
        self.graph = nx.MultiDiGraph()
        self.encoder = RoleFillerEncoder(dim=hrr_dim, seed=seed)
        self._entity_edges: dict[str, list[np.ndarray]] = defaultdict(list)
        self._entity_trace: dict[str, np.ndarray] = {}

    # ---------- construction ----------

    def add_triples(self, triples: list[Triple]):
        for t in triples:
            self.graph.add_edge(t.subj, t.obj, rel=t.rel, source=t.source_doc)
            edge_vec = bind(
                self.encoder.symbol(f"rel::{t.rel}"),
                self.encoder.symbol(f"ent::{t.obj}"),
            )
            self._entity_edges[t.subj].append(edge_vec)
        # (re)bundle traces for touched entities
        for subj in {t.subj for t in triples}:
            self._entity_trace[subj] = bundle(self._entity_edges[subj])

    def build_from_docs(self, docs: dict[str, str], spacy_nlp):
        for doc_id, text in docs.items():
            self.add_triples(extract_triples(text, doc_id, spacy_nlp))

    # ---------- symbolic queries (exact, via networkx) ----------

    def neighbors(self, entity: str, max_edges: int = 10) -> list[Triple]:
        out = []
        if entity not in self.graph:
            return out
        for _, obj, data in self.graph.out_edges(entity, data=True):
            out.append(Triple(entity, data["rel"], obj, data.get("source", "")))
            if len(out) >= max_edges:
                break
        return out

    def multi_hop(self, start: str, end: str, cutoff: int = 3) -> list[list[Triple]]:
        """All simple paths start->end up to `cutoff` hops, as triple chains."""
        if start not in self.graph or end not in self.graph:
            return []
        paths = []
        for node_path in self.nx.all_simple_paths(self.graph, start, end, cutoff=cutoff):
            chain = []
            for a, b in zip(node_path, node_path[1:]):
                data = list(self.graph.get_edge_data(a, b).values())[0]
                chain.append(Triple(a, data["rel"], b, data.get("source", "")))
            paths.append(chain)
        return paths

    # ---------- vector queries (approximate, via HRR) ----------

    def hrr_query(self, entity: str, rel: str, top_k: int = 3):
        """One-shot relational query: unbind rel from the entity's bundled
        trace, clean up to candidate entities. Returns [(name, sim)].
        Degrades gracefully as edge count grows — measure that curve, it's
        a capacity experiment for free."""
        trace = self._entity_trace.get(entity)
        if trace is None:
            return []
        noisy = unbind(trace, self.encoder.symbol(f"rel::{rel}"))
        return [
            (name, sim)
            for name, sim in self.encoder.cleanup_mem.cleanup(noisy, top_k=top_k)
            if name.startswith("ent::")
        ]

    def edge_count(self, entity: str) -> int:
        return len(self._entity_edges.get(entity, []))
