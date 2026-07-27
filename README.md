# Corrective RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-35%20passing-brightgreen)](#testing)

**A self-corrective RAG system for 0.5B-3B models with NLI verification, citation tracking, and structural retrieval.**

Unlike standard RAG that blindly accepts retrieved chunks, Corrective RAG adds a **three-signal verification layer** — embedding similarity, HRR structural analysis, and NLI entailment — that decides whether each chunk is *correct*, *ambiguous*, or *incorrect* before generation.

```
┌─────────────────────────────────────────────────────────────┐
│                      Query Processing                        │
├─────────────────────────────────────────────────────────────┤
│  Query  →  Retrieve Top-K  →  Verify Each Chunk             │
│                              │  1. Embedding cosine          │
│                              │  2. HRR role-filler check     │
│                              │  3. NLI entailment score      │
│                              └→  CORRECT / AMBIGUOUS / INCORRECT
│                                        │                     │
│        ┌───────────────────────────────┘                     │
│        ▼                                                     │
│  Enough CORRECT?  ──yes──→  Generate  →  Self-Check  →  Answer
│        │                          │                          │
│        no                         │                          │
│        │                     SUPPORTED                       │
│        ▼                     PARTIAL → Regenerate             │
│  Rewrite Query             UNSUPPORTED → ABSTAIN             │
│  Retry (max 2)                                          ──→ │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

- **Three-Signal Verification** — Embedding, HRR structural, NLI entailment scores fused into a single verdict per chunk
- **CRAG-Style Correction** — Automatically rewrites queries when retrieved chunks are insufficient
- **Citation Tracking** — Every answer includes source documents with relevance scores
- **Streaming API** — Real-time token-by-token responses with decision traces
- **Multi-Format Loading** — PDF, DOCX, PPTX, TXT, Markdown, CSV, HTML, JSONL, images (OCR)
- **Ablation Framework** — Built-in A0-A4 configs for systematic verification analysis
- **Edge-Device Optimized** — Runs on CPU with 6GB RAM; Qwen2.5-0.5B and 1.5B supported
- **Professional CLI** — Rich-formatted output with spinners, tables, and progress bars
- **Web Interface** — Chat UI with streaming, markdown rendering, and trace panels

## Quick Start

```bash
# Install
git clone https://github.com/your-username/corrective-rag.git
cd corrective-rag
pip install -e ".[cli]"
python -m spacy download en_core_web_sm

# Download models (~2GB)
python scripts/download_models.py

# Ask a question
rag query "Who directed Inception?"

# Start the API server
rag serve
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `rag query "question"` | Ask a question with citation tracking |
| `rag query "..." --verbose` | Show full decision trace |
| `rag serve` | Start FastAPI server with Web UI |
| `rag serve --port 8080` | Custom port |
| `rag index docs.jsonl` | Index documents (JSONL format) |
| `rag index /path/to/dir/` | Index all supported files in directory |
| `rag benchmark --dataset hotpot_qa --samples 100` | Run benchmark |
| `rag benchmark --ablation` | Run A0-A4 ablation study |
| `rag download-models` | Download all ML models |

### API Server

```bash
# Query with streaming
curl -N -X POST localhost:8000/query/stream \
  -H 'content-type: application/json' \
  -d '{"question": "Who directed Inception?"}'

# Query with citations
curl -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "Who directed Inception?"}'
```

Response:
```json
{
  "answer": "Christopher Nolan",
  "confidence": "HIGH",
  "latency_s": 4.2,
  "citations": [
    {"doc_id": "nolan", "chunk_id": 0, "text": "Christopher Nolan directed Inception...", "score": 0.85}
  ]
}
```

### Web Interface

Navigate to `http://localhost:8000` after starting the server for a chat interface with:
- Real-time streaming responses
- Collapsible decision traces
- Source citation panels
- Document upload with progress tracking

## Benchmark Results

### Ablation Study (HotpotQA, 10 samples, Qwen2.5-0.5B)

| Config | Embed | NLI | HRR | Exact Match | Avg Latency | Notes |
|--------|-------|-----|-----|-------------|-------------|-------|
| A0 (Vanilla) | — | — | — | 30% | 42.1s | No verification |
| A1 (Embed Only) | ✓ | — | — | 30% | 44.7s | Embedding threshold only |
| **A2 (Embed + NLI)** | ✓ | ✓ | — | **40%** | 33.4s | **Best accuracy + fastest** |
| A3 (Full) | ✓ | ✓ | ✓ | 20% | 44.7s | HRR noise at 0.5B scale |
| A4 (Embed + HRR) | ✓ | — | ✓ | 20% | 42.7s | HRR structural hurts |

### Key Findings

1. **NLI verification improves accuracy** — A2 achieves 40% EM vs 30% vanilla (+10% improvement)
2. **NLI also reduces latency** — 33.4s vs 42.1s (21% faster) by filtering bad chunks early
3. **HRR structural hurts at 0.5B** — A3/A4 both show -10% EM vs vanilla due to noisy role-filler extraction
4. **NLI is the critical signal** — Embedding alone doesn't help; NLI provides the discrimination needed

### Edge Device Profile

| Spec | Value |
|------|-------|
| RAM | 6.2 GB |
| Swap | 10 GB |
| CPU | 4 cores |
| Model | Qwen2.5-0.5B-Instruct |
| Embed | BAAI/bge-small-en-v1.5 (33M) |
| NLI | cross-encoder/nli-deberta-v3-base |
| Latency (A2) | 33.4s per query |

## Architecture

```
corrective-rag/
├── src/
│   ├── pipeline.py         # Corrective RAG loop with citation tracking
│   ├── retriever.py        # Dense retrieval (FAISS + BM25 hybrid)
│   ├── verifier.py         # 3-signal verifier with ablation flags
│   ├── generator.py        # Qwen2.5 generation + self-check
│   ├── hrr.py              # Holographic Reduced Representation ops
│   ├── graph_store.py      # HRR-encoded knowledge graph
│   ├── agent.py            # ReAct-style agentic flow
│   ├── benchmarks.py       # HotpotQA/ASQA/PopQA evaluators
│   ├── multi_loader.py     # PDF, DOCX, PPTX, HTML, OCR loader
│   ├── config.py           # Environment-driven settings
│   └── evaluate.py         # EM/F1 metrics, abstention accuracy
├── static/
│   └── index.html          # Web UI with streaming + citations
├── tests/
│   └── test_pipeline.py    # 35 passing unit tests
├── results/                # Benchmark outputs
├── server.py               # FastAPI server
├── cli.py                  # Rich CLI entry point
└── pyproject.toml          # Package configuration
```

## Configuration

All settings are environment-overridable with the `RAG_` prefix:

```bash
cp .env.example .env
# Edit .env with your settings
export $(grep -v '^#' .env | xargs)
```

Key settings:
- `RAG_GEN_MODEL` — Generator model (default: `Qwen/Qwen2.5-0.5B-Instruct`)
- `RAG_EMBED_MODEL` — Embedding model (default: `BAAI/bge-small-en-v1.5`)
- `RAG_T_CORRECT` — Threshold for CORRECT verdict (default: 0.5)
- `RAG_T_INCORRECT` — Threshold for INCORRECT verdict (default: 0.3)
- `RAG_TOP_K` — Number of chunks to retrieve (default: 3)

## Supported Datasets

- **HotpotQA** (distractor) — Multi-hop reasoning
- **ASQA** — Ambiguous long-form QA
- **PopQA** — Tail knowledge, tests abstention

```bash
# Single dataset
rag benchmark --dataset hotpot_qa --samples 100

# Full ablation (A0-A4)
rag benchmark --ablation --samples 50
```

## Docker

```bash
docker compose up --build
# or
make docker
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Citation

If you use this work in your research, please cite:

```bibtex
@software{corrective_rag_2026,
  title={Corrective RAG: Self-Corrective Retrieval-Augmented Generation with NLI Verification},
  year={2026},
  url={https://github.com/your-username/corrective-rag}
}
```

## License

[MIT](LICENSE)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
