# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-27

### Added
- **Citation tracking** — Every answer now includes source documents with relevance scores
- **CLI citation display** — Sources table shown after each answer
- **API citation response** — `/query` and `/query/stream` return `citations` field
- **Web UI citation panel** — Collapsible sources panel in chat interface
- **Multi-format document loader** — PDF, DOCX, PPTX, TXT, Markdown, CSV, HTML, JSONL, images (OCR)
- **Professional CLI** — Rich library with logo, panels, tables, spinners, progress bars
- **Streaming SSE** — Token-by-token responses with decision traces
- **MCP Protocol** — Model Context Protocol integration
- **Semantic caching** — Query result caching for repeated questions
- **Plugin system** — Extensible architecture for custom components
- **CORS support** — Cross-origin resource sharing for web clients
- **File upload** — Web UI supports drag-and-drop document upload
- **Ablation framework** — Built-in A0-A4 configs for systematic verification analysis

### Changed
- **Generator prompt** — Updated for short-answer-only responses (HotpotQA compatibility)
- **Verifier disabled signals** — Now return 0.0 (not 0.5 neutral) for proper weight normalization
- **Benchmark evaluator** — Now indexes context paragraphs per question for accurate retrieval
- **Edge device optimization** — Float16 fallback for large models on CPU

### Fixed
- **14 critical bugs** — Server, pipeline, verifier, generator, retriever issues resolved
- **All local imports** — Fixed to use `src.` prefix across 10 files
- **Dataset IDs** — HotpotQA, ASQA, PopQA corrected for HuggingFace Hub
- **ASQA field names** — `ambiguous_question` and `qa_pairs` access fixed
- **PopQA parsing** — `possible_answers` JSON string handling corrected
- **transformers 5.14.1** — BatchEncoding and BitsAndBytesConfig compatibility
- **Noise padding** — Added 30 random noise paragraphs per question for realistic benchmarks

### Removed
- Development-only files (IMPLEMENTATION_CHECKLIST.md, PRODUCTION_GUIDE.md, etc.)
- Redundant production configs (Makefile_production, requirements_production.txt, server_production.py)

## [0.2.0] - 2026-07-18

### Added
- Three-signal verification (embedding, HRR structural, NLI entailment)
- CRAG-style correction loop with query rewriting
- Agentic ReAct-style flow with graph store
- HotpotQA, ASQA, PopQA benchmark evaluators
- FastAPI server with streaming SSE
- Docker support
- 35 passing unit tests

## [0.1.0] - 2026-07-15

### Added
- Initial release
- Basic RAG pipeline
- Dense retrieval with FAISS
- Qwen2.5 generation
