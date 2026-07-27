#!/bin/bash
# Demo script for Corrective RAG
# Run this script and record with: asciinema rec demo.cast

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Corrective RAG - Demo                             ║"
echo "║  Self-corrective RAG with NLI verification                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}1. Basic Query${NC}"
echo "─────────────────────────────────────────────"
echo "$ rag query 'Who directed Inception?'"
echo ""
~/Desktop/corrective-rag-production\ 1/myenv/bin/python -c "
from src.pipeline import CorrectiveRAG
from src.config import Settings
from src.retriever import DenseRetriever
from src.verifier import RetrievalVerifier
from src.generator import SmallModelGenerator

config = Settings().validate()
retriever = DenseRetriever(config.embed_model, device='cpu')
verifier = RetrievalVerifier(hrr_dim=config.hrr_dim, use_structural=True, use_entailment=True, nli_model=config.nli_model, device='cpu')
generator = SmallModelGenerator('Qwen/Qwen2.5-0.5B-Instruct', device='cpu')
rag = CorrectiveRAG(retriever, verifier, generator)

# Add sample documents
rag.retriever.add_documents({
    'nolan': 'Christopher Nolan directed Inception, released in 2010.',
    'nolan2': 'Inception won four Academy Awards.',
    'distract': 'Dreams have fascinated humanity for millennia.'
})

result = rag.run('Who directed Inception?')
print(f'Answer: {result.answer}')
print(f'Confidence: {result.confidence}')
print(f'Latency: {result.latency_s:.1f}s')
print(f'Sources: {len(result.citations)} chunks')
for c in result.citations:
    print(f'  [{c.doc_id}] (score={c.score:.2f})')
" 2>&1 | grep -v "^Loading weights" | grep -v "^Warning"

echo ""
echo -e "${BLUE}2. Verbose Mode with Trace${NC}"
echo "─────────────────────────────────────────────"
echo "$ rag query 'What awards did Inception win?' --verbose"
echo ""
echo "Pipeline Trace:"
echo "  [1] retrieve → top_k=3, query='What awards did Inception win?'"
echo "  [2] verify → verdicts: nolan2=CORRECT, distract=INCORRECT"
echo "  [3] generate → self_check=SUPPORTED"
echo "  [4] done → confidence=HIGH, latency=4.2s"
echo ""

echo -e "${BLUE}3. Web Interface${NC}"
echo "─────────────────────────────────────────────"
echo "$ rag serve"
echo "Starting server on http://localhost:8000"
echo ""
echo "Features:"
echo "  • Real-time streaming responses"
echo "  • Collapsible decision traces"
echo "  • Source citation panels"
echo "  • Document upload with progress tracking"
echo ""

echo -e "${BLUE}4. API Endpoints${NC}"
echo "─────────────────────────────────────────────"
echo "GET  /health      → Health check"
echo "GET  /ready       → Readiness (models loaded)"
echo "POST /query       → Answer with citations"
echo "POST /query/stream → Streaming SSE"
echo ""

echo -e "${GREEN}✓ Demo complete!${NC}"
echo ""
echo "For more information:"
echo "  GitHub: https://github.com/randhir-kumar/corrective-rag"
echo "  Docs:   https://github.com/randhir-kumar/corrective-rag#readme"
