"""
Production API server.

  uvicorn server:app --host 0.0.0.0 --port 8000

Endpoints:
  GET  /health          liveness (always fast)
  GET  /ready           readiness (models loaded + index present)
  POST /index           {"docs": {"doc_id": "text", ...}}  -> build/extend index
  POST /query           {"question": "...", "mode": "corrective"|"agentic"}
  POST /query/stream    SSE streaming response
  POST /upload          multipart file upload (PDF, DOCX, etc.)
  POST /mcp             MCP protocol endpoint
  GET  /                Web UI

Design:
  - Components load once at startup (fail fast if models are broken).
  - Requests run in a thread pool so the event loop stays responsive.
  - Typed errors map to proper HTTP codes.
  - Every response carries the decision trace for observability.
"""

import asyncio
import json
import logging
import time
import uuid
import tempfile
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import Settings
from src.exceptions import IndexNotBuiltError, ModelLoadError, RAGError
from src.logging_utils import setup_logging

log = logging.getLogger("rag.server")

settings = Settings().validate()
setup_logging(settings.log_level, settings.log_json)

_components: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.agent import GraphAgenticRAG
    from src.generator import SmallModelGenerator
    from src.graph_store import HRRGraphStore
    from src.pipeline import CorrectiveRAG
    from src.retriever import DenseRetriever
    from src.verifier import RetrievalVerifier

    t0 = time.time()
    log.info("loading components (device=%s)...", settings.device)
    try:
        retriever = DenseRetriever(settings.embed_model, device="cpu")
        retriever.load(settings.index_dir)
        verifier = RetrievalVerifier(
            hrr_dim=settings.hrr_dim,
            use_structural=settings.use_structural,
            use_entailment=settings.use_entailment,
            nli_model=settings.nli_model,
            thresholds=(settings.t_correct, settings.t_incorrect),
            weights=(settings.w_embedding, settings.w_structural, settings.w_entailment),
        )
        generator = SmallModelGenerator(
            settings.gen_model, device=settings.device,
            max_new_tokens=settings.max_new_tokens,
        )
        graph = HRRGraphStore(hrr_dim=max(settings.hrr_dim, 4096))
    except ModelLoadError:
        raise
    except Exception as e:
        raise ModelLoadError(f"component load failed: {e}") from e

    _components.update(
        retriever=retriever,
        verifier=verifier,
        generator=generator,
        graph=graph,
        corrective=CorrectiveRAG(
            retriever, verifier, generator,
            top_k=settings.top_k, min_correct=settings.min_correct,
            max_retries=settings.max_retries,
        ),
    )
    try:
        _components["agentic"] = GraphAgenticRAG(
            retriever, graph, verifier, generator,
            max_steps=settings.agent_max_steps,
        )
    except Exception:
        _components["agentic"] = _components["corrective"]

    log.info("components loaded in %.2fs", time.time() - t0)
    yield
    log.info("shutting down")


app = FastAPI(
    title="Corrective Graph-Agentic RAG",
    version="0.2.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_components: dict = {}


class IndexRequest(BaseModel):
    docs: dict[str, str] = Field(..., min_length=1)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    mode: str = Field("corrective", pattern="^(corrective|agentic)$")


class CitationResponse(BaseModel):
    doc_id: str
    chunk_id: int
    text: str
    score: float


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    confidence: str
    mode: str
    latency_s: float
    n_context_chunks: int
    citations: list[CitationResponse] = []
    trace: list[dict]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    r = _components.get("retriever")
    return {
        "models_loaded": bool(_components),
        "index_chunks": len(r.chunks) if r else 0,
        "graph_entities": _components["graph"].graph.number_of_nodes() if _components else 0,
    }


@app.post("/index")
async def build_index(req: IndexRequest):
    retriever = _components["retriever"]
    graph = _components["graph"]
    verifier = _components["verifier"]

    def _do():
        retriever.add_documents(req.docs)
        retriever.save(settings.index_dir)
        if verifier.nlp is not None:
            graph.build_from_docs(req.docs, verifier.nlp)

    await asyncio.get_running_loop().run_in_executor(None, _do)
    return {"indexed_docs": len(req.docs), "total_chunks": len(retriever.chunks),
            "graph_entities": graph.graph.number_of_nodes()}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    rid = uuid.uuid4().hex[:12]
    t0 = time.time()
    engine = _components[req.mode if req.mode == "agentic" else "corrective"]
    try:
        result = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, engine.run, req.question),
            timeout=settings.request_timeout_s,
        )
    except asyncio.TimeoutError:
        log.error("request %s timed out", rid)
        raise HTTPException(504, "query timed out")
    except IndexNotBuiltError as e:
        raise HTTPException(409, str(e))
    except RAGError as e:
        log.exception("request %s failed", rid)
        raise HTTPException(500, str(e))

    latency = time.time() - t0
    log.info("request %s mode=%s conf=%s latency=%.2fs",
             rid, req.mode, result.confidence, latency)
    return QueryResponse(
        request_id=rid,
        answer=result.answer,
        confidence=result.confidence,
        mode=req.mode,
        latency_s=round(latency, 3),
        n_context_chunks=len(result.used_chunks),
        citations=[CitationResponse(doc_id=c.doc_id, chunk_id=c.chunk_id, text=c.text, score=c.score)
                   for c in result.citations],
        trace=result.trace.steps,
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    engine = _components["corrective"]
    rid = uuid.uuid4().hex[:12]
    t0 = time.time()

    try:
        result = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, engine.run, req.question),
            timeout=settings.request_timeout_s,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "query timed out")
    except IndexNotBuiltError as e:
        raise HTTPException(409, str(e))
    except RAGError as e:
        raise HTTPException(500, str(e))

    latency = time.time() - t0

    async def event_stream():
        yield f"data: {json.dumps({'type': 'meta', 'chunks': len(result.used_chunks), 'request_id': rid})}\n\n"
        for word in result.answer.split():
            yield f"data: {json.dumps({'type': 'token', 'text': word + ' '})}\n\n"
        citations_json = [{'doc_id': c.doc_id, 'chunk_id': c.chunk_id, 'text': c.text, 'score': c.score}
                          for c in result.citations]
        yield f"data: {json.dumps({'type': 'done', 'answer': result.answer, 'self_check': result.confidence, 'latency_s': round(latency, 3), 'request_id': rid, 'trace': result.trace.steps, 'citations': citations_json})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ==================== File Upload ====================

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and index a document file (PDF, DOCX, TXT, MD, CSV, HTML, etc.)"""
    from src.multi_loader import load_file, chunk_documents

    suffix = Path(file.filename or "unknown").suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        docs = load_file(tmp_path)
        if not docs:
            raise HTTPException(400, f"Could not extract text from {file.filename}")

        chunked = chunk_documents(docs)
        retriever = _components["retriever"]
        retriever.add_documents(chunked)
        retriever.save(settings.index_dir)

        return {"filename": file.filename, "chunks": len(chunked), "pages": len(docs)}
    finally:
        os.unlink(tmp_path)


# ==================== MCP Protocol ====================

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """MCP (Model Context Protocol) JSON-RPC endpoint"""
    from src.mcp_server import create_mcp_server

    body = await request.json()
    server = create_mcp_server(
        _components.get("corrective"),
        _components.get("graph"),
    )
    response = server.handle_jsonrpc(body)
    return JSONResponse(response)


# ==================== Web UI ====================

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Serve the Web UI"""
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h1>Corrective RAG</h1><p>Static files not found. Run from the project root.</p>")


# Mount static files
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
