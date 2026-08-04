"""
===================================================================
 VECTOR DB SHOWDOWN — UI Backend
 FastAPI wrapper around the ..\\VectorDB_Showdown engine (FAISS + Chroma),
 exposing endpoints the frontend dashboard uses to build indexes, run
 side-by-side queries, and tail the engine's own log file live.

 Run with: uvicorn server:app --reload --port 8010
===================================================================
"""

import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

if sys.platform == "win32":
    # Windows consoles default to a legacy codepage that chokes on the arrows/emoji
    # in the shared log format — force UTF-8 before uvicorn's own logging kicks in.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# The engine lives in the sibling "VectorDB_Showdown" program folder —
# add it to sys.path so we can import it without duplicating any code.
ENGINE_DIR = Path(__file__).parent.parent / "VectorDB_Showdown"
sys.path.insert(0, str(ENGINE_DIR))

from ui_logger_config import LOG_FILE as UI_LOG_FILE, api_logger  # noqa: E402

# These come from the engine folder above (added to sys.path), not from this
# package — that's what lets the UI stay a thin wrapper with zero duplicated logic.
from case_studies import CASE_STUDIES  # noqa: E402
from logger_config import LOG_FILE as ENGINE_LOG_FILE  # noqa: E402
from vector_db_engine import MODEL_PROVIDER_DEFAULT, build_both, clear_indexes, run_comparison  # noqa: E402

api_logger.info("=" * 70)
api_logger.info("Vector DB Showdown UI — starting up")
api_logger.info(f"Engine dir     : {ENGINE_DIR}")
api_logger.info(f"Engine log file: {ENGINE_LOG_FILE}")
api_logger.info("=" * 70)

app = FastAPI(
    title="Vector DB Showdown — FAISS vs Chroma",
    description="Lecture demo API: builds and queries FAISS + Chroma side by side.",
    version="1.0.0",
)

# Wide open on purpose — this is a local lecture demo served to a static HTML file
# opened directly from disk (file:// origin), not a deployed multi-tenant service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs every request/response with timing through api_logger — this is what
    lets the UI log panel show HTTP traffic alongside the engine's own build/query
    trace when tailing /api/logs/latest?source=ui."""
    t0 = time.perf_counter()
    api_logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    api_logger.info(f"← {request.method} {request.url.path} → {response.status_code} ({elapsed:.1f} ms)")
    return response


# ===========================================================================
# Schemas — request bodies for the POST routes below (FastAPI validates + docs
# these automatically from the Pydantic models)
# ===========================================================================
class BuildRequest(BaseModel):
    case_study: str
    provider: Optional[str] = None
    force_rebuild: bool = False  # True = bypass the engine's in-memory cache and rebuild


class QueryRequest(BaseModel):
    case_study: str
    query: str
    provider: Optional[str] = None
    k: int = 3
    use_filter: bool = False  # apply the case study's predefined filter_demo, if it has one


class ClearRequest(BaseModel):
    # Both optional: omit both to wipe every persisted store; set one or both to
    # scope the wipe to a specific case study and/or provider (see clear_indexes()).
    case_study: Optional[str] = None
    provider: Optional[str] = None


# ===========================================================================
# Routes
# ===========================================================================
@app.get("/api/health")
def health():
    """Liveness probe the frontend polls every 10s to flip the status dot and
    know which provider ('ollama'/'openai') is selected by default on load."""
    return {
        "status": "online",
        "default_provider": MODEL_PROVIDER_DEFAULT,
        "case_studies": list(CASE_STUDIES.keys()),
    }


@app.get("/api/case-studies")
def get_case_studies():
    """Feeds the left-hand case-study picker: name/description for the cards,
    sample_queries for the clickable chips, filter_demo to enable/disable the
    metadata-filter toggle per case study. Loader functions are never exposed."""
    return {
        case_id: {
            "name": meta["name"],
            "description": meta["description"],
            "sample_queries": meta["sample_queries"],
            "filter_demo": meta["filter_demo"],
        }
        for case_id, meta in CASE_STUDIES.items()
    }


@app.post("/api/build")
def build(req: BuildRequest):
    """Builds (or reuses the cached) FAISS + Chroma engines without running a query —
    backs the "(Re)build indexes" button so build-phase stats can be shown on their own."""
    if req.case_study not in CASE_STUDIES:
        raise HTTPException(status_code=400, detail=f"Unknown case study '{req.case_study}'")
    try:
        faiss_demo, chroma_demo = build_both(
            req.case_study, provider=req.provider, force_rebuild=req.force_rebuild
        )
        return {"faiss": faiss_demo.stats(), "chroma": chroma_demo.stats()}
    except RuntimeError as e:
        # RuntimeError is raised deliberately (e.g. missing OPENAI_API_KEY) — treat
        # it as a client-fixable 400 rather than a 500 server error.
        api_logger.error(f"Build failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        api_logger.error(f"Build failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
def query(req: QueryRequest):
    """Main comparison endpoint — builds both engines if needed, then runs the same
    query through each and returns one merged payload for the results panel + latency bars."""
    if req.case_study not in CASE_STUDIES:
        raise HTTPException(status_code=400, detail=f"Unknown case study '{req.case_study}'")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        result = run_comparison(
            req.case_study, req.query.strip(), provider=req.provider, k=req.k, use_filter=req.use_filter
        )
        return result
    except RuntimeError as e:
        api_logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        api_logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
def clear(req: ClearRequest):
    """Wipes persisted FAISS/Chroma stores (scoped or all) and the matching build
    cache entries — backs the "Clear all indexes" button so the lecture can reset
    between scenarios without touching a terminal."""
    if req.case_study and req.case_study not in CASE_STUDIES:
        raise HTTPException(status_code=400, detail=f"Unknown case study '{req.case_study}'")
    try:
        cleared = clear_indexes(case_id=req.case_study, provider=req.provider)
        return {"cleared": cleared, "count": len(cleared)}
    except Exception as e:
        api_logger.error(f"Clear failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs/latest")
def latest_logs(lines: int = 60, source: str = "engine"):
    """Tails the engine log (default) or the UI log — lets the frontend show
    live log lines during the lecture without opening a text editor."""
    log_path = ENGINE_LOG_FILE if source == "engine" else UI_LOG_FILE
    if not log_path.exists():
        return {"source": source, "path": str(log_path), "lines": []}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        all_lines = f.readlines()
    tail = [ln.rstrip("\n") for ln in all_lines[-max(1, min(lines, 500)):]]
    return {"source": source, "path": str(log_path), "lines": tail}


if __name__ == "__main__":
    # Lets `python server.py` work directly; the documented/preferred way is still
    # `uvicorn server:app --reload --port 8010` run from this folder.
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8010, reload=True)
