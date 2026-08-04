"""
===================================================================
 VECTOR DB SHOWDOWN — Core Engine
 Wraps FAISS and Chroma behind a common interface (build / query / stats)
 so both the CLI demo and the comparison UI can drive identical case
 studies through either database, with every phase logged for the lecture.

 Embeddings: OpenAI (gpt / text-embedding-3-small) or Ollama (nomic-embed-text),
 selectable per-call — see get_embeddings(provider=...).
===================================================================
"""

import os
import shutil
import time
from pathlib import Path
from typing import Optional

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))  # picks up the shared c:\AI Agent study\.env

from logger_config import chroma_logger, compare_logger, embed_logger, faiss_logger, log_banner

BASE_DIR = Path(__file__).parent  # this program folder — persisted indexes/logs live under here

# Shared defaults, all overridable via the root .env (see [[model_provider_toggle]] memory) or
# per-call `provider=` argument — this is how the CLI's --provider flag and the UI's toggle work.
MODEL_PROVIDER_DEFAULT = os.getenv("MODEL_PROVIDER", "ollama").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_embeddings(provider: Optional[str] = None):
    """Returns (embeddings_instance, resolved_provider) for 'openai' or 'ollama'.

    Both FAISS and Chroma get the exact same embeddings object per build, so any
    difference observed between them is purely about the index/database, not the model.
    """
    # Fall back to the shared .env toggle when no provider is passed explicitly.
    provider = (provider or MODEL_PROVIDER_DEFAULT).strip().lower()
    t0 = time.perf_counter()

    if provider == "openai":
        # Imported lazily so the ollama-only path never needs the openai package installed.
        from langchain_openai import OpenAIEmbeddings

        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
            embed_logger.error("OPENAI_API_KEY missing or placeholder in the shared .env file")
            raise RuntimeError(
                r"OPENAI_API_KEY is not set. Add it to c:\AI Agent study\.env then retry."
            )
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
        embed_logger.info(
            f"OpenAIEmbeddings ({OPENAI_EMBEDDING_MODEL}) ready in {time.perf_counter() - t0:.3f}s"
        )
    else:
        # Default path — local Ollama server, no API key required.
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        embed_logger.info(
            f"OllamaEmbeddings ({OLLAMA_EMBEDDING_MODEL} @ {OLLAMA_BASE_URL}) ready in "
            f"{time.perf_counter() - t0:.3f}s"
        )
    return embeddings, provider


def _dir_size_bytes(path: Path) -> int:
    """Recursively sums file sizes under `path` — used to report each engine's on-disk
    footprint (FAISS's flat binary files vs. Chroma's SQLite store) in the stats table."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ===========================================================================
# FAISS — pure ANN index, no native metadata filtering, manual persistence
# ===========================================================================
class FAISSDemo:
    """Thin wrapper around a single LangChain `FAISS` vectorstore instance for one
    (case_study, provider) pair. FAISS itself is just an ANN index library — it has
    no server, no metadata engine, and no built-in persistence API beyond flat files,
    which is exactly what this class's `build`/`query`/`stats` make visible."""

    def __init__(self, case_id: str, embeddings, provider: str):
        self.case_id = case_id
        self.embeddings = embeddings
        self.provider = provider
        self.store = None  # populated by build(); None means "not built yet"
        self.persist_dir = BASE_DIR / "faiss_index" / f"{case_id}_{provider}"
        self.build_time_s = 0.0
        self.doc_count = 0

    def build(self, documents):
        """Embeds every document and builds an in-memory FAISS index from scratch,
        then dumps it to disk. There is no incremental/upsert path used here — each
        run is a full rebuild, which keeps the timing comparison against Chroma fair."""
        from langchain_community.vectorstores import FAISS

        log_banner(
            faiss_logger,
            f"FAISS — building index | case_study={self.case_id} | provider={self.provider} | "
            f"docs={len(documents)}",
        )
        t0 = time.perf_counter()
        self.store = FAISS.from_documents(documents, self.embeddings)
        self.build_time_s = time.perf_counter() - t0
        self.doc_count = len(documents)
        faiss_logger.info(
            f"Index built in {self.build_time_s:.3f}s for {self.doc_count} vectors "
            f"({self.build_time_s / max(self.doc_count, 1) * 1000:.1f} ms/vector)"
        )

        # FAISS has no "database" to write to — save_local() just pickles the
        # docstore and dumps the raw index bytes to two files in persist_dir.
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(self.persist_dir))
        faiss_logger.info(
            f"Persisted via FAISS.save_local() → {self.persist_dir} "
            "(binary index.faiss + index.pkl — no query engine, just raw files)"
        )
        return self.stats()

    def query(self, text: str, k: int = 3, metadata_filter: Optional[tuple] = None):
        """Runs a similarity search and, if a metadata filter was requested, applies it
        AFTER retrieval — FAISS has no concept of metadata inside the index itself."""
        if self.store is None:
            raise RuntimeError("FAISS index not built yet — call .build() first")

        note = (
            f"metadata_filter={metadata_filter} — FAISS has NO native filter, "
            "so we over-fetch then filter in Python"
            if metadata_filter
            else "no metadata filter requested"
        )
        faiss_logger.info(f"Query: {text!r} (k={k}) — {note}")

        t0 = time.perf_counter()
        # Over-fetch (k*5) when filtering so post-filtering still has enough candidates
        # left to return k results — a pure ANN index can't push the filter into the search.
        fetch_k = k * 5 if metadata_filter else k
        raw = self.store.similarity_search_with_score(text, k=fetch_k)
        if metadata_filter:
            field, value = metadata_filter
            raw = [(d, s) for d, s in raw if d.metadata.get(field) == value][:k]
        else:
            raw = raw[:k]
        latency_ms = (time.perf_counter() - t0) * 1000

        faiss_logger.info(f"Query done in {latency_ms:.1f} ms — {len(raw)} result(s) returned")
        for i, (d, s) in enumerate(raw):
            faiss_logger.debug(f"  [#{i+1}] score={s:.4f} | {d.page_content[:90]!r}")

        return {
            "results": [
                {"text": d.page_content, "metadata": d.metadata, "score": float(s)} for d, s in raw
            ],
            "latency_ms": round(latency_ms, 2),
        }

    def stats(self):
        """Snapshot used by both the stats table (build phase) and the query result
        payload (merged with query() output) — kept identical in shape to ChromaDemo.stats()
        so the UI/CLI can render both engines through one shared code path."""
        return {
            "engine": "FAISS",
            "provider": self.provider,
            "build_time_s": round(self.build_time_s, 4),
            "doc_count": self.doc_count,
            "disk_size_kb": round(_dir_size_bytes(self.persist_dir) / 1024, 2),
            "supports_native_metadata_filter": False,
            "persistence": "Manual: save_local() / load_local() to a folder of flat binary files",
        }


# ===========================================================================
# Chroma — full vector database: native metadata filtering + auto persistence
# ===========================================================================
class ChromaDemo:
    """Thin wrapper around a single LangChain `Chroma` vectorstore instance for one
    (case_study, provider) pair. Unlike FAISS, Chroma is a real embedded database:
    it stores vectors + metadata together in SQLite and can filter on that metadata
    natively inside the query itself — this class's `query()` is where that shows up."""

    def __init__(self, case_id: str, embeddings, provider: str):
        self.case_id = case_id
        self.embeddings = embeddings
        self.provider = provider
        self.store = None  # populated by build(); None means "not built yet"
        self.persist_dir = BASE_DIR / "chroma_store" / f"{case_id}_{provider}"
        self.collection_name = f"{case_id}_{provider}"[:63]  # Chroma caps collection names at 63 chars
        self.build_time_s = 0.0
        self.doc_count = 0

    def build(self, documents):
        """Embeds every document and creates a fresh Chroma collection, auto-persisted
        to SQLite as it's built — there's no separate save_local()-style step like FAISS."""
        from langchain_community.vectorstores import Chroma

        log_banner(
            chroma_logger,
            f"Chroma — building collection | case_study={self.case_id} | provider={self.provider} | "
            f"docs={len(documents)}",
        )
        # Rebuild fresh each run so re-running the lecture demo is idempotent.
        if self.persist_dir.exists():
            shutil.rmtree(self.persist_dir, ignore_errors=True)
            chroma_logger.info(f"Cleared previous persisted collection at {self.persist_dir}")

        t0 = time.perf_counter()
        self.store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_dir),
        )
        self.build_time_s = time.perf_counter() - t0
        self.doc_count = len(documents)
        chroma_logger.info(
            f"Collection built in {self.build_time_s:.3f}s for {self.doc_count} vectors "
            f"({self.build_time_s / max(self.doc_count, 1) * 1000:.1f} ms/vector)"
        )
        chroma_logger.info(
            f"Auto-persisted to SQLite-backed store → {self.persist_dir} "
            "(collection is queryable immediately, metadata stored alongside each vector)"
        )
        return self.stats()

    def query(self, text: str, k: int = 3, metadata_filter: Optional[tuple] = None):
        """Runs a similarity search with the metadata filter (if any) pushed straight
        into Chroma's `where` clause — the filter narrows the search itself, it never
        has to over-fetch and post-filter the way FAISSDemo.query() does."""
        if self.store is None:
            raise RuntimeError("Chroma collection not built yet — call .build() first")

        where = {metadata_filter[0]: metadata_filter[1]} if metadata_filter else None
        chroma_logger.info(f"Query: {text!r} (k={k}) — native `where` filter={where}")

        t0 = time.perf_counter()
        raw = self.store.similarity_search_with_score(text, k=k, filter=where)
        latency_ms = (time.perf_counter() - t0) * 1000

        chroma_logger.info(f"Query done in {latency_ms:.1f} ms — {len(raw)} result(s) returned")
        for i, (d, s) in enumerate(raw):
            chroma_logger.debug(f"  [#{i+1}] score={s:.4f} | {d.page_content[:90]!r}")

        return {
            "results": [
                {"text": d.page_content, "metadata": d.metadata, "score": float(s)} for d, s in raw
            ],
            "latency_ms": round(latency_ms, 2),
        }

    def stats(self):
        """Kept identical in shape to FAISSDemo.stats() so both engines render through
        one shared UI/CLI code path — only the values differ."""
        return {
            "engine": "Chroma",
            "provider": self.provider,
            "build_time_s": round(self.build_time_s, 4),
            "doc_count": self.doc_count,
            "disk_size_kb": round(_dir_size_bytes(self.persist_dir) / 1024, 2),
            "supports_native_metadata_filter": True,
            "persistence": "Automatic: SQLite-backed persist_directory, queryable as a live collection",
        }


# ===========================================================================
# Comparison cache — avoids rebuilding both indexes on every single query
# ===========================================================================
# Keyed by (case_id, provider) → (FAISSDemo, ChromaDemo). Process-lifetime only —
# there is no eviction beyond what clear_indexes() removes explicitly.
_engine_cache: dict[tuple, tuple] = {}


def build_both(case_id: str, provider: Optional[str] = None, force_rebuild: bool = False):
    """Builds (or returns cached) FAISS + Chroma engines for a case study/provider pair.

    Both engines are built from the SAME `documents` list and the SAME `embeddings`
    instance, so any timing/size/behavior difference reported downstream is caused by
    the vector database itself, never by different input data.
    """
    from case_studies import load_case_study

    embeddings, provider = get_embeddings(provider)
    cache_key = (case_id, provider)

    # Reuse whatever was already built this process unless the caller explicitly
    # wants a fresh build (UI "(Re)build" button) — keeps repeated queries fast.
    if not force_rebuild and cache_key in _engine_cache:
        compare_logger.info(f"Reusing cached FAISS + Chroma indexes for {cache_key}")
        return _engine_cache[cache_key]

    log_banner(compare_logger, f"BUILD BOTH — case_study={case_id} | provider={provider}")
    documents = load_case_study(case_id)
    compare_logger.info(f"Loaded {len(documents)} document chunk(s) for case study '{case_id}'")

    faiss_demo = FAISSDemo(case_id, embeddings, provider)
    faiss_stats = faiss_demo.build(documents)

    chroma_demo = ChromaDemo(case_id, embeddings, provider)
    chroma_stats = chroma_demo.build(documents)

    compare_logger.info(
        f"BUILD SUMMARY — FAISS: {faiss_stats['build_time_s']}s / {faiss_stats['disk_size_kb']} KB  |  "
        f"Chroma: {chroma_stats['build_time_s']}s / {chroma_stats['disk_size_kb']} KB"
    )

    _engine_cache[cache_key] = (faiss_demo, chroma_demo)
    return faiss_demo, chroma_demo


def clear_indexes(case_id: Optional[str] = None, provider: Optional[str] = None) -> list[str]:
    """Deletes persisted FAISS/Chroma stores from disk and drops matching entries
    from the in-memory build cache, so the next build starts completely fresh.
    Omit case_id/provider to wipe every persisted store instead of just one pair."""
    log_banner(compare_logger, f"CLEAR INDEXES — case_study={case_id or 'ALL'} | provider={provider or 'ALL'}")

    # Drop matching in-memory engines first — without this, build_both() would keep
    # happily serving the now-deleted-from-disk index out of _engine_cache.
    for cache_key in list(_engine_cache.keys()):
        cached_case, cached_provider = cache_key
        if (case_id is None or cached_case == case_id) and (provider is None or cached_provider == provider):
            del _engine_cache[cache_key]

    # Persisted store folders are named "<case_id>_<provider>" (see FAISSDemo/ChromaDemo
    # persist_dir) — filter on that naming convention rather than tracking a separate list.
    cleared = []
    for root in (BASE_DIR / "faiss_index", BASE_DIR / "chroma_store"):
        if not root.exists():
            continue
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            if case_id and not entry.name.startswith(f"{case_id}_"):
                continue
            if provider and not entry.name.endswith(f"_{provider}"):
                continue
            shutil.rmtree(entry, ignore_errors=True)
            cleared.append(str(entry))

    compare_logger.info(f"CLEAR DONE — removed {len(cleared)} persisted store(s)")
    for path in cleared:
        compare_logger.info(f"  removed: {path}")
    return cleared


def run_comparison(
    case_id: str,
    query: str,
    provider: Optional[str] = None,
    k: int = 3,
    use_filter: bool = False,
    force_rebuild: bool = False,
):
    """Builds both engines (cached) then runs the same query through each, returning a
    single comparison dict — this is the function driven by both run_demo.py and the UI."""
    from case_studies import CASE_STUDIES

    faiss_demo, chroma_demo = build_both(case_id, provider, force_rebuild=force_rebuild)

    # Resolve the case study's predefined filter_demo (field/value) into the
    # (field, value) tuple both engines' .query() expect — only when the caller
    # opted in AND the case study actually defines one (see case_studies/__init__.py).
    metadata_filter = None
    if use_filter:
        filter_demo = CASE_STUDIES[case_id]["filter_demo"]
        if filter_demo:
            metadata_filter = (filter_demo["field"], filter_demo["value"])

    log_banner(compare_logger, f"QUERY — {query!r} | case_study={case_id} | filter={metadata_filter}")
    # Same query, same k, same filter — run against both engines back to back so the
    # latency numbers are directly comparable.
    faiss_result = faiss_demo.query(query, k=k, metadata_filter=metadata_filter)
    chroma_result = chroma_demo.query(query, k=k, metadata_filter=metadata_filter)

    # >1 means FAISS was faster (took a smaller fraction of Chroma's time); reported
    # as "FAISS is Nx the speed of Chroma" in both the CLI table and the UI verdict line.
    speedup = (
        round(chroma_result["latency_ms"] / faiss_result["latency_ms"], 2)
        if faiss_result["latency_ms"] > 0
        else None
    )
    compare_logger.info(
        f"RESULT — FAISS {faiss_result['latency_ms']} ms vs Chroma {chroma_result['latency_ms']} ms "
        f"(FAISS is {speedup}x the speed of Chroma)" if speedup else "RESULT — timing inconclusive"
    )

    # Merge each engine's build-time stats() with this query's result dict so callers
    # get one flat payload per engine instead of having to join two calls themselves.
    return {
        "case_study": case_id,
        "query": query,
        "provider": provider or MODEL_PROVIDER_DEFAULT,
        "metadata_filter_applied": metadata_filter,
        "faiss": {**faiss_demo.stats(), **faiss_result},
        "chroma": {**chroma_demo.stats(), **chroma_result},
        "faiss_vs_chroma_latency_ratio": speedup,
    }
