"""
===================================================================
 LangChain RAG AI Agent - FastAPI Backend Server
 Framework : FastAPI + LangChain + ChromaDB + PyPDF
 LLM Options:
   • OpenAI  — gpt-4.1-nano  (cheap, fast, pay-as-you-go, no local GPU needed)
   • Ollama  — gemma4:31b-cloud  (fully local, requires `ollama serve`)

 HOW TO SWITCH:
   Edit MODEL_PROVIDER in the shared .env file (c:\\AI Agent study\\.env)
   to "openai" or "ollama" — no code changes needed, just restart the server.
   Put your real key in that same file: OPENAI_API_KEY=sk-...
===================================================================
Run with: uvicorn server:app --reload --port 8000
"""

import io
import os
import sys
import shutil
import time
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(find_dotenv(usecwd=True))  # picks up the shared c:\AI Agent study\.env (searches this dir and parents)

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ===========================================================================
# LOGGING SETUP
# ===========================================================================
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"rag_agent_{datetime.now().strftime('%Y-%m-%d')}.log"

# Log format — includes time, level, and the logger name (module area)
LOG_FORMAT = "%(asctime)s  [%(levelname)-8s]  %(name)-28s │ %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ── Root logger ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        # Rotating file handler — keeps last 5 × 5 MB log files
        logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        ),
        # Console handler — coloured output to terminal
        logging.StreamHandler(sys.stdout),
    ],
)

# ── Per-module loggers ─────────────────────────────────────────────────────
logger          = logging.getLogger("rag_server")          # General server events
embed_logger    = logging.getLogger("rag_server.embedding")  # Embedding operations
chunk_logger    = logging.getLogger("rag_server.chunking")   # Chunking operations
vector_logger   = logging.getLogger("rag_server.vectorstore") # Chroma VectorStore ops
retrieval_logger= logging.getLogger("rag_server.retrieval")   # Query retrieval ops
llm_logger      = logging.getLogger("rag_server.llm")         # LLM chain invocations
api_logger      = logging.getLogger("rag_server.api")         # FastAPI request/response

# Quieten noisy libraries that spam DEBUG
for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "langchain", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger.info("=" * 70)
logger.info("  🚀 LangChain RAG AI Agent Server — Initialising")
logger.info(f"  📁 Log file: {LOG_FILE}")
logger.info("=" * 70)


# ===========================================================================
# MODEL PROVIDER — switch between "openai" and "ollama"
# ===========================================================================
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  "openai"  → Uses OpenAI API (gpt-4o-mini + text-embedding-3-small)    │
# │             Requires: pip install langchain-openai                      │
# │             Set env var: OPENAI_API_KEY=sk-...                          │
# │             Free tier: https://platform.openai.com (new accounts get   │
# │             $5 free credit; gpt-4o-mini is the cheapest model ~$0.15/M)│
# │                                                                         │
# │  "ollama"  → Uses local Ollama (no API key, no cost, runs offline)     │
# │             Requires: ollama serve + ollama pull nomic-embed-text       │
# └─────────────────────────────────────────────────────────────────────────┘
# Read from the shared .env (c:\AI Agent study\.env) — change MODEL_PROVIDER there
# to "openai" or "ollama" to switch every project's model at once, no code edits needed.
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "ollama").strip().lower()

# OpenAI model names (used when MODEL_PROVIDER == "openai")
OPENAI_LLM_MODEL       = os.environ.get("OPENAI_LLM_MODEL", "gpt-4.1-nano")       # cheap, fast OpenAI model
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")  # cheapest OpenAI embedding model

# Ollama model names (used when MODEL_PROVIDER == "ollama")
OLLAMA_LLM_MODEL       = os.environ.get("OLLAMA_LLM_MODEL", "gemma4:31b-cloud")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL        = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


# ===========================================================================
# LangChain Imports
# ===========================================================================
logger.info("Importing LangChain modules...")
logger.info(f"   Model provider selected: {MODEL_PROVIDER.upper()}")
try:
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.documents import Document

    if MODEL_PROVIDER == "openai":
        # ── OpenAI imports ─────────────────────────────────────────────
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        logger.info("✅ OpenAI LangChain modules imported (langchain-openai).")
    else:
        # ── Ollama imports ─────────────────────────────────────────────
        from langchain_ollama import OllamaEmbeddings, ChatOllama
        logger.info("✅ Ollama LangChain modules imported (langchain-ollama).")

    logger.info("✅ All LangChain core modules imported successfully.")
except ImportError as err:
    logger.critical(f"❌ LangChain dependency missing: {err}")
    if MODEL_PROVIDER == "openai":
        logger.critical("Run: pip install langchain langchain-community langchain-core langchain-openai chromadb pypdf fastapi uvicorn")
    else:
        logger.critical("Run: pip install langchain langchain-community langchain-core langchain-ollama chromadb pypdf fastapi uvicorn")
    sys.exit(1)


# ===========================================================================
# QUERY CLASSIFIER — detects broad / whole-document queries
# ===========================================================================
# Keywords that indicate the user wants the ENTIRE document, not just a snippet
BROAD_QUERY_KEYWORDS = {
    "summarize", "summary", "summarise", "overview", "outline",
    "explain the document", "explain the pdf", "explain the file",
    "what is the document", "what is the pdf", "whole document",
    "entire document", "full document", "all pages", "entire pdf",
    "complete document", "key points", "main points", "all topics",
    "what does the document cover", "what does this document say",
    "describe the document", "tell me about this document",
    "tell me about this pdf", "give me an overview",
}

def classify_query(query: str) -> str:
    """
    Returns 'broad' if the query targets the whole document (summary, overview, etc.)
    or 'specific' for targeted factual questions.
    """
    q_lower = query.lower()
    for kw in BROAD_QUERY_KEYWORDS:
        if kw in q_lower:
            retrieval_logger.info(f"   🏷️  Query classified as BROAD (matched keyword: '{kw}')")
            return "broad"
    retrieval_logger.info("   🏷️  Query classified as SPECIFIC")
    return "specific"


# ===========================================================================
# LangChain RAG Manager Class
# ===========================================================================
class LangChainRAGManager:
    """
    RAG Manager using LangChain for document chunking, embedding generation,
    Chroma VectorStore persistence, and retriever management.
    """

    def __init__(self, db_dir: Optional[str] = None):
        logger.info("─" * 60)
        logger.info("🔧 [RAGManager.__init__] Initialising LangChain RAG Manager")

        if db_dir is None:
            db_dir = os.path.join(os.path.dirname(__file__), "chroma_db_langchain")
        
        self.db_dir = db_dir
        self.enabled = True
        self.uploaded_files: List[Dict[str, Any]] = []
        self.vectorstore: Optional[Chroma] = None
        self.retriever = None
        self.embeddings = None

        logger.info(f"   📂 ChromaDB persist directory : {self.db_dir}")
        logger.info(f"   🔁 RAG enabled                : {self.enabled}")

        self._init_embeddings()
        self._init_vectorstore()

    # ------------------------------------------------------------------
    def _init_embeddings(self):
        """Initialises embedding model based on MODEL_PROVIDER setting."""
        embed_logger.info("─" * 60)
        embed_logger.info(f"🔌 [_init_embeddings] Provider = {MODEL_PROVIDER.upper()}")

        t0 = time.perf_counter()

        if MODEL_PROVIDER == "openai":
            # ── OpenAI Embeddings ──────────────────────────────────────────
            # Model  : text-embedding-3-small (~$0.02 per 1M tokens, very cheap)
            # Requires OPENAI_API_KEY environment variable to be set.
            embed_logger.info(f"   Model  : {OPENAI_EMBEDDING_MODEL}")
            embed_logger.info("   Host   : api.openai.com (requires OPENAI_API_KEY)")
            try:
                api_key = os.environ.get("OPENAI_API_KEY", "")
                if not api_key or api_key == "your_openai_api_key_here":
                    embed_logger.error("❌ OPENAI_API_KEY is not set!")
                    embed_logger.error(r"   Set it in c:\AI Agent study\.env, then restart the server.")
                    self.embeddings = None
                    return
                self.embeddings = OpenAIEmbeddings(
                    model=OPENAI_EMBEDDING_MODEL,
                    openai_api_key=api_key,
                )
                elapsed = time.perf_counter() - t0
                embed_logger.info(f"✅ OpenAIEmbeddings ({OPENAI_EMBEDDING_MODEL}) initialised in {elapsed:.3f}s")
            except Exception as e:
                embed_logger.error(f"❌ OpenAIEmbeddings init FAILED: {e}")
                embed_logger.warning("   📌 Check your OPENAI_API_KEY and run: pip install langchain-openai")
                self.embeddings = None
        else:
            # ── Ollama Embeddings (local, no API key needed) ───────────────
            # Model  : nomic-embed-text  (runs fully offline via Ollama)
            # Requires Ollama running: ollama serve && ollama pull nomic-embed-text
            embed_logger.info(f"   Model  : {OLLAMA_EMBEDDING_MODEL}")
            embed_logger.info(f"   Host   : {OLLAMA_BASE_URL}")
            try:
                self.embeddings = OllamaEmbeddings(
                    model=OLLAMA_EMBEDDING_MODEL,
                    base_url=OLLAMA_BASE_URL,
                )
                elapsed = time.perf_counter() - t0
                embed_logger.info(f"✅ OllamaEmbeddings ({OLLAMA_EMBEDDING_MODEL}) initialised in {elapsed:.3f}s")
            except Exception as e:
                embed_logger.error(f"❌ OllamaEmbeddings init FAILED: {e}")
                embed_logger.warning("   📌 Make sure Ollama is running: `ollama serve`")
                self.embeddings = None

    # ------------------------------------------------------------------
    def _init_vectorstore(self):
        """Loads or creates Chroma VectorStore with logging."""
        vector_logger.info("─" * 60)
        vector_logger.info("🗄️  [_init_vectorstore] Loading/creating ChromaDB VectorStore...")
        vector_logger.info(f"   Collection : pdf_knowledge_base")
        vector_logger.info(f"   Persist dir: {self.db_dir}")

        if not self.embeddings:
            vector_logger.warning("⚠️  Skipping VectorStore init — embedding model unavailable.")
            return

        t0 = time.perf_counter()
        try:
            self.vectorstore = Chroma(
                collection_name="pdf_knowledge_base",
                embedding_function=self.embeddings,
                persist_directory=self.db_dir,
            )
            count = self.vectorstore._collection.count()
            elapsed = time.perf_counter() - t0
            vector_logger.info(f"✅ ChromaDB connected in {elapsed:.3f}s")
            vector_logger.info(f"   📊 Existing chunks in collection: {count}")

            if count > 0:
                # MMR (Maximal Marginal Relevance) returns diverse chunks,
                # reducing duplicate/overlapping content in the context window.
                self.retriever = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}
                )
                vector_logger.info("   🔍 Retriever activated (MMR, top-k=8, fetch_k=20)")
            else:
                vector_logger.info("   ℹ️  Collection is empty — retriever will activate after first PDF upload.")
        except Exception as err:
            elapsed = time.perf_counter() - t0
            vector_logger.error(f"❌ ChromaDB init error after {elapsed:.3f}s: {err}")
            self.vectorstore = None

    # ------------------------------------------------------------------
    def add_pdf_document(self, filename: str, content_bytes: bytes) -> Dict[str, Any]:
        """Parses PDF, chunks text, generates embeddings, and indexes into ChromaDB."""
        logger.info("=" * 70)
        logger.info(f"📄 [add_pdf_document] Processing upload: '{filename}'")
        logger.info(f"   File size: {len(content_bytes):,} bytes")

        if not self.embeddings:
            logger.error("❌ Embeddings unavailable — cannot process document.")
            raise RuntimeError("Ollama embedding model (nomic-embed-text) is unavailable. Make sure Ollama is running.")

        # ── Step 1: Parse document ──────────────────────────────────
        ext = os.path.splitext(filename)[1].lower()
        text_content = ""
        page_count = 1
        logger.info(f"   📑 File extension detected: '{ext}'")

        if ext == ".pdf":
            logger.info("   🔍 Method: pypdf.PdfReader — extracting text from PDF pages...")
            t0 = time.perf_counter()
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(content_bytes))
                page_count = len(reader.pages)
                logger.info(f"   📖 Total pages found: {page_count}")

                pages_text = []
                for i, page in enumerate(reader.pages):
                    txt = page.extract_text() or ""
                    char_count = len(txt.strip())
                    logger.debug(f"      Page {i+1:>3}: {char_count:>6} chars extracted")
                    if txt.strip():
                        pages_text.append(f"[Page {i+1}]\n{txt}")

                text_content = "\n\n".join(pages_text)
                elapsed = time.perf_counter() - t0
                logger.info(f"   ✅ PDF parsed in {elapsed:.3f}s — total chars: {len(text_content):,}")
            except Exception as pdf_err:
                elapsed = time.perf_counter() - t0
                logger.warning(f"   ⚠️  pypdf error after {elapsed:.3f}s: {pdf_err}")
                logger.warning("   🔄 Fallback: decoding as raw UTF-8 text")
                text_content = content_bytes.decode("utf-8", errors="ignore")
        else:
            logger.info("   📝 Method: UTF-8 decode (plain text / markdown)")
            text_content = content_bytes.decode("utf-8", errors="ignore")
            logger.info(f"   ✅ Decoded {len(text_content):,} chars")

        if not text_content.strip():
            logger.error(f"   ❌ No extractable text found in '{filename}'")
            raise ValueError(f"Could not extract readable text content from '{filename}'.")

        # ── Step 2: Chunking ────────────────────────────────────────
        chunk_logger.info("─" * 60)
        chunk_logger.info("✂️  [RecursiveCharacterTextSplitter] Starting chunking...")
        chunk_logger.info("   chunk_size    : 900 chars  (increased for richer context per chunk)")
        chunk_logger.info("   chunk_overlap : 120 chars  (increased to preserve cross-boundary context)")
        chunk_logger.info("   separators    : ['\\n\\n', '\\n', ' ', '']")

        t0 = time.perf_counter()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,    # Larger chunks → more content per retrieved chunk
            chunk_overlap=120, # More overlap → no information lost at chunk boundaries
            separators=["\n\n", "\n", " ", ""]
        )

        doc = Document(page_content=text_content, metadata={"source": filename, "pages": page_count})
        chunks = text_splitter.split_documents([doc])
        elapsed = time.perf_counter() - t0

        chunk_logger.info(f"✅ Chunking complete in {elapsed:.3f}s")
        chunk_logger.info(f"   Total chunks produced    : {len(chunks)}")
        chunk_logger.info(f"   Avg chunk size           : {sum(len(c.page_content) for c in chunks) // max(len(chunks),1)} chars")
        chunk_logger.info(f"   Total chars in document  : {len(text_content):,}")
        chunk_logger.info(f"   Coverage per top-8 chunks: ~{min(100, 8 * 900 * 100 // max(len(text_content),1))}% of document")
        for i, c in enumerate(chunks[:3]):  # preview first 3 chunks
            chunk_logger.debug(f"   [Chunk #{i+1}] {len(c.page_content)} chars | {c.page_content[:80].replace(chr(10),' ')}…")
        if len(chunks) > 3:
            chunk_logger.debug(f"   ... and {len(chunks)-3} more chunks (enable DEBUG to see all)")

        # ── Step 3: Embed & Store in ChromaDB ──────────────────────
        embed_logger.info("─" * 60)
        embed_logger.info("🧠 [OllamaEmbeddings] Generating embeddings & indexing into ChromaDB...")
        embed_logger.info(f"   Embedding model : nomic-embed-text")
        embed_logger.info(f"   Chunks to embed : {len(chunks)}")

        t0 = time.perf_counter()
        if self.vectorstore is None:
            embed_logger.info("   🆕 No existing collection — calling Chroma.from_documents()")
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name="pdf_knowledge_base",
                persist_directory=self.db_dir,
            )
        else:
            embed_logger.info("   ➕ Existing collection — calling vectorstore.add_documents()")
            self.vectorstore.add_documents(chunks)

        elapsed = time.perf_counter() - t0
        total_in_db = self.vectorstore._collection.count()
        embed_logger.info(f"✅ Embeddings generated & stored in {elapsed:.3f}s")
        embed_logger.info(f"   Avg time per chunk     : {elapsed/max(len(chunks),1)*1000:.1f} ms")
        embed_logger.info(f"   Total vectors in ChromaDB: {total_in_db}")

        # ── Step 4: Update Retriever ────────────────────────────────
        vector_logger.info("🔍 [as_retriever] Updating retriever (MMR, k=8, fetch_k=20)...")
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}
        )
        vector_logger.info("✅ Retriever updated — MMR diversity search active.")

        file_info = {
            "filename": filename,
            "char_count": len(text_content),
            "chunk_count": len(chunks),
            "pages": page_count,
            "uploaded_at": datetime.now(timezone.utc).strftime("%H:%M:%S")
        }

        existing = [f["filename"] for f in self.uploaded_files]
        if filename not in existing:
            self.uploaded_files.append(file_info)
        else:
            for f in self.uploaded_files:
                if f["filename"] == filename:
                    f.update(file_info)

        logger.info(f"🎉 Document '{filename}' fully indexed!")
        logger.info(f"   Pages: {page_count}  |  Chunks: {len(chunks)}  |  Total DB vectors: {total_in_db}")
        logger.info("=" * 70)
        return file_info

    # ------------------------------------------------------------------
    def get_relevant_chunks(self, query: str, query_type: str = "specific") -> List[Document]:
        """
        Retrieves document chunks for the given query.

        Strategy:
          - 'broad'  queries (summarize, overview, etc.) → fetch ALL chunks for
            the source document(s) so the LLM sees the complete document.
          - 'specific' queries → MMR similarity search returning top-8 diverse chunks.
        """
        retrieval_logger.info("─" * 60)
        retrieval_logger.info("🔎 [get_relevant_chunks] Starting retrieval...")
        retrieval_logger.info(f"   Query       : {query!r}")
        retrieval_logger.info(f"   Query type  : {query_type.upper()}")
        retrieval_logger.info(f"   RAG enabled : {self.enabled}")
        retrieval_logger.info(f"   Vectorstore : {'ready' if self.vectorstore else 'NOT initialised'}")
        retrieval_logger.info(f"   Retriever   : {'ready' if self.retriever else 'NOT ready'}")

        if not self.enabled or not self.vectorstore or not self.retriever:
            retrieval_logger.warning("⚠️  Retrieval skipped — conditions not met.")
            return []

        t0 = time.perf_counter()
        try:
            if query_type == "broad":
                # ── BROAD query: fetch every chunk in the collection ──────
                retrieval_logger.info("   📖 BROAD query detected — fetching ALL document chunks for full context.")
                total_in_db = self.vectorstore._collection.count()
                retrieval_logger.info(f"   Total vectors in DB: {total_in_db} — fetching all via similarity_search")
                # Fetch all chunks by using a very high k
                docs = self.vectorstore.similarity_search(query, k=min(total_in_db, 200))
                elapsed = time.perf_counter() - t0
                total_chars = sum(len(d.page_content) for d in docs)
                retrieval_logger.info(f"✅ BROAD retrieval: {len(docs)} chunks | {total_chars:,} total chars fetched in {elapsed*1000:.1f} ms")
                retrieval_logger.info(f"   📊 Context coverage: ALL {total_in_db} indexed chunks provided to LLM")
            else:
                # ── SPECIFIC query: MMR top-8 diversity search ────────────
                retrieval_logger.info("   🎯 SPECIFIC query — running MMR similarity search (k=8, fetch_k=20)...")
                docs = self.retriever.invoke(query)
                elapsed = time.perf_counter() - t0
                total_chars = sum(len(d.page_content) for d in docs)
                retrieval_logger.info(f"✅ SPECIFIC retrieval: {len(docs)} chunks | {total_chars:,} chars in {elapsed*1000:.1f} ms")

            for i, d in enumerate(docs):
                src = d.metadata.get("source", "unknown")
                preview = d.page_content[:100].replace("\n", " ")
                retrieval_logger.info(f"   [Chunk #{i+1:>3}] source={src!r} | {preview}…")

            return docs
        except Exception as e:
            elapsed = time.perf_counter() - t0
            retrieval_logger.error(f"❌ Retrieval error after {elapsed*1000:.1f} ms: {e}", exc_info=True)
            return []

    # ------------------------------------------------------------------
    def clear_all(self) -> Dict[str, Any]:
        """Clears all stored PDF document vectors and resets database."""
        logger.info("🧹 [clear_all] Clearing ChromaDB vector store and file list...")
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
                vector_logger.info("   ChromaDB collection deleted.")
            if os.path.exists(self.db_dir):
                shutil.rmtree(self.db_dir, ignore_errors=True)
                vector_logger.info(f"   Persist directory removed: {self.db_dir}")
        except Exception as err:
            logger.error(f"❌ Error during clear: {err}")

        self.vectorstore = None
        self.retriever = None
        self.uploaded_files = []
        self._init_vectorstore()
        logger.info("✅ Knowledge base cleared and VectorStore re-initialised.")
        return {"status": "success", "message": "Knowledge base cleared."}

    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """Returns RAG system status."""
        total_chunks = 0
        if self.vectorstore:
            try:
                total_chunks = self.vectorstore._collection.count()
            except Exception:
                total_chunks = sum(f.get("chunk_count", 0) for f in self.uploaded_files)

        return {
            "enabled": self.enabled,
            "total_files": len(self.uploaded_files),
            "total_chunks": total_chunks,
            "files": self.uploaded_files
        }


# ===========================================================================
# Instantiate RAG Manager & LLM
# ===========================================================================
logger.info("─" * 60)
logger.info("🛠️  Instantiating RAGManager...")
rag = LangChainRAGManager()

logger.info("─" * 60)
if MODEL_PROVIDER == "openai":
    # ── OpenAI LLM ────────────────────────────────────────────────────────
    # Model       : gpt-4.1-nano (cheap, fast OpenAI model)
    # Temperature : 0.2          (factual, low randomness for RAG answers)
    # -------------------------------------------------------------------------
    # To switch back to Ollama, set MODEL_PROVIDER=ollama in the shared .env.
    logger.info("🤖 [ChatOpenAI] Initialising OpenAI LLM...")
    logger.info(f"   Model       : {OPENAI_LLM_MODEL}")
    logger.info("   Temperature : 0.2")
    logger.info("   Host        : api.openai.com")
    _openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not _openai_key or _openai_key == "your_openai_api_key_here":
        logger.error("❌ OPENAI_API_KEY not set — LLM calls will fail!")
        logger.error(r"   Set it in c:\AI Agent study\.env, then restart the server.")
    llm = ChatOpenAI(
        model=OPENAI_LLM_MODEL,
        temperature=0.2,
        openai_api_key=_openai_key or "MISSING_KEY",
    )
    llm_logger.info(f"✅ ChatOpenAI ({OPENAI_LLM_MODEL}) instance ready.")
else:
    # ── Ollama LLM (local, no API key, no cost) ───────────────────────────
    # Model       : gemma4:31b-cloud  (runs via local Ollama server)
    # Requires    : ollama serve && ollama pull gemma4:31b-cloud
    # -------------------------------------------------------------------------
    # To switch to OpenAI, set MODEL_PROVIDER=openai in the shared .env.
    logger.info("🤖 [ChatOllama] Initialising local Ollama LLM...")
    logger.info(f"   Model       : {OLLAMA_LLM_MODEL}")
    logger.info("   Temperature : 0.2")
    logger.info(f"   Host        : {OLLAMA_BASE_URL}")
    llm = ChatOllama(
        model=OLLAMA_LLM_MODEL,
        temperature=0.2,
        base_url=OLLAMA_BASE_URL,
    )
    llm_logger.info(f"✅ ChatOllama ({OLLAMA_LLM_MODEL}) instance ready.")

# RAG Prompt Template
rag_prompt = ChatPromptTemplate.from_template("""
You are an intelligent AI Assistant powered by LangChain. Answer the user's question accurately using the uploaded PDF document context provided below.

Instructions:
1. Base your answer on the provided PDF Context.
2. Be precise, structured, clear, and easy to read.
3. If the context does not contain enough information to answer the question, state politely that the answer isn't present in the uploaded PDF.

PDF Document Context:
{context}

User Question: {question}

Answer:""")

# General (no-PDF) prompt
general_prompt = ChatPromptTemplate.from_template("""
You are a helpful, smart, and friendly AI assistant powered by LangChain and Ollama.
Answer the user's question clearly and concisely.

User Question: {question}

Answer:""")

# LangChain LCEL Chains
def format_docs(docs: List[Document]) -> str:
    formatted = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'PDF')}]\n{d.page_content}" for d in docs
    )
    llm_logger.debug(f"   format_docs() → {len(docs)} docs, {len(formatted)} total chars in context")
    return formatted

rag_chain = (
    {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
    | rag_prompt
    | llm
    | StrOutputParser()
)

general_chain = (
    {"question": lambda x: x["question"]}
    | general_prompt
    | llm
    | StrOutputParser()
)

logger.info("✅ LangChain LCEL RAG chain & general chain compiled.")


# ===========================================================================
# FastAPI Web Application
# ===========================================================================
app = FastAPI(
    title="LangChain RAG AI Agent API",
    description="FastAPI backend for LangChain AI Agent with PDF Upload & RAG QA support",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response middleware — logs every API call ──────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    api_logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    api_logger.info(f"← {request.method} {request.url.path} → {response.status_code} ({elapsed:.1f} ms)")
    return response


# ===========================================================================
# Request / Response Schemas
# ===========================================================================
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    rag_used: bool
    sources: List[Dict[str, str]]
    timestamp: str

class RAGToggleRequest(BaseModel):
    enabled: bool


# ===========================================================================
# API Routes
# ===========================================================================
@app.get("/api/health")
def health():
    api_logger.info("🏥 Health check requested.")
    return {
        "status": "online",
        "agent_framework": "LangChain",
        "model_provider": MODEL_PROVIDER,
        "model": OPENAI_LLM_MODEL if MODEL_PROVIDER == "openai" else OLLAMA_LLM_MODEL,
        "rag_enabled": rag.enabled
    }


@app.get("/api/rag/status")
def get_rag_status():
    status = rag.get_status()
    api_logger.info(f"📊 RAG status: {status['total_files']} files | {status['total_chunks']} chunks | enabled={status['enabled']}")
    return status


@app.post("/api/rag/toggle")
def toggle_rag(req: RAGToggleRequest):
    rag.enabled = bool(req.enabled)
    api_logger.info(f"🔁 RAG toggled → enabled={rag.enabled}")
    return rag.get_status()


@app.post("/api/upload-pdf")
@app.post("/api/upload-document")
async def upload_pdf(file: UploadFile = File(...)):
    """Handles PDF file upload, extracts text chunks, and populates Chroma vectorstore."""
    api_logger.info(f"📥 Upload received: filename='{file.filename}' content_type='{file.content_type}'")

    if not file.filename:
        api_logger.warning("⚠️  Upload rejected — no filename provided.")
        raise HTTPException(status_code=400, detail="No file selected for upload.")

    content_bytes = await file.read()
    api_logger.info(f"   Bytes read: {len(content_bytes):,}")

    if not content_bytes:
        api_logger.warning("⚠️  Upload rejected — file is empty.")
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        file_info = rag.add_pdf_document(file.filename, content_bytes)
        api_logger.info(f"✅ Upload success: {file_info}")
        return {
            "status": "success",
            "message": f"Successfully indexed PDF document '{file.filename}' into LangChain vector database.",
            "file": file_info,
            "rag_status": rag.get_status()
        }
    except Exception as err:
        api_logger.error(f"❌ Upload processing failed: {err}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(err)}")


@app.delete("/api/rag/documents")
def clear_documents():
    api_logger.info("🧹 Clear all documents requested.")
    return rag.clear_all()


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    query = req.message.strip()
    llm_logger.info("=" * 70)
    llm_logger.info("💬 [/api/chat] New chat request received")
    llm_logger.info(f"   User query  : {query!r}")
    llm_logger.info(f"   RAG enabled : {rag.enabled}")
    llm_logger.info(f"   VectorStore : {'ready' if rag.vectorstore else 'empty'}")

    if not query:
        api_logger.warning("⚠️  Chat rejected — empty query.")
        raise HTTPException(status_code=400, detail="Message query cannot be empty.")

    sources_info = []
    rag_used = False
    t_total = time.perf_counter()

    if rag.enabled and rag.vectorstore is not None:
        # ── Phase 1: Classify query intent ───────────────────────
        llm_logger.info("🏷️  Phase 1/3: Classifying query intent...")
        query_type = classify_query(query)
        llm_logger.info(f"   Strategy selected: {'FULL-DOCUMENT fetch' if query_type == 'broad' else 'MMR similarity search (top-8)'}")

        # ── Phase 2: Retrieval ────────────────────────────────────
        llm_logger.info("🔍 Phase 2/3: Retrieving relevant PDF chunks...")
        docs = rag.get_relevant_chunks(query, query_type=query_type)

        if docs:
            rag_used = True
            total_context_chars = sum(len(d.page_content) for d in docs)
            llm_logger.info(f"   ✅ {len(docs)} chunk(s) retrieved")
            llm_logger.info(f"   📏 Total context chars fed to LLM: {total_context_chars:,}")

            for d in docs:
                sources_info.append({
                    "source": str(d.metadata.get("source", "PDF")),
                    "snippet": d.page_content.replace("\n", " ").strip()[:200]
                })

            # ── Phase 3: LLM RAG Chain ────────────────────────────
            llm_logger.info("🤖 Phase 3/3: Invoking RAG chain (Prompt → ChatOllama → StrOutputParser)...")
            llm_logger.info(f"   Prompt template  : rag_prompt")
            llm_logger.info(f"   Context chunks   : {len(docs)}")
            llm_logger.info(f"   Context chars    : {total_context_chars:,}")
            t0 = time.perf_counter()
            try:
                ai_response = rag_chain.invoke({"docs": docs, "question": query})
                elapsed = time.perf_counter() - t0
                llm_logger.info(f"✅ RAG chain responded in {elapsed:.2f}s")
                llm_logger.info(f"   Response length  : {len(ai_response)} chars")
                llm_logger.debug(f"   Response preview : {ai_response[:200]}…")
            except Exception as chain_err:
                elapsed = time.perf_counter() - t0
                llm_logger.error(f"❌ RAG chain error after {elapsed:.2f}s: {chain_err}", exc_info=True)
                llm_logger.info("🔄 Fallback: invoking general_chain without context...")
                ai_response = general_chain.invoke({"question": query})
                rag_used = False
        else:
            llm_logger.info("ℹ️  No matching chunks — falling back to general LLM chain.")
            llm_logger.info("🤖 Invoking general_chain (no PDF context)...")
            t0 = time.perf_counter()
            ai_response = general_chain.invoke({"question": query})
            elapsed = time.perf_counter() - t0
            llm_logger.info(f"✅ General chain responded in {elapsed:.2f}s | {len(ai_response)} chars")
    else:
        llm_logger.info("ℹ️  RAG disabled or no PDF uploaded — using general_chain directly.")
        t0 = time.perf_counter()
        ai_response = general_chain.invoke({"question": query})
        elapsed = time.perf_counter() - t0
        llm_logger.info(f"✅ General chain responded in {elapsed:.2f}s | {len(ai_response)} chars")

    total_elapsed = time.perf_counter() - t_total
    llm_logger.info(f"⏱️  Total /api/chat round-trip: {total_elapsed:.2f}s")
    llm_logger.info(f"   rag_used={rag_used}  |  sources={len(sources_info)}")
    llm_logger.info("=" * 70)

    return ChatResponse(
        response=ai_response.strip(),
        rag_used=rag_used,
        sources=sources_info,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


# ===========================================================================
# Startup / Shutdown Events
# ===========================================================================
@app.on_event("startup")
async def on_startup():
    logger.info("─" * 60)
    logger.info("🟢 FastAPI application STARTED")
    logger.info(f"   Docs available at : http://localhost:8000/docs")
    logger.info(f"   Log file          : {LOG_FILE}")
    logger.info("─" * 60)

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("─" * 60)
    logger.info("🔴 FastAPI application SHUTTING DOWN")
    logger.info("─" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
