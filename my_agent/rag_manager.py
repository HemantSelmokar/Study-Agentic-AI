"""
Modular RAG Manager for AI Agent Knowledge Base
Handles document uploading, chunking, Chroma vector storage, and dynamic ON/OFF toggling.
"""

import os
import sys
import shutil
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv, find_dotenv

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv(find_dotenv(usecwd=True))  # picks up the shared c:\AI Agent study\.env

# ---------------------------------------------------------------------------
# Embedding provider toggle — set MODEL_PROVIDER=openai or MODEL_PROVIDER=ollama
# in the shared .env file (c:\AI Agent study\.env) to switch models.
# ---------------------------------------------------------------------------
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama").strip().lower()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    if MODEL_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
    else:
        from langchain_ollama import OllamaEmbeddings
except ImportError as e:
    print(f"[Warning] RAG dependencies not fully loaded: {e}")


class RAGManager:
    """
    Modular manager for RAG document indexing and retrieval.
    Can be dynamically enabled or disabled.
    """

    def __init__(self, db_dir: Optional[str] = None):
        if db_dir is None:
            db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
        
        self.db_dir = db_dir
        self.enabled = True
        self.uploaded_files: List[Dict[str, Any]] = []
        self.vectorstore: Optional[Chroma] = None
        self.retriever = None
        self.embeddings = None
        
        # Initialize embeddings model (provider chosen via MODEL_PROVIDER in .env)
        try:
            if MODEL_PROVIDER == "openai":
                self.embeddings = OpenAIEmbeddings(
                    model=OPENAI_EMBEDDING_MODEL,
                    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                )
            else:
                self.embeddings = OllamaEmbeddings(
                    model=OLLAMA_EMBEDDING_MODEL,
                    base_url=OLLAMA_BASE_URL,
                )
        except Exception as err:
            print(f"[RAGManager] Warning: embeddings initialization error ({MODEL_PROVIDER}): {err}")

        # Initialize vector store
        self._init_vectorstore()

    def _init_vectorstore(self):
        """Initializes or loads existing Chroma vectorstore persistent instance."""
        if not self.embeddings:
            return
        
        try:
            self.vectorstore = Chroma(
                collection_name="user_uploaded_docs",
                embedding_function=self.embeddings,
                persist_directory=self.db_dir,
            )
            # Count existing items if any
            count = self.vectorstore._collection.count()
            if count > 0:
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        except Exception as err:
            print(f"[RAGManager] Notice initializing Chroma store: {err}")
            self.vectorstore = None

    def is_enabled(self) -> bool:
        """Return current RAG toggle state."""
        return self.enabled

    def set_enabled(self, state: bool) -> bool:
        """Enable or disable RAG document search."""
        self.enabled = bool(state)
        return self.enabled

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def add_document(self, filename: str, content_bytes: bytes) -> Dict[str, Any]:
        """
        Parses text, markdown, or PDF content, splits it into chunks,
        and adds it to the Chroma vector store.
        """
        if not self.embeddings:
            raise RuntimeError("Embedding model is not available. Please check Ollama connection.")

        # Decode document text based on file type
        text_content = ""
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            try:
                import pypdf
                import io
                reader = pypdf.PdfReader(io.BytesIO(content_bytes))
                pages_text = [page.extract_text() or "" for page in reader.pages]
                text_content = "\n".join(pages_text)
            except Exception as pdf_err:
                # Fallback to UTF-8 decoding if pypdf fails
                text_content = content_bytes.decode("utf-8", errors="ignore")
        else:
            text_content = content_bytes.decode("utf-8", errors="ignore")

        if not text_content.strip():
            raise ValueError(f"No extractable text content found in '{filename}'.")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )

        doc = Document(page_content=text_content, metadata={"source": filename})
        chunks = text_splitter.split_documents([doc])

        # Add to Chroma VectorStore
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name="user_uploaded_docs",
                persist_directory=self.db_dir,
            )
        else:
            self.vectorstore.add_documents(chunks)

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        file_info = {
            "filename": filename,
            "char_count": len(text_content),
            "chunk_count": len(chunks)
        }

        # Keep record of uploaded file
        existing_names = [f["filename"] for f in self.uploaded_files]
        if filename not in existing_names:
            self.uploaded_files.append(file_info)
        else:
            # Update chunk info
            for f in self.uploaded_files:
                if f["filename"] == filename:
                    f["char_count"] = len(text_content)
                    f["chunk_count"] = len(chunks)

        return file_info

    def clear_documents(self) -> Dict[str, Any]:
        """Clear all stored documents and reset vector database."""
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
            if os.path.exists(self.db_dir):
                shutil.rmtree(self.db_dir, ignore_errors=True)
        except Exception as err:
            print(f"[RAGManager] Notice clearing collection: {err}")

        self.vectorstore = None
        self.retriever = None
        self.uploaded_files = []
        self._init_vectorstore()
        return {"status": "cleared", "total_files": 0}

    def get_status(self) -> Dict[str, Any]:
        """Returns current RAG status, toggle state, and loaded files."""
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

    def search(self, query: str) -> str:
        """
        Retrieves matching document snippets for the given query.
        Returns empty or descriptive text if disabled or empty.
        """
        if not self.enabled:
            return "Document search feature is currently turned OFF by the user."

        if not self.vectorstore or not self.retriever:
            return "No user documents have been uploaded to the knowledge base yet."

        try:
            docs = self.retriever.invoke(query)
            if not docs:
                return "No matching information found in the uploaded documents."

            formatted = []
            for d in docs:
                source = d.metadata.get("source", "Uploaded Document")
                content = d.page_content.replace("\n", " ").strip()
                formatted.append(f"📄 [Source: {source}]: {content}")

            return "\n\n".join(formatted)
        except Exception as err:
            return f"Error querying uploaded document knowledge base: {err}"
