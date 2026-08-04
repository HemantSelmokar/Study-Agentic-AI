"""
===================================================================
 Shared logging setup for the FAISS vs Chroma Vector DB Showdown.
 Every phase of index building / querying logs through here so the
 log file can be opened during the lecture to show students exactly
 what each vector database did under the hood, and how long it took.
===================================================================
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"vectordb_showdown_{datetime.now().strftime('%Y-%m-%d')}.log"

LOG_FORMAT = "%(asctime)s  [%(levelname)-8s]  %(name)-24s │ %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Attach handlers directly to the "vdb" namespace logger (propagate=False) rather
# than calling logging.basicConfig(), which only configures the ROOT logger and is
# a no-op if something else (e.g. the UI server's own logger_config) already called
# it first in the same process — that collision was silently swallowing every
# engine log line whenever this module ran embedded inside VectorDB_Showdown_UI.
_vdb_root_logger = logging.getLogger("vdb")
_vdb_root_logger.setLevel(logging.DEBUG)
_vdb_root_logger.propagate = False

if not _vdb_root_logger.handlers:
    _formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    _file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    _file_handler.setFormatter(_formatter)

    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setFormatter(_formatter)

    _vdb_root_logger.addHandler(_file_handler)
    _vdb_root_logger.addHandler(_console_handler)

# Per-module loggers — each maps to one concept explained in the lecture. All are
# children of "vdb" so they inherit the handlers attached above without any of them
# needing their own setup.
embed_logger    = logging.getLogger("vdb.embedding")     # embedding generation (OpenAI / Ollama)
faiss_logger    = logging.getLogger("vdb.faiss")         # FAISS index build / query / persistence
chroma_logger   = logging.getLogger("vdb.chroma")        # Chroma collection build / query / persistence
compare_logger  = logging.getLogger("vdb.compare")       # head-to-head comparison results
case_logger     = logging.getLogger("vdb.case_study")    # case-study loading / chunking

# Third-party libraries log INFO/DEBUG noise (HTTP request bodies, connection pool
# chatter) that would drown out the lecture-relevant lines above — cap them at WARNING.
for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def log_banner(logger: logging.Logger, title: str) -> None:
    """Prints a visible section banner into the log — makes it easy to point at
    a specific phase on-screen while walking students through the log file."""
    logger.info("=" * 74)
    logger.info(f"  {title}")
    logger.info("=" * 74)
