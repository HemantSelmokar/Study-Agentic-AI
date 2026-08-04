"""Logging setup for the Vector DB Showdown UI's FastAPI server (separate from
the engine's own logs in ..\\VectorDB_Showdown\\logs — this one tracks HTTP
requests/responses so both layers can be shown during the lecture).

Named `ui_logger_config` (not `logger_config`) deliberately — server.py adds
the sibling VectorDB_Showdown folder to sys.path, which also has its own
`logger_config.py`; identical module names would collide in sys.modules."""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"vectordb_ui_{datetime.now().strftime('%Y-%m-%d')}.log"

LOG_FORMAT = "%(asctime)s  [%(levelname)-8s]  %(name)-20s │ %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Attach handlers to the "vdb_ui" namespace logger directly (propagate=False)
# instead of logging.basicConfig() — basicConfig only touches the ROOT logger and
# is a no-op after the first call in a process, so whichever of this module or the
# engine's logger_config.py imports first would otherwise silently win and the
# other's log file would stop receiving any lines.
_vdb_ui_root_logger = logging.getLogger("vdb_ui")
_vdb_ui_root_logger.setLevel(logging.DEBUG)
_vdb_ui_root_logger.propagate = False

if not _vdb_ui_root_logger.handlers:
    _formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    _file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    _file_handler.setFormatter(_formatter)

    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setFormatter(_formatter)

    _vdb_ui_root_logger.addHandler(_file_handler)
    _vdb_ui_root_logger.addHandler(_console_handler)

# Same rationale as the engine's logger_config.py: third-party HTTP/DB library
# chatter would otherwise drown out the request/response lines this file cares about.
for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Single logger for this module — server.py's log_requests() middleware is the only
# thing that writes through it, so there's no need for further per-concept splitting.
api_logger = logging.getLogger("vdb_ui.api")
