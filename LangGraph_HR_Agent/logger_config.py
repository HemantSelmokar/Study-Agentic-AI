"""
===================================================================
 LangGraph HR Agent - Central Logger Configuration
 Writes logs to both terminal and rotating log files inside logs/
===================================================================
"""

import sys
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

# Logging Directory
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"hr_agent_{datetime.now().strftime('%Y-%m-%d')}.log"

# Formatting
LOG_FORMAT = "%(asctime)s  [%(levelname)-8s]  %(name)-28s │ %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Configure Root Logger
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,  # 5 MB per log file
            backupCount=5,
            encoding="utf-8"
        ),
        logging.StreamHandler(sys.stdout)
    ]
)

# Named Loggers
logger            = logging.getLogger("hr_agent")              # General events
api_logger        = logging.getLogger("hr_agent.api")          # FastAPI requests/responses
graph_logger      = logging.getLogger("hr_agent.orchestrator")   # LangGraph StateGraph routing
leave_logger      = logging.getLogger("hr_agent.leave")        # Leave Application Agent & DB
salary_logger     = logging.getLogger("hr_agent.salary")       # Salary Agent & Calculations
policy_logger     = logging.getLogger("hr_agent.policy")       # HR Policy RAG & ChromaDB
db_logger         = logging.getLogger("hr_agent.db")           # SQLite database events

# Quieten noisy third-party libraries
for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "langchain"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger.info("=" * 70)
logger.info("  🚀 LangGraph HR Multi-Agent Suite Logger Initialized")
logger.info(f"  📁 Log file: {LOG_FILE}")
logger.info("=" * 70)
