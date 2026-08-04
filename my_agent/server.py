"""
FastAPI backend for the AI Agent chat interface with Modular RAG Knowledge Base support.
Run with: uvicorn server:app --reload --port 8000
"""

import io  # provides io.StringIO(), an in-memory text buffer used to capture the agent's printed output
import os  # reads MODEL_PROVIDER / OPENAI_API_KEY from the environment (populated via .env below)
import sys  # imported for general stdio access (not directly used beyond stdout redirection support)
from contextlib import redirect_stdout  # context manager that temporarily reroutes print() output into our buffer, so we can detect which tools the agent called
from datetime import datetime, timezone  # used to stamp each chat response with a UTC timestamp
from typing import Any, List, Optional  # type hints for readability/IDE support (not all are used, kept for future schema fields)

from dotenv import load_dotenv, find_dotenv  # loads OPENAI_API_KEY / MODEL_PROVIDER from the shared .env file
from fastapi import FastAPI, HTTPException, UploadFile, File  # FastAPI: the web framework/app object; HTTPException: standardized error responses; UploadFile/File: handle multipart file uploads
from fastapi.middleware.cors import CORSMiddleware  # lets the frontend (different origin/port) call this API without being blocked by the browser
from pydantic import BaseModel  # base class for request/response schemas with automatic validation
from strands import Agent, tool  # Agent: the LLM agent runtime that orchestrates the model + tools; tool: decorator to expose a Python function as something the agent can call
from strands.models.ollama import OllamaModel  # wraps a locally-hosted Ollama model so the Agent can send it prompts
from strands.models.openai import OpenAIModel  # wraps an OpenAI model so the Agent can send it prompts
from strands_tools import calculator, current_time  # pre-built tools (math, current time) bundled with strands_tools

from rag_manager import RAGManager  # custom module that manages the Retrieval-Augmented-Generation knowledge base (embeddings/vector search)

load_dotenv(find_dotenv(usecwd=True))  # picks up the shared c:\AI Agent study\.env (searches this dir and parents)

# Initialize modular RAG Manager
rag_manager = RAGManager()  # single shared instance used across all requests to store/search uploaded documents

# ---------------------------------------------------------------------------
# Model provider toggle — set MODEL_PROVIDER=openai or MODEL_PROVIDER=ollama
# in the shared .env file (c:\AI Agent study\.env) to switch models.
# ---------------------------------------------------------------------------
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama").strip().lower()
OPENAI_MODEL_ID = os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-nano")
OLLAMA_MODEL_ID = os.getenv("OLLAMA_LLM_MODEL", "gemma4:31b-cloud")
OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

if MODEL_PROVIDER == "openai":
    model = OpenAIModel(
        client_args={"api_key": os.getenv("OPENAI_API_KEY", "")},
        model_id=OPENAI_MODEL_ID,
    )
    ACTIVE_MODEL_NAME = OPENAI_MODEL_ID
else:
    # Cloud-routed Ollama models fail with streaming; use additional_args to disable it
    model = OllamaModel(
        model_id=OLLAMA_MODEL_ID,  # which model Ollama should route the request to
        host=OLLAMA_HOST,  # local Ollama server address the SDK talks to (Ollama itself proxies to the cloud model)
        additional_args={"stream": False},  # disables token streaming; required for cloud-routed models since the agent code below expects one complete response, not a stream
    )
    ACTIVE_MODEL_NAME = OLLAMA_MODEL_ID


# ---------------------------------------------------------------------------
# Custom tools
# ---------------------------------------------------------------------------
@tool  # registers this function as an agent-callable tool; the docstring below is what the LLM reads to decide when/how to call it
def letter_counter(word: str, letter: str) -> int:
    """
    Count occurrences of a specific letter in a word.

    Args:
        word (str): The input word to search in
        letter (str): The specific letter to count

    Returns:
        int: The number of occurrences of the letter in the word
    """
    if not isinstance(word, str) or not isinstance(letter, str):
        return 0  # guard against the LLM passing malformed/non-string args instead of raising
    if len(letter) != 1:
        raise ValueError("The 'letter' parameter must be a single character")  # enforce single-character contract implied by the docstring
    return word.lower().count(letter.lower())  # case-insensitive count


@tool  # exposes RAG search as a callable tool so the agent can answer questions using uploaded documents
def search_uploaded_documents(query: str) -> str:
    """
    Search the user's uploaded knowledge base documents, notes, PDFs, or text files.

    Args:
        query (str): The search topic or question regarding uploaded documents.

    Returns:
        str: Relevant document excerpts retrieved from the vector database.
    """
    return rag_manager.search(query)  # delegates to the vector-store search implemented in rag_manager.py


SYSTEM_PROMPT = """You are a warm, friendly, intelligent, and natural AI assistant.
Respond in a conversational, engaging, and human-like manner:
- Be helpful, clear, authentic, and empathetic.
- Keep your tone natural and conversational, avoiding stiff or overly robotic phrasing.
- Provide concise answers or emails when requested.
- Use available tools naturally when needed. When answering questions about user-uploaded documents or files, search using `search_uploaded_documents`.
- Seamlessly integrate tool results into your response without over-explaining technical tool mechanics.
"""  # instructions prepended to every conversation, steering the model's tone and telling it when to use the RAG tool

agent = Agent(
    model=model,  # the LLM backend defined above (OpenAI or Ollama, per MODEL_PROVIDER)
    tools=[calculator, current_time, letter_counter, search_uploaded_documents],  # full set of functions the agent is allowed to invoke
    system_prompt=SYSTEM_PROMPT,
)  # single shared Agent instance reused for every /api/chat request

TOOL_NAMES = {"calculator", "current_time", "letter_counter", "search_uploaded_documents"}  # whitelist used later to validate tool names parsed out of captured stdout


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="AI Agent API with Modular RAG", version="1.1.0")  # creates the ASGI application; title/version show up in the auto-generated OpenAPI docs at /docs

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # accept requests from any origin (fine for local dev; would need tightening in production)
    allow_credentials=True,  # allow cookies/auth headers to be sent cross-origin
    allow_methods=["*"],  # allow all HTTP methods (GET, POST, DELETE, etc.)
    allow_headers=["*"],  # allow all request headers
)  # without this, a browser-based frontend on a different port would be blocked from calling this API


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str  # the user's chat input; FastAPI validates the incoming JSON body has this field


class ChatResponse(BaseModel):
    response: str  # the agent's final text reply
    tools_used: list[str]  # names of tools the agent invoked while producing the reply (for UI transparency/debugging)
    timestamp: str  # ISO-8601 UTC timestamp of when the response was generated


class RAGToggleRequest(BaseModel):
    enabled: bool  # request body for turning the RAG knowledge base on/off


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")  # simple liveness/readiness check endpoint
def health():
    return {
        "status": "ok",
        "model_provider": MODEL_PROVIDER,
        "model": ACTIVE_MODEL_NAME,  # reports whichever model is actually configured for the active provider
        "rag_enabled": rag_manager.is_enabled()  # lets the frontend know at a glance whether document search is active
    }


@app.get("/api/rag/status")  # returns details about the knowledge base (e.g. doc count, enabled flag) for the frontend to display
def get_rag_status():
    return rag_manager.get_status()


@app.post("/api/rag/toggle")  # lets the frontend turn RAG on/off without restarting the server
def toggle_rag(req: RAGToggleRequest):
    rag_manager.set_enabled(req.enabled)
    return rag_manager.get_status()  # return the updated status so the frontend can refresh its UI in one round trip


@app.post("/api/upload-document")  # handles file uploads to add documents into the RAG knowledge base
async def upload_document(file: UploadFile = File(...)):  # async because reading the upload stream is an I/O-bound operation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")  # reject requests with no file attached

    content_bytes = await file.read()  # read the entire uploaded file into memory
    if not content_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")  # reject empty files early, before touching the RAG pipeline

    try:
        res = rag_manager.add_document(file.filename, content_bytes)  # parses, chunks, embeds, and stores the document
        return {
            "status": "success",
            "message": f"Successfully ingested '{file.filename}'",
            "file": res,  # metadata about the ingested file (e.g. chunk count) returned by RAGManager
            "rag_status": rag_manager.get_status()  # include fresh status so the frontend doesn't need a second call
        }
    except Exception as err:
        # any failure during parsing/embedding is surfaced as a 500 with the underlying error message
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(err)}")


@app.delete("/api/rag/documents")  # wipes all ingested documents from the knowledge base
def clear_rag_documents():
    return rag_manager.clear_documents()


@app.post("/api/chat", response_model=ChatResponse)  # main chat endpoint; response_model validates/serializes the return value as ChatResponse
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")  # reject blank/whitespace-only messages before invoking the LLM

    # Capture stdout to detect which tools were invoked
    buffer = io.StringIO()  # in-memory buffer to hold whatever the agent prints while running
    with redirect_stdout(buffer):  # strands' Agent prints tool-invocation logs (e.g. "Tool #1: calculator") to stdout as it runs;
        result = agent(req.message)  # this temporarily reroutes that stdout into `buffer` instead of the server's real console, so it can be parsed below

    stdout_text = buffer.getvalue()  # extract everything that was printed during the agent call, as a single string

    # Parse tool names from captured stdout ("Tool #N: tool_name")
    tools_used: list[str] = []
    for line in stdout_text.splitlines():  # walk each captured line of output
        line = line.strip()
        if line.startswith("Tool #"):  # strands logs tool calls in the format "Tool #<N>: <tool_name>"
            parts = line.split(":", 1)  # split into ["Tool #N", " tool_name"] on the first colon
            if len(parts) == 2:
                tool_name = parts[1].strip()
                if tool_name in TOOL_NAMES:  # only keep names that match our known tool set, guarding against false positives in the log text
                    tools_used.append(tool_name)

    response_text = str(result).strip()  # the agent's return value isn't guaranteed to be a plain str, so coerce it and trim whitespace

    return ChatResponse(
        response=response_text,
        tools_used=tools_used,
        timestamp=datetime.now(timezone.utc).isoformat(),  # current UTC time in ISO-8601 format, e.g. "2026-08-03T12:34:56+00:00"
    )
