"""
===================================================================
 TUTORIAL PART 2: RAG AI AGENT (WITH DYNAMIC TOOL CALLING)
 Frameworks: LangChain (for VectorStore/RAG) + Strands / Ollama (for Agent)
 Purpose: Demonstrating how an AI Agent dynamically uses a RAG Tool
===================================================================
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from dotenv import load_dotenv, find_dotenv
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from strands import Agent, tool
    from strands.models.ollama import OllamaModel
    from strands.models.openai import OpenAIModel
except ImportError:
    print("[Error] Missing required libraries. Please run in your venv:")
    print("pip install langchain langchain-community langchain-core chromadb langchain-ollama langchain-openai strands-agents python-dotenv")
    sys.exit(1)

load_dotenv(find_dotenv(usecwd=True))  # picks up the shared c:\AI Agent study\.env

# ---------------------------------------------------------------------------
# Model provider toggle — set MODEL_PROVIDER=openai or MODEL_PROVIDER=ollama
# in the shared .env file (c:\AI Agent study\.env) to switch models.
# ---------------------------------------------------------------------------
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama").strip().lower()
OPENAI_MODEL_ID = os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-nano")
OLLAMA_MODEL_ID = os.getenv("OLLAMA_LLM_MODEL", "gemma4:31b-cloud")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

if MODEL_PROVIDER == "openai":
    from langchain_openai import OpenAIEmbeddings
else:
    from langchain_ollama import OllamaEmbeddings


# Global retriever reference
retriever = None


def init_vectorstore():
    """Builds VectorStore and initializes retriever"""
    global retriever
    data_file = os.path.join(os.path.dirname(__file__), "sample_knowledge.txt")

    print("📌 Loading knowledge document & building Chroma VectorStore...")
    loader = TextLoader(data_file, encoding="utf-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    if MODEL_PROVIDER == "openai":
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, openai_api_key=os.getenv("OPENAI_API_KEY", ""))
    else:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    print("   --> RAG Vector Database index ready!")


# Define the RAG Tool using Strands @tool decorator
@tool
def search_campus_policy(query: str) -> str:
    """
    Search engineering campus rules, examination attendance requirements, lab hours, capstone guidelines, and scholarships.

    Args:
        query (str): The search term or policy question.

    Returns:
        str: Relevant policy snippets retrieved from the vector database.
    """
    if retriever is None:
        return "Error: Vector database retriever is not initialized."

    print(f"\n   🛠️ [AGENT TOOL CALLED] search_campus_policy(query='{query}')")
    docs = retriever.invoke(query)
    results = "\n\n".join(f"- {d.page_content.replace('\n', ' ')}" for d in docs)
    return results


def main():
    print("=========================================================")
    print("        🤖 LESSON 2: RAG REASONING AI AGENT             ")
    print("=========================================================\n")

    # Initialize RAG Vector Database
    init_vectorstore()

    # Configure the LLM model for the Agent (OpenAI or Ollama, per MODEL_PROVIDER)
    if MODEL_PROVIDER == "openai":
        model = OpenAIModel(
            client_args={"api_key": os.getenv("OPENAI_API_KEY", "")},
            model_id=OPENAI_MODEL_ID,
        )
    else:
        model = OllamaModel(
            model_id=OLLAMA_MODEL_ID,
            host=OLLAMA_BASE_URL,
            additional_args={"stream": False},
        )

    # Initialize Agent with RAG search tool
    print("📌 Initializing AI Agent with RAG Tool...")
    agent = Agent(
        model=model,
        tools=[search_campus_policy],
    )

    print("\n=========================================================")
    print("🤖 RAG AGENT IS READY! Try asking questions:")
    print("  • 'What is the attendance requirement?' (Triggers RAG Tool)")
    print("  • 'Hi, what is 15 + 25?' (Does NOT call RAG Tool - pure LLM)")
    print("=========================================================\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye! 👋")
            break

        agent(user_input)
        print()


if __name__ == "__main__":
    main()
