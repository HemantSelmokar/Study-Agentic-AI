"""
===================================================================
 TUTORIAL: Building a Retrieval-Augmented Generation (RAG) System
 Framework: LangChain
 Purpose: Lecture demonstration for Engineering Students
===================================================================

WHAT IS RAG (Retrieval-Augmented Generation)?
---------------------------------------------
Standard LLMs answer based ONLY on their pre-trained parameters.
RAG enhances LLMs by:
1. Retrieval: Searching an external Knowledge Base (e.g. private docs, PDFs, database).
2. Augmentation: Inserting relevant content into the LLM prompt.
3. Generation: Producing an accurate answer backed by real factual source text.

5 STAGES OF RAG:
[ Document ] ──> [ 1. Chunking ] ──> [ 2. Embedding ] ──> [ 3. VectorStore ]
                                                              │ (Query match)
[ User Question ] ───────────────────────────────────────────> [ 4. Retrieval ] ──> [ 5. Generation (LLM) ]
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows terminal unicode support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ------------------------------------------------------------------
# Step 0: Package check & Imports
# ------------------------------------------------------------------
try:
    from dotenv import load_dotenv, find_dotenv
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
except ImportError:
    print("[Error] Missing required libraries. Please run:")
    print("pip install langchain langchain-community langchain-core chromadb langchain-ollama langchain-openai python-dotenv")
    sys.exit(1)

load_dotenv(find_dotenv(usecwd=True))  # picks up the shared c:\AI Agent study\.env

# ---------------------------------------------------------------------------
# Model provider toggle — set MODEL_PROVIDER=openai or MODEL_PROVIDER=ollama
# in the shared .env file (c:\AI Agent study\.env) to switch models.
# ---------------------------------------------------------------------------
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama").strip().lower()
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-nano")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gemma4:31b-cloud")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

if MODEL_PROVIDER == "openai":
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
else:
    from langchain_ollama import OllamaEmbeddings, ChatOllama


def main():
    print("=========================================================")
    print("      🚀 LESSON 1: RAG PIPELINE WITH LANGCHAIN          ")
    print("=========================================================\n")

    # Path to sample text file
    data_file = os.path.join(os.path.dirname(__file__), "sample_knowledge.txt")

    # --------------------------------------------------------------
    # STEP 1: Load Document (Data Ingestion)
    # --------------------------------------------------------------
    print("📌 STEP 1: Loading Source Document...")
    loader = TextLoader(data_file, encoding="utf-8")
    docs = loader.load()
    print(f"   --> Loaded {len(docs)} document. Total characters: {len(docs[0].page_content)}")

    # --------------------------------------------------------------
    # STEP 2: Text Chunking (Document Splitting)
    # --------------------------------------------------------------
    print("\n📌 STEP 2: Chunking Text into smaller segments...")
    # Why chunking? LLMs have context limits, and vector search works better on small focused paragraphs.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,       # Max characters per chunk
        chunk_overlap=50,     # Overlap ensures contextual continuity between adjacent chunks
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"   --> Split document into {len(chunks)} chunks.")
    print(f"   --> Sample Chunk #1 preview:\n       \"{chunks[0].page_content.strip()[:100]}...\"\n")

    # --------------------------------------------------------------
    # STEP 3: Vector Embeddings & Vector Database Storage
    # --------------------------------------------------------------
    print("📌 STEP 3: Generating Embeddings & Storing in Vector Store (ChromaDB)...")
    print(f"   --> Model provider: {MODEL_PROVIDER.upper()}")
    embeddings = None

    if MODEL_PROVIDER == "openai":
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, openai_api_key=os.getenv("OPENAI_API_KEY", ""))
        print(f"   --> Using embedding model: {OPENAI_EMBEDDING_MODEL}")
    else:
        for model_name in [OLLAMA_EMBEDDING_MODEL, OLLAMA_LLM_MODEL]:
            try:
                emb = OllamaEmbeddings(model=model_name, base_url=OLLAMA_BASE_URL)
                emb.embed_query("test")
                embeddings = emb
                print(f"   --> Using embedding model: {model_name}")
                break
            except Exception:
                continue

    if embeddings is None:
        print("❌ No embedding model found. Pulling nomic-embed-text via Ollama...")
        sys.exit(1)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="engineering_faq"
    )
    print("   --> Vector Database index successfully created!")

    # --------------------------------------------------------------
    # STEP 4: Retriever Setup
    # --------------------------------------------------------------
    # The retriever turns the VectorStore into an object that accepts queries and returns top-K documents.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Top 2 most relevant chunks

    # --------------------------------------------------------------
    # STEP 5: Define RAG Prompt & LLM Chain
    # --------------------------------------------------------------
    print("\n📌 STEP 5: Setting up RAG Prompt Template & LLM Chain...")

    # Initialize LLM (OpenAI or Ollama, per MODEL_PROVIDER)
    if MODEL_PROVIDER == "openai":
        llm = ChatOpenAI(
            model=OPENAI_LLM_MODEL,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        )
    else:
        llm = ChatOllama(
            model=OLLAMA_LLM_MODEL,  # Or llama3, mistral, gemma, etc.
            temperature=0,             # Low temperature for factual RAG output
            base_url=OLLAMA_BASE_URL,
        )

    # Prompt Engineering for RAG: Restrict LLM to use ONLY provided context
    system_prompt_template = """
You are an AI assistant for Engineering Students. Answer the question using ONLY the provided context below.
If the answer is not mentioned in the context, reply "I do not have this information in my knowledge base."

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(system_prompt_template)

    # Helper function to format retrieved documents into single context string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LangChain LCEL (LangChain Expression Language) Chain:
    # 1. RunnablePassthrough forwards question to retriever
    # 2. Retrieved docs are formatted by format_docs
    # 3. Prompt formats context + question
    # 4. LLM processes prompt
    # 5. Output parser converts response to string
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("   --> RAG Pipeline compiled successfully!\n")

    # --------------------------------------------------------------
    # STEP 6: Interactive Terminal Query Loop
    # --------------------------------------------------------------
    print("=========================================================")
    print("🎓 RAG AGENT READY! Test questions for your lecture:")
    print("  1. What is the attendance requirement for exams?")
    print("  2. Where is the Robotics lab located?")
    print("  3. What is the tuition fee discount for Dean's scholarship?")
    print("  4. Who won the 2024 World Cup? (Tests out-of-scope response)")
    print("=========================================================\n")

    while True:
        try:
            query = input("\n❓ Ask Question (type 'exit' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Class dismissed! 👋")
            break

        print("\n🔍 Step 1 [Retrieval]: Searching vector database for relevant chunks...")
        retrieved_docs = retriever.invoke(query)
        for idx, doc in enumerate(retrieved_docs, 1):
            print(f"   [Chunk #{idx}]: {doc.page_content.replace('\n', ' ')}")

        print("\n🤖 Step 2 [Generation]: LLM synthesizing answer based on retrieved context...")
        response = rag_chain.invoke(query)
        print("\n💡 AI ANSWER:")
        print(response)


if __name__ == "__main__":
    main()
